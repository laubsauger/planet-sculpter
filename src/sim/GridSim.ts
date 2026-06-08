// Equirect single-grid simulation (replaces the 6-face Simulation + all 7 seam
// systems). Stage 3: pipe-model water. Stage 4: hydraulic + thermal erosion +
// flow-aware evaporation. Longitude wraps / latitude clamps — no seam machinery.
// Tick order: flux -> velocity -> water update (flow-aware evap) -> erosion ->
// sediment advect -> thermal slumping.

import type { WebGPURenderer } from 'three/webgpu';
import { Vector3, type Texture } from 'three';
import {
  Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int,
  float, vec3, vec4, length, smoothstep, max, uniform,
} from 'three/tsl';
import { lonLatDirNode } from '../tsl/latlongNode';
import { GridField, buildGridCopy, buildGridFill, buildGridSeed } from './gridStore';
import { gridFlux, gridUpdate } from './gridWater';
import { gridVelocity, gridErosion, gridAdvect, gridThermal } from './gridErosion';
import { waterUniforms } from './passes/water';

/* eslint-disable @typescript-eslint/no-explicit-any */

export class GridSim {
  readonly water: GridField; // depth as VOLUME
  readonly flux: GridField; // rgba L,R,T,B
  readonly velocity: GridField; // rg velocity (drives erosion + evap)
  readonly sediment: GridField; // suspended sediment
  readonly loose: GridField; // soft-material thickness (erodes first)
  readonly source: GridField; // per-texel emitter rate (rivers)
  erosionEnabled = true;

  private readonly fluxN;
  private readonly fluxC;
  private readonly velN;
  private readonly velC;
  private readonly updateN;
  private readonly waterC;
  private readonly erodeN;
  private readonly bC;
  private readonly looseC;
  private readonly sErodeC;
  private readonly advectN;
  private readonly sAdvectC;
  private readonly thermalN;
  private readonly bThermalC;
  // river-source stamp (additive emitter into source field).
  private readonly srcCenter = uniform(new Vector3(1, 0, 0));
  private readonly srcRadius = uniform(0.06);
  private readonly srcRate = uniform(0.0);
  private readonly srcStampN;
  private readonly srcCopyN;

  constructor(
    private readonly renderer: WebGPURenderer,
    height: GridField,
    rainfall: Texture,
    cellArea: Texture,
    looseSeed: Texture,
    hardness: Texture,
    readonly w: number,
    readonly h: number,
  ) {
    this.water = new GridField(w, h);
    this.flux = new GridField(w, h, true);
    this.velocity = new GridField(w, h, true);
    this.sediment = new GridField(w, h);
    this.loose = new GridField(w, h);
    this.source = new GridField(w, h);
    // start empty (water/sediment/source); seed loose from the loose map.
    renderer.compute(buildGridFill(this.water.main, w, h, 0));
    renderer.compute(buildGridFill(this.velocity.main, w, h, 0));
    renderer.compute(buildGridFill(this.sediment.main, w, h, 0));
    renderer.compute(buildGridFill(this.source.main, w, h, 0));
    renderer.compute(buildGridSeed(looseSeed as never, this.loose.main, w, h));

    const b = height.main;
    const area = cellArea as never;
    // --- water ---
    this.fluxN = gridFlux(b, this.water.main, area, this.flux.main, this.flux.scratch, this.sediment.main, w, h, waterUniforms);
    this.fluxC = buildGridCopy(this.flux.scratch, this.flux.main, w, h);
    this.velN = gridVelocity(this.flux.main, this.water.main, area, this.velocity.main, this.velocity.scratch, w, h);
    this.velC = buildGridCopy(this.velocity.scratch, this.velocity.main, w, h);
    this.updateN = gridUpdate(this.water.main, this.flux.main, b, this.source.main, rainfall as never, area, this.velocity.main, this.water.scratch, w, h, waterUniforms);
    this.waterC = buildGridCopy(this.water.scratch, this.water.main, w, h);
    // --- erosion ---
    this.erodeN = gridErosion(b, this.loose.main, this.sediment.main, this.velocity.main, this.water.main, hardness as never, this.source.main, area, height.scratch, this.loose.scratch, this.sediment.scratch, w, h);
    this.bC = buildGridCopy(height.scratch, b, w, h);
    this.looseC = buildGridCopy(this.loose.scratch, this.loose.main, w, h);
    this.sErodeC = buildGridCopy(this.sediment.scratch, this.sediment.main, w, h);
    this.advectN = gridAdvect(this.velocity.main, this.sediment.main, this.sediment.scratch, w, h);
    this.sAdvectC = buildGridCopy(this.sediment.scratch, this.sediment.main, w, h);
    this.thermalN = gridThermal(b, this.water.main, area, height.scratch, w, h);
    this.bThermalC = buildGridCopy(height.scratch, b, w, h);

    // river-source stamp: add a soft blob of emitter rate at a clicked dir.
    const W = w, H = h;
    const stampFn = Fn(() => {
      const x = instanceIndex.mod(uint(W));
      const y = instanceIndex.div(uint(W));
      const ix = int(x), iy = int(y);
      const u = x.toFloat().div(W);
      const v = y.toFloat().div(H - 1);
      const dir = lonLatDirNode(u, v);
      const dist = length(dir.sub(vec3(this.srcCenter)));
      const wgt = float(1).sub(smoothstep(float(0), this.srcRadius, dist));
      const cur = textureLoad(this.source.main, ivec2(ix, iy)).x;
      const out = max(float(0), cur.add(this.srcRate.mul(wgt)));
      textureStore(this.source.scratch, uvec2(x, y), vec4(out, 0, 0, 1)).toWriteOnly();
    });
    this.srcStampN = stampFn().compute(W * H) as any;
    this.srcCopyN = buildGridCopy(this.source.scratch, this.source.main, w, h);
  }

  setRain(rate: number): void {
    waterUniforms.source.value = rate;
  }

  setEvap(rate: number): void {
    waterUniforms.evapProp.value = rate;
  }

  /** Add (rate>0) or erase (rate<0) a river emitter blob at a planet dir. */
  placeSource(centerDir: Vector3, rate: number, radius = 0.06): void {
    this.srcCenter.value.copy(centerDir).normalize();
    this.srcRate.value = rate;
    this.srcRadius.value = radius;
    this.renderer.compute(this.srcStampN);
    this.renderer.compute(this.srcCopyN);
  }

  clearSources(): void {
    this.renderer.compute(buildGridFill(this.source.main, this.w, this.h, 0));
  }

  clearWater(): void {
    const r = this.renderer;
    r.compute(buildGridFill(this.water.main, this.w, this.h, 0));
    r.compute(buildGridFill(this.flux.main, this.w, this.h, 0));
    r.compute(buildGridFill(this.velocity.main, this.w, this.h, 0));
  }

  tick(dt: number): void {
    waterUniforms.dt.value = dt;
    const r = this.renderer;
    // water: flux -> velocity -> fused update (flow-aware evap).
    r.compute(this.fluxN);
    r.compute(this.fluxC);
    r.compute(this.velN);
    r.compute(this.velC);
    r.compute(this.updateN);
    r.compute(this.waterC);
    if (this.erosionEnabled) {
      r.compute(this.erodeN);
      r.compute(this.bC);
      r.compute(this.looseC);
      r.compute(this.sErodeC);
      r.compute(this.advectN);
      r.compute(this.sAdvectC);
      r.compute(this.thermalN);
      r.compute(this.bThermalC);
    }
  }
}
