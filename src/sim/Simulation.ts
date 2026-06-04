// Sim orchestrator (T13). Owns water depth `d` + flux `f` fields, references
// the height fields `b`. Runs pipe-model passes in fixed order (V7):
// addWater -> flux -> depth (+evaporation). Each pass: read main -> write
// scratch -> copy back (V2). Phases run across all faces before the next phase.

import type { WebGPURenderer } from 'three/webgpu';
import { FACES, type FaceName } from '../config';
import { type HeightFields, FieldSet, buildCopyCompute, buildFillZero } from './fields';
import { buildFlux, buildFluidUpdate, waterUniforms } from './passes/water';
import {
  buildVelocity,
  buildErosion,
  buildAdvect,
  buildThermal,
  erosionUniforms,
} from './passes/erosion';
import { SeamSync } from './passes/seamCopy';
import { SeamFlux } from './passes/seamFlux';
import { NormalBaker } from './passes/normals';
import { buildSeamTable, type SeamTable } from '../planet/seamTable';
import { textureLoad } from 'three/tsl';
import type { SampleFace } from '../tsl/surface';
import type { SimHooks } from '../app/Engine';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

interface FacePasses {
  flux: ComputeNode;
  copyF: ComputeNode;
  update: ComputeNode; // fused flow + loss + source + sea-fill
  copyD: ComputeNode;
  // erosion (M5)
  velocity: ComputeNode;
  copyVel: ComputeNode;
  erosion: ComputeNode;
  copyB: ComputeNode;
  copyLoose: ComputeNode;
  copyS1: ComputeNode;
  advect: ComputeNode;
  copyS2: ComputeNode;
  thermal: ComputeNode;
}

export class Simulation implements SimHooks {
  readonly water: FieldSet; // depth d
  readonly waterSource: FieldSet; // per-texel emission rate (river sources)
  readonly flux: FieldSet; // rgba L,R,T,B
  readonly sediment: FieldSet; // suspended sediment s
  readonly loose: FieldSet; // soft material (soil/sand) thickness atop hard rock
  readonly hardness: FieldSet; // static per-cell erosion-resistance variation
  readonly rainfall: FieldSet; // static regional rainfall map (climate zones)
  readonly velocity: FieldSet; // rgba (vx,vy)
  readonly waterNormals: FieldSet; // baked object-space normals of (b+d)
  private readonly waterBaker: NormalBaker;
  erosionEnabled = false;
  private readonly passes = new Map<FaceName, FacePasses>();
  /** Cross-seam water flux (flow across faces) + edge averaging (exact edge
   *  continuity so the water surface has no seam, mirrors terrain's b sync). */
  private readonly waterSeamFlux: SeamFlux;
  private readonly waterSeam: SeamSync;
  private readonly heightSeam: SeamSync;
  private tickCount = 0;
  /** true on ticks where erosion mutated bedrock (Engine rebakes normals then). */
  terrainChanged = false;

  constructor(
    private readonly renderer: WebGPURenderer,
    private readonly height: HeightFields,
  ) {
    const n = height.n;
    const table: SeamTable = buildSeamTable(n - 1);
    this.water = new FieldSet(n, false);
    this.waterSource = new FieldSet(n, false);
    this.flux = new FieldSet(n, true);
    this.sediment = new FieldSet(n, false);
    this.loose = new FieldSet(n, false);
    this.hardness = new FieldSet(n, false);
    this.rainfall = new FieldSet(n, false);
    this.velocity = new FieldSet(n, true);
    this.waterSeamFlux = new SeamFlux(height, this.water, table, n, waterUniforms);
    this.waterSeam = new SeamSync(this.water, table, n);
    this.heightSeam = new SeamSync(this.height, table, n);

    this.waterNormals = new FieldSet(n, true);
    const sampleWater: SampleFace = (f, coord) =>
      textureLoad(height.field(f).main, coord).x.add(textureLoad(this.water.field(f).main, coord).x);
    this.waterBaker = new NormalBaker(sampleWater, table, this.waterNormals, n);

    for (const face of FACES) {
      const b = height.field(face);
      const d = this.water.field(face);
      const f = this.flux.field(face);
      const s = this.sediment.field(face);
      const vel = this.velocity.field(face);
      this.passes.set(face, {
        flux: buildFlux(b.main, d.main, f.main, f.scratch, n, waterUniforms),
        copyF: buildCopyCompute(f.scratch, f.main, n) as ComputeNode,
        update: buildFluidUpdate(
          d.main,
          f.main,
          b.main,
          this.waterSource.field(face).main,
          this.rainfall.field(face).main,
          d.scratch,
          n,
          waterUniforms,
          true,
        ),
        copyD: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
        velocity: buildVelocity(f.main, d.main, vel.scratch, n),
        copyVel: buildCopyCompute(vel.scratch, vel.main, n) as ComputeNode,
        erosion: buildErosion(
          b.main,
          this.loose.field(face).main,
          s.main,
          vel.main,
          d.main,
          this.hardness.field(face).main,
          this.waterSource.field(face).main,
          b.scratch,
          this.loose.field(face).scratch,
          s.scratch,
          n,
        ),
        copyB: buildCopyCompute(b.scratch, b.main, n) as ComputeNode,
        copyLoose: buildCopyCompute(
          this.loose.field(face).scratch,
          this.loose.field(face).main,
          n,
        ) as ComputeNode,
        copyS1: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
        advect: buildAdvect(s.main, vel.main, s.scratch, n),
        copyS2: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
        thermal: buildThermal(b.main, b.scratch, n),
      });
    }
  }

  /** depth field for a face (sampled by the water material). */
  depthField(face: FaceName) {
    return this.water.field(face);
  }

  setRain(rate: number): void {
    waterUniforms.source.value = rate;
  }

  /** Reset all water depth + flux to zero. */
  clearWater(): void {
    for (const face of FACES) {
      const d = this.water.field(face);
      const f = this.flux.field(face);
      this.renderer.compute(buildFillZero(d.main, d.n));
      this.renderer.compute(buildFillZero(d.scratch, d.n));
      this.renderer.compute(buildFillZero(f.main, f.n));
      this.renderer.compute(buildFillZero(f.scratch, f.n));
    }
  }

  tick(dt: number): void {
    waterUniforms.dt.value = dt;
    const r = this.renderer;
    this.tickCount++;
    this.terrainChanged = false;
    // erosion is gradual -> run it (and its bedrock/sediment seam syncs) every
    // 3rd tick for perf; scale its dt up so erosion speed stays consistent.
    const doErosion = this.erosionEnabled && this.tickCount % 3 === 0;
    erosionUniforms.dt.value = dt * 3;

    // water (every tick for responsive flow): flux, then fused
    // flow+loss+source+sea-fill update.
    for (const p of this.passes.values()) {
      r.compute(p.flux);
      r.compute(p.copyF);
    }
    for (const p of this.passes.values()) {
      r.compute(p.update);
      r.compute(p.copyD);
    }

    // phase 4: hydraulic erosion (throttled) — velocity -> erode/deposit -> advect -> thermal.
    if (doErosion) {
      for (const p of this.passes.values()) {
        r.compute(p.velocity);
        r.compute(p.copyVel);
      }
      for (const p of this.passes.values()) {
        r.compute(p.erosion);
        r.compute(p.copyB);
        r.compute(p.copyLoose);
        r.compute(p.copyS1);
      }
      for (const p of this.passes.values()) {
        r.compute(p.advect);
        r.compute(p.copyS2);
      }
      for (const p of this.passes.values()) {
        r.compute(p.thermal);
        r.compute(p.copyB);
      }
    }

    // phase 5: water cross-seam flux (flow) + edge averaging (exact edge
    // continuity -> no water-surface seam, mirrors terrain's b sync).
    this.waterSeamFlux.sync(r);
    this.waterSeam.sync(r);
    this.heightSeam.sync(r); // always keep bedrock edge continuous (cheap)
    if (doErosion) this.terrainChanged = true;

    // phase 6: rebake water-surface normals (throttled).
    if (this.tickCount % 3 === 0) this.waterBaker.bake(r);
  }

  setErosion(on: boolean): void {
    this.erosionEnabled = on;
  }

  async warmup(): Promise<void> {
    const r = this.renderer;
    for (const p of this.passes.values()) {
      r.compute(p.flux);
      r.compute(p.copyF);
      r.compute(p.update);
      r.compute(p.copyD);
    }
    this.waterBaker.bake(r);
  }
}
