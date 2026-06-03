// Sim orchestrator (T13). Owns water depth `d` + flux `f` fields, references
// the height fields `b`. Runs pipe-model passes in fixed order (V7):
// addWater -> flux -> depth (+evaporation). Each pass: read main -> write
// scratch -> copy back (V2). Phases run across all faces before the next phase.

import type { WebGPURenderer } from 'three/webgpu';
import { FACES, type FaceName } from '../config';
import { type HeightFields, FieldSet, buildCopyCompute, buildFillZero } from './fields';
import { buildAddWater, buildFlux, buildDepth, waterUniforms } from './passes/water';
import { buildVelocity, buildErosion, buildAdvect, erosionUniforms } from './passes/erosion';
import { SeamSync } from './passes/seamCopy';
import { buildSeamTable, type SeamTable } from '../planet/seamTable';
import type { SimHooks } from '../app/Engine';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

interface FacePasses {
  addWater: ComputeNode;
  copyD1: ComputeNode;
  flux: ComputeNode;
  copyF: ComputeNode;
  depth: ComputeNode;
  copyD2: ComputeNode;
  // erosion (M5)
  velocity: ComputeNode;
  copyVel: ComputeNode;
  erosion: ComputeNode;
  copyB: ComputeNode;
  copyS1: ComputeNode;
  advect: ComputeNode;
  copyS2: ComputeNode;
}

export class Simulation implements SimHooks {
  readonly water: FieldSet; // depth d
  readonly flux: FieldSet; // rgba L,R,T,B
  readonly sediment: FieldSet; // suspended sediment s
  readonly velocity: FieldSet; // rgba (vx,vy)
  erosionEnabled = false;
  private readonly passes = new Map<FaceName, FacePasses>();
  /** Diffuses fields across coincident face-edge cells -> cross-seam (V5). */
  private readonly waterSeam: SeamSync;
  private readonly heightSeam: SeamSync;
  private readonly sedimentSeam: SeamSync;

  constructor(
    private readonly renderer: WebGPURenderer,
    private readonly height: HeightFields,
  ) {
    const n = height.n;
    const table: SeamTable = buildSeamTable(n - 1);
    this.water = new FieldSet(n, false);
    this.flux = new FieldSet(n, true);
    this.sediment = new FieldSet(n, false);
    this.velocity = new FieldSet(n, true);
    this.waterSeam = new SeamSync(this.water, table, n);
    this.heightSeam = new SeamSync(this.height, table, n);
    this.sedimentSeam = new SeamSync(this.sediment, table, n);

    for (const face of FACES) {
      const b = height.field(face);
      const d = this.water.field(face);
      const f = this.flux.field(face);
      const s = this.sediment.field(face);
      const vel = this.velocity.field(face);
      this.passes.set(face, {
        addWater: buildAddWater(d.main, d.scratch, n),
        copyD1: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
        flux: buildFlux(b.main, d.main, f.main, f.scratch, n),
        copyF: buildCopyCompute(f.scratch, f.main, n) as ComputeNode,
        depth: buildDepth(d.main, f.main, d.scratch, n),
        copyD2: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
        velocity: buildVelocity(f.main, d.main, vel.scratch, n),
        copyVel: buildCopyCompute(vel.scratch, vel.main, n) as ComputeNode,
        erosion: buildErosion(b.main, s.main, vel.main, d.main, b.scratch, s.scratch, n),
        copyB: buildCopyCompute(b.scratch, b.main, n) as ComputeNode,
        copyS1: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
        advect: buildAdvect(s.main, vel.main, s.scratch, n),
        copyS2: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
      });
    }
  }

  /** depth field for a face (sampled by the water material). */
  depthField(face: FaceName) {
    return this.water.field(face);
  }

  setRain(rate: number): void {
    waterUniforms.rain.value = rate;
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
    erosionUniforms.dt.value = dt;
    const r = this.renderer;
    // phase 1: add water
    for (const p of this.passes.values()) {
      r.compute(p.addWater);
      r.compute(p.copyD1);
    }
    // phase 2: flux
    for (const p of this.passes.values()) {
      r.compute(p.flux);
      r.compute(p.copyF);
    }
    // phase 3: depth + evaporation
    for (const p of this.passes.values()) {
      r.compute(p.depth);
      r.compute(p.copyD2);
    }
    // phase 4: hydraulic erosion (optional) — velocity -> erode/deposit -> advect.
    if (this.erosionEnabled) {
      for (const p of this.passes.values()) {
        r.compute(p.velocity);
        r.compute(p.copyVel);
      }
      for (const p of this.passes.values()) {
        r.compute(p.erosion);
        r.compute(p.copyB);
        r.compute(p.copyS1);
      }
      for (const p of this.passes.values()) {
        r.compute(p.advect);
        r.compute(p.copyS2);
      }
    }

    // phase 5: diffuse across face seams (continuity + cross-seam flow).
    this.waterSeam.sync(r);
    if (this.erosionEnabled) {
      this.heightSeam.sync(r); // erosion modified bedrock -> keep seams continuous
      this.sedimentSeam.sync(r);
    }
  }

  setErosion(on: boolean): void {
    this.erosionEnabled = on;
  }

  async warmup(): Promise<void> {
    const r = this.renderer;
    for (const p of this.passes.values()) {
      r.compute(p.addWater);
      r.compute(p.copyD1);
      r.compute(p.flux);
      r.compute(p.copyF);
      r.compute(p.depth);
      r.compute(p.copyD2);
    }
  }
}
