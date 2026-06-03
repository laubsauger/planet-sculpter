// Sim orchestrator (T13). Owns water depth `d` + flux `f` fields, references
// the height fields `b`. Runs pipe-model passes in fixed order (V7):
// addWater -> flux -> depth (+evaporation). Each pass: read main -> write
// scratch -> copy back (V2). Phases run across all faces before the next phase.

import type { WebGPURenderer } from 'three/webgpu';
import { FACES, type FaceName } from '../config';
import { type HeightFields, FieldSet, buildCopyCompute, buildFillZero } from './fields';
import { buildAddWater, buildFlux, buildDepth, waterUniforms } from './passes/water';
import { SeamSync } from './passes/seamCopy';
import { buildSeamTable } from '../planet/seamTable';
import type { SimHooks } from '../app/Engine';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

interface FacePasses {
  addWater: ComputeNode;
  copyD1: ComputeNode;
  flux: ComputeNode;
  copyF: ComputeNode;
  depth: ComputeNode;
  copyD2: ComputeNode;
}

export class Simulation implements SimHooks {
  readonly water: FieldSet; // depth d
  readonly flux: FieldSet; // rgba L,R,T,B
  private readonly passes = new Map<FaceName, FacePasses>();
  /** Diffuses water across coincident face-edge cells -> cross-seam flow (V5). */
  private readonly waterSeam: SeamSync;

  constructor(
    private readonly renderer: WebGPURenderer,
    height: HeightFields,
  ) {
    const n = height.n;
    this.water = new FieldSet(n, false);
    this.flux = new FieldSet(n, true);
    this.waterSeam = new SeamSync(this.water, buildSeamTable(n - 1), n);

    for (const face of FACES) {
      const b = height.field(face).main;
      const d = this.water.field(face);
      const f = this.flux.field(face);
      this.passes.set(face, {
        addWater: buildAddWater(d.main, d.scratch, n),
        copyD1: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
        flux: buildFlux(b, d.main, f.main, f.scratch, n),
        copyF: buildCopyCompute(f.scratch, f.main, n) as ComputeNode,
        depth: buildDepth(d.main, f.main, d.scratch, n),
        copyD2: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
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
    // phase 4: diffuse water across face seams (continuity + cross-seam flow).
    this.waterSeam.sync(r);
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
