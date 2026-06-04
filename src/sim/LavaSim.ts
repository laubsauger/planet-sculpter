// Lava simulation (M8, T17). Reuses the fluid solver (addSource/flux/depth) +
// a cooling pass. Lava flows over bedrock; cold lava solidifies into bedrock
// (builds cones/flows -> modifies shared height -> Engine rebakes normals).
// Lean for perf: no velocity/heat-advection (heat decays in place; vents stay
// hot). Throttled to every 3rd tick at normal dt (slow + stable, not dt-scaled
// which made the flux explode).

import type { WebGPURenderer } from 'three/webgpu';
import { FACES, type FaceName } from '../config';
import { type HeightFields, FieldSet, buildCopyCompute, buildFillZero } from './fields';
import { buildFlux, buildFluidUpdate } from './passes/water';
import { buildLavaCool, lavaUniforms, lavaCool } from './passes/lava';
import { SeamSync } from './passes/seamCopy';
import type { SeamTable } from '../planet/seamTable';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

interface FacePasses {
  flux: ComputeNode;
  copyFlux: ComputeNode;
  update: ComputeNode; // fused flow + source
  copyLava: ComputeNode;
  cool: ComputeNode;
  copyLava3: ComputeNode;
  copyHeat: ComputeNode;
  copyB: ComputeNode;
}

export class LavaSim {
  readonly lava: FieldSet;
  readonly lavaSource: FieldSet;
  readonly heat: FieldSet;
  readonly flux: FieldSet;
  active = false;
  terrainChanged = false;

  private readonly passes = new Map<FaceName, FacePasses>();
  private readonly lavaSeam: SeamSync;
  private readonly heightSeam: SeamSync;
  private tickCount = 0;

  constructor(
    private readonly renderer: WebGPURenderer,
    height: HeightFields,
    table: SeamTable,
  ) {
    const n = height.n;
    this.lava = new FieldSet(n, false);
    this.lavaSource = new FieldSet(n, false);
    this.heat = new FieldSet(n, false);
    this.flux = new FieldSet(n, true);
    this.lavaSeam = new SeamSync(this.lava, table, n);
    this.heightSeam = new SeamSync(height, table, n);

    for (const face of FACES) {
      const b = height.field(face);
      const lv = this.lava.field(face);
      const src = this.lavaSource.field(face).main;
      const ht = this.heat.field(face);
      const f = this.flux.field(face);
      this.passes.set(face, {
        flux: buildFlux(b.main, lv.main, f.main, f.scratch, n, lavaUniforms),
        copyFlux: buildCopyCompute(f.scratch, f.main, n) as ComputeNode,
        update: buildFluidUpdate(lv.main, f.main, b.main, src, src, lv.scratch, n, lavaUniforms, false),
        copyLava: buildCopyCompute(lv.scratch, lv.main, n) as ComputeNode,
        cool: buildLavaCool(lv.main, ht.main, b.main, src, lv.scratch, ht.scratch, b.scratch, n),
        copyLava3: buildCopyCompute(lv.scratch, lv.main, n) as ComputeNode,
        copyHeat: buildCopyCompute(ht.scratch, ht.main, n) as ComputeNode,
        copyB: buildCopyCompute(b.scratch, b.main, n) as ComputeNode,
      });
    }
  }

  tick(dt: number): void {
    this.terrainChanged = false;
    if (!this.active) return;
    this.tickCount = (this.tickCount + 1) % 3;
    if (this.tickCount !== 0) return; // run every 3rd tick (normal dt -> stable)
    lavaUniforms.dt.value = dt;
    lavaCool.dt.value = dt;
    const r = this.renderer;
    const P = [...this.passes.values()];

    for (const p of P) {
      r.compute(p.flux);
      r.compute(p.copyFlux);
    }
    for (const p of P) {
      r.compute(p.update);
      r.compute(p.copyLava);
    }
    for (const p of P) {
      r.compute(p.cool);
      r.compute(p.copyLava3);
      r.compute(p.copyHeat);
      r.compute(p.copyB);
    }
    this.lavaSeam.sync(r);
    this.heightSeam.sync(r); // lava solidified into bedrock
    this.terrainChanged = true;
  }

  clear(): void {
    for (const face of FACES) {
      for (const fld of [this.lavaSource, this.lava, this.heat, this.flux]) {
        const f = fld.field(face);
        this.renderer.compute(buildFillZero(f.main, f.n));
        this.renderer.compute(buildFillZero(f.scratch, f.n));
      }
    }
    this.active = false;
  }
}
