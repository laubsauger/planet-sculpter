// Sim orchestrator (T13). Owns water depth `d` + flux `f` fields, references
// the height fields `b`. Runs pipe-model passes in fixed order (V7):
// addWater -> flux -> depth (+evaporation). Each pass: read main -> write
// scratch -> copy back (V2). Phases run across all faces before the next phase.

import type { WebGPURenderer } from 'three/webgpu';
import { FACES, type FaceName } from '../config';
import { type HeightFields, FieldSet, buildCopyCompute, buildFillZero } from './fields';
import { buildFlux, buildFluidUpdate, buildSurfaceCombine, waterUniforms } from './passes/water';
import {
  buildVelocity,
  buildErosion,
  buildAdvect,
  buildThermal,
  erosionUniforms,
} from './passes/erosion';
import { SeamSync, NormalSeamSync, NormalBandSmooth, VelocityMagSeam } from './passes/seamCopy';
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
  copyViz: ComputeNode;
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
  readonly erosionViz: FieldSet; // rg: fresh erosion / deposition, decaying (terrain tint)
  readonly hardness: FieldSet; // static per-cell erosion-resistance variation
  readonly rainfall: FieldSet; // static regional rainfall map (climate zones)
  readonly cellArea: FieldSet; // static per-cell area (mean 1); depth = vol/area
  readonly velocity: FieldSet; // rgba (vx,vy)
  readonly waterNormals: FieldSet; // baked object-space normals of (b+d)
  readonly waterSurface: FieldSet; // b + vol/area; normal bake samples this (1 tex/face)
  private readonly waterBaker: NormalBaker;
  private readonly waterNormalSeam: NormalSeamSync;
  private readonly waterNormalBand: NormalBandSmooth;
  private readonly surfaceCombine: ComputeNode[] = [];
  erosionEnabled = false;
  private readonly passes = new Map<FaceName, FacePasses>();
  /** Cross-seam water flux (flow across faces) + edge averaging (exact edge
   *  continuity so the water surface has no seam, mirrors terrain's b sync). */
  private readonly waterSeamFlux: SeamFlux;
  private readonly waterSeam: SeamSync;
  private readonly heightSeam: SeamSync;
  private readonly looseSeam: SeamSync;
  private readonly sedimentSeam: SeamSync;
  private readonly velSeam: VelocityMagSeam;
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
    this.erosionViz = new FieldSet(n, true);
    this.hardness = new FieldSet(n, false);
    this.rainfall = new FieldSet(n, false);
    this.cellArea = new FieldSet(n, false);
    this.velocity = new FieldSet(n, true);
    this.waterSurface = new FieldSet(n, false); // b + vol/area (built each tick)
    this.waterSeamFlux = new SeamFlux(
      this.waterSurface,
      this.water,
      this.sediment,
      table,
      n,
      waterUniforms,
    );
    this.waterSeam = new SeamSync(this.water, table, n);
    this.heightSeam = new SeamSync(this.height, table, n);
    this.looseSeam = new SeamSync(this.loose, table, n);
    this.sedimentSeam = new SeamSync(this.sediment, table, n);
    this.velSeam = new VelocityMagSeam(this.velocity, table, n);

    this.waterNormals = new FieldSet(n, true);
    const sampleB: SampleFace = (f, coord) => textureLoad(height.field(f).main, coord).x;
    // surface = b + depth (vol/area) precomputed into one tex/face (combine pass
    // below) so the cross-seam normal bake samples 1 tex/face, not 3 (>16 limit).
    const sampleWater: SampleFace = (f, coord) => textureLoad(this.waterSurface.field(f).main, coord).x;
    this.waterBaker = new NormalBaker(sampleWater, table, this.waterNormals, n);
    this.waterNormalSeam = new NormalSeamSync(this.waterNormals, table, n);
    this.waterNormalBand = new NormalBandSmooth(this.waterNormals, n);
    for (const face of FACES) {
      this.surfaceCombine.push(
        buildSurfaceCombine(
          height.field(face).main,
          this.water.field(face).main,
          this.cellArea.field(face).main,
          this.waterSurface.field(face).main,
          n,
        ),
      );
    }

    for (const face of FACES) {
      const b = height.field(face);
      const d = this.water.field(face);
      const f = this.flux.field(face);
      const s = this.sediment.field(face);
      const vel = this.velocity.field(face);
      const area = this.cellArea.field(face).main;
      this.passes.set(face, {
        flux: buildFlux(b.main, d.main, f.main, f.scratch, n, waterUniforms, area, s.main),
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
          area,
        ),
        copyD: buildCopyCompute(d.scratch, d.main, n) as ComputeNode,
        velocity: buildVelocity(f.main, d.main, vel.scratch, n, area, vel.main),
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
          this.erosionViz.field(face).main,
          this.erosionViz.field(face).scratch,
          n,
          face,
          table,
          sampleB,
          area,
        ),
        copyB: buildCopyCompute(b.scratch, b.main, n) as ComputeNode,
        copyLoose: buildCopyCompute(
          this.loose.field(face).scratch,
          this.loose.field(face).main,
          n,
        ) as ComputeNode,
        copyS1: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
        copyViz: buildCopyCompute(
          this.erosionViz.field(face).scratch,
          this.erosionViz.field(face).main,
          n,
        ) as ComputeNode,
        advect: buildAdvect(vel.main, s.scratch, n, face, table, (f, coord) =>
          textureLoad(this.sediment.field(f).main, coord).x,
        ),
        copyS2: buildCopyCompute(s.scratch, s.main, n) as ComputeNode,
        thermal: buildThermal(b.main, b.scratch, n, face, table, sampleB, d.main, area),
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
    const doErosion = this.erosionEnabled && this.tickCount % 4 === 0;
    erosionUniforms.dt.value = dt * 4;

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
    // velocity runs only on erosion ticks (perf): flow-viz streaks are off by
    // default; when on they tolerate the every-4th-tick velocity update.
    if (doErosion) {
      for (const p of this.passes.values()) {
        r.compute(p.velocity);
        r.compute(p.copyVel);
      }
      // make flow SPEED consistent across seams so the shared edge erodes equally
      // (⊥ trench/pileup along the seam from per-face velocity divergence).
      this.velSeam.sync(r);
      for (const p of this.passes.values()) {
        r.compute(p.erosion);
        r.compute(p.copyB);
        r.compute(p.copyLoose);
        r.compute(p.copyS1);
        r.compute(p.copyViz);
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
    // refresh surface (b + vol/area) first: seamFlux reads it (post flux/erosion).
    for (const c of this.surfaceCombine) r.compute(c);
    this.waterSeamFlux.sync(r);
    this.waterSeam.sync(r);
    if (doErosion) {
      this.heightSeam.sync(r); // bedrock changed -> keep edge continuous
      this.looseSeam.sync(r); // keep loose-material (color) continuous across seams
      this.sedimentSeam.sync(r); // suspended sediment can't cross seam in advect ->
      // edge-average it so it doesn't pile up + deposit a ridge along the seam.
      this.terrainChanged = true;
    }

    // phase 6: rebake water-surface normals (throttled).
    if (this.tickCount % 3 === 0) {
      for (const c of this.surfaceCombine) r.compute(c); // b + vol/area -> surface tex
      this.waterBaker.bake(r);
      this.waterNormalBand.sync(r);
      this.waterNormalSeam.sync(r);
    }
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
    for (const c of this.surfaceCombine) r.compute(c);
    this.waterBaker.bake(r);
    this.waterNormalBand.sync(r);
    this.waterNormalSeam.sync(r);
  }
}
