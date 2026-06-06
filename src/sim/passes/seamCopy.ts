// Seam sync (T11, V5). Per face: copy main->scratch for all texels; for
// boundary texels, average with the matching neighbor-face texel(s). A corner
// texel lies on 2 edges -> averages self + its 2 edge-neighbors, which are the
// other 2 faces meeting there -> all 3 faces converge to the same 3-way mean.
// Then scratch->main (buildCopyCompute) so `main` stays canonical (V2).

import type { WebGPURenderer } from 'three/webgpu';
import {
  Fn,
  instanceIndex,
  textureLoad,
  textureStore,
  ivec2,
  uvec2,
  uint,
  int,
  float,
  vec4,
  If,
} from 'three/tsl';
import { normalize, length, max, min, mix, smoothstep } from 'three/tsl';
import { FACES, type FaceName } from '../../config';
import type { FieldProvider } from '../fields';
import { buildCopyCompute } from '../fields';
import { EDGES, type SeamTable, type EdgeId } from '../../planet/seamTable';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

function buildSeamProgram(
  face: FaceName,
  fields: FieldProvider,
  table: SeamTable,
  n: number,
): ComputeNode {
  const res = n - 1;
  const self = fields.field(face);
  const N = uint(n);

  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);
    const last = int(res);

    const val = textureLoad(self.main, ivec2(ix, iy)).x.toVar();
    const cnt = float(1).toVar();

    for (const edge of EDGES) {
      const seam = table[face][edge as EdgeId];
      const nTex = fields.field(seam.nFace).main;

      const cond =
        edge === 0
          ? ix.equal(int(0))
          : edge === 1
            ? ix.equal(last)
            : edge === 2
              ? iy.equal(int(0))
              : iy.equal(last);

      const i = edge === 0 || edge === 1 ? iy : ix;
      const vary = seam.varyReverse ? last.sub(i) : i;
      const ncoord = seam.nFixedIsX
        ? ivec2(int(seam.nFixedVal), vary)
        : ivec2(vary, int(seam.nFixedVal));

      If(cond, () => {
        val.addAssign(textureLoad(nTex, ncoord).x);
        cnt.addAssign(float(1));
      });
    }

    textureStore(self.scratch, uvec2(x, y), vec4(val.div(cnt), 0, 0, 1)).toWriteOnly();
  });

  return fn().compute(n * n) as ComputeNode;
}

// --- vec3 (object-space normal) seam sync ------------------------------------
// Baked per-face normals at a shared edge can disagree (each face uses its own
// finite-difference stencil, degenerate near corners) -> lighting crease. Avg
// the coincident edge normals across faces + renormalize so BOTH sides carry
// the identical direction -> seamless shading. Mirror of buildSeamProgram but
// vector-valued and direction-averaged.
function buildNormalSeamProgram(
  face: FaceName,
  fields: FieldProvider,
  table: SeamTable,
  n: number,
): ComputeNode {
  const res = n - 1;
  const self = fields.field(face);
  const N = uint(n);

  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);
    const last = int(res);

    const acc = textureLoad(self.main, ivec2(ix, iy)).xyz.toVar();

    for (const edge of EDGES) {
      const seam = table[face][edge as EdgeId];
      const nTex = fields.field(seam.nFace).main;
      const cond =
        edge === 0
          ? ix.equal(int(0))
          : edge === 1
            ? ix.equal(last)
            : edge === 2
              ? iy.equal(int(0))
              : iy.equal(last);
      const i = edge === 0 || edge === 1 ? iy : ix;
      const vary = seam.varyReverse ? last.sub(i) : i;
      const ncoord = seam.nFixedIsX
        ? ivec2(int(seam.nFixedVal), vary)
        : ivec2(vary, int(seam.nFixedVal));
      If(cond, () => {
        acc.addAssign(textureLoad(nTex, ncoord).xyz);
      });
    }

    textureStore(self.scratch, uvec2(x, y), vec4(normalize(acc), 1)).toWriteOnly();
  });

  return fn().compute(n * n) as ComputeNode;
}

/** Averages baked object-space normals across face seams (renormalized). */
export class NormalSeamSync {
  private readonly seam = new Map<FaceName, ComputeNode>();
  private readonly copy = new Map<FaceName, ComputeNode>();

  constructor(fields: FieldProvider, table: SeamTable, n: number) {
    for (const face of FACES) {
      this.seam.set(face, buildNormalSeamProgram(face, fields, table, n));
      const f = fields.field(face);
      this.copy.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  sync(renderer: WebGPURenderer): void {
    for (const node of this.seam.values()) renderer.compute(node);
    for (const node of this.copy.values()) renderer.compute(node);
  }
}

/** Builds + runs the seam sync across all 6 faces. */
export class SeamSync {
  private readonly seam = new Map<FaceName, ComputeNode>();
  private readonly copy = new Map<FaceName, ComputeNode>();

  constructor(fields: FieldProvider, table: SeamTable, n: number) {
    for (const face of FACES) {
      this.seam.set(face, buildSeamProgram(face, fields, table, n));
      const f = fields.field(face);
      this.copy.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  /** Two-phase: all faces read main->write scratch, then all scratch->main,
   *  so every seam read sees the pre-sync state. */
  sync(renderer: WebGPURenderer): void {
    for (const node of this.seam.values()) renderer.compute(node);
    for (const node of this.copy.values()) renderer.compute(node);
  }
}

// --- velocity MAGNITUDE seam sync --------------------------------------------
// Edge cells take their own face's first-interior velocity, so the SHARED edge
// gets different flow speed on each face -> erodes by different amounts -> a
// trench + pileup along the seam. Speed (|v|) is frame-independent, so average
// it across the coincident edge cells (keep each face's local direction) -> the
// edge erodes identically on both sides -> no seam trench.
function buildVelMagSeam(face: FaceName, fields: FieldProvider, table: SeamTable, n: number): ComputeNode {
  const res = n - 1;
  const self = fields.field(face);
  const N = uint(n);
  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);
    const last = int(res);

    const sv = textureLoad(self.main, ivec2(ix, iy)).xy;
    const selfMag = length(sv);
    const magSum = selfMag.toVar();
    const cnt = float(1).toVar();

    for (const edge of EDGES) {
      const seam = table[face][edge as EdgeId];
      const nTex = fields.field(seam.nFace).main;
      const cond =
        edge === 0
          ? ix.equal(int(0))
          : edge === 1
            ? ix.equal(last)
            : edge === 2
              ? iy.equal(int(0))
              : iy.equal(last);
      const i = edge === 0 || edge === 1 ? iy : ix;
      const vary = seam.varyReverse ? last.sub(i) : i;
      const ncoord = seam.nFixedIsX
        ? ivec2(int(seam.nFixedVal), vary)
        : ivec2(vary, int(seam.nFixedVal));
      If(cond, () => {
        magSum.addAssign(length(textureLoad(nTex, ncoord).xy));
        cnt.addAssign(float(1));
      });
    }

    // rescale this cell's velocity to the averaged speed (direction preserved).
    const dir = sv.div(max(selfMag, float(1e-6)));
    const out = dir.mul(magSum.div(cnt));
    textureStore(self.scratch, uvec2(x, y), vec4(out.x, out.y, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

// --- normal band smooth (shading-only crease fix) ----------------------------
// The shared EDGE normal is made consistent by NormalSeamSync, but the 1-2 cells
// INWARD on each face reflect that face's own slope -> a shading crease flanking
// the seam. Diffuse the baked normal in a narrow band toward its neighbor avg
// (Laplacian, edge-proximity weighted) so the shading transitions smoothly. This
// touches ONLY the normal texture -> can't gap or groove (no height change).
function buildNormalBandSmooth(face: FaceName, fields: FieldProvider, n: number): ComputeNode {
  const res = n - 1;
  const self = fields.field(face);
  const N = uint(n);
  const band = 3.0;
  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);
    const cur = textureLoad(self.main, ivec2(ix, iy)).xyz;
    const out = cur.toVar();
    const edist = min(
      min(ix.toFloat(), float(res).sub(ix.toFloat())),
      min(iy.toFloat(), float(res).sub(iy.toFloat())),
    );
    If(edist.lessThan(float(band)), () => {
      const cl = (a: any) => a.toFloat().max(float(0)).min(float(res)).toInt();
      const s0 = textureLoad(self.main, ivec2(cl(ix.sub(1)), iy)).xyz;
      const s1 = textureLoad(self.main, ivec2(cl(ix.add(1)), iy)).xyz;
      const s2 = textureLoad(self.main, ivec2(ix, cl(iy.sub(1)))).xyz;
      const s3 = textureLoad(self.main, ivec2(ix, cl(iy.add(1)))).xyz;
      const avg = normalize(cur.add(s0).add(s1).add(s2).add(s3));
      const w = smoothstep(float(band), float(0), edist).mul(0.7);
      out.assign(normalize(mix(cur, avg, w)));
    });
    textureStore(self.scratch, uvec2(x, y), vec4(out, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Smooths the baked normal in the seam band (shading-only crease fix). Run
 *  AFTER the bake; NormalSeamSync after locks the exact edge cross-seam. */
export class NormalBandSmooth {
  private readonly smooth = new Map<FaceName, ComputeNode>();
  private readonly copy = new Map<FaceName, ComputeNode>();

  constructor(fields: FieldProvider, n: number) {
    for (const face of FACES) {
      this.smooth.set(face, buildNormalBandSmooth(face, fields, n));
      const f = fields.field(face);
      this.copy.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  sync(renderer: WebGPURenderer): void {
    for (const node of this.smooth.values()) renderer.compute(node);
    for (const node of this.copy.values()) renderer.compute(node);
  }
}

/** Cross-seam velocity-MAGNITUDE averaging so the shared edge erodes equally. */
export class VelocityMagSeam {
  private readonly seam = new Map<FaceName, ComputeNode>();
  private readonly copy = new Map<FaceName, ComputeNode>();

  constructor(fields: FieldProvider, table: SeamTable, n: number) {
    for (const face of FACES) {
      this.seam.set(face, buildVelMagSeam(face, fields, table, n));
      const f = fields.field(face);
      this.copy.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  sync(renderer: WebGPURenderer): void {
    for (const node of this.seam.values()) renderer.compute(node);
    for (const node of this.copy.values()) renderer.compute(node);
  }
}
