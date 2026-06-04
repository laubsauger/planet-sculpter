// Shared displaced-surface math (texel-exact, seam-aware).
// The baked normal is computed from the HEIGHT GRADIENT + analytic sphere
// tangents: cross-seam neighbor heights are cheap textureLoads, and only ~5
// faceDirNode evals are needed per texel (vs 25 for the position-difference
// approach) — the key perf lever for higher res + erosion.

import {
  uv,
  ivec2,
  int,
  float,
  cross,
  dot,
  sign,
  normalize,
  modelViewMatrix,
  textureLoad,
} from 'three/tsl';
import type { Texture } from 'three';
import { faceDirNode } from './warpNode';
import { PLANET, type FaceName } from '../config';
import { heightScaleUniform } from './heightScale';
import { type SeamTable, type EdgeId } from '../planet/seamTable';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface Surface {
  position: any;
  viewNormal: any;
  slope: any;
  height: any;
}

/** Height sampler: (faceName, ivec2 texel) -> height float node. */
export type SampleFace = (face: FaceName, coord: any) => any;

/** Height at integer texel (tx,ty) which may be one cell outside [0,res];
 *  outside -> read the neighbor face's adjacent cell via the seam table. Cheap
 *  (textureLoads only, no faceDirNode). */
export function seamHeight(
  face: FaceName,
  sample: SampleFace,
  table: SeamTable,
  tx: any,
  ty: any,
): any {
  const res = PLANET.res;
  const R = int(res);
  const Z = int(0);
  const hSelf = sample(face, ivec2(tx.max(Z).min(R), ty.max(Z).min(R)));

  const nH = (edge: EdgeId, vary: any) => {
    const seam = table[face][edge];
    const varyI = seam.varyReverse ? R.sub(vary) : vary;
    const nx = seam.nFixedIsX ? int(seam.nInwardVal) : varyI;
    const ny = seam.nFixedIsX ? varyI : int(seam.nInwardVal);
    return sample(seam.nFace, ivec2(nx, ny));
  };

  let h = tx.lessThan(Z).select(nH(0 as EdgeId, ty), hSelf);
  h = tx.greaterThan(R).select(nH(1 as EdgeId, ty), h);
  h = ty.lessThan(Z).select(nH(2 as EdgeId, tx), h);
  h = ty.greaterThan(R).select(nH(3 as EdgeId, tx), h);
  return h;
}

/** Object-space normal at integer texel (cx,cy) from the cross product of
 *  displaced neighbor positions. Cross-seam neighbors use the NEIGHBOR face's
 *  own direction+height (seamless on flat AND sloped surfaces). Each offset
 *  evaluates only its one relevant edge -> ~9 faceDirNode (vs 25). */
export function objNormalAt(
  face: FaceName,
  sample: SampleFace,
  table: SeamTable,
  cx: any,
  cy: any,
): any {
  const res = PLANET.res;
  const R = int(res);
  const Z = int(0);
  const baseR = float(PLANET.baseRadius);
  const s = heightScaleUniform;

  const selfPos = (tx: any, ty: any) => {
    const h = sample(face, ivec2(tx.max(Z).min(R), ty.max(Z).min(R)));
    const u = tx.toFloat().div(res).mul(2).sub(1);
    const v = ty.toFloat().div(res).mul(2).sub(1);
    return faceDirNode(face, u, v).mul(baseR.add(h.mul(s)));
  };
  const neighborPos = (edge: EdgeId, vary: any) => {
    const seam = table[face][edge];
    const varyI = seam.varyReverse ? R.sub(vary) : vary;
    const ncx = seam.nFixedIsX ? int(seam.nInwardVal) : varyI;
    const ncy = seam.nFixedIsX ? varyI : int(seam.nInwardVal);
    const h = sample(seam.nFace, ivec2(ncx, ncy));
    const u = ncx.toFloat().div(res).mul(2).sub(1);
    const v = ncy.toFloat().div(res).mul(2).sub(1);
    return faceDirNode(seam.nFace, u, v).mul(baseR.add(h.mul(s)));
  };

  const pc = selfPos(cx, cy);
  const pXp = cx.add(1).greaterThan(R).select(neighborPos(1 as EdgeId, cy), selfPos(cx.add(1), cy));
  const pXm = cx.sub(1).lessThan(Z).select(neighborPos(0 as EdgeId, cy), selfPos(cx.sub(1), cy));
  const pYp = cy.add(1).greaterThan(R).select(neighborPos(3 as EdgeId, cx), selfPos(cx, cy.add(1)));
  const pYm = cy.sub(1).lessThan(Z).select(neighborPos(2 as EdgeId, cx), selfPos(cx, cy.sub(1)));

  const outward = normalize(pc);
  let n = normalize(cross(pXp.sub(pXm), pYp.sub(pYm)));
  n = n.mul(sign(dot(n, outward)));
  return n;
}

/**
 * Cheap material surface: texel-exact center displacement + normal SAMPLED from
 * a pre-baked normal texture (object space) -> view space.
 */
export function bakedSurface(
  face: FaceName,
  sampleHeight: (coord: any) => any,
  normalTex: Texture,
): Surface {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const h = sampleHeight(coord);
  const u = cx.toFloat().div(res).mul(2).sub(1);
  const v = cy.toFloat().div(res).mul(2).sub(1);
  const pos = faceDirNode(face, u, v).mul(float(PLANET.baseRadius).add(h.mul(heightScaleUniform)));

  const objN = textureLoad(normalTex, coord).xyz;
  const outward = normalize(pos);
  return {
    position: pos,
    viewNormal: objN.transformDirection(modelViewMatrix),
    slope: float(1).sub(dot(objN, outward).abs()),
    height: h,
  };
}
