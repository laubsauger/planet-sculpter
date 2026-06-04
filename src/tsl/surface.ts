// Shared displaced-surface math (texel-exact, seam-aware).
// Split into reusable pieces so the analytic normal can be BAKED into a texture
// by a compute pass (cheap per-fragment sampling) instead of recomputed per
// fragment (~25 dir evals). Position sampling is texel-exact (vertex k -> texel
// k); the normal gradient reads neighbor faces across seams using the neighbor
// face's OWN direction so it is continuous (no seam line).

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

/** Height sampler: (faceName, ivec2 texel) -> height float node. */
export type SampleFace = (face: FaceName, coord: any) => any;

export interface Surface {
  position: any;
  viewNormal: any;
  slope: any;
  height: any;
}

/**
 * Cheap material surface: texel-exact center displacement + normal SAMPLED from
 * a pre-baked normal texture (object space) -> view space. No per-fragment
 * gradient/seam work.
 * @param sampleHeight (ivec2 coord) -> total height for THIS face's center.
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

/** Build a seam-aware "displaced position at integer texel (tx,ty)" function. */
export function makePosAt(face: FaceName, sample: SampleFace, table: SeamTable) {
  const res = PLANET.res;
  const R = int(res);
  const Z = int(0);
  const baseR = float(PLANET.baseRadius);

  const selfPos = (tx: any, ty: any) => {
    const h = sample(face, ivec2(tx.max(Z).min(R), ty.max(Z).min(R)));
    const u = tx.toFloat().div(res).mul(2).sub(1);
    const v = ty.toFloat().div(res).mul(2).sub(1);
    return faceDirNode(face, u, v).mul(baseR.add(h.mul(heightScaleUniform)));
  };

  const neighborPos = (edge: EdgeId, vary: any) => {
    const seam = table[face][edge];
    const varyI = seam.varyReverse ? R.sub(vary) : vary;
    const ncx = seam.nFixedIsX ? int(seam.nInwardVal) : varyI;
    const ncy = seam.nFixedIsX ? varyI : int(seam.nInwardVal);
    const h = sample(seam.nFace, ivec2(ncx, ncy));
    const u = ncx.toFloat().div(res).mul(2).sub(1);
    const v = ncy.toFloat().div(res).mul(2).sub(1);
    return faceDirNode(seam.nFace, u, v).mul(baseR.add(h.mul(heightScaleUniform)));
  };

  return (tx: any, ty: any) => {
    let p = selfPos(tx, ty);
    p = tx.lessThan(Z).select(neighborPos(0 as EdgeId, ty), p);
    p = tx.greaterThan(R).select(neighborPos(1 as EdgeId, ty), p);
    p = ty.lessThan(Z).select(neighborPos(2 as EdgeId, tx), p);
    p = ty.greaterThan(R).select(neighborPos(3 as EdgeId, tx), p);
    return p;
  };
}

/** Object-space outward normal at integer texel (cx,cy). */
export function objNormalAt(posAt: ReturnType<typeof makePosAt>, cx: any, cy: any) {
  const pc = posAt(cx, cy);
  const tu = posAt(cx.add(1), cy).sub(posAt(cx.sub(1), cy));
  const tv = posAt(cx, cy.add(1)).sub(posAt(cx, cy.sub(1)));
  const outward = normalize(pc);
  let n = normalize(cross(tu, tv));
  n = n.mul(sign(dot(n, outward)));
  return n;
}

/** Cheap center displaced position (self texel only). */
export function centerPosition(face: FaceName, sample: SampleFace, cx: any, cy: any) {
  const res = PLANET.res;
  const u = cx.toFloat().div(res).mul(2).sub(1);
  const v = cy.toFloat().div(res).mul(2).sub(1);
  const h = sample(face, ivec2(cx, cy));
  return faceDirNode(face, u, v).mul(float(PLANET.baseRadius).add(h.mul(heightScaleUniform)));
}
