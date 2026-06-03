// Shared displaced-surface math for terrain & water (texel-exact, seam-aware).
// Samples height by INTEGER texel (vertex k -> texel k). The normal is the
// analytic gradient of neighbor DISPLACED POSITIONS; at a face border the
// neighbor sample uses the NEIGHBOR FACE'S OWN direction + height (via the seam
// table) — not this face's warp extrapolated past its edge — so the tangent
// across the seam is the true surface tangent and the normal is continuous
// (no seam crease). Output normal is VIEW space.

import { uv, ivec2, int, float, cross, dot, sign, normalize, modelViewMatrix } from 'three/tsl';
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

export function computeSurface(face: FaceName, sample: SampleFace, table: SeamTable): Surface {
  const res = PLANET.res;
  const R = int(res);
  const Z = int(0);
  const baseR = float(PLANET.baseRadius);

  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();

  // displaced position at texel (tx,ty) using THIS face.
  const selfPos = (tx: any, ty: any) => {
    const h = sample(face, ivec2(tx.max(Z).min(R), ty.max(Z).min(R)));
    const u = tx.toFloat().div(res).mul(2).sub(1);
    const v = ty.toFloat().div(res).mul(2).sub(1);
    return faceDirNode(face, u, v).mul(baseR.add(h.mul(heightScaleUniform)));
  };

  // displaced position of the cell across an edge, using the NEIGHBOR face.
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

  // position at (tx,ty) which may be one cell outside [0,res] -> cross seam.
  const posAt = (tx: any, ty: any) => {
    let p = selfPos(tx, ty);
    p = tx.lessThan(Z).select(neighborPos(0 as EdgeId, ty), p); // x<0  left
    p = tx.greaterThan(R).select(neighborPos(1 as EdgeId, ty), p); // x>res right
    p = ty.lessThan(Z).select(neighborPos(2 as EdgeId, tx), p); // y<0  bottom
    p = ty.greaterThan(R).select(neighborPos(3 as EdgeId, tx), p); // y>res top
    return p;
  };

  const pc = posAt(cx, cy);
  const tu = posAt(cx.add(1), cy).sub(posAt(cx.sub(1), cy));
  const tv = posAt(cx, cy.add(1)).sub(posAt(cx, cy.sub(1)));

  const outward = normalize(pc);
  let objN = normalize(cross(tu, tv));
  objN = objN.mul(sign(dot(objN, outward)));

  const viewNormal = objN.transformDirection(modelViewMatrix);
  const slope = float(1).sub(dot(objN, outward).abs());

  return { position: pc, viewNormal, slope, height: sample(face, ivec2(cx, cy)) };
}
