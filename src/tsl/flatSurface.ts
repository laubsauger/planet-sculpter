// Flat heightfield surface (From-Dust pivot). Plane in XZ, displaced in Y by the
// height texture (bilinear, clamp both edges — no wrap, no poles). Normal from the
// height gradient, then perturbed by a procedural FRAGMENT detail-normal so the
// surface reads crisp WITHOUT heavy displaced geometry (detail in fragment, per
// the perf decision). Uniform grid -> uniform world cells, zero distortion.

import {
  uv, ivec2, float, vec3, normalize, cross, mix, sign, textureLoad, uniform,
  cameraViewMatrix, transformDirection, varying,
} from 'three/tsl';
import type { Texture } from 'three';
import { FLAT } from '../config';

/* eslint-disable @typescript-eslint/no-explicit-any */

export const flatHeightScale = uniform(FLAT.heightScale);
export const flatSeaLevel = uniform(FLAT.seaLevel);
/** Fragment detail-normal strength + frequency (subtle surface texture). */
export const detailStrength = uniform(0.22);
export const detailFreq = uniform(11);

const W = FLAT.gridW;
const H = FLAT.gridH;
export const flatGridX = W - 1;
export const flatGridY = H - 1;
const SIZE = FLAT.worldSize;

// Catmull-Rom cubic weights (interpolates THROUGH the samples, C1-smooth).
const cwm1 = (t: any) => t.mul(t).mul(t).mul(-0.5).add(t.mul(t)).sub(t.mul(0.5));
const cw0 = (t: any) => t.mul(t).mul(t).mul(1.5).sub(t.mul(t).mul(2.5)).add(1);
const cw1 = (t: any) => t.mul(t).mul(t).mul(-1.5).add(t.mul(t).mul(2)).add(t.mul(0.5));
const cw2 = (t: any) => t.mul(t).mul(t).mul(0.5).sub(t.mul(t).mul(0.5));

/** Bicubic (Catmull-Rom) sample, clamp both axes. Smooth C1 -> no facets/blocks. */
export function bicubic(sample: (c: any) => any, fx: any, fy: any): any {
  const ix = fx.floor(), iy = fy.floor();
  const tx = fx.sub(ix), ty = fy.sub(iy);
  const wx = [cwm1(tx), cw0(tx), cw1(tx), cw2(tx)];
  const wy = [cwm1(ty), cw0(ty), cw1(ty), cw2(ty)];
  const cx = (o: number) => ix.add(o).max(float(0)).min(float(W - 1)).toInt();
  const cy = (o: number) => iy.add(o).max(float(0)).min(float(H - 1)).toInt();
  let acc: any = float(0);
  for (let j = -1; j <= 2; j++) {
    let row: any = float(0);
    for (let i = -1; i <= 2; i++) row = row.add(sample(ivec2(cx(i), cy(j))).mul(wx[i + 1]));
    acc = acc.add(row.mul(wy[j + 1]));
  }
  return acc;
}

/** Bicubic reconstruction clamped to the local bilinear cell range. Useful for
 * nonnegative fluid surfaces, where Catmull-Rom overshoot creates tall spikes. */
export function bicubicClamped(sample: (c: any) => any, fx: any, fy: any): any {
  const smooth = bicubic(sample, fx, fy);
  const x0 = fx.floor(), y0 = fy.floor();
  const cx = (x: any) => x.max(float(0)).min(float(W - 1)).toInt();
  const cy = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
  const a = sample(ivec2(cx(x0), cy(y0)));
  const b = sample(ivec2(cx(x0.add(1)), cy(y0)));
  const c = sample(ivec2(cx(x0), cy(y0.add(1))));
  const d = sample(ivec2(cx(x0.add(1)), cy(y0.add(1))));
  return smooth.max(a.min(b).min(c).min(d)).min(a.max(b).max(c).max(d));
}

export interface FlatSurface {
  position: any;
  worldNormal: any;
  viewNormal: any;
  height: any;
  slope: any; // 0 flat .. 1 vertical
}

/** Bilinear sample, clamp both axes. fx=u*W, fy=v*H. */
export function bilinear(sample: (c: any) => any, fx: any, fy: any): any {
  const x0 = fx.floor(), y0 = fy.floor();
  const tx = fx.sub(x0), ty = fy.sub(y0);
  const cx = (x: any) => x.max(float(0)).min(float(W - 1)).toInt();
  const cy = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
  const h00 = sample(ivec2(cx(x0), cy(y0)));
  const h10 = sample(ivec2(cx(x0.add(1)), cy(y0)));
  const h01 = sample(ivec2(cx(x0), cy(y0.add(1))));
  const h11 = sample(ivec2(cx(x0.add(1)), cy(y0.add(1))));
  return mix(mix(h00, h10, tx), mix(h01, h11, tx), ty);
}

export function flatSurface(sampleHeight: (coord: any) => any, clampCubic = false): FlatSurface {
  const u = uv().x, v = uv().y;
  // BICUBIC height (C1-smooth) -> smooth geometry AND smooth normals, no per-cell
  // facets/blocks regardless of grid resolution.
  const hAt = (uu: any, vv: any) => {
    const fx = uu.mul(flatGridX), fy = vv.mul(flatGridY);
    return clampCubic ? bicubicClamped(sampleHeight, fx, fy) : bicubic(sampleHeight, fx, fy);
  };
  const worldPos = (uu: any, vv: any) =>
    vec3(uu.sub(0.5).mul(SIZE), hAt(uu, vv).mul(flatHeightScale), vv.sub(0.5).mul(SIZE));

  const e = float(1.2 / flatGridX);
  const p = worldPos(u, v);
  const px = worldPos(u.add(e), v), pxm = worldPos(u.sub(e), v);
  const pz = worldPos(u, v.add(e)), pzm = worldPos(u, v.sub(e));
  let n = normalize(cross(pz.sub(pzm), px.sub(pxm)));
  n = n.mul(sign(n.y)); // force upward
  // PERF: compute the (bicubic, ~64-tap) normal PER-VERTEX and interpolate, not
  // per-fragment — that was the 15fps killer. varying() moves it to the vertex stage.
  const nObj: any = varying(n);

  return {
    position: p,
    worldNormal: nObj,
    // `p` and its derived normal are already in world coordinates. PBR's
    // `normalNode` contract is view space, so apply only world -> view here.
    // Using modelViewMatrix treated this world normal as object-local and made
    // the diffuse response change under camera rotation.
    viewNormal: transformDirection(cameraViewMatrix, nObj),
    height: hAt(u, v),
    slope: float(1).sub(nObj.y.clamp(0, 1)),
  };
}

export function flatSurfaceFromTex(tex: Texture): FlatSurface {
  return flatSurface((c) => textureLoad(tex, c).x);
}
