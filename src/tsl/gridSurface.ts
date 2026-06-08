// Single equirect grid surface with HIGH-RES output from a LOW-RES sim grid:
//   - BILINEAR height sampling (4-tap lerp; r32float can't hardware-filter) so
//     displacement/shading is smooth between texels, ⊥ blocky nearest sampling.
//   - PROCEDURAL detail noise (dir-based fbm) adds sub-texel crispness the sim
//     grid doesn't store -> "high resolution" look without raising sim cost.
//   - normal sampled at SUB-texel offsets -> picks up both bilinear + detail.
// WRAP-X (longitude) / CLAMP-Y (latitude) — no seam table.

import {
  uv,
  ivec2,
  float,
  cross,
  dot,
  sign,
  mix,
  normalize,
  modelViewMatrix,
  textureLoad,
  uniform,
  sin,
  smoothstep,
  mx_fractal_noise_float,
} from 'three/tsl';
import type { Texture } from 'three';
import { lonLatDirNode } from './latlongNode';
import { PLANET } from '../config';
import { heightScaleUniform } from './heightScale';

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface Surface {
  position: any;
  viewNormal: any;
  objNormal: any; // object-space normal (for custom/toon lighting vs sunDir)
  slope: any;
  height: any;
}

/** Procedural fine-detail amount (height units) + frequency. GUI-tunable. */
export const detailStrength = uniform(0.024);
export const detailFreq = uniform(20);

const W = PLANET.lonRes;
const H = PLANET.latRes;

/** Bilinear height at continuous texel coords (fx=u*W, fy=v*(H-1)). wrap/clamp.
 *  Plain (C0) bilinear: the gradient is constant inside a cell, so the surface
 *  NORMAL is faceted at grid scale. We mask that with strong procedural detail in
 *  surfH (below) rather than smootherstep — smootherstep's derivative pulses to
 *  zero at every cell edge -> visible grid-aligned corrugation in the normal. */
export function bilinear(sample: (c: any) => any, fx: any, fy: any): any {
  const x0 = fx.floor();
  const y0 = fy.floor();
  const tx = fx.sub(x0);
  const ty = fy.sub(y0);
  const wrap = (x: any) => x.add(float(W)).mod(float(W)).toInt();
  const clampY = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
  const ix0 = wrap(x0);
  const ix1 = wrap(x0.add(1));
  const iy0 = clampY(y0);
  const iy1 = clampY(y0.add(1));
  const h00 = sample(ivec2(ix0, iy0));
  const h10 = sample(ivec2(ix1, iy0));
  const h01 = sample(ivec2(ix0, iy1));
  const h11 = sample(ivec2(ix1, iy1));
  return mix(mix(h00, h10, tx), mix(h01, h11, tx), ty);
}

export function gridSurface(sampleHeight: (coord: any) => any): Surface {
  const u = uv().x;
  const v = uv().y;

  // bilinear grid height + TWO octaves of procedural sub-texel detail. The detail
  // is what makes the low-res grid read as high-res: it adds high-freq variation to
  // both displacement AND the (per-fragment) normal, masking the bilinear facets.
  const surfH = (uu: any, vv: any) => {
    const base = bilinear(sampleHeight, uu.mul(W), vv.mul(H - 1));
    const dir = lonLatDirNode(uu, vv);
    // fade detail toward the poles: the uv->dir map is singular there, so detail
    // (and its normal) smear into radial streaks. cos(lat)=sin(v*PI).
    const poleFade = smoothstep(float(0.12), float(0.4), sin(vv.mul(Math.PI)));
    const d1 = mx_fractal_noise_float(dir.mul(detailFreq), 5).mul(detailStrength);
    const d2 = mx_fractal_noise_float(dir.mul(detailFreq.mul(3.3)), 3).mul(detailStrength.mul(0.4));
    return base.add(d1.add(d2).mul(poleFade));
  };
  const posC = (uu: any, vv: any) =>
    lonLatDirNode(uu, vv).mul(float(PLANET.baseRadius).add(surfH(uu, vv).mul(heightScaleUniform)));

  const e = float(0.6 / W); // sub-texel offset (captures bilinear + detail gradient)
  const pc = posC(u, v);
  const pXp = posC(u.add(e), v);
  const pXm = posC(u.sub(e), v);
  const pYp = posC(u, v.add(e));
  const pYm = posC(u, v.sub(e));

  const outward = normalize(pc);
  let n = normalize(cross(pXp.sub(pXm), pYp.sub(pYm)));
  n = n.mul(sign(dot(n, outward)));

  return {
    position: pc,
    viewNormal: n.transformDirection(modelViewMatrix),
    objNormal: n,
    slope: float(1).sub(dot(n, outward).abs()),
    height: bilinear(sampleHeight, u.mul(W), v.mul(H - 1)),
  };
}

/** Convenience: surface from a single height texture. */
export function gridSurfaceFromTex(tex: Texture): Surface {
  return gridSurface((coord) => textureLoad(tex, coord).x);
}
