// Flat heightfield surface (From-Dust pivot). Plane in XZ, displaced in Y by the
// height texture (bilinear, clamp both edges — no wrap, no poles). Normal from the
// height gradient, then perturbed by a procedural FRAGMENT detail-normal so the
// surface reads crisp WITHOUT heavy displaced geometry (detail in fragment, per
// the perf decision). Uniform grid -> uniform world cells, zero distortion.

import {
  uv, ivec2, float, vec2, vec3, normalize, mix, textureLoad, texture as sampleTexture,
  uniform, cameraViewMatrix, transformDirection, varying,
} from 'three/tsl';
import type { Texture } from 'three';
import { FLAT } from '../config';

/* eslint-disable @typescript-eslint/no-explicit-any */

export const flatHeightScale = uniform(FLAT.heightScale);
export const flatSeaLevel = uniform(FLAT.seaLevel);
/** Fragment detail-normal strength + frequency (subtle surface texture). */
export const detailStrength = uniform(0.08);
export const detailFreq = uniform(17);

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
// d/dt of the Catmull-Rom weights (for the analytic surface gradient).
const dwm1 = (t: any) => t.mul(t).mul(-1.5).add(t.mul(2)).sub(0.5);
const dw0 = (t: any) => t.mul(t).mul(4.5).sub(t.mul(5));
const dw1 = (t: any) => t.mul(t).mul(-4.5).add(t.mul(4)).add(0.5);
const dw2 = (t: any) => t.mul(t).mul(1.5).sub(t);

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

/** Bicubic value + analytic gradient (d/dfx, d/dfy) in ONE 16-tap loop. The same
 * taps feed three accumulators, so this replaces FIVE separate bicubic
 * evaluations (value + 4 central-difference neighbours = 80 taps) with 16. The
 * gradient is the exact derivative of the identical Catmull-Rom surface. */
export function bicubicGrad(sample: (c: any) => any, fx: any, fy: any): { value: any; dx: any; dy: any } {
  const ix = fx.floor(), iy = fy.floor();
  const tx = fx.sub(ix), ty = fy.sub(iy);
  const wx = [cwm1(tx), cw0(tx), cw1(tx), cw2(tx)];
  const wy = [cwm1(ty), cw0(ty), cw1(ty), cw2(ty)];
  const dwx = [dwm1(tx), dw0(tx), dw1(tx), dw2(tx)];
  const dwy = [dwm1(ty), dw0(ty), dw1(ty), dw2(ty)];
  const cx = (o: number) => ix.add(o).max(float(0)).min(float(W - 1)).toInt();
  const cy = (o: number) => iy.add(o).max(float(0)).min(float(H - 1)).toInt();
  let value: any = float(0), dx: any = float(0), dy: any = float(0);
  for (let j = -1; j <= 2; j++) {
    let row: any = float(0), rowD: any = float(0);
    for (let i = -1; i <= 2; i++) {
      const s = sample(ivec2(cx(i), cy(j)));
      row = row.add(s.mul(wx[i + 1]));
      rowD = rowD.add(s.mul(dwx[i + 1]));
    }
    value = value.add(row.mul(wy[j + 1]));
    dx = dx.add(rowD.mul(wy[j + 1]));
    dy = dy.add(row.mul(dwy[j + 1]));
  }
  return { value, dx, dy };
}

/** Catmull-Rom bicubic via NINE hardware-bilinear fetches (exact: per axis the
 * center texel pair has nonnegative weights cw0,cw1 on [0,1], so one fractional
 * fetch at w1/(w0+w1) reproduces their sum; the ±edge texels keep their own
 * (possibly negative) weights as scalars). Same surface as `bicubic()` (16 loads),
 * for FRAGMENT use where the texture is LinearFilter float32. Returns .x channel. */
export function bicubicTex(tex: Texture, fx: any, fy: any): any {
  const axis = (f: any) => {
    const i = f.floor();
    const t = f.sub(i);
    const wm = cwm1(t), w0 = cw0(t), w1 = cw1(t), w2 = cw2(t);
    const gC = w0.add(w1);
    // positions in texel space; weight of each of the 3 taps
    return {
      pos: [i.sub(1), i.add(w1.div(gC.max(1e-8))), i.add(2)],
      w: [wm, gC, w2],
    };
  };
  const ax = axis(fx), ay = axis(fy);
  let acc: any = float(0);
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) {
      acc = acc.add(sampleTexture(tex, vec2(ax.pos[i].add(0.5).div(W), ay.pos[j].add(0.5).div(H))).x.mul(ax.w[i]).mul(ay.w[j]));
    }
  }
  return acc;
}

/** `bicubicTex` clamped to the local bilinear cell range (the hardware twin of
 * `bicubicClamped`): kills Catmull-Rom overshoot spikes on nonnegative fluid
 * surfaces. Corner min/max uses 4 exact textureLoads. */
export function bicubicClampedTex(tex: Texture, fx: any, fy: any): any {
  const smooth = bicubicTex(tex, fx, fy);
  const x0 = fx.floor(), y0 = fy.floor();
  const cx = (x: any) => x.max(float(0)).min(float(W - 1)).toInt();
  const cy = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
  const a = textureLoad(tex, ivec2(cx(x0), cy(y0))).x;
  const b = textureLoad(tex, ivec2(cx(x0.add(1)), cy(y0))).x;
  const c = textureLoad(tex, ivec2(cx(x0), cy(y0.add(1)))).x;
  const d = textureLoad(tex, ivec2(cx(x0.add(1)), cy(y0.add(1)))).x;
  return smooth.max(a.min(b).min(c).min(d)).min(a.max(b).max(c).max(d));
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

/** HARDWARE-filtered bilinear sample (vec4) at texel coords fx∈[0,W-1], fy∈[0,H-1].
 * Exactly the lerp `bilinear()` does manually, but one texture-unit fetch instead of
 * 4 textureLoads + shader ALU. Texel i center ↔ uv (i+0.5)/W; clamp-to-edge wrap
 * reproduces the manual coordinate clamp. Textures are LinearFilter float32
 * (float32-filterable). Take .x/.xy off the result as needed. */
export function bilinearTex(tex: Texture, fx: any, fy: any): any {
  return sampleTexture(tex, vec2(fx.add(0.5).div(W), fy.add(0.5).div(H)));
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
  // facets/blocks regardless of grid resolution. ONE 16-tap evaluation yields the
  // value AND the analytic gradient (was 5 evaluations / 80 taps: value + central
  // differences — the gradient here is the exact derivative of that same surface).
  const fx = u.mul(flatGridX), fy = v.mul(flatGridY);
  const g = bicubicGrad(sampleHeight, fx, fy);
  let h: any = g.value;
  if (clampCubic) {
    // clamp to the local bilinear cell range (kills Catmull-Rom overshoot spikes
    // on nonnegative fluid surfaces) — same as bicubicClamped.
    const x0 = fx.floor(), y0 = fy.floor();
    const cx = (x: any) => x.max(float(0)).min(float(W - 1)).toInt();
    const cy = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
    const a = sampleHeight(ivec2(cx(x0), cy(y0)));
    const b = sampleHeight(ivec2(cx(x0.add(1)), cy(y0)));
    const c = sampleHeight(ivec2(cx(x0), cy(y0.add(1))));
    const d = sampleHeight(ivec2(cx(x0.add(1)), cy(y0.add(1))));
    h = h.max(a.min(b).min(c).min(d)).min(a.max(b).max(c).max(d));
  }
  const p = vec3(u.sub(0.5).mul(SIZE), h.mul(flatHeightScale), v.sub(0.5).mul(SIZE));
  // Surface normal from the analytic gradient: n ∝ (-dh/dx_world·scale, 1, -dh/dz_world·scale).
  // dfx/dx_world = flatGridX/SIZE, so scale by flatGridX and use SIZE as the y term.
  // y component is always +SIZE -> inherently upward, no sign() flip needed.
  const n = normalize(vec3(
    g.dx.mul(flatHeightScale).mul(flatGridX).negate(),
    float(SIZE),
    g.dy.mul(flatHeightScale).mul(flatGridY).negate(),
  ));
  // PERF: compute the (bicubic, 16-tap) normal PER-VERTEX and interpolate, not
  // per-fragment — that was the 15fps killer. varying() moves it to the vertex stage.
  const nObj: any = varying(n);

  return {
    position: p,
    worldNormal: nObj,
    // `p` and its derived normal are already in world coordinates. PBR's
    // `normalNode` contract is view space, so apply only world -> view here.
    // Using modelViewMatrix treated this world normal as object-local and made
    // the diffuse response change under camera rotation.
    viewNormal: transformDirection(cameraViewMatrix as any, nObj),
    height: h,
    slope: float(1).sub(nObj.y.clamp(0, 1)),
  };
}

export function flatSurfaceFromTex(tex: Texture): FlatSurface {
  return flatSurface((c) => textureLoad(tex, c).x);
}
