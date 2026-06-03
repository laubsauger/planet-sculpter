// Hydraulic erosion passes (M5, T14, V7). After the water depth update:
//   velocity:  water velocity from flux imbalance
//   erosion:   sediment capacity C = Kc*sin(tilt)*|v|; erode bedrock if s<C,
//              deposit if s>C (moves material between bedrock `b` and sediment `s`)
//   advect:    transport suspended sediment with the velocity (semi-Lagrangian)
// Canonical read main -> write scratch -> copy back (V2). Cross-seam handled by
// per-tick seam sync of b/s in Simulation. Ref: Mei et al.

import type { WebGPURenderer } from 'three/webgpu';
import type { StorageTexture } from 'three/webgpu';
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
  vec2,
  vec4,
  max,
  length,
  uniform,
} from 'three/tsl';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

/* eslint-disable @typescript-eslint/no-explicit-any */

export const erosionUniforms = {
  sedimentCapacity: uniform(0.6), // Kc
  dissolve: uniform(0.3), // Ks
  deposit: uniform(0.3), // Kd
  minSlope: uniform(0.02),
  advectScale: uniform(1.0),
  dt: uniform(1 / 60),
};

const EPS = 1e-6;

function xy(n: number) {
  const N = uint(n);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}
const clampI = (i: any, res: number) => i.toFloat().max(float(0)).min(float(res)).toInt();

/** Water velocity (vx,vy) from flux imbalance, written to rg of `velOut`. */
export function buildVelocity(
  f: StorageTexture,
  d: StorageTexture,
  velOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const xm = clampI(ix.sub(1), res);
    const xp = clampI(ix.add(1), res);
    const ym = clampI(iy.sub(1), res);
    const yp = clampI(iy.add(1), res);

    const self = textureLoad(f, ivec2(ix, iy));
    const Lr = textureLoad(f, ivec2(xm, iy)).y; // left neighbor's R (into us +x)
    const Rl = textureLoad(f, ivec2(xp, iy)).x; // right neighbor's L (into us -x)
    const Bt = textureLoad(f, ivec2(ix, ym)).z; // bottom neighbor's T (into us +y)
    const Tb = textureLoad(f, ivec2(ix, yp)).w; // top neighbor's B (into us -y)

    const dc = max(textureLoad(d, ivec2(ix, iy)).x, float(EPS));
    const vx = Lr.sub(self.x).add(self.y.sub(Rl)).mul(0.5).div(dc);
    const vy = Bt.sub(self.w).add(self.z.sub(Tb)).mul(0.5).div(dc);

    textureStore(velOut, uvec2(x, y), vec4(vx, vy, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Erode/deposit between bedrock `b` and suspended sediment `s`. */
export function buildErosion(
  b: StorageTexture,
  s: StorageTexture,
  vel: StorageTexture,
  d: StorageTexture,
  bOut: StorageTexture,
  sOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const xm = clampI(ix.sub(1), res);
    const xp = clampI(ix.add(1), res);
    const ym = clampI(iy.sub(1), res);
    const yp = clampI(iy.add(1), res);

    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const sc = textureLoad(s, ivec2(ix, iy)).x;
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const v = textureLoad(vel, ivec2(ix, iy));

    // terrain tilt from bedrock gradient.
    const dbx = textureLoad(b, ivec2(xp, iy)).x.sub(textureLoad(b, ivec2(xm, iy)).x).mul(0.5);
    const dby = textureLoad(b, ivec2(ix, yp)).x.sub(textureLoad(b, ivec2(ix, ym)).x).mul(0.5);
    const tilt = length(vec2(dbx, dby));
    const sinTilt = max(tilt, u.minSlope.toVar());

    const speed = length(vec2(v.x, v.y));
    const hasWater = dc.greaterThan(float(EPS)).select(float(1), float(0));
    const capacity = u.sedimentCapacity.mul(sinTilt).mul(speed).mul(hasWater);

    // s < C -> erode bedrock into sediment; s > C -> deposit.
    const erode = max(float(0), capacity.sub(sc)).mul(u.dissolve);
    const dep = max(float(0), sc.sub(capacity)).mul(u.deposit);
    const bNew = bc.sub(erode).add(dep);
    const sNew = max(float(0), sc.add(erode).sub(dep));

    textureStore(bOut, uvec2(x, y), vec4(max(bNew, float(0)), 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Semi-Lagrangian advection of sediment along velocity (nearest backtrace). */
export function buildAdvect(
  s: StorageTexture,
  vel: StorageTexture,
  sOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const v = textureLoad(vel, ivec2(ix, iy));
    const step = erosionUniforms.dt.mul(erosionUniforms.advectScale);
    const bx = ix.toFloat().sub(v.x.mul(step));
    const by = iy.toFloat().sub(v.y.mul(step));
    const sx = bx.max(float(0)).min(float(res)).toInt();
    const sy = by.max(float(0)).min(float(res)).toInt();
    const sVal = textureLoad(s, ivec2(sx, sy)).x;
    textureStore(sOut, uvec2(x, y), vec4(sVal, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
