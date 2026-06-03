// Pipe-model hydraulic water passes (M4, T12, V7). Per face, per tick:
//   addWater: d += rain*dt
//   flux:     outflow per pipe from water-surface height diff, scale-clamped (V4)
//   depth:    d += dt*(inflow - outflow)/l^2, then evaporate
// Each pass reads canonical `main`, writes `scratch`; Simulation copies back (V2).
// Cross-seam flow not yet handled (no apron) -> face borders act as walls; M5+
// adds aprons. Reference: Mei et al., "Fast Hydraulic Erosion Simulation".

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
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
  max,
  min,
  uniform,
} from 'three/tsl';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

/** Shared, tunable sim constants (mutated from UI later). */
export const waterUniforms = {
  rain: uniform(0),
  evaporation: uniform(0.02),
  gravity: uniform(9.81),
  pipeArea: uniform(1),
  pipeLength: uniform(1),
  dt: uniform(1 / 60),
};

const EPS = 1e-6;

/* eslint-disable @typescript-eslint/no-explicit-any */
// TSL node types don't thread cleanly through helpers; use loose typing.

function coords(n: number): { x: any; y: any; ix: any; iy: any } {
  const N = uint(n);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}

/** Clamp an integer node coordinate to [0, res] via float (int lacks min/max). */
function clamp(i: any, res: number): any {
  return i.toFloat().max(float(0)).min(float(res)).toInt();
}

/** d += rain*dt */
export function buildAddWater(d: StorageTexture, dOut: StorageTexture, n: number): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(n);
    const cur = textureLoad(d, ivec2(ix, iy)).x;
    const next = cur.add(waterUniforms.rain.mul(waterUniforms.dt));
    textureStore(dOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Pipe outflow flux (L,R,T,B) -> rgba, scale-clamped to available water (V4). */
export function buildFlux(
  b: StorageTexture,
  d: StorageTexture,
  fPrev: StorageTexture,
  fOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(n);
    const xm = clamp(ix.sub(1), res);
    const xp = clamp(ix.add(1), res);
    const ym = clamp(iy.sub(1), res);
    const yp = clamp(iy.add(1), res);

    const surf = (cx: any, cy: any) =>
      textureLoad(b, ivec2(cx, cy)).x.add(textureLoad(d, ivec2(cx, cy)).x);

    const hc = surf(ix, iy);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const prev = textureLoad(fPrev, ivec2(ix, iy));

    const k = waterUniforms.dt
      .mul(waterUniforms.pipeArea)
      .mul(waterUniforms.gravity)
      .div(waterUniforms.pipeLength);

    const fL = max(float(0), prev.x.add(k.mul(hc.sub(surf(xm, iy)))));
    const fR = max(float(0), prev.y.add(k.mul(hc.sub(surf(xp, iy)))));
    const fT = max(float(0), prev.z.add(k.mul(hc.sub(surf(ix, yp)))));
    const fB = max(float(0), prev.w.add(k.mul(hc.sub(surf(ix, ym)))));

    const sum = fL.add(fR).add(fT).add(fB);
    // scale so total outflow <= water present this step.
    const l2 = waterUniforms.pipeLength.mul(waterUniforms.pipeLength);
    const scale = min(float(1), dc.mul(l2).div(max(sum.mul(waterUniforms.dt), float(EPS))));

    textureStore(
      fOut,
      uvec2(x, y),
      vec4(fL.mul(scale), fR.mul(scale), fT.mul(scale), fB.mul(scale)),
    ).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** d += dt*(inflow - outflow)/l^2, then evaporate. */
export function buildDepth(
  d: StorageTexture,
  f: StorageTexture,
  dOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(n);
    const xm = clamp(ix.sub(1), res);
    const xp = clamp(ix.add(1), res);
    const ym = clamp(iy.sub(1), res);
    const yp = clamp(iy.add(1), res);

    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const self = textureLoad(f, ivec2(ix, iy));
    const outflow = self.x.add(self.y).add(self.z).add(self.w);

    // inflow = neighbor's flux toward this cell.
    const inL = textureLoad(f, ivec2(xm, iy)).y; // left neighbor's R
    const inR = textureLoad(f, ivec2(xp, iy)).x; // right neighbor's L
    const inT = textureLoad(f, ivec2(ix, yp)).w; // top neighbor's B
    const inB = textureLoad(f, ivec2(ix, ym)).z; // bottom neighbor's T
    const inflow = inL.add(inR).add(inT).add(inB);

    const l2 = waterUniforms.pipeLength.mul(waterUniforms.pipeLength);
    let next = dc.add(waterUniforms.dt.mul(inflow.sub(outflow)).div(l2));
    next = next.mul(float(1).sub(waterUniforms.evaporation.mul(waterUniforms.dt)));
    next = max(next, float(0));

    textureStore(dOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
