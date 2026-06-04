// Reusable pipe-model FLUID passes (M4 water; M8 lava reuses these). Per tick:
//   addSource: d += sourceRate*dt   (rain for water; vent injection for lava)
//   flux:      outflow per pipe from fluid-surface height diff, scale-clamped (V4)
//   depth:     d += dt*(inflow - outflow)/l^2, then loss*dt (evap / cooling)
// Parameterized by a FluidUniforms set so water & lava share the same solver;
// only the constants (viscosity via pipeArea/gravity, loss rate, source) and
// the terrain coupling (erosion vs solidify, done in separate passes) differ.
// Borders sealed for conservation; cross-face via seam diffusion. Ref: Mei et al.

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

/** Tunable constants for one fluid (water or lava). source=uniform add rate,
 *  loss=subtractive loss/sec (evap/cooling), pipeArea=inverse viscosity. */
export function makeFluidUniforms(o: {
  source?: number;
  loss: number;
  gravity?: number;
  pipeArea?: number;
  pipeLength?: number;
}) {
  return {
    source: uniform(o.source ?? 0),
    loss: uniform(o.loss),
    gravity: uniform(o.gravity ?? 9.81),
    pipeArea: uniform(o.pipeArea ?? 1),
    pipeLength: uniform(o.pipeLength ?? 1),
    dt: uniform(1 / 60),
  };
}

/** Type carries the rich TSL node methods (.mul etc.) via inference. */
export type FluidUniforms = ReturnType<typeof makeFluidUniforms>;

/** Water instance. Low loss so rain accumulates; pipeArea 2 = brisk drainage. */
export const waterUniforms = makeFluidUniforms({ loss: 0.0012, pipeArea: 2 });

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

/** d += source*dt (uniform add: rain / vent baseline). */
export function buildAddSource(
  d: StorageTexture,
  dOut: StorageTexture,
  n: number,
  p: FluidUniforms,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(n);
    const cur = textureLoad(d, ivec2(ix, iy)).x;
    const next = cur.add(p.source.mul(p.dt));
    textureStore(dOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Pipe outflow flux (L,R,T,B) -> rgba, scale-clamped to available fluid (V4). */
export function buildFlux(
  b: StorageTexture,
  d: StorageTexture,
  fPrev: StorageTexture,
  fOut: StorageTexture,
  n: number,
  p: FluidUniforms,
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

    const k = p.dt.mul(p.pipeArea).mul(p.gravity).div(p.pipeLength);

    let fL = max(float(0), prev.x.add(k.mul(hc.sub(surf(xm, iy)))));
    let fR = max(float(0), prev.y.add(k.mul(hc.sub(surf(xp, iy)))));
    let fT = max(float(0), prev.z.add(k.mul(hc.sub(surf(ix, yp)))));
    let fB = max(float(0), prev.w.add(k.mul(hc.sub(surf(ix, ym)))));

    // Seal face borders: no flux into a wall (clamped neighbor would otherwise
    // read self -> phantom flux -> non-conservation -> water explosion).
    // Cross-face transfer is handled by the conservative seam diffusion.
    fL = ix.lessThan(int(1)).select(float(0), fL);
    fR = ix.greaterThan(int(res - 1)).select(float(0), fR);
    fT = iy.greaterThan(int(res - 1)).select(float(0), fT);
    fB = iy.lessThan(int(1)).select(float(0), fB);

    const sum = fL.add(fR).add(fT).add(fB);
    // scale so total outflow <= fluid present this step.
    const l2 = p.pipeLength.mul(p.pipeLength);
    const scale = min(float(1), dc.mul(l2).div(max(sum.mul(p.dt), float(EPS))));

    textureStore(
      fOut,
      uvec2(x, y),
      vec4(fL.mul(scale), fR.mul(scale), fT.mul(scale), fB.mul(scale)),
    ).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** d += dt*(inflow - outflow)/l^2, then subtract loss (evap / cooling). */
export function buildDepth(
  d: StorageTexture,
  f: StorageTexture,
  dOut: StorageTexture,
  n: number,
  p: FluidUniforms,
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

    // inflow = neighbor's flux toward this cell. Border mask (0/1): no neighbor
    // -> 0 (clamped read would return self's flux -> phantom water).
    const mL = ix.greaterThan(int(0)).select(float(1), float(0));
    const mR = ix.lessThan(int(res)).select(float(1), float(0));
    const mT = iy.lessThan(int(res)).select(float(1), float(0));
    const mB = iy.greaterThan(int(0)).select(float(1), float(0));
    const inL = textureLoad(f, ivec2(xm, iy)).y.mul(mL); // left neighbor's R
    const inR = textureLoad(f, ivec2(xp, iy)).x.mul(mR); // right neighbor's L
    const inT = textureLoad(f, ivec2(ix, yp)).w.mul(mT); // top neighbor's B
    const inB = textureLoad(f, ivec2(ix, ym)).z.mul(mB); // bottom neighbor's T
    const inflow = inL.add(inR).add(inT).add(inB);

    const l2 = p.pipeLength.mul(p.pipeLength);
    const flowDelta = inflow.sub(outflow).mul(p.dt).div(l2);
    const lossAmt = p.loss.mul(p.dt); // subtractive loss (evap / cooling)
    const next: any = dc.add(flowDelta).sub(lossAmt).max(float(0));

    textureStore(dOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
