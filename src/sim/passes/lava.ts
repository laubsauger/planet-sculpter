// Lava sim (M8, T17). Reuses the fluid solver (addSource/flux/depth) with
// viscous, slow constants, plus a heat field (advected with the flow) and a
// cooling pass: heat decays, cold lava SOLIDIFIES into bedrock (builds cones /
// flows), vents stay hot. Emissive glow ramps with heat.

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
import { makeFluidUniforms } from './water';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];
/* eslint-disable @typescript-eslint/no-explicit-any */

/** Flowing but viscous: enough flow to travel downslope before solidifying. */
export const lavaUniforms = makeFluidUniforms({
  loss: 0,
  gravity: 8,
  pipeArea: 2.2,
  damping: 0.85,
});

export const lavaCool = {
  heatDecay: uniform(0.12), // heat lost / sec (slow -> stays hot + flows)
  solidify: uniform(0.012), // lava -> rock rate / sec when cold (lower -> flows further)
  dt: uniform(1 / 60),
};

/**
 * Cool + solidify. heat decays (vents stay hot via the source map); cold lava
 * converts to bedrock. Writes lava, heat, and bedrock.
 */
export function buildLavaCool(
  lava: StorageTexture,
  heat: StorageTexture,
  b: StorageTexture,
  src: StorageTexture,
  lavaOut: StorageTexture,
  heatOut: StorageTexture,
  bOut: StorageTexture,
  n: number,
): ComputeNode {
  const c = lavaCool;
  const fn = Fn(() => {
    const x = instanceIndex.mod(uint(n));
    const y = instanceIndex.div(uint(n));
    const co = ivec2(int(x), int(y));

    const lavaC = textureLoad(lava, co).x;
    const heatC = textureLoad(heat, co).x;
    const bc = textureLoad(b, co).x;
    const srcC = textureLoad(src, co).x;

    let h: any = max(float(0), heatC.sub(c.heatDecay.mul(c.dt)));
    h = srcC.greaterThan(float(0.0001)).select(float(1), h); // vents stay hot
    h = lavaC.greaterThan(float(0.0001)).select(h, float(0)); // no lava -> no heat

    // solidify faster when cold (low heat).
    const rate = c.solidify.mul(c.dt).mul(float(1).sub(h.mul(0.85)));
    const cooled = min(lavaC, rate);
    const newLava: any = max(float(0), lavaC.sub(cooled));
    const newB: any = bc.add(cooled);

    textureStore(lavaOut, uvec2(x, y), vec4(newLava, 0, 0, 1)).toWriteOnly();
    textureStore(heatOut, uvec2(x, y), vec4(h, 0, 0, 1)).toWriteOnly();
    textureStore(bOut, uvec2(x, y), vec4(newB, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
