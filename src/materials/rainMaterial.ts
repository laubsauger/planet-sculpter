// Rain veil (T23, V26). A shell below the clouds, visible under storm regions
// (same cloud noise -> aligned with the dark storm clouds). A SLOW, stable streak
// curtain with an opacity floor in storm regions (never blinks to 0) reads as a
// steady downpour instead of strobing sparkle. Dark grey-blue, translucent.

import { MeshBasicNodeMaterial, FrontSide } from 'three/webgpu';
import {
  uniform,
  vec3,
  float,
  normalWorld,
  positionWorld,
  time,
  smoothstep,
  mx_noise_float,
} from 'three/tsl';
import { cloudNoiseNode, storminess } from './cloudMaterial';

export const rainStrength = uniform(0.9);

export function makeRainMaterial(): MeshBasicNodeMaterial {
  // broad storm region (where cloud is dense) scaled by storminess -> rain sits
  // under the storm clouds; softer/wider threshold than the cloud-core darkening.
  const region = smoothstep(float(0.45), float(0.8), cloudNoiseNode()).mul(storminess);
  // SLOW streak curtain (no strobing); floor at 0.6 so rain stays present in the
  // region and only its intensity ripples -> steady downpour, not 1-frame pops.
  const curtain = mx_noise_float(positionWorld.mul(12).add(normalWorld.mul(time.mul(-0.9))));
  const streak = float(0.6).add(smoothstep(float(0.35), float(0.85), curtain).mul(0.4));
  const veil = region.mul(streak).mul(rainStrength);

  const mat = new MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: FrontSide });
  mat.colorNode = vec3(0.34, 0.39, 0.5); // dark grey-blue downpour
  mat.opacityNode = veil.min(float(0.7));
  return mat;
}
