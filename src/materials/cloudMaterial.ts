// Cloud shell (T22, V25/V28/V29). A sphere just above the peaks, textured by
// dir-based 3D fbm noise -> clouds wrap the planet seamlessly (3D world dir, no
// face seams, V29). Coverage uniform thresholds density; wind offset animates
// drift (time uniform, V28). Sun-lit, semi-transparent, depthWrite off (V25).

import { MeshBasicNodeMaterial, FrontSide } from 'three/webgpu';
import {
  uniform,
  vec3,
  float,
  normalWorld,
  time,
  max,
  smoothstep,
  mx_fractal_noise_float,
} from 'three/tsl';
import { sunDirUniform } from '../tsl/lighting';

export const cloudCoverage = uniform(0.42); // 0 = clear sky, 1 = overcast
export const cloudScale = uniform(5.0); // noise frequency (higher = smaller clouds)
export const cloudOpacity = uniform(0.85);
export const windDir = uniform(vec3(1, 0, 0.3));
export const windSpeed = uniform(0.004);
export const storminess = uniform(0.25); // 0 = fair weather, 1 = heavy storms

/** Dir-based cloud fbm (wind-drifted). Same field used by the rain veil so rain
 *  falls under the storm clouds. Resolves per-material via normalWorld. */
export function cloudNoiseNode(): ReturnType<typeof mx_fractal_noise_float> {
  const wind = windDir.mul(time.mul(windSpeed));
  // 3 octaves (was 5): this runs PER FRAGMENT every frame over a full-screen
  // transparent shell -> octaves are the dominant cost. 3 is plenty stylized.
  return mx_fractal_noise_float(normalWorld.mul(cloudScale).add(wind), 3);
}

/** Storm intensity (dense cloud cores × storminess) for a given direction. */
export function stormMaskNode(): any {
  return smoothstep(0.55, 0.95, cloudNoiseNode()).mul(storminess);
}

export function makeCloudMaterial(): MeshBasicNodeMaterial {
  const n = cloudNoiseNode();
  // WIDE gentle fade so drifting clouds don't blink in/out across a sharp edge.
  const dens = smoothstep(cloudCoverage.oneMinus(), cloudCoverage.oneMinus().add(0.4), n);
  // sun-lit clouds: bright on the day side, dim (not black) on the night side.
  const sunlit = max(normalWorld.dot(sunDirUniform), float(-0.2)).mul(0.5).add(0.55);
  // puffy shading: denser tops read brighter -> volume, ⊥ flat white.
  const shade = float(0.6).add(n.mul(0.4));
  // storm cores (dense cloud × storminess) go dark + heavy. SAME mask -> rain veil.
  const storm = smoothstep(0.5, 0.9, n).mul(storminess);
  const base: any = vec3(0.98, 0.98, 1.0).mul(sunlit).mul(shade);
  const col = base.mix(vec3(0.22, 0.24, 0.3).mul(sunlit), storm); // dark storm grey

  const mat = new MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: FrontSide });
  mat.colorNode = col;
  // opacity from coverage only (adding storm here caused popping); storms are
  // already in dense (high-n) regions so they read thick.
  mat.opacityNode = dens.mul(cloudOpacity).min(float(1));
  return mat;
}
