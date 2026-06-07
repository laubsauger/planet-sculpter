// Lighting rig (T20, V24). Directional SUN + opposite FILL + hemispheric AMBIENT
// so the night hemisphere stays readable (artistic, not physical). Sun direction
// is exposed as a uniform for the weather shaders (atmosphere rim, clouds, day/
// night terminator). GUI-controllable via `lightingSettings`.

import { uniform, vec3 } from 'three/tsl';
import { Vector3 } from 'three';

export interface LightingSettings {
  azimuth: number; // radians around the pole
  elevation: number; // radians above the equator
  sunIntensity: number;
  fill: number; // opposite-side directional fill
  ambient: number; // hemispheric ambient (dark-side readability)
}

export const lightingSettings: LightingSettings = {
  azimuth: 0.7,
  elevation: 0.7,
  sunIntensity: 2.6,
  fill: 0.6, // anti-sun directional fill -> dark hemisphere readable, but not washed blue
  ambient: 0.35, // low flat hemispheric ambient -> keeps color contrast/saturation
};

/** World-space unit direction TOWARD the sun (for shaders). */
export const sunDirUniform = uniform(vec3(0, 1, 0));

/** Direction toward the sun from azimuth/elevation. */
export function sunDirection(s: LightingSettings): Vector3 {
  const ce = Math.cos(s.elevation);
  return new Vector3(ce * Math.cos(s.azimuth), Math.sin(s.elevation), ce * Math.sin(s.azimuth)).normalize();
}
