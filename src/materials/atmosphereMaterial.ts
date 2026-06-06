// Atmosphere shell (T21, V24/V25). A slightly-larger back-side sphere rendered
// additive: a Fresnel rim glows at the planet's limb, brighter on the sun-facing
// side, fading on the night side -> soft atmospheric halo. Cheap scatter approx,
// no raymarch. Drawn after opaque, depthWrite off (V25).

import { MeshBasicNodeMaterial, AdditiveBlending, BackSide } from 'three/webgpu';
import {
  uniform,
  vec3,
  float,
  normalView,
  normalWorld,
  positionViewDirection,
  max,
  pow,
} from 'three/tsl';
import { sunDirUniform } from '../tsl/lighting';

export const atmosphereColor = uniform(vec3(0.4, 0.6, 1.0));
export const atmosphereStrength = uniform(1.0);

export function makeAtmosphereMaterial(): MeshBasicNodeMaterial {
  // limb fresnel: surface edge-on to the camera -> rim ~1 (back-side shell -> the
  // glow wraps the planet silhouette).
  const fres = float(1).sub(normalView.dot(positionViewDirection).abs());
  const rim = pow(fres, float(3));
  // brighter on the day side, dim (but not black) on the night side.
  const sunFacing = max(normalWorld.dot(sunDirUniform), float(-0.3)).mul(0.5).add(0.5);
  const intensity = rim.mul(sunFacing).mul(atmosphereStrength);

  const mat = new MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    side: BackSide,
  });
  mat.colorNode = atmosphereColor;
  mat.opacityNode = intensity;
  return mat;
}
