// Global ocean shell (T16 sea level). A smooth icosphere at the sea-level
// radius; terrain pokes through above it -> instant coastlines. Radius is
// driven by a uniform so the sea level can be changed live. Subtle animated
// ripples + Fresnel for a stylized water look.

import { Mesh, IcosahedronGeometry } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  positionGeometry,
  positionWorld,
  normalize,
  cameraPosition,
  dot,
  sin,
  time,
  mix,
  vec3,
  float,
} from 'three/tsl';
import { seaRadiusUniform } from '../tsl/heightScale';

const SHALLOW = vec3(0.16, 0.5, 0.62);
const DEEP = vec3(0.02, 0.13, 0.36);

export function makeSeaMesh(): Mesh {
  // unit icosphere (no poles), displaced to the sea radius in the vertex node.
  const geo = new IcosahedronGeometry(1, 32);
  const dir = normalize(positionGeometry);

  const mat = new MeshStandardNodeMaterial({
    transparent: true,
    roughness: 0.15,
    metalness: 0,
    depthWrite: false,
  });
  mat.positionNode = dir.mul(seaRadiusUniform);

  // depth-ish tint by view grazing + subtle ripple shimmer.
  const viewDir = normalize(cameraPosition.sub(positionWorld));
  const fres = float(1).sub(dot(viewDir, dir).max(0));
  const ripple = sin(positionWorld.x.mul(40).add(time.mul(1.5)))
    .mul(sin(positionWorld.z.mul(40).sub(time.mul(1.1))))
    .mul(0.5)
    .add(0.5);
  mat.colorNode = mix(DEEP, SHALLOW, fres.mul(0.6).add(ripple.mul(0.15)));
  mat.opacityNode = float(0.78);

  const mesh = new Mesh(geo, mat);
  mesh.renderOrder = 1; // after terrain
  return mesh;
}
