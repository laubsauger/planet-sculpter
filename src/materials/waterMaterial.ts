// Water surface material (M4). Per-face mesh (same cube-sphere geometry as
// terrain) displaced to the water-surface height (b + d). Hidden where depth
// is negligible (opacity 0). Stylized: depth-tinted blue, faceted normals.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  texture as sampleTex,
  uv,
  positionGeometry,
  normalGeometry,
  normalFlat,
  mix,
  smoothstep,
  vec3,
} from 'three/tsl';
import { heightScaleUniform } from './terrainMaterial';

const SHALLOW = vec3(0.30, 0.62, 0.78);
const DEEP = vec3(0.05, 0.20, 0.42);
/** below this water depth, fully transparent. */
const MIN_DEPTH = 0.002;

export function makeWaterMaterial(heightTex: Texture, depthTex: Texture): MeshStandardNodeMaterial {
  const b = sampleTex(heightTex, uv()).x;
  const d = sampleTex(depthTex, uv()).x;

  // surface sits at terrain + water column.
  const surf = b.add(d);
  const displaced = positionGeometry.add(normalGeometry.mul(surf.mul(heightScaleUniform)));

  const col = mix(SHALLOW, DEEP, smoothstep(0.0, 0.15, d));
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 3, d).mul(0.85);

  const mat = new MeshStandardNodeMaterial({
    roughness: 0.2,
    metalness: 0,
    transparent: true,
  });
  mat.positionNode = displaced;
  mat.normalNode = normalFlat;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
