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
/** below this water depth, fully transparent. Flat ground now dries to 0
 *  (subtractive evaporation), so this can be small — only real puddles show. */
const MIN_DEPTH = 0.004;

export function makeWaterMaterial(heightTex: Texture, depthTex: Texture): MeshStandardNodeMaterial {
  const b = sampleTex(heightTex, uv()).x;
  const d = sampleTex(depthTex, uv()).x;

  // surface sits at terrain + water column.
  const surf = b.add(d);
  const displaced = positionGeometry.add(normalGeometry.mul(surf.mul(heightScaleUniform)));

  const col = mix(SHALLOW, DEEP, smoothstep(0.0, 0.15, d));
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 3, d).mul(0.85);

  const mat = new MeshStandardNodeMaterial({
    // higher roughness -> softer specular, hides the per-face flat-normal crease
    // at mesh seams on the water surface.
    roughness: 0.5,
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
