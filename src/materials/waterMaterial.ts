// Water surface material (M4). Cheap: center displacement at (b+d) + baked
// normal. Depth-tinted, hidden where dry.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { texture as sampleTex, textureLoad, uv, mix, smoothstep, vec3 } from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import type { FaceName } from '../config';

const SHALLOW = vec3(0.30, 0.62, 0.78);
const DEEP = vec3(0.05, 0.20, 0.42);
/** below this water depth, fully transparent. */
const MIN_DEPTH = 0.004;

export function makeWaterMaterial(
  face: FaceName,
  heightTex: Texture,
  depthTex: Texture,
  normalTex: Texture,
): MeshStandardNodeMaterial {
  const s = bakedSurface(
    face,
    (coord) => textureLoad(heightTex, coord).x.add(textureLoad(depthTex, coord).x),
    normalTex,
  );
  const d = sampleTex(depthTex, uv()).x;

  const col = mix(SHALLOW, DEEP, smoothstep(0.0, 0.15, d));
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 3, d).mul(0.85);

  const mat = new MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0, transparent: true });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
