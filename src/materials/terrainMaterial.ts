// Terrain node material (T8). Cheap: texel-exact center displacement + normal
// sampled from a pre-baked normal texture (seam-continuous). Biome color by
// height + slope. One material per face.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { textureLoad, mix, smoothstep, vec3 } from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { heightScaleUniform } from '../tsl/heightScale';
import type { FaceName } from '../config';

export { heightScaleUniform };

const SAND = vec3(0.76, 0.70, 0.50);
const GRASS = vec3(0.28, 0.45, 0.20);
const ROCK = vec3(0.40, 0.37, 0.33);
const SNOW = vec3(0.92, 0.94, 0.97);

export interface TerrainMaterial {
  material: MeshStandardNodeMaterial;
}

export function makeTerrainMaterial(
  face: FaceName,
  heightTex: Texture,
  normalTex: Texture,
): TerrainMaterial {
  const s = bakedSurface(face, (coord) => textureLoad(heightTex, coord).x, normalTex);
  const h = s.height;

  let col = mix(SAND, GRASS, smoothstep(0.10, 0.18, h));
  col = mix(col, ROCK, smoothstep(0.42, 0.58, h));
  col = mix(col, ROCK, smoothstep(0.55, 0.85, s.slope));
  col = mix(col, SNOW, smoothstep(0.72, 0.82, h));

  const mat = new MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  return { material: mat };
}
