// Terrain node material (T8). Cheap baked-normal surface + material-aware color:
// exposed hard rock (grey) where the loose layer is thin or slopes are steep,
// soil/grass/sand/snow where soft loose material sits. One material per face.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { textureLoad, ivec2, uv, mix, smoothstep, max, float, vec3 } from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { heightScaleUniform, seaLevelUniform } from '../tsl/heightScale';
import { PLANET, type FaceName } from '../config';

export { heightScaleUniform };

const SAND = vec3(0.82, 0.74, 0.5);
const GRASS = vec3(0.33, 0.55, 0.22);
const DEEP_GRASS = vec3(0.2, 0.4, 0.16);
const ROCK_BROWN = vec3(0.46, 0.36, 0.26);
const ROCK_GREY = vec3(0.42, 0.41, 0.39);
const SNOW = vec3(0.95, 0.96, 0.99);
/** loose thickness considered "full cover" (matches erosion looseFull). */
const LOOSE_FULL = 0.025;

export interface TerrainMaterial {
  material: MeshStandardNodeMaterial;
}

export function makeTerrainMaterial(
  face: FaceName,
  heightTex: Texture,
  looseTex: Texture,
  normalTex: Texture,
): TerrainMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(face, (c) => textureLoad(heightTex, c).x, normalTex);
  const h = s.height;
  const looseRatio = textureLoad(looseTex, coord).x.div(LOOSE_FULL).min(float(1));

  // loose (soft) surface color by elevation: sand -> grass -> alpine -> rock -> snow.
  let loose = mix(SAND, GRASS, smoothstep(0.05, 0.14, h));
  loose = mix(loose, DEEP_GRASS, smoothstep(0.18, 0.34, h));
  loose = mix(loose, ROCK_BROWN, smoothstep(0.42, 0.6, h));
  loose = mix(loose, SNOW, smoothstep(0.66, 0.78, h));

  // hard grey rock shows where loose is thin OR the slope is steep (no snow on cliffs).
  const exposure = max(float(1).sub(looseRatio), smoothstep(0.55, 0.85, s.slope)).min(float(1));
  let col = mix(loose, ROCK_GREY, exposure);

  // coastline: sandy beach band around sea level; darken submerged terrain.
  const above = h.sub(seaLevelUniform);
  col = mix(col, SAND, smoothstep(0.05, 0.0, above.abs()).mul(0.7));
  const wet = smoothstep(0.0, -0.05, above); // 1 below sea level
  col = col.mul(mix(float(1), float(0.5), wet));

  const mat = new MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  return { material: mat };
}
