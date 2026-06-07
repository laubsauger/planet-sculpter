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

const SAND = vec3(0.86, 0.76, 0.5); // beach
const ARID = vec3(0.8, 0.66, 0.32); // dry savanna / steppe (low moisture)
const LUSH = vec3(0.3, 0.62, 0.16); // lush grassland (high moisture)
const FOREST = vec3(0.13, 0.4, 0.12); // wet highland forest
const ROCK_BROWN = vec3(0.5, 0.37, 0.24); // alpine soil/rock
const ROCK_GREY = vec3(0.44, 0.43, 0.4); // bare soft rock
const ROCK_RED = vec3(0.55, 0.32, 0.24); // resistant rock (iron/basalt look)
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
  vizTex: Texture,
  rainfallTex: Texture,
  hardnessTex: Texture,
): TerrainMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(face, (c) => textureLoad(heightTex, c).x, normalTex);
  const h = s.height;
  const looseRatio = textureLoad(looseTex, coord).x.div(LOOSE_FULL).min(float(1));
  const moisture = textureLoad(rainfallTex, coord).x; // 0 desert .. 1 rainforest
  const erodibility = textureLoad(hardnessTex, coord).x; // ~0.1 hard .. ~2.9 soft

  // BIOME by MOISTURE: arid steppe (dry) -> lush grass -> wet forest, plus snow
  // up high. Distinct climate zones instead of one flat green.
  const veg = mix(ARID, LUSH, smoothstep(0.18, 0.62, moisture));
  const highVeg = mix(veg, FOREST, smoothstep(0.35, 0.75, moisture)); // forest if wet
  let loose = mix(SAND, veg, smoothstep(0.06, 0.12, h));
  loose = mix(loose, highVeg, smoothstep(0.22, 0.34, h));
  loose = mix(loose, ROCK_BROWN, smoothstep(0.46, 0.56, h));
  loose = mix(loose, SNOW, smoothstep(0.72, 0.82, h)); // snow only on the highest peaks

  // ROCK color by HARDNESS: resistant (low erodibility) reads as red/dark rock,
  // soft as grey -> the erosion-resistance map is visible in the terrain.
  const resistant = float(1).sub(smoothstep(0.4, 1.2, erodibility));
  const rockCol = mix(ROCK_GREY, ROCK_RED, resistant);
  // bare rock where loose is thin OR slope steep (cliffs/ridges read as rock).
  const exposure = max(float(1).sub(looseRatio), smoothstep(0.32, 0.6, s.slope)).min(float(1));
  let col = mix(loose, rockCol, exposure);

  // coastline: sandy beach band around sea level; darken submerged terrain.
  const above = h.sub(seaLevelUniform);
  col = mix(col, SAND, smoothstep(0.05, 0.0, above.abs()).mul(0.7));
  const wet = smoothstep(0.0, -0.05, above); // 1 below sea level
  col = col.mul(mix(float(1), float(0.5), wet));

  // From-Dust style activity tint (V35): fresh erosion = dark wet-earth streaks,
  // fresh deposition = light sediment fans. Decays in the sim (erosionViz).
  const vizD = textureLoad(vizTex, coord);
  // only strong fresh activity tints (avoid washing the whole surface).
  const eroded = smoothstep(0.25, 1.0, vizD.x);
  const deposited = smoothstep(0.25, 1.0, vizD.y);
  col = mix(col, vec3(0.3, 0.22, 0.16), eroded.mul(0.3)); // dark fresh-cut earth
  col = mix(col, vec3(0.78, 0.72, 0.56), deposited.mul(0.28)); // pale sediment fan

  const mat = new MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  return { material: mat };
}
