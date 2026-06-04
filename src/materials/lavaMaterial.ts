// Lava surface material (M8, T17). Displaced to (bedrock + lava), lifted a hair
// above terrain to avoid z-fighting. Smooth radial normal (molten + emissive,
// so surface detail is unneeded and we avoid the terrain-normal shading shards).
// Dark basalt that glows ember -> orange -> white-hot by heat. Hidden where dry.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  textureLoad,
  ivec2,
  uv,
  mix,
  smoothstep,
  vec3,
  normalize,
  modelViewMatrix,
} from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { PLANET, type FaceName } from '../config';

const BASALT = vec3(0.07, 0.06, 0.06);
const EMBER = vec3(0.42, 0.06, 0.015);
const HOT = vec3(0.9, 0.32, 0.06);
const WHITE_HOT = vec3(0.98, 0.62, 0.2);
const MIN_LAVA = 0.0015;

export function makeLavaMaterial(
  face: FaceName,
  heightTex: Texture,
  lavaTex: Texture,
  heatTex: Texture,
  normalTex: Texture,
): MeshStandardNodeMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(
    face,
    (c) => textureLoad(heightTex, c).x.add(textureLoad(lavaTex, c).x),
    normalTex,
  );
  const lava = textureLoad(lavaTex, coord).x;
  const heat = textureLoad(heatTex, coord).x;

  const glow = mix(HOT, WHITE_HOT, smoothstep(0.8, 1.0, heat));
  const present = smoothstep(MIN_LAVA, MIN_LAVA * 3, lava);
  const emissive = mix(EMBER, glow, smoothstep(0.08, 0.5, heat)).mul(present).mul(0.85);

  const mat = new MeshStandardNodeMaterial({ roughness: 0.7, metalness: 0, transparent: true });
  // lift a hair above terrain (radial scale) to avoid z-fighting.
  mat.positionNode = s.position.mul(1.0015);
  // smooth radial normal: molten + emissive, avoids terrain-normal shards.
  mat.normalNode = (normalize(s.position) as any).transformDirection(modelViewMatrix);
  mat.colorNode = BASALT;
  mat.emissiveNode = emissive;
  mat.opacityNode = present;
  mat.depthWrite = false;
  return mat;
}
