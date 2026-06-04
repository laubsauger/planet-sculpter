// Debug field visualization (M9). Same displaced terrain surface, but colored
// by sim fields: R = suspended sediment, G = loose cover, B = water depth.
// Toggled with the 'v' key to inspect the simulation.

import type { Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { textureLoad, ivec2, uv, vec3, float } from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { PLANET, type FaceName } from '../config';

export function makeDebugMaterial(
  face: FaceName,
  heightTex: Texture,
  normalTex: Texture,
  waterTex: Texture,
  sedimentTex: Texture,
  looseTex: Texture,
): MeshBasicNodeMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(face, (c) => textureLoad(heightTex, c).x, normalTex);
  const sed = textureLoad(sedimentTex, coord).x.mul(8).min(float(1));
  const loose = textureLoad(looseTex, coord).x.div(0.04).min(float(1));
  const water = textureLoad(waterTex, coord).x.mul(20).min(float(1));

  const mat = new MeshBasicNodeMaterial();
  mat.positionNode = s.position;
  mat.colorNode = vec3(sed, loose, water);
  return mat;
}
