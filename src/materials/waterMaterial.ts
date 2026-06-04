// Water surface material (M4). Baked-normal surface at (b+d). Color by water
// DEPTH: shallow = turquoise, deep = deep blue. Hidden where dry.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { textureLoad, ivec2, uv, mix, smoothstep, vec3 } from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { PLANET, type FaceName } from '../config';

const SHALLOW = vec3(0.28, 0.66, 0.74); // turquoise
const DEEP = vec3(0.03, 0.16, 0.42); // deep blue
const MIN_DEPTH = 0.003;

export function makeWaterMaterial(
  face: FaceName,
  heightTex: Texture,
  depthTex: Texture,
  normalTex: Texture,
): MeshStandardNodeMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(
    face,
    (c) => textureLoad(heightTex, c).x.add(textureLoad(depthTex, c).x),
    normalTex,
  );
  // exact texel depth (texture(uv) on r32float storage is unreliable).
  const d = textureLoad(depthTex, coord).x;

  const col = mix(SHALLOW, DEEP, smoothstep(0.004, 0.06, d));
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 3, d).mul(0.88);

  const mat = new MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0, transparent: true });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
