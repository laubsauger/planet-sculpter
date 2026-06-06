// Water surface material (M4). Baked-normal surface at (b+d). Color by water
// DEPTH: shallow = turquoise, deep = deep blue. Hidden where dry.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  textureLoad,
  ivec2,
  uv,
  mix,
  smoothstep,
  vec3,
  float,
  max,
  sin,
  time,
} from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { PLANET, type FaceName } from '../config';

const SHALLOW = vec3(0.28, 0.66, 0.74); // turquoise
const DEEP = vec3(0.03, 0.16, 0.42); // deep blue
const MIN_DEPTH = 0.0008; // reveal thin fast flow (steep rivers) -> continuous

export function makeWaterMaterial(
  face: FaceName,
  heightTex: Texture,
  depthTex: Texture,
  normalTex: Texture,
  areaTex: Texture,
): MeshStandardNodeMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  // depthTex stores VOLUME -> depth = vol/area (V31).
  const s = bakedSurface(
    face,
    (c) =>
      textureLoad(heightTex, c).x.add(
        textureLoad(depthTex, c).x.div(max(textureLoad(areaTex, c).x, float(1e-6))),
      ),
    normalTex,
  );
  // exact texel depth (texture(uv) on r32float storage is unreliable).
  const d = textureLoad(depthTex, coord).x.div(max(textureLoad(areaTex, coord).x, float(1e-6)));

  // Surface motion comes from the SIM (depth changes -> baked normal moves),
  // which follows real flow. No procedural spatial ripple grid (looked tiled /
  // uncorrelated). Only depth-based foam + slope-based rapids, gently pulsed.
  let col = mix(SHALLOW, DEEP, smoothstep(0.004, 0.06, d));
  const pulse = sin(time.mul(2.5)).mul(0.2).add(0.8); // subtle brightness pulse

  // foam line where water meets land (very shallow).
  const foam = float(1).sub(smoothstep(0.002, 0.016, d));
  col = mix(col, vec3(0.82, 0.9, 0.95), foam.mul(0.4));

  // rapids/whitewater: steep slope + flowing water -> white churn (flow-correlated).
  const rapids = smoothstep(0.28, 0.6, s.slope).mul(smoothstep(0.0008, 0.012, d));
  col = mix(col, vec3(0.92, 0.96, 0.98), rapids.mul(pulse).mul(0.7));

  // thin flow becomes visible quickly; deeper water more opaque.
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 4, d).mul(0.55).add(smoothstep(0.01, 0.05, d).mul(0.35));

  const mat = new MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0, transparent: true });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
