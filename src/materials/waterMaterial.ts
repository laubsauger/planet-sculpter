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
  length,
  normalize,
  dot,
  uniform,
} from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { faceDirNode } from '../tsl/warpNode';
import { PLANET, type FaceName } from '../config';

const SHALLOW = vec3(0.28, 0.66, 0.74); // turquoise
const DEEP = vec3(0.03, 0.16, 0.42); // deep blue
const MIN_DEPTH = 0.0008; // reveal thin fast flow (steep rivers) -> continuous

/** Strength of the flow-direction streaks on the water surface (V27). */
// default OFF: the scrolling streaks read as a "wave sweeping the globe" on open
// water. opt in via GUI for rivers. (machinery kept; gated to shallow+fast.)
export const flowVizStrength = uniform(0.0);

export function makeWaterMaterial(
  face: FaceName,
  heightTex: Texture,
  depthTex: Texture,
  normalTex: Texture,
  areaTex: Texture,
  velocityTex: Texture,
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

  // FLOW DIRECTION (V27/T31): world-space procedural streaks scrolling ALONG the
  // velocity field -> flow direction is visible. World-space phase (not face uv)
  // => seamless across faces. face-local velocity -> world via face tangents.
  // SMOOTH the velocity over a 3x3 neighborhood: raw per-texel velocity is noisy
  // (flux imbalance) -> incoherent direction -> shimmer. The average is a stable
  // flow direction -> coherent streaks that are visible AND don't flicker.
  const cl = (i: any) => i.toFloat().max(float(0)).min(float(res)).toInt();
  const vAt = (ox: number, oy: number) =>
    textureLoad(velocityTex, ivec2(cl(cx.add(ox)), cl(cy.add(oy)))).xy;
  const velS = vAt(0, 0)
    .add(vAt(1, 0))
    .add(vAt(-1, 0))
    .add(vAt(0, 1))
    .add(vAt(0, -1))
    .add(vAt(1, 1))
    .add(vAt(-1, -1))
    .add(vAt(1, -1))
    .add(vAt(-1, 1))
    .mul(1 / 9);
  const speed = length(velS);
  const e = float(1 / res);
  const u2 = cx.toFloat().div(res).mul(2).sub(1);
  const v2 = cy.toFloat().div(res).mul(2).sub(1);
  const tU = faceDirNode(face, u2.add(e), v2).sub(faceDirNode(face, u2.sub(e), v2));
  const tV = faceDirNode(face, u2, v2.add(e)).sub(faceDirNode(face, u2, v2.sub(e)));
  const flowWorld = normalize(tU.mul(velS.x).add(tV.mul(velS.y)).add(vec3(1e-5, 0, 0)));
  // phase travels along flow over world position. LOW freq + constant scroll =
  // no aliasing/jitter. shown on any flowing water (smoothed dir keeps it clean).
  const phase = dot(s.position, flowWorld).mul(30).sub(time.mul(2.5));
  const streak = smoothstep(0.45, 1.0, sin(phase));
  // streaks belong on flowing RIVERS (shallow + faster), NOT the deep slow ocean
  // (whose smoothed drift crosses the speed threshold globally -> whole-ocean
  // flicker). depth BAND (rises at shore, falls off in deep water) + speed floor.
  const shallow = smoothstep(0.0015, 0.008, d).mul(float(1).sub(smoothstep(0.05, 0.12, d)));
  const flowing = smoothstep(0.03, 0.12, speed).mul(shallow);
  col = mix(col, col.add(vec3(0.16, 0.18, 0.2)), streak.mul(flowing).mul(flowVizStrength));

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
