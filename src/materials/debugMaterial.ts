// Debug field visualization (M9, M11 T32). Same displaced surface, colored by a
// selected sim field. Explicit debug MODES (uniform-switched) instead of one
// fixed RGB packing (V36) -> inspect water depth, flow speed/direction, sediment,
// erosion, deposition, soil, cell area, active gate. Cycled with the 'v' key.

import type { Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad,
  ivec2,
  uv,
  vec3,
  vec2,
  float,
  max,
  min,
  mix,
  length,
  normalize,
  uniform,
} from 'three/tsl';
import { bakedSurface } from '../tsl/surface';
import { PLANET, type FaceName } from '../config';

/** 0 = off (handled by Engine swapping back to terrain). See DEBUG_MODES. */
export const debugModeUniform = uniform(1);

/** Display labels by index; index 0 is "off". */
export const DEBUG_MODES = [
  'off',
  'waterDepth',
  'flowSpeed',
  'flowDir',
  'sediment',
  'erosion',
  'deposition',
  'soilDepth',
  'cellArea',
  'activeGate',
];

/* eslint-disable @typescript-eslint/no-explicit-any */

export function makeDebugMaterial(
  face: FaceName,
  heightTex: Texture,
  normalTex: Texture,
  waterTex: Texture,
  sedimentTex: Texture,
  looseTex: Texture,
  velocityTex: Texture,
  areaTex: Texture,
  vizTex: Texture,
): MeshBasicNodeMaterial {
  const res = PLANET.res;
  const cx = uv().x.mul(res).add(0.5).floor().toInt();
  const cy = uv().y.mul(res).add(0.5).floor().toInt();
  const coord = ivec2(cx, cy);

  const s = bakedSurface(face, (c) => textureLoad(heightTex, c).x, normalTex);

  const area = max(textureLoad(areaTex, coord).x, float(1e-6));
  const vol = textureLoad(waterTex, coord).x;
  const depth = vol.div(area);
  const sed = textureLoad(sedimentTex, coord).x;
  const loose = textureLoad(looseTex, coord).x;
  const vel = textureLoad(velocityTex, coord).xy;
  const speed = length(vel);
  const viz = textureLoad(vizTex, coord);

  // cold->hot ramp for scalar fields.
  const ramp = (t: any) => mix(vec3(0.04, 0.1, 0.4), vec3(0.95, 0.85, 0.2), min(t, float(1)));
  const m = debugModeUniform;

  // material colorNode = pure expression -> mode switch via `select` (NOT `If`,
  // which needs a Fn shader context and is null at material build time).
  const dir = vec3(normalize(vel.add(vec2(1e-5, 0))).mul(0.5).add(0.5), float(0));
  const areaCol = vec3(
    max(area.sub(1), float(0)).mul(3),
    float(0.4),
    max(float(1).sub(area), float(0)).mul(3),
  );
  const active = depth.greaterThan(float(0.0008)).or(sed.greaterThan(float(1e-5)));
  const gateCol = active.select(vec3(0.1, 0.8, 0.2), vec3(0.12, 0.12, 0.14));

  let col: any = vec3(0);
  col = m.equal(float(1)).select(ramp(depth.mul(20)), col);
  col = m.equal(float(2)).select(ramp(speed.mul(2)), col);
  col = m.equal(float(3)).select(dir, col);
  col = m.equal(float(4)).select(ramp(sed.mul(8)), col);
  col = m.equal(float(5)).select(ramp(viz.x), col);
  col = m.equal(float(6)).select(ramp(viz.y), col);
  col = m.equal(float(7)).select(ramp(loose.div(0.04)), col);
  col = m.equal(float(8)).select(areaCol, col);
  col = m.equal(float(9)).select(gateCol, col);

  const mat = new MeshBasicNodeMaterial();
  mat.positionNode = s.position;
  mat.colorNode = col;
  return mat;
}
