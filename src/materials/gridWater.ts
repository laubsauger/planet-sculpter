// Water material for the equirect grid. Surface = gridSurface(b + vol/area)
// (bilinear, in-shader normal — no bake, no seams). Depth-tinted, transparent.

import type { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { textureLoad, uv, mix, smoothstep, vec3, float, max, sin, time, mx_fractal_noise_float, positionLocal } from 'three/tsl';
import { gridSurface, bilinear } from '../tsl/gridSurface';
import { PLANET } from '../config';

const SHALLOW = vec3(0.28, 0.66, 0.74);
const DEEP = vec3(0.03, 0.16, 0.42);
const MIN_DEPTH = 0.0008;

export function makeGridWater(heightTex: Texture, volTex: Texture, areaTex: Texture): MeshStandardNodeMaterial {
  const W = PLANET.lonRes;
  const H = PLANET.latRes;

  const s = gridSurface((c) =>
    textureLoad(heightTex, c).x.add(
      textureLoad(volTex, c).x.div(max(textureLoad(areaTex, c).x, float(1e-6))),
    ),
  );
  // BILINEAR depth (vol/area) so color/foam/opacity are smooth, ⊥ blocky nearest.
  // + a small high-freq dither on the depth used for the EDGE so the wet/dry line
  // (a 0->deep jump over one texel) breaks up instead of stairstepping the grid.
  const depthAt = (c: any) =>
    textureLoad(volTex, c).x.div(max(textureLoad(areaTex, c).x, float(1e-6)));
  const d = bilinear(depthAt, uv().x.mul(W), uv().y.mul(H - 1));
  const dither = mx_fractal_noise_float(positionLocal.mul(120), 3).mul(0.0012);
  const dEdge = d.add(dither).max(float(0));

  let col = mix(SHALLOW, DEEP, smoothstep(0.004, 0.06, d));
  const pulse = sin(time.mul(2.5)).mul(0.2).add(0.8);
  const foam = float(1).sub(smoothstep(0.002, 0.016, dEdge));
  col = mix(col, vec3(0.82, 0.9, 0.95), foam.mul(0.4));
  const rapids = smoothstep(0.28, 0.6, s.slope).mul(smoothstep(0.0008, 0.012, d));
  col = mix(col, vec3(0.92, 0.96, 0.98), rapids.mul(pulse).mul(0.7));
  // wider opacity ramp + dithered edge -> soft shoreline, not a hard grid stairstep.
  const opacity = smoothstep(MIN_DEPTH, 0.006, dEdge).mul(0.6).add(smoothstep(0.008, 0.05, d).mul(0.35));

  const mat = new MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0, transparent: true });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
