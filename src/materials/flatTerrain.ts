// Flat From-Dust-style terrain. PBR (MeshStandardNodeMaterial) so the real light
// rig (sun + sky) lights it; crispness comes from the fragment detail-normal in
// flatSurface. Material blend: sand/grass/rock/snow by height, slope, moisture,
// hardness, with a coastal sand band. Sediment (loose) shades toward sand.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { textureLoad, uv, mix, smoothstep, max, dot, float, vec3 } from 'three/tsl';
import { flatSurface, bilinear, flatSeaLevel } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU, ambientU } from '../tsl/lighting';
import { FLAT } from '../config';

const SUN_COL = vec3(1.0, 0.93, 0.82);
const SKY_COL = vec3(0.45, 0.6, 0.82);

/* eslint-disable @typescript-eslint/no-explicit-any */

const SAND = vec3(0.84, 0.74, 0.5);
const DRY = vec3(0.62, 0.52, 0.3);
const GRASS = vec3(0.33, 0.5, 0.18);
const GRASS_LUSH = vec3(0.2, 0.42, 0.14);
const ROCK = vec3(0.42, 0.39, 0.35);
const ROCK_DARK = vec3(0.28, 0.25, 0.23);
const ROCK_RED = vec3(0.5, 0.34, 0.26);
const SNOW = vec3(0.95, 0.96, 0.98);

export function makeFlatTerrain(
  heightTex: Texture,
  looseTex: Texture,
  moistureTex: Texture,
  hardnessTex: Texture,
): MeshBasicNodeMaterial {
  const W = FLAT.gridW, H = FLAT.gridH;
  const fx = uv().x.mul(W), fy = uv().y.mul(H);
  const bl = (t: Texture) => bilinear((c: any) => textureLoad(t, c).x, fx, fy);

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x);
  const h = s.height;
  const slope = s.slope;
  const moisture = bl(moistureTex);
  const erod = bl(hardnessTex);
  const looseRatio = bl(looseTex).div(0.022).min(float(1));

  // ground cover by moisture + elevation.
  const grass = mix(GRASS, GRASS_LUSH, smoothstep(0.4, 0.75, moisture));
  let albedo: any = mix(DRY, grass, smoothstep(0.3, 0.5, moisture));
  albedo = mix(SAND, albedo, smoothstep(0.04, 0.12, h.sub(flatSeaLevel))); // low = sand
  albedo = mix(albedo, SNOW, smoothstep(0.62, 0.72, h)); // peaks snow

  // rock on steep slopes / where hard rock exposed (low erodibility) / bare.
  const hard = float(1).sub(smoothstep(0.4, 1.3, erod));
  const rock = mix(mix(ROCK, ROCK_RED, hard), ROCK_DARK, smoothstep(0.4, 0.75, slope));
  const exposure = max(smoothstep(0.32, 0.55, slope), float(1).sub(looseRatio)).min(float(1));
  albedo = mix(albedo, rock, exposure);

  // coastal sand band at the waterline + darker wet just under it.
  const above = h.sub(flatSeaLevel);
  albedo = mix(albedo, SAND, smoothstep(0.035, 0.0, above.abs()).mul(0.7));
  albedo = albedo.mul(mix(float(1), float(0.55), smoothstep(0.0, -0.06, above)));

  // WORLD-space lighting (camera-independent): worldNormal · world sun dir. Soft
  // wrap so the shadow side stays readable; sky term from up-facing. Unlit material
  // -> no view-space PBR -> rotating the camera never changes the lighting.
  const wN = s.worldNormal;
  const ndl = max(float(0), dot(wN, sunDirUniform));
  const half = ndl.mul(0.78).add(0.22);
  const skyAmt = wN.y.mul(0.5).add(0.5);
  const light = SUN_COL.mul(half.mul(sunIntensityU.mul(0.32)))
    .add(SKY_COL.mul(skyAmt.mul(ambientU)))
    .add(vec3(0.05, 0.05, 0.06));
  const col = albedo.mul(light);

  const mat = new MeshBasicNodeMaterial({ side: DoubleSide });
  mat.positionNode = s.position;
  mat.colorNode = col;
  return mat;
}
