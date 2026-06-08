// Stylized TOON terrain for the single equirect grid. UNLIT (MeshBasicNodeMaterial):
// does its own quantized 2-3 step lighting from the sun uniform (⊥ PBR), crisp
// elevation color BANDS, anti-sun fill + ambient for dark-side readability, and a
// fresnel RIM light. The flat/banded cartoon look (From Dust style) deliberately
// hides the low sim-grid resolution. Surface (pos + normals) from gridSurface.

import type { Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, dot, float, vec3,
  mx_fractal_noise_float, positionLocal, sin,
} from 'three/tsl';
import { gridSurface, bilinear } from '../tsl/gridSurface';
import { lonLatDirNode } from '../tsl/latlongNode';
import { seaLevelUniform } from '../tsl/heightScale';
import { sunDirUniform, sunIntensityU, fillU, ambientU } from '../tsl/lighting';
import { erosionUniforms } from '../sim/passes/erosion';
import { PLANET } from '../config';

const SAND = vec3(0.86, 0.76, 0.5);
const ARID = vec3(0.82, 0.64, 0.3);
const LUSH = vec3(0.34, 0.64, 0.18);
const FOREST = vec3(0.12, 0.4, 0.13);
const ROCK_BROWN = vec3(0.5, 0.37, 0.24);
const ROCK_GREY = vec3(0.46, 0.45, 0.42);
const ROCK_RED = vec3(0.56, 0.32, 0.23);
const SNOW = vec3(0.96, 0.97, 1.0);
const RIM_COL = vec3(0.55, 0.72, 0.95);
const LOOSE_FULL = 0.025;

/* eslint-disable @typescript-eslint/no-explicit-any */

export function makeGridTerrain(
  heightTex: Texture,
  looseTex: Texture,
  rainfallTex: Texture,
  hardnessTex: Texture,
): MeshBasicNodeMaterial {
  const W = PLANET.lonRes;
  const H = PLANET.latRes;
  const fx = uv().x.mul(W);
  const fy = uv().y.mul(H - 1);
  const sampleBL = (tex: Texture) => bilinear((c: any) => textureLoad(tex, c).x, fx, fy);

  const s = gridSurface((c: any) => textureLoad(heightTex, c).x);
  const h = s.height;
  const looseRatio = sampleBL(looseTex).div(LOOSE_FULL).min(float(1));
  // low-freq blotch noise to break flat biomes (high-freq = grainy speckle/jank).
  const noise = mx_fractal_noise_float(positionLocal.mul(6), 4);
  const moisture = sampleBL(rainfallTex).add(noise.mul(0.06));
  // erodibility = 2D province * SAME 3D volumetric noise the erosion pass uses
  // (sampled at bedrock point dir*(R+h)), so exposed hard knobs render as rock.
  const dir = lonLatDirNode(uv().x, uv().y);
  const pos3d = dir.mul(float(PLANET.baseRadius).add(h));
  const n3 = mx_fractal_noise_float(pos3d.mul(erosionUniforms.hardness3dFreq), 4);
  const erodibility = sampleBL(hardnessTex)
    .mul(float(1).add(n3.mul(erosionUniforms.hardness3dAmp)))
    .max(float(0.05));

  // --- albedo: biome by moisture/elevation, CRISP bands (narrow transitions) ---
  const veg = mix(ARID, LUSH, smoothstep(0.24, 0.4, moisture));
  const highVeg = mix(veg, FOREST, smoothstep(0.45, 0.62, moisture));
  let albedo: any = mix(SAND, veg, smoothstep(0.07, 0.1, h));
  albedo = mix(albedo, highVeg, smoothstep(0.24, 0.28, h));
  albedo = mix(albedo, ROCK_BROWN, smoothstep(0.48, 0.52, h));
  albedo = mix(albedo, SNOW, smoothstep(0.72, 0.76, h));

  const resistant = float(1).sub(smoothstep(0.4, 1.2, erodibility));
  const rockCol = mix(ROCK_GREY, ROCK_RED, resistant);
  const exposure = max(float(1).sub(looseRatio), smoothstep(0.34, 0.5, s.slope)).min(float(1));
  albedo = mix(albedo, rockCol, exposure);

  // coastline + submerged darkening.
  const above = h.sub(seaLevelUniform);
  albedo = mix(albedo, SAND, smoothstep(0.04, 0.0, above.abs()).mul(0.7));
  const wet = smoothstep(0.0, -0.05, above);
  albedo = albedo.mul(mix(float(1), float(0.55), wet));

  // polar ice caps -> snow.
  const cosLat = sin(uv().y.mul(Math.PI));
  albedo = mix(albedo, SNOW, smoothstep(0.38, 0.18, cosLat));

  // --- TOON lighting: quantized 3-step sun term + fill + ambient ---
  const ndl = max(float(0), dot(s.objNormal, sunDirUniform));
  const band = smoothstep(0.02, 0.08, ndl).mul(0.34)
    .add(smoothstep(0.34, 0.4, ndl).mul(0.33))
    .add(smoothstep(0.66, 0.72, ndl).mul(0.33));
  // lit face ≈ albedo (factor ~1), shadow ~0.45 -> contrast without blowing out.
  const litF = ambientU.mul(0.35).add(sunIntensityU.mul(0.3));
  const shadeF = ambientU.mul(0.45).add(fillU.mul(0.45)); // dark side stays readable
  let col: any = albedo.mul(mix(shadeF, litF, band));

  // fresnel rim (cartoon edge highlight); subtle cool tint.
  const rim = float(1).sub(s.viewNormal.z.clamp(0, 1)).pow(4.0).mul(0.16);
  col = col.add(RIM_COL.mul(rim)).min(vec3(1.0, 1.0, 1.0));

  const mat = new MeshBasicNodeMaterial();
  mat.positionNode = s.position;
  mat.colorNode = col;
  return mat;
}
