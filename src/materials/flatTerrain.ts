// Flat From-Dust-style terrain. PBR (MeshStandardNodeMaterial) so the real light
// rig (sun + sky) lights it; crispness comes from the fragment detail-normal in
// flatSurface. Material blend: sand/grass/rock/snow by height, slope, moisture,
// hardness, with a coastal sand band. Sediment (loose) shades toward sand.

import { FrontSide, type Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, min, float, vec3, normalize,
  mx_noise_float, mx_fractal_noise_float, cameraViewMatrix, transformDirection, sin, time, uniform, fract,
} from 'three/tsl';
import {
  flatSurface, bilinear, flatSeaLevel, flatGridX, flatGridY, detailFreq, detailStrength,
} from '../tsl/flatSurface';

/* eslint-disable @typescript-eslint/no-explicit-any */

const SAND = vec3(0.84, 0.74, 0.5);
const DRY = vec3(0.62, 0.52, 0.3);
const GRASS = vec3(0.33, 0.5, 0.18);
const GRASS_LUSH = vec3(0.2, 0.42, 0.14);
const ROCK = vec3(0.42, 0.39, 0.35);
const ROCK_DARK = vec3(0.28, 0.25, 0.23);
const ROCK_RED = vec3(0.5, 0.34, 0.26);
const SNOW = vec3(0.95, 0.96, 0.98);
const WET_EARTH = vec3(0.42, 0.3, 0.18);
const FRESH_CUT = vec3(0.34, 0.2, 0.11);
const FRESH_DEPOSIT = vec3(0.76, 0.58, 0.31);
const SEABED_SHALLOW = vec3(0.46, 0.66, 0.6); // sandy-turquoise just under the surface
const SEABED_DEEP = vec3(0.05, 0.1, 0.16);    // dark cool deep-ocean floor
// Debug A/B toggle: 0 disables the animated lapping wet-sand darkening entirely.
export const shoreWetEnabled = uniform(1);
// Debug inspection: 1 tiles the 4 material types into vertical strips (sand|grass|rock|snow)
// with clean base colors, so each per-type detail normal can be inspected in isolation
// (pair with the flat `materialTest` benchmark).
export const materialDebugGrid = uniform(0);
// Height-contour line overlay (on by default — an elegant elevation read).
export const contourOverlay = uniform(1);
export const contourCount = uniform(44);

export function makeFlatTerrain(
  heightTex: Texture,
  looseTex: Texture,
  moistureTex: Texture,
  hardnessTex: Texture,
  waterTex: Texture,
  sedimentTex: Texture,
  activityTex: Texture,
): MeshStandardNodeMaterial {
  const fx = uv().x.mul(flatGridX), fy = uv().y.mul(flatGridY);
  const bl = (t: Texture) => bilinear((c: any) => textureLoad(t, c).x, fx, fy);
  const blVec = (t: Texture) => bilinear((c: any) => textureLoad(t, c), fx, fy);

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x);
  const h = s.height;
  const slope = s.slope;
  const moisture = bl(moistureTex);
  const erod = bl(hardnessTex);
  const looseRatio = bl(looseTex).div(0.022).min(float(1));
  const water = bl(waterTex);
  const sediment = bl(sedimentTex);
  const activity = blVec(activityTex);

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

  // Broad material mottling in ALBEDO — low-frequency ONLY. The old strata `sin(p.y*13)`
  // and `fracture` painted flat contour/crack LINES (the "squiggly line" artifact). Rock
  // layering/shale now lives in the per-type detail NORMAL below, where it reads as lit
  // relief instead of a painted line.
  const p = s.position;
  const broadRock = mx_noise_float(p.mul(0.65)).mul(0.5).add(0.5);
  const province = mx_noise_float(p.mul(1.3).add(vec3(4.3, 1.7, -2.8))).mul(0.5).add(0.5);
  const rockStructure = broadRock.mul(0.18).sub(0.09).add(province.mul(0.1).sub(0.05));
  albedo = albedo.mul(float(1).add(rockStructure.mul(exposure)));

  const sandMask = looseRatio.mul(float(1).sub(exposure))
    .mul(float(1).sub(smoothstep(0.16, 0.42, h.sub(flatSeaLevel))));
  const sandWarp = mx_noise_float(p.mul(1.4)).mul(1.2);
  const sandMottle = mx_noise_float(p.mul(2.4).add(sandWarp)).mul(0.5).add(0.5);
  albedo = albedo.mul(float(1).add(sandMottle.sub(0.5).mul(0.09).mul(sandMask)));

  // Material weights that PARTITION the surface (sum to ~1, dirt fills the remainder) so
  // EVERY fragment gets the full detail of its dominant material — not a faint fraction.
  // Previously gentle/dry terrain had all weights ~0 -> detailH ~0 -> dead-flat diffuse.
  const wSnowB = smoothstep(0.55, 0.72, h);
  const wRockB = exposure.mul(float(1).sub(wSnowB));
  const wSandB = sandMask.mul(float(1).sub(wSnowB)).mul(float(1).sub(wRockB));
  const wGrassB = smoothstep(0.28, 0.5, moisture)
    .mul(float(1).sub(exposure)).mul(float(1).sub(wSnowB)).mul(float(1).sub(wSandB));
  const wDirtB = max(float(0), float(1).sub(wSnowB).sub(wRockB).sub(wSandB).sub(wGrassB));
  // Debug grid override: 4 vertical strips, one pure material each (dirt off in grid).
  const gcell = uv().x.mul(4.0);
  const gseg = (lo: number, hi: number) => gcell.greaterThanEqual(float(lo)).and(gcell.lessThan(float(hi))).select(float(1), float(0));
  const wSand = mix(wSandB, gseg(0, 1), materialDebugGrid);
  const wGrass = mix(wGrassB, gseg(1, 2), materialDebugGrid);
  const wRock = mix(wRockB, gseg(2, 3), materialDebugGrid);
  const wSnow = mix(wSnowB, gseg(3, 4.01), materialDebugGrid);
  const wDirt = mix(wDirtB, float(0), materialDebugGrid);

  // Fresh material motion remains visible for a while rather than appearing as
  // an abrupt grid-colored edit.
  albedo = mix(albedo, FRESH_CUT, smoothstep(0.04, 0.65, activity.x).mul(0.62));
  albedo = mix(albedo, FRESH_DEPOSIT, smoothstep(0.04, 0.65, activity.y).mul(0.72));

  // Water darkens the material already present instead of replacing it with a
  // bright water-colored fringe. Sediment adds only a restrained warm earth tint.
  const wet = smoothstep(0.0004, 0.018, water);
  const muddy = smoothstep(0.002, 0.08, sediment);
  albedo = albedo.mul(mix(float(1), float(0.68), wet.mul(0.72)));
  albedo = mix(albedo, WET_EARTH, muddy.mul(0.28));
  // Lingering wetness (activity.z): subtly darken ground that was recently under water,
  // fading back to dry over a few seconds after runoff.
  albedo = albedo.mul(mix(float(1), float(0.7), activity.z.min(float(1))));

  // Coastal sand band at the waterline.
  const above = h.sub(flatSeaLevel);
  albedo = mix(albedo, SAND, smoothstep(0.035, 0.0, above.abs()).mul(0.7));

  // Lapping wet sand: a THIN wash band right at the waterline whose wet line creeps up the
  // sand and recedes on a slow tide (two rhythms). Driven by the height-above-sea CONTOUR
  // so it stays shore-parallel and thin, with only a tiny STATIC spatial warp. Deliberately
  // NO time-animated 3D noise here: that smeared wandering "wet cloud" blobs across the
  // whole lower slope (the green band sits just above sea level, inside the old wide band).
  // `nearShore` clamps the band tight so it can never reach the grassy mid-slope.
  // Band hugs the waterline: top creeps only a few thousandths ABOVE sea level so the wet
  // sand meets the wave/foam zone instead of floating a wide stripe up the dry beach.
  const shoreNoise = mx_noise_float(p.mul(5.0)).mul(0.002);
  // Band sits IN the wave-wash zone: from just under the waterline (where the foam laps) to
  // only a hair above it. Top barely clears sea level so the wet sand visually connects to
  // the approaching waves instead of floating a stripe up the dry beach.
  const reachA = mix(float(-0.006), float(-0.001), sin(time.mul(0.9)).mul(0.5).add(0.5));
  const reachB = mix(float(-0.004), float(0.002), sin(time.mul(1.5).add(1.3)).mul(0.5).add(0.5));
  const wetLine = max(
    smoothstep(reachA.add(shoreNoise), reachA.sub(float(0.004)), above),
    smoothstep(reachB.add(shoreNoise), reachB.sub(float(0.004)), above).mul(0.7),
  );
  const nearShore = smoothstep(float(-0.028), float(-0.01), above); // reach under the waterline into the wash
  const lapWet = wetLine.mul(nearShore).min(float(1)).mul(shoreWetEnabled);
  albedo = albedo.mul(mix(float(1), float(0.74), lapWet));

  // Submerged seabed: smooth blend from sandy-turquoise shallows to a dark cool deep floor
  // as the bed descends — a gradual depth ramp, NO step at the waterline, so there is no
  // hard color edge where land meets sea. The transparent water tint sits on top of this.
  const sub = max(float(0), above.mul(-1));
  const shallowBed = mix(SAND, SEABED_SHALLOW, smoothstep(0.0, 0.03, sub).mul(0.85));
  const seabed: any = mix(shallowBed, SEABED_DEEP, smoothstep(0.03, 0.26, sub));
  // Underwater texture: sand ripples + broad mottling so the seabed reads as a real bottom,
  // not a flat color ramp. Strongest in the clear shallows (visible through the water),
  // fading out with depth where it would be invisible anyway.
  const bedRipple = sin(p.x.mul(7.0).add(p.z.mul(2.6)).add(mx_noise_float(p.mul(1.8)).mul(2.0))).mul(0.5).add(0.5);
  const bedMottle = mx_noise_float(p.mul(2.6).add(vec3(5, 0, 5))).mul(0.5).add(0.5);
  const rippleFade = float(1).sub(smoothstep(0.03, 0.13, sub));
  const seabedTex = seabed.mul(float(1).add(bedRipple.sub(0.5).mul(0.14).add(bedMottle.sub(0.5).mul(0.1)).mul(rippleFade)));
  const underwater = smoothstep(0.001, 0.012, sub);
  albedo = mix(albedo, seabedTex, underwater);

  // Unified PERFORMANT multi-scale detail. ONE fractal field (3 octaves) at a HIGH base
  // frequency — high enough to read as real surface texture given the large world stretch
  // (old low freqs put ~1 feature across a whole biome -> looked flat) — plus a cheap per-type
  // macro pattern (dune crests in sand, shale layers in rock). Sampled 3x (center + 2 forward
  // offsets) = ~9 noise taps (was ~80) feeding BOTH the fragment normal (relief) and the
  // albedo/roughness value structure. World-position driven -> tiles seamlessly across the
  // whole map, randomized by the fractal noise. detailFreq/detailStrength sliders still scale.
  const detFreq = detailFreq.mul(0.4);
  const det = (q: any) => {
    const grain = mx_fractal_noise_float(q.mul(detFreq), 3);
    const dune = sin(q.x.mul(detFreq.mul(0.7)).add(q.z.mul(detFreq.mul(0.4))));
    const strata = sin(q.y.mul(detFreq.mul(1.6)));
    return grain.add(wSand.mul(dune).mul(0.7)).add(wRock.mul(strata).mul(0.8));
  };
  const structCenter = det(p);
  const dX = det(p.add(vec3(0.04, 0, 0))).sub(structCenter);
  const dZ = det(p.add(vec3(0, 0, 0.04))).sub(structCenter);
  const detailAmp = wSand.mul(1.4).add(wGrass.mul(1.0)).add(wRock.mul(2.2)).add(wSnow.mul(0.6)).add(wDirt.mul(1.0)).add(0.3);
  const bumpedWorldNormal = normalize(s.worldNormal.sub(vec3(dX, 0, dZ).mul(detailStrength.mul(detailAmp).mul(6.0))));
  const structContrast = wSand.mul(0.1).add(wGrass.mul(0.16)).add(wRock.mul(0.2)).add(wSnow.mul(0.07)).add(wDirt.mul(0.13)).add(0.04);
  albedo = albedo.mul(float(1).add(structCenter.mul(structContrast)));

  const mat = new MeshStandardNodeMaterial({
    side: FrontSide,
    roughness: 0.94,
    metalness: 0,
  });
  mat.positionNode = s.position;
  mat.normalNode = (transformDirection as any)(cameraViewMatrix, bumpedWorldNormal);
  // Debug grid: clean per-type base colour so only the detail normal + roughness vary.
  const gridColor = gseg(0, 1).mul(SAND).add(gseg(1, 2).mul(GRASS)).add(gseg(2, 3).mul(ROCK)).add(gseg(3, 4.01).mul(SNOW));
  let outColor: any = mix(albedo, gridColor, materialDebugGrid);
  // Optional height-contour overlay: thin dark lines at evenly spaced elevations (dev tool).
  // FLAT darkening lines only — purely a color overlay, NO relief/highlight (the lines are a
  // map-style elevation read, not geometry or sim, so they must not suggest 3D depressions).
  const cf = fract(h.mul(contourCount));
  const cLine = float(1).sub(smoothstep(float(0.0), float(0.03), min(cf, float(1).sub(cf))));
  outColor = outColor.mul(float(1).sub(cLine.mul(0.4).mul(contourOverlay)));
  mat.colorNode = outColor;
  // Per-type roughness so materials read distinctly under the sun: sand matte, grass soft,
  // rock varied/slightly sheened, snow brightest. Sequential mix-by-weight; wet = shinier.
  let rough: any = float(0.93); // dry dirt default
  rough = mix(rough, float(0.97), wSand);
  rough = mix(rough, float(0.88), wGrass);
  rough = mix(rough, float(0.8), wRock);
  rough = mix(rough, float(0.58), wSnow);
  // Structure modulates sheen too — crests/troughs of the per-type pattern catch the sun
  // slightly differently, giving specular life instead of a uniform matte field.
  rough = rough.add(structCenter.mul(0.07)).clamp(float(0.3), float(1));
  rough = mix(rough, float(0.6), max(max(wet, activity.z), lapWet).min(float(1)));
  mat.roughnessNode = rough;
  return mat;
}
