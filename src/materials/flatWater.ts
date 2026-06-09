// Flat water, WORLD-space lit (camera-independent base, like the terrain). The
// turquoise depth color is ALWAYS visible from any angle; sun glint + fresnel sky
// reflection are additive bonuses, not the only thing (so it never goes dark/blank
// when you rotate). Animated wave + flow normals (calm, not noise-soup). Surface =
// bedrock + depth; shallow = see-through turquoise, deep = blue; shoreline foam.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, clamp, float, vec2, vec3, length,
  normalize, dot, pow, sin, time, cameraPosition, mx_fractal_noise_float, uniform,
} from 'three/tsl';
import { flatSurface, bilinear, bicubicClamped, flatGridX, flatGridY, flatSeaLevel } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU } from '../tsl/lighting';

/* eslint-disable @typescript-eslint/no-explicit-any */

const SHALLOW = vec3(0.1, 0.65, 0.64);
const MID = vec3(0.025, 0.4, 0.58);
const DEEP = vec3(0.015, 0.12, 0.3);
const SKY_REFLECT = vec3(0.6, 0.78, 0.95);
const FOAM = vec3(0.95, 0.98, 1.0);
const SILT = vec3(0.27, 0.2, 0.11);
export const flowBandStrength = uniform(0.24);
export const flowBandScale = uniform(7.0);

export function makeFlatWater(
  heightTex: Texture,
  waterTex: Texture,
  fluxTex: Texture,
  velTex: Texture,
  sedimentTex: Texture,
): MeshBasicNodeMaterial {
  const fx = uv().x.mul(flatGridX), fy = uv().y.mul(flatGridY);

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x.add(textureLoad(waterTex, c).x), true);
  // bicubic depth -> smooth (C1) color + alpha edge instead of grid-aligned bilinear
  // stair-steps that read as pixelation along the waterline.
  const depth = bicubicClamped((c: any) => textureLoad(waterTex, c).x, fx, fy);
  const depthL = bicubicClamped((c: any) => textureLoad(waterTex, c).x, fx.sub(2), fy);
  const depthR = bicubicClamped((c: any) => textureLoad(waterTex, c).x, fx.add(2), fy);
  const depthB = bicubicClamped((c: any) => textureLoad(waterTex, c).x, fx, fy.sub(2));
  const depthT = bicubicClamped((c: any) => textureLoad(waterTex, c).x, fx, fy.add(2));
  const depthGradient = length(vec2(depthR.sub(depthL), depthT.sub(depthB)));
  const vel = bilinear((c: any) => textureLoad(velTex, c).xy, fx, fy);
  const flux = bilinear((c: any) => textureLoad(fluxTex, c), fx, fy);
  const sediment = bilinear((c: any) => textureLoad(sedimentTex, c).x, fx, fy);
  const speed = length(vec2(vel.x, vel.y));
  // Use the production solver's actual outgoing discharge for visual direction.
  // Reconstructed velocity can point upstream around confluences and obstacles;
  // ambiguous pipe discharge should fade out instead of drawing a confident lie.
  const outgoing = flux.x.add(flux.y).add(flux.z).add(flux.w);
  const flowVector = vec2(flux.y.sub(flux.x), flux.z.sub(flux.w));
  const flowDir = normalize(flowVector.add(vec2(1e-6, 0)));
  const directionConfidence = smoothstep(0.18, 0.72, length(flowVector).div(max(outgoing, float(1e-6))));
  const visualFlow = flowDir.mul(speed).mul(directionConfidence);

  // calm wave + flow normals (subtle ripple, not noise soup).
  const posXZ = vec2(s.position.x, s.position.z);
  const e = float(0.06);
  const grad = (q: any, freq: number) => {
    const nz = (p: any) => mx_fractal_noise_float(vec3(p.x, p.y, 0).mul(freq), 2);
    return vec3(nz(q.add(vec2(e, 0))).sub(nz(q.sub(vec2(e, 0)))), float(0), nz(q.add(vec2(0, e))).sub(nz(q.sub(vec2(0, e)))));
  };
  // Flow ripples: noise advected ALONG the local flow -> rivers ripple downstream,
  // each follows its own direction (not a shared global wave).
  const shallowFlow = float(1).sub(smoothstep(0.025, 0.08, depth));
  const flowR = grad(posXZ.sub(visualFlow.mul(time.mul(0.22))), 1.5)
    .mul(speed.min(float(1.2)).mul(0.11).add(0.012))
    .mul(shallowFlow).mul(directionConfidence);
  // Ambient swell belongs to the OPEN OCEAN ONLY (bedrock below sea level). All water
  // on land — rivers, rain runoff, puddles — must read by its OWN flow (flowR), never
  // share one global wave. Gate the swell by an ocean-basin mask so it can't bleed onto
  // land water and produce that homogeneous diagonal pattern when it rains / on rivers.
  const bed = bilinear((c: any) => textureLoad(heightTex, c).x, fx, fy);
  const oceanMask = float(1).sub(smoothstep(flatSeaLevel.sub(0.03), flatSeaLevel, bed));
  const still = float(1).sub(smoothstep(0.03, 0.3, speed));
  const flatWater = float(1).sub(smoothstep(0.006, 0.045, depthGradient));
  const swell = grad(posXZ.add(vec2(time.mul(0.07), time.mul(-0.05))), 1.7)
    .add(grad(posXZ.add(vec2(time.mul(-0.05), time.mul(0.085))), 3.3).mul(0.55))
    .mul(0.035).mul(still).mul(flatWater);
  const nW: any = normalize(s.worldNormal.add(flowR).add(swell));

  // depth color (view-INDEPENDENT base, always visible).
  let col: any = mix(SHALLOW, MID, smoothstep(0.01, 0.09, depth));
  col = mix(col, DEEP, smoothstep(0.09, 0.32, depth));
  const concentration = sediment.div(max(depth, float(0.003)));
  const sedimentLoad = smoothstep(0.012, 0.22, concentration);
  const plumeNoise = mx_fractal_noise_float(
    vec3(posXZ.sub(visualFlow.mul(time.mul(0.12))).mul(2.4), time.mul(0.035)), 3,
  ).mul(0.5).add(0.5);
  // Suspended sediment forms warm plumes within still-readable blue-green water,
  // rather than replacing the entire surface with a flat brown sheet.
  const turbidity = smoothstep(0.18, 0.72, sedimentLoad.mul(plumeNoise.mul(0.75).add(0.4)));
  col = mix(col, SILT, turbidity.mul(0.66));

  // Long, narrow streaks run ALONG the validated downhill flow and their phase
  // travels downstream. Avoid transverse contour bands: those read as uphill
  // waves even when the velocity vector itself is correct.
  const acrossDir = vec2(flowDir.y.mul(-1), flowDir.x);
  const along = dot(posXZ, flowDir);
  const across = dot(posXZ, acrossDir);
  const travel = time.mul(speed.mul(1.7).add(0.25));
  const laneWarp = sin(along.mul(1.15).sub(travel.mul(0.35))).mul(0.55);
  const lanes = sin(across.mul(flowBandScale).add(laneWarp)).mul(0.5).add(0.5);
  const movingSegments = sin(along.mul(3.4).sub(travel.mul(2.0))
    .add(sin(across.mul(1.8)).mul(0.5))).mul(0.5).add(0.5);
  const streak = smoothstep(0.58, 0.9, lanes).mul(smoothstep(0.22, 0.72, movingSegments));
  const movingWater = smoothstep(0.008, 0.32, speed);
  const riverMask = float(1).sub(oceanMask);
  const streakStrength = streak.mul(movingWater).mul(directionConfidence).mul(riverMask)
    .mul(smoothstep(0.00006, 0.05, depth))
    .mul(float(1).sub(smoothstep(0.04, 0.2, depth)));
  col = mix(col, SKY_REFLECT, streakStrength.mul(flowBandStrength));

  // world-space lighting: gentle sun diffuse keeps base bright + readable any angle.
  const viewW = normalize(cameraPosition.sub(s.position));
  const ndl = max(float(0), dot(nW, sunDirUniform));
  col = col.mul(ndl.mul(0.35).add(0.75));
  // fresnel sky reflection (rim, additive bonus).
  const fres = pow(float(1).sub(max(float(0), dot(nW, viewW))), float(4));
  col = mix(col, SKY_REFLECT, fres.mul(0.18).mul(float(1).sub(turbidity.mul(0.75))));
  // sun specular glint (Blinn-Phong, additive sparkle).
  const half = normalize(sunDirUniform.add(viewW));
  const spec = pow(max(float(0), dot(nW, half)), float(55)).mul(sunIntensityU.mul(0.1));
  col = col.add(vec3(1.0, 0.97, 0.9).mul(spec).mul(float(1).sub(turbidity.mul(0.82))));
  // Breakers follow local bathymetry contours instead of a global diagonal wave.
  // They only exist in shallow ocean water with a real shore-facing depth gradient.
  const shoreBand = smoothstep(0.001, 0.006, depth)
    .mul(float(1).sub(smoothstep(0.025, 0.075, depth)));
  const coastSlope = smoothstep(0.004, 0.045, depthGradient);
  const breakerPhase = sin(depth.mul(260).sub(time.mul(2.8))
    .add(mx_fractal_noise_float(vec3(posXZ.mul(1.8), time.mul(0.08)), 2).mul(2.2)))
    .mul(0.5).add(0.5);
  const breakerCrest = smoothstep(0.72, 0.94, breakerPhase);
  const breakers = shoreBand.mul(coastSlope).mul(oceanMask).mul(breakerCrest);

  // Rapids are a moving-land-water effect. Steep depth changes and speed produce
  // broken highlights, without painting every flowing cell white.
  const rapids = smoothstep(0.4, 1.4, speed)
    .mul(smoothstep(0.005, 0.055, depthGradient))
    .mul(smoothstep(0.0004, 0.025, depth))
    .mul(riverMask);
  const foamAmt = max(breakers.mul(0.72), rapids.mul(0.32))
    .mul(float(1).sub(turbidity.mul(0.7))).min(float(1));
  col = mix(col, FOAM, foamAmt);

  // LAND water (rivers/puddles): stay readable even thin (depth + flow floor).
  const steepRunoff = smoothstep(0.12, 0.5, s.slope);
  const depthOpacity = smoothstep(0.00015, 0.0015, depth).mul(0.62)
    .add(smoothstep(0.0012, 0.05, depth).mul(0.35));
  const flowOpacity = smoothstep(0.04, 0.45, speed).mul(0.36)
    .mul(smoothstep(0.00015, 0.001, depth));
  const channelOrPool = max(
    smoothstep(0.00045, 0.004, depth.mul(speed)),
    smoothstep(0.012, 0.045, depth),
  );
  const landOpacity = max(
    depthOpacity.mul(channelOrPool).mul(mix(float(1), float(0.12), steepRunoff)),
    flowOpacity.mul(channelOrPool),
  );
  const waterBodyOpacity = smoothstep(0.0005, 0.018, depth).mul(0.22);
  const muddyOpacity = turbidity.mul(smoothstep(0.0005, 0.018, depth)).mul(0.82);
  // OCEAN: clear shallows reveal the sandy seabed, ramping to opaque deep blue (the
  // From-Dust look) instead of a flat turquoise sheet.
  const oceanOpacity = smoothstep(0.004, 0.22, depth).mul(0.9).add(0.08);
  // foam (shoreline + rapids) stays opaque even over transparent shallows so it reads.
  // Gate EVERYTHING by water presence so dry land is fully transparent (no phantom
  // water/foam painted over terrain).
  const hasWater = smoothstep(0.00015, 0.001, depth);
  const opacity = clamp(
    max(
      max(max(max(mix(landOpacity, oceanOpacity, oceanMask), waterBodyOpacity), muddyOpacity), foamAmt.mul(0.85)),
      streakStrength.mul(0.12),
    ).mul(hasWater),
    float(0), float(0.97),
  );

  const mat = new MeshBasicNodeMaterial({ transparent: true, side: DoubleSide });
  const visualLift = smoothstep(0.0004, 0.006, depth).mul(0.006);
  mat.positionNode = vec3(s.position.x, s.position.y.add(visualLift), s.position.z);
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
