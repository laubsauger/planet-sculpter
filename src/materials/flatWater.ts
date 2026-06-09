// Flat water, WORLD-space lit (camera-independent base, like the terrain). The
// turquoise depth color is ALWAYS visible from any angle; sun glint + fresnel sky
// reflection are additive bonuses, not the only thing (so it never goes dark/blank
// when you rotate). Animated wave + flow normals (calm, not noise-soup). Surface =
// bedrock + depth; shallow = see-through turquoise, deep = blue; shoreline foam.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, clamp, float, vec2, vec3, length,
  normalize, dot, pow, sin, cos, time, cameraPosition, mx_fractal_noise_float, uniform,
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
// Perf / A-B toggles for potentially heavy stylized effects (1 = on, 0 = off).
// (causticsEnabled moved to flatTerrain — caustics now render on the seabed.)
export const shoreFoamEnabled = uniform(1);
export const oceanSwellEnabled = uniform(1);

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
  // Gradient neighbours use cheap BILINEAR (not bicubic) — 4 taps each vs 16. The waterline
  // smoothness comes from the bicubic `depth`; the gradient only needs an approximate slope.
  const depthL = bilinear((c: any) => textureLoad(waterTex, c).x, fx.sub(2), fy);
  const depthR = bilinear((c: any) => textureLoad(waterTex, c).x, fx.add(2), fy);
  const depthB = bilinear((c: any) => textureLoad(waterTex, c).x, fx, fy.sub(2));
  const depthT = bilinear((c: any) => textureLoad(waterTex, c).x, fx, fy.add(2));
  const depthGradient = length(vec2(depthR.sub(depthL), depthT.sub(depthB)));
  // Wide spatial blur of depth, used ONLY for the deep color/opacity cues. The bathymetric
  // shelf is steep (depth jumps over a few cells), so any sharp depth->color/opacity ramp
  // draws a hard line right there. Averaging depth over a wide kernel spreads that transition
  // smoothly across the surface. `depthColor` stays CRISP in the shallows (clean waterline +
  // caustics + foam keep their sharp `depth`) and blends to the blurred field in deeper water.
  const dW = (ox: number, oy: number) => bilinear((c: any) => textureLoad(waterTex, c).x, fx.add(ox), fy.add(oy));
  const depthWide = dW(10, 10).add(dW(-10, 10)).add(dW(10, -10)).add(dW(-10, -10)).mul(1 / 4);
  const depthColor = mix(depth, depthWide, smoothstep(0.02, 0.12, depth));
  const vel = bilinear((c: any) => textureLoad(velTex, c).xy, fx, fy);
  const flux = bilinear((c: any) => textureLoad(fluxTex, c), fx, fy);
  // Center-weighted blur of sediment so muddy plumes have SOFT edges that merge into the
  // ocean, instead of hard-contrast shapes. (Turbidity ramp below is also widened.)
  const sed = (ox: number, oy: number) => bilinear((c: any) => textureLoad(sedimentTex, c).x, fx.add(ox), fy.add(oy));
  const sediment = sed(0, 0).mul(0.5)
    .add(sed(4, 0).add(sed(-4, 0)).add(sed(0, 4)).add(sed(0, -4)).mul(0.125));
  const speed = length(vec2(vel.x, vel.y));
  // Use the production solver's actual outgoing discharge for visual direction.
  // Reconstructed velocity can point upstream around confluences and obstacles;
  // ambiguous pipe discharge should fade out instead of drawing a confident lie.
  const outgoing = flux.x.add(flux.y).add(flux.z).add(flux.w);
  const flowVector = vec2(flux.y.sub(flux.x), flux.z.sub(flux.w));
  const pipeDir = normalize(flowVector.add(vec2(1e-6, 0)));
  const velocityDir = normalize(vec2(vel.x, vel.y).add(vec2(1e-6, 0)));
  // The direction overlay shows the neighborhood-reconstructed velocity, which can
  // be clear even when a cell splits its outgoing flux among several pipes. Follow
  // that clear route and use local pipe flux only to reject a real contradiction.
  const fluxAgreement = smoothstep(-0.15, 0.3, dot(velocityDir, pipeDir));
  const flowDir = velocityDir;
  const directionConfidence = smoothstep(0.008, 0.12, speed)
    .mul(mix(float(0.35), float(1), fluxAgreement))
    .mul(smoothstep(0.00001, 0.001, outgoing));
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
  // Open-ocean stylized swell: long rolling crest lines along a fixed wind direction,
  // plus a finer cross-chop. One COHERENT wave drives both the surface normal (travelling
  // glints) and crest/trough shading (below) so the ocean reads as moving water instead of
  // a flat sheet. Gated to genuine ocean depth (not the breaker-zone shoreline) so it can't
  // bleed onto land water / rivers.
  // WIDE depth fade-in (not 0.03->0.13): a narrow onset made the swell appear abruptly just
  // past the bathymetric shelf -> a visible line where "deep ocean starts". oceanMask already
  // restricts to real ocean, so the flatWater(depth-gradient) gate is dropped here — at the
  // steep shelf that gate notched to 0 and helped draw the very edge we're trying to remove.
  const deepOcean = oceanMask.mul(smoothstep(0.015, 0.32, depthColor)).mul(still).mul(oceanSwellEnabled);
  const swellDir = normalize(vec2(0.82, 0.57));
  const swellPhase = dot(posXZ, swellDir).mul(1.05).sub(time.mul(0.5));
  // crest slope tilts the normal along the wave direction -> specular/fresnel travel.
  const swellSlope = cos(swellPhase).mul(1.05);
  const oceanSwellNormal = vec3(swellDir.x.mul(swellSlope), float(0), swellDir.y.mul(swellSlope))
    .mul(0.09).mul(deepOcean);
  // faint noise micro-ripple so crests aren't a perfectly clean machine sine.
  const microRipple = grad(posXZ.add(vec2(time.mul(0.07), time.mul(-0.05))), 3.0)
    .mul(0.03).mul(deepOcean);
  const nW: any = normalize(s.worldNormal.add(flowR).add(oceanSwellNormal).add(microRipple));

  // depth color (view-INDEPENDENT base, always visible). Wider ramps than before so the
  // shallow shelf reads as a turquoise->blue GRADIENT instead of saturating to deep blue
  // within a couple of cells (the "hard cut" at the steep bathymetric shelf).
  // shallow = light turquoise (over bright seabed), deepening to dark blue further out.
  // DEEP is reached by a moderate depth so the OPEN ocean genuinely darkens (was staying
  // mid-turquoise while fresnel lit the distance -> looked inverted).
  let col: any = mix(SHALLOW, MID, smoothstep(0.008, 0.1, depthColor));
  col = mix(col, DEEP, smoothstep(0.06, 0.32, depthColor)); // tight enough that deep reads DARK
  // Open-ocean shade: ORGANIC low-frequency drifting fractal noise, NOT crossing sines. Two
  // pure sines interfered into a regular grid of oval blobs that looked awful at distance.
  // Fractal noise gives irregular, gently moving light/dark patches; kept subtle so the deep
  // ocean stays calm rather than homogeneous. Travelling specular glints still come from the
  // swell-tilted normal below.
  const swellShade = mx_fractal_noise_float(vec3(posXZ.mul(0.45), time.mul(0.035)), 3);
  col = mix(col, DEEP.mul(0.85), smoothstep(0.1, -0.5, swellShade).mul(0.08).mul(deepOcean));
  col = mix(col, SKY_REFLECT, smoothstep(0.2, 0.6, swellShade).mul(0.03).mul(deepOcean));

  // (Caustics moved to the SEABED in flatTerrain — projected on the real surface, so they
  // stay put from any angle instead of sliding around on the transparent water sheet.)
  const concentration = sediment.div(max(depth, float(0.003)));
  // WIDE, gentle ramps everywhere on the turbidity path. The previous tight thresholds,
  // modulated by the animated plume noise, sat right at their edge for marginal sediment
  // and toggled on/off frame to frame -> visible flicker. Gradual ramps + lower-contrast
  // noise mean a cell fades smoothly through partial turbidity instead of blinking.
  const sedimentLoad = smoothstep(0.008, 0.3, concentration);
  const plumeNoise = mx_fractal_noise_float(
    vec3(posXZ.sub(visualFlow.mul(time.mul(0.12))).mul(2.4), time.mul(0.035)), 3,
  ).mul(0.5).add(0.5);
  // Suspended sediment forms warm plumes within still-readable blue-green water,
  // rather than replacing the entire surface with a flat brown sheet.
  const turbidity = smoothstep(0.12, 0.82, sedimentLoad.mul(plumeNoise.mul(0.5).add(0.55)));
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
  // fresnel sky reflection (rim, additive bonus). DEPTH-GATED: deep water must NOT get
  // sky-washed lighter, or the distant/grazing open ocean reads lighter than the shallows
  // (inverted). Fade fresnel out with depth so deep stays dark; shallows keep their sheen.
  const fres = pow(float(1).sub(max(float(0), dot(nW, viewW))), float(4));
  const fresDeepFade = float(1).sub(smoothstep(0.04, 0.34, depthColor));
  col = mix(col, SKY_REFLECT, fres.mul(0.12).mul(fresDeepFade).mul(float(1).sub(turbidity.mul(0.75))));
  // sun specular glint (Blinn-Phong, additive sparkle).
  const half = normalize(sunDirUniform.add(viewW));
  const spec = pow(max(float(0), dot(nW, half)), float(55)).mul(sunIntensityU.mul(0.1));
  col = col.add(vec3(1.0, 0.97, 0.9).mul(spec).mul(float(1).sub(turbidity.mul(0.82))));
  // Lapping shore foam: several depth-contour wave trains at different wavelengths and
  // speeds (MULTIPLE patterns, not one ring), warped by noise so each lap is irregular.
  // Hugs the shore (very shallow band) and fades before open water. Mixed to FOAM (white)
  // with its own opacity below so it reads as white foam, not tinted sand.
  const shoreWarp = mx_fractal_noise_float(vec3(posXZ.mul(2.6), time.mul(0.09)), 2).mul(2.0);
  const shoreWarp2 = mx_fractal_noise_float(vec3(posXZ.mul(5.0).add(11.0), time.mul(0.13)), 2).mul(1.4);
  // + time (not -): constant-phase contour has depth DECREASING with time -> crests run
  // shoreward (up onto the shore), not out to sea.
  const lapA = sin(depth.mul(180).add(time.mul(1.9)).add(shoreWarp)).mul(0.5).add(0.5);
  const lapB = sin(depth.mul(360).add(time.mul(3.1)).add(shoreWarp2)).mul(0.5).add(0.5);
  const lapC = sin(depth.mul(640).add(time.mul(4.6)).add(shoreWarp.mul(1.7))).mul(0.5).add(0.5);
  const lapField = max(
    max(smoothstep(0.55, 0.9, lapA), smoothstep(0.6, 0.92, lapB).mul(0.75)),
    smoothstep(0.66, 0.95, lapC).mul(0.5),
  );
  // band hugging the shore (very shallow -> closer to shore than before).
  const shoreBand = smoothstep(0.0006, 0.0035, depth)
    .mul(float(1).sub(smoothstep(0.016, 0.05, depth)));
  const breakers = shoreBand.mul(oceanMask).mul(lapField).mul(shoreFoamEnabled);
  // Swash sheet: the thinnest film right at the waterline gets a translucent white wash
  // that pulses up/down the sand (lapping) on its own slow cycle, independent of breakers.
  const swashPulse = sin(time.mul(1.3).add(shoreWarp.mul(0.6))).mul(0.5).add(0.5);
  const swash = smoothstep(0.0004, 0.0014, depth)
    .mul(float(1).sub(smoothstep(0.0035, 0.012, depth)))
    .mul(oceanMask).mul(mix(float(0.35), float(1), swashPulse)).mul(shoreFoamEnabled);

  // Rapids are a moving-land-water effect. Steep depth changes and speed produce
  // broken highlights, without painting every flowing cell white.
  const rapids = smoothstep(0.4, 1.4, speed)
    .mul(smoothstep(0.005, 0.055, depthGradient))
    .mul(smoothstep(0.0004, 0.025, depth))
    .mul(riverMask);
  const foamAmt = max(max(breakers, swash.mul(0.8)), rapids.mul(0.32))
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
  // OCEAN: clear shallows reveal the sandy seabed + caustics, deepening gradually. WIDE
  // depth ramp so the steep bathymetric shelf no longer maps to a hard transparent->opaque
  // CUT (the visible edge). Capped below full opacity so the dark deep seabed keeps showing
  // through -> the depth darkening comes from BOTH water tint and the seabed gradient.
  // Floor keeps shallows see-through (seabed + caustics); moderate cap so the dark deep
  // SEABED shows through and supplies the deep-blue (no opaque plane under the grid anymore
  // — the ocean continuation is now a frame around the grid, so deep water just reveals the
  // dark seabed). depthColor (blurred) keeps the shallow->deep ramp smooth across the shelf.
  const oceanOpacity = smoothstep(0.004, 0.2, depthColor).mul(0.6).add(0.27);
  // foam (shoreline + rapids) stays opaque even over transparent shallows so it reads.
  // Gate EVERYTHING by water presence so dry land is fully transparent (no phantom
  // water/foam painted over terrain).
  const hasWater = smoothstep(0.00015, 0.001, depth);
  const opacity = clamp(
    max(
      max(max(max(mix(landOpacity, oceanOpacity, oceanMask), waterBodyOpacity), muddyOpacity), foamAmt.mul(0.96)),
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
