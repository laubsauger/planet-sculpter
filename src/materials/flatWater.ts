// Flat water, WORLD-space lit (camera-independent base, like the terrain). The
// turquoise depth color is ALWAYS visible from any angle; sun glint + fresnel sky
// reflection are additive bonuses, not the only thing (so it never goes dark/blank
// when you rotate). Animated wave + flow normals (calm, not noise-soup). Surface =
// bedrock + depth; shallow = see-through turquoise, deep = blue; shoreline foam.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, clamp, float, vec2, vec3, length, Fn, Discard, fract,
  normalize, dot, pow, sin, cos, time, cameraPosition, positionWorld, mx_fractal_noise_float, uniform,
} from 'three/tsl';
import { flatSurface, bilinearTex, bicubicClampedTex, flatGridX, flatGridY, flatSeaLevel } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU } from '../tsl/lighting';

/* eslint-disable @typescript-eslint/no-explicit-any */

const SHALLOW = vec3(0.1, 0.65, 0.64);
const MID = vec3(0.025, 0.4, 0.58);
const DEEP = vec3(0.015, 0.12, 0.3);
const SKY_REFLECT = vec3(0.6, 0.78, 0.95);
const FOAM = vec3(0.95, 0.98, 1.0);
const SILT = vec3(0.27, 0.2, 0.11);
export const flowBandStrength = uniform(0.38);
export const flowBandScale = uniform(7.0);
// Perf / A-B toggles for potentially heavy stylized effects (1 = on, 0 = off).
// (causticsEnabled moved to flatTerrain — caustics now render on the seabed.)
export const shoreFoamEnabled = uniform(1);
export const oceanSwellEnabled = uniform(1);

/** `oceanOnly`: build the OPEN-OCEAN subset for the skirt. Outside the grid every
 * sampler clamps to the border ring, where sediment≡0, velocity≡0 and bed≡0
 * (deep-ocean pin) — so the river/flow/turbidity/foam terms are EXACTLY zero
 * there and can be skipped at graph level (verified by border readback). The
 * kept terms run the identical formulas, so the skirt matches the grid edge
 * pixel-for-pixel while skipping ~half the per-fragment noise/sample work. */
export function makeFlatWater(
  heightTex: Texture,
  waterTex: Texture,
  fluxTex: Texture,
  velTex: Texture,
  sedimentTex: Texture,
  oceanOnly = false,
): MeshBasicNodeMaterial {
  const fx = uv().x.mul(flatGridX), fy = uv().y.mul(flatGridY);

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x.add(textureLoad(waterTex, c).x), true);
  // bicubic depth -> smooth (C1) color + alpha edge instead of grid-aligned bilinear
  // stair-steps that read as pixelation along the waterline.
  const depth = bicubicClampedTex(waterTex, fx, fy);
  // Gradient neighbours use cheap BILINEAR (not bicubic) — hardware-filtered, 1 fetch each.
  // The waterline smoothness comes from the bicubic `depth`; the gradient only needs an
  // approximate slope.
  const depthGradient = oceanOnly ? float(0) : length(vec2(
    bilinearTex(waterTex, fx.add(2), fy).x.sub(bilinearTex(waterTex, fx.sub(2), fy).x),
    bilinearTex(waterTex, fx, fy.add(2)).x.sub(bilinearTex(waterTex, fx, fy.sub(2)).x),
  ));
  // Wide spatial blur of depth, used ONLY for the deep color/opacity cues. The bathymetric
  // shelf is steep (depth jumps over a few cells), so any sharp depth->color/opacity ramp
  // draws a hard line right there. Averaging depth over a wide kernel spreads that transition
  // smoothly across the surface. `depthColor` stays CRISP in the shallows (clean waterline +
  // caustics + foam keep their sharp `depth`) and blends to the blurred field in deeper water.
  const dW = (ox: number, oy: number) => bilinearTex(waterTex, fx.add(ox), fy.add(oy)).x;
  const depthWide = dW(10, 10).add(dW(-10, 10)).add(dW(10, -10)).add(dW(-10, -10)).mul(1 / 4);
  const depthColor = mix(depth, depthWide, smoothstep(0.02, 0.12, depth));
  const vel = oceanOnly ? vec2(0, 0) : bilinearTex(velTex, fx, fy).xy;
  const flux = oceanOnly ? vec3(0).xxxx : bilinearTex(fluxTex, fx, fy);
  // Center-weighted blur of sediment so muddy plumes have SOFT edges that merge into the
  // ocean, instead of hard-contrast shapes. (Turbidity ramp below is also widened.)
  const sed = (ox: number, oy: number) => bilinearTex(sedimentTex, fx.add(ox), fy.add(oy)).x;
  const sediment = oceanOnly ? float(0) : sed(0, 0).mul(0.5)
    .add(sed(4, 0).add(sed(-4, 0)).add(sed(0, 4)).add(sed(0, -4)).mul(0.125));
  const speed = oceanOnly ? float(0) : length(vec2(vel.x, vel.y));
  // Use the production solver's actual outgoing discharge for visual direction.
  // Reconstructed velocity can point upstream around confluences and obstacles;
  // ambiguous pipe discharge should fade out instead of drawing a confident lie.
  const outgoing = flux.x.add(flux.y).add(flux.z).add(flux.w);
  const flowVector = vec2(flux.y.sub(flux.x), flux.z.sub(flux.w));
  const pipeDir = normalize(flowVector.add(vec2(1e-6, 0)));
  const velocityDir = normalize(vec2(vel.x, vel.y).add(vec2(1e-6, 0)));
  // The reconstructed velocity is smooth but can point UPSTREAM near confluences
  // and obstacles. The net pipe discharge direction cannot (flux follows the head
  // gradient), so where the two disagree the pattern must follow the PIPE
  // direction — never advect against the flow. Fading alone (the old 0.35
  // confidence floor) still let bands visibly crawl uphill, which breaks the
  // whole read the moment it happens.
  const fluxAgreement = smoothstep(0.0, 0.3, dot(velocityDir, pipeDir));
  const flowDir = normalize(mix(pipeDir, velocityDir, fluxAgreement));
  const directionConfidence = oceanOnly ? float(0) : smoothstep(0.008, 0.12, speed)
    .mul(mix(float(0.35), float(1), fluxAgreement))
    .mul(smoothstep(0.00001, 0.001, outgoing));
  const visualFlow = flowDir.mul(speed).mul(directionConfidence);

  // calm wave + flow normals (subtle ripple, not noise soup). positionWorld is the
  // rasterizer-interpolated displaced surface — identical to re-deriving from uv
  // (x/z are linear in uv), with zero fragment recomputation.
  const posXZ = vec2(positionWorld.x, positionWorld.z);
  const e = float(0.06);
  const grad = (q: any, freq: number) => {
    const nz = (p: any) => mx_fractal_noise_float(vec3(p.x, p.y, 0).mul(freq), 2);
    return vec3(nz(q.add(vec2(e, 0))).sub(nz(q.sub(vec2(e, 0)))), float(0), nz(q.add(vec2(0, e))).sub(nz(q.sub(vec2(0, e)))));
  };
  // Flow ripples: noise advected ALONG the local flow -> rivers ripple downstream,
  // each follows its own direction (not a shared global wave).
  // Full ripple on shallow flow; deep moving water keeps ~35% instead of cutting
  // to a dead-flat mirror at depth 0.08 (deep slow rivers looked static).
  const shallowFlow = float(1).sub(smoothstep(0.025, 0.08, depth).mul(0.65));
  // FLOWMAP advection (bounded shear): naive `pos - flow·time` with a spatially
  // varying flow shears any pattern into ever-finer filaments as time grows —
  // after a minute every strong flow was fingerprint-fine static. Two advection
  // windows half a period apart, triangle cross-faded, keep the pattern moving
  // with the current while resetting the shear every `period` seconds.
  const flowSampled = (fnq: (q: any) => any, rate: number, period: number) => {
    const ph1 = fract(time.div(period));
    const ph2 = fract(time.div(period).add(0.5));
    const w1 = float(1).sub(ph1.sub(0.5).abs().mul(2));
    const q1 = posXZ.sub(visualFlow.mul(ph1.mul(period * rate)));
    const q2 = posXZ.sub(visualFlow.mul(ph2.mul(period * rate)));
    return fnq(q1).mul(w1).add(fnq(q2).mul(float(1).sub(w1)));
  };
  // Ripples advect at ~the actual current speed (0.22 read as molasses — the
  // pattern crawled while the sim water clearly moved faster).
  const flowR = oceanOnly ? vec3(0, 0, 0) : flowSampled((q: any) => grad(q, 1.5), 0.85, 3.0)
    .mul(speed.min(float(1.2)).mul(0.11).add(0.012))
    .mul(shallowFlow).mul(directionConfidence);
  // Ambient swell belongs to the OPEN OCEAN ONLY (bedrock below sea level). All water
  // on land — rivers, rain runoff, puddles — must read by its OWN flow (flowR), never
  // share one global wave. Gate the swell by an ocean-basin mask so it can't bleed onto
  // land water and produce that homogeneous diagonal pattern when it rains / on rivers.
  const bed = bilinearTex(heightTex, fx, fy).x;
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
  const deepOcean = oceanOnly
    ? oceanMask.mul(smoothstep(0.015, 0.32, depthColor)).mul(oceanSwellEnabled) // still≡1 (speed≡0)
    : oceanMask.mul(smoothstep(0.015, 0.32, depthColor)).mul(still).mul(oceanSwellEnabled);
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
  // Suspended sediment forms warm plumes within still-readable blue-green water,
  // rather than replacing the entire surface with a flat brown sheet.
  const turbidity = oceanOnly ? float(0) : smoothstep(0.12, 0.82, sedimentLoad.mul(
    mx_fractal_noise_float(
      vec3(posXZ.sub(visualFlow.mul(time.mul(0.12))).mul(2.4), time.mul(0.035)), 3,
    ).mul(0.5).add(0.5).mul(0.5).add(0.55),
  ));
  col = oceanOnly ? col : mix(col, SILT, turbidity.mul(0.66));

  // Long, narrow streaks run ALONG the validated downhill flow and their phase
  // travels downstream. Avoid transverse contour bands: those read as uphill
  // waves even when the velocity vector itself is correct.
  const riverMask = float(1).sub(oceanMask);
  let streakStrength: any = float(0);
  if (!oceanOnly) {
    // Flow read = noise ADVECTED with the current and SMEARED along the local flow
    // direction (3 samples offset downstream → long soft bands). The pattern moves
    // at the simulated speed, stretches along the gravity-driven route, and cannot
    // moiré: no sin(dot(pos, flowDir)) phase — that fringes into fine rings
    // wherever flowDir rotates (eddies/confluences) and flickers at distance.
    // LOW frequency (shore-wave register, a notch smaller) — band strength is
    // GRADED by speed so a lazy drift reads faint and a strong current bold.
    const alongSmear = (q: any) => {
      const sm = (o: number) => mx_fractal_noise_float(
        vec3(q.sub(flowDir.mul(o)).mul(flowBandScale.mul(0.25)), time.mul(0.05)), 2,
      );
      return sm(0).add(sm(0.3)).add(sm(0.6)).mul(1 / 3);
    };
    const elong = flowSampled(alongSmear, 2.2, 3.0).mul(0.5).add(0.5);
    const streak = smoothstep(0.55, 0.85, elong);
    const speedRamp = smoothstep(0.02, 0.5, speed);
    // Deep-river fade widened (was gone by depth 0.2): a deep slow river still
    // reads as MOVING water, not a static mirror.
    streakStrength = streak.mul(speedRamp).mul(directionConfidence).mul(riverMask)
      .mul(smoothstep(0.00006, 0.05, depth))
      .mul(float(1).sub(smoothstep(0.12, 0.45, depth)));
    col = mix(col, SKY_REFLECT, streakStrength.mul(flowBandStrength));
  }

  // world-space lighting: gentle sun diffuse keeps base bright + readable any angle.
  const viewW = normalize(cameraPosition.sub(positionWorld));
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
  let foamAmt: any = float(0);
  if (!oceanOnly) {
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
    // WHITEWATER on steep falls: where the BED is steep and the current fast, the
    // water aerates — broad churning white masses streaking downslope, readable
    // from a distance (the masks are low-frequency; churn/rush only modulate).
    // This also replaces the old "steep runoff goes transparent" read: a fall now
    // shows as opaque rushing white instead of breaking up into invisible film.
    // STREAM POWER (bed slope × current speed) drives aeration — an 8° grade with a
    // fast current froths (measured river core: slope ~0.009, speed 0.3-0.8 → power
    // 0.003-0.007); a steep fall at any real speed saturates. Separate slope/speed
    // gates multiplied to ~0 here even when both were individually significant.
    // STYLIZED like the shore waves: broad SMOOTH froth patches (low-frequency
    // advected noise through a wide smoothstep), not fine grain — high-freq churn
    // read as TV static from any distance. Patch COVERAGE grows with stream power
    // (light rapids = sparse white tufts, strong falls = near-solid froth) so
    // weak vs strong whitewater read differently. NO sin(dot(pos, flowDir))
    // stripes: a rotating flowDir sweeps that phase through arbitrary
    // frequencies → moiré rings, flicker, uncoupled from the real flow.
    const streamPower = s.slope.mul(speed);
    const power = smoothstep(0.0008, 0.01, streamPower);
    // Big bands PERPENDICULAR to the flow (standing-wave read, visible from a
    // distance): the advected noise is smeared ACROSS the current — 3 samples
    // offset along the perpendicular axis — at half the old frequency. Bands
    // travel downstream with the flow; still no sin-phase anywhere, so no moiré.
    const acrossDir = vec2(flowDir.y.negate(), flowDir.x);
    // Slow z-evolution (0.12, was 0.45): the noise field's OWN drift reads as
    // waves radiating 360° from a point once it rivals the advection speed —
    // motion must come from the flow, the field itself only simmers.
    const crossSmear = (q: any) => {
      const cs = (o: number) => mx_fractal_noise_float(
        vec3(q.sub(acrossDir.mul(o)).mul(0.9), time.mul(0.12)), 2,
      );
      return cs(0).add(cs(0.25)).add(cs(0.5)).mul(1 / 3);
    };
    const churn = flowSampled(crossSmear, 1.4, 3.5).mul(0.5).add(0.5);
    const patches = smoothstep(float(0.62).sub(power.mul(0.4)), float(0.82).sub(power.mul(0.18)), churn);
    const whitewater = power.mul(float(0.3).add(patches.mul(0.7)))
      .mul(smoothstep(0.00008, 0.0015, depth))
      .mul(riverMask);
    foamAmt = max(max(max(breakers, swash.mul(0.8)), rapids.mul(0.32)), whitewater.mul(0.9))
      .mul(float(1).sub(turbidity.mul(0.7))).min(float(1));
    col = mix(col, FOAM, foamAmt);
  }

  // LAND water (rivers/puddles): stay readable even thin (depth + flow floor).
  // Skipped for the open-ocean skirt — it's multiplied by (1-oceanMask)≡0 there.
  let landOpacity: any = float(0);
  if (!oceanOnly) {
    const steepRunoff = smoothstep(0.12, 0.5, s.slope);
    const depthOpacity = smoothstep(0.00015, 0.0015, depth).mul(0.62)
      .add(smoothstep(0.0012, 0.05, depth).mul(0.35));
    const flowOpacity = smoothstep(0.04, 0.45, speed).mul(0.36)
      .mul(smoothstep(0.00015, 0.001, depth));
    const channelOrPool = max(
      smoothstep(0.00045, 0.004, depth.mul(speed)),
      smoothstep(0.012, 0.045, depth),
    );
    // A thin but connected pipe route is still valid flowing water. Give it a
    // restrained visibility floor so low-discharge channels can be inspected
    // without making the simulation physically deeper or more viscous.
    // Depth floor lowered (was 0.00012): an ultra-thin ROUTED film is exactly the
    // "flow breaks up on the flats" gap — moving water must stay readable.
    const connectedFlow = smoothstep(0.00001, 0.0007, outgoing)
      .mul(smoothstep(0.00003, 0.0004, depth));
    landOpacity = max(
      depthOpacity.mul(max(channelOrPool, connectedFlow.mul(0.62))).mul(mix(float(1), float(0.12), steepRunoff)),
      flowOpacity.mul(max(channelOrPool, connectedFlow)),
    );
    // Thin-film floor: a connected MOVING film keeps a modest constant opacity even
    // where depthOpacity/flowOpacity (deeper-water ramps) are still ~0.
    landOpacity = max(landOpacity, connectedFlow.mul(smoothstep(0.02, 0.3, speed)).mul(0.38));
  }
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
  // Flowing films count as water below the still-water depth floor — the routed
  // thin sheet crossing a flat must not vanish (matches the discard threshold).
  const hasWater = oceanOnly
    ? smoothstep(0.00015, 0.001, depth)
    : max(smoothstep(0.00015, 0.001, depth),
        smoothstep(0.00001, 0.0007, outgoing).mul(smoothstep(0.00003, 0.0004, depth)).mul(0.9));
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
  // Dry land: opacity is exactly 0 below the hasWater floor — discard those
  // fragments up front so spatially-coherent dry regions (most of the island)
  // skip the noise/lighting work instead of blending an invisible result.
  // The skirt is always deep water, so it never discards (skip the test there).
  mat.colorNode = oceanOnly ? col : Fn(() => {
    // Below the thin-film floor (matches hasWater's lowest onset) — truly dry.
    Discard(depth.lessThanEqual(float(0.00003)));
    return col;
  })();
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
