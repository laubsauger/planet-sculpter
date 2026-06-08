// Pipe-model water on the single equirect grid. Neighbor indexing: longitude
// WRAPS (mod W), latitude CLAMPS at poles (masked so poles don't inflow phantom
// water). Water stored as VOLUME; depth = vol/area (cos-lat). No seam table, no
// cross-face anything — the wrap is one line. Reuses FluidUniforms (water.ts).

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import {
  Fn,
  instanceIndex,
  textureLoad,
  textureStore,
  ivec2,
  uvec2,
  uint,
  int,
  float,
  vec2,
  vec4,
  length,
  max,
  min,
  mix,
  sin,
  smoothstep,
  uniform,
} from 'three/tsl';
import { seaLevelUniform } from '../tsl/heightScale';
import { type FluidUniforms, mudViscosityFactor } from './passes/water';

/** Flow-aware evaporation (V42): fast moving water evaporates LESS than
 *  standing/slow water, so rivers reach the sea while pools/lakes still lose
 *  enough to prevent flooding. Calibrated to the actual velocity scale (k≈1.96 ->
 *  river speeds ~2-3): protection only kicks in WELL above standing-water speed
 *  (deadzone at 0.4*ref) or thin sheet flow stays protected and the planet floods. */
export const evapFlowReduce = uniform(0.7); // max evap reduction for fast flow
export const evapSpeedRef = uniform(2.5); // speed at which reduction saturates
/** Deep water also evaporates slowly: proportional evap removes a FRACTION of the
 *  column, so a deep still lake (speed≈0, no flow protection) loses huge absolute
 *  depth and drains to a puddle. Protect deep water too — saturating, so an
 *  equilibrium depth still exists (no flood). Protected if MOVING or DEEP; only
 *  shallow still films evaporate at full rate. */
export const evapDeepReduce = uniform(0.5); // deep water shrinks slowly (⊥ immortal pools)
export const evapDeepRef = uniform(0.06); // depth at which deep protection saturates
export const evapShallowRef = uniform(0.008); // below this depth = full evap (dries films)
/** Polar ice caps: water frozen (no liquid sim) where cos(lat) below this, so the
 *  equirect pole singularity (cell area -> 0 -> depth = vol/area explodes) can't
 *  spike/flood. cos(lat)=sin(v*PI); 0.18 ≈ poleward of ~80° latitude. */
export const poleCapCos = uniform(0.26);
const PI = float(Math.PI);

/** Orographic rainfall (nature-approx): rain falls mostly on high terrain / mountain
 *  faces, little over oceans & plains. Concentrates water input UP-slope so it flows
 *  down and forms rivers, instead of raining everywhere and just pooling. 0 = uniform
 *  rain, 1 = fully elevation-weighted. rainHighRef = height above sea for full rain. */
export const rainOrographic = uniform(0.9);
export const rainHighRef = uniform(0.2);

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];
/* eslint-disable @typescript-eslint/no-explicit-any */
const EPS = 1e-6;

function coords(w: number) {
  const N = uint(w);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}
/** wrap longitude (mod W), clamp latitude. */
const wrapX = (x: any, w: number) => x.add(int(w)).mod(int(w));
const clampY = (y: any, h: number) => y.toFloat().max(float(0)).min(float(h - 1)).toInt();

/** Pipe outflow flux (L,R,T,B) on the grid. */
export function gridFlux(
  b: StorageTexture,
  d: StorageTexture,
  area: StorageTexture,
  fPrev: StorageTexture,
  fOut: StorageTexture,
  sediment: StorageTexture,
  w: number,
  h: number,
  p: FluidUniforms,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const surf = (cx: any, cy: any) => {
      const c = ivec2(wrapX(cx, w), clampY(cy, h));
      return textureLoad(b, c).x.add(textureLoad(d, c).x.div(max(textureLoad(area, c).x, float(EPS))));
    };
    const hc = surf(ix, iy);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const prev = textureLoad(fPrev, ivec2(ix, iy));

    let k: any = p.dt.mul(p.pipeArea).mul(p.gravity).div(p.pipeLength);
    const conc = textureLoad(sediment, ivec2(ix, iy)).x.div(max(dc, float(EPS)));
    k = k.div(float(1).add(conc.mul(mudViscosityFactor)).min(float(8)));

    const fL = max(float(0), prev.x.mul(p.damping).add(k.mul(hc.sub(surf(ix.sub(1), iy)))));
    const fR = max(float(0), prev.y.mul(p.damping).add(k.mul(hc.sub(surf(ix.add(1), iy)))));
    const fT = max(float(0), prev.z.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.add(1))))));
    const fB = max(float(0), prev.w.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.sub(1))))));
    // poles: clamp makes the beyond-pole neighbor == self -> diff 0 -> flux 0
    // automatically (natural seal). Longitude wraps -> no seal needed in X.
    const sum = fL.add(fR).add(fT).add(fB);
    const l2 = p.pipeLength.mul(p.pipeLength);
    const scale = min(float(1), dc.mul(l2).div(max(sum.mul(p.dt), float(EPS))));
    textureStore(fOut, uvec2(x, y), vec4(fL.mul(scale), fR.mul(scale), fT.mul(scale), fB.mul(scale))).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}

/** Fused update: flow + evap (flat+proportional) + rain/source + sea-fill. */
export function gridUpdate(
  d: StorageTexture,
  f: StorageTexture,
  b: StorageTexture,
  source: StorageTexture,
  rainfall: StorageTexture,
  area: StorageTexture,
  vel: StorageTexture,
  dOut: StorageTexture,
  w: number,
  h: number,
  p: FluidUniforms,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const self = textureLoad(f, ivec2(ix, iy));
    const outflow = self.x.add(self.y).add(self.z).add(self.w);
    const fAt = (cx: any, cy: any) => textureLoad(f, ivec2(wrapX(cx, w), clampY(cy, h)));
    // X always wraps; Y masked at poles (no neighbor beyond).
    const mT = iy.lessThan(int(h - 1)).select(float(1), float(0));
    const mB = iy.greaterThan(int(0)).select(float(1), float(0));
    const inflow = fAt(ix.sub(1), iy).y
      .add(fAt(ix.add(1), iy).x)
      .add(fAt(ix, iy.add(1)).w.mul(mT))
      .add(fAt(ix, iy.sub(1)).z.mul(mB));

    const areaC = max(textureLoad(area, ivec2(ix, iy)).x, float(EPS));
    const l2 = p.pipeLength.mul(p.pipeLength);
    const emit = textureLoad(source, ivec2(ix, iy)).x;
    // orographic rain: weight by elevation so mountains catch the rain, oceans/
    // plains catch little -> water enters up-slope and runs off into rivers.
    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const oro = smoothstep(seaLevelUniform, seaLevelUniform.add(rainHighRef), bc);
    const rainW = textureLoad(rainfall, ivec2(ix, iy)).x.mul(mix(float(1), oro, rainOrographic));
    const rain = p.source.mul(rainW);
    let next: any = inflow.sub(outflow).mul(p.dt).div(l2).add(dc);
    next = next.sub(p.loss.mul(p.dt).mul(areaC));
    next = next.add(rain.add(emit).mul(p.dt).mul(areaC));
    // flow-aware proportional evap (V42): moving water evaporates less so rivers
    // persist; standing water evaporates full rate -> no flood. deadzone below
    // 0.4*ref so slow/standing & thin sheet flow are NOT protected.
    const velC = textureLoad(vel, ivec2(ix, iy));
    const speed = length(vec2(velC.x, velC.y));
    const flowKeep = smoothstep(evapSpeedRef.mul(0.4), evapSpeedRef, speed).mul(evapFlowReduce);
    // deep water protected too (saturating) so still lakes persist; equilibrium
    // still exists -> no flood. protected if MOVING or DEEP; only shallow still
    // films evaporate at full rate.
    const depthNow = next.div(areaC);
    const deepKeep = smoothstep(evapShallowRef, evapDeepRef, depthNow).mul(evapDeepReduce);
    const keep = max(flowKeep, deepKeep).min(float(0.97));
    const evapEff = p.evapProp.mul(float(1).sub(keep));
    next = next.mul(max(float(0), float(1).sub(evapEff.mul(p.dt))));
    // sea fill: pin deep ocean to sea level (one flat global level).
    const below = bc.mul(-1).add(seaLevelUniform);
    const need = below.max(float(0)).mul(areaC);
    const ocean = smoothstep(0.0, 0.03, below).mul(0.95);
    next = mix(next.max(need), need, ocean);
    next = max(next, float(0));
    // POLAR ICE CAP: freeze (zero liquid) where the equirect cell shrinks toward
    // the pole singularity -> no vol/area depth explosion / spikes.
    const cosLat = sin(iy.toFloat().div(float(h - 1)).mul(PI));
    const liquid = smoothstep(poleCapCos.mul(0.5), poleCapCos, cosLat);
    next = next.mul(liquid);
    textureStore(dOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}
