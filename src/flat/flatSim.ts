// Flat-grid sim: pipe-model water + velocity + hydraulic/thermal erosion. Uniform
// cell area (depth = volume directly), edges CLAMPED + sealed (water drains into
// the ocean border, which the sea-fill pins to sea level). No wrap, no poles, no
// cos-lat — the sphere machinery is gone. Reuses the tuned uniforms.

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import { Vector2 } from 'three';
import {
  Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int,
  float, vec2, vec4, max, min, length, mix, smoothstep, If, uniform,
  mx_fractal_noise_float, sqrt, vec3, time, clamp,
} from 'three/tsl';
import { GridField, buildGridCopy, buildGridFill, buildGridSeed } from '../sim/gridStore';
import { mudViscosityFactor, waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import {
  evapFlowReduce, evapSpeedRef, evapDeepReduce, evapDeepRef, evapShallowRef,
  rainOrographic, rainHighRef,
} from '../sim/gridWater';
import { flatSeaLevel } from '../tsl/flatSurface';
import { momentumSubstepCount } from './momentumCfl';

type CN = Parameters<WebGPURenderer['compute']>[0];
export const FLAT_WATER_SOLVERS = ['pipe', 'momentum'] as const;
export type FlatWaterSolver = (typeof FLAT_WATER_SOLVERS)[number];
/* eslint-disable @typescript-eslint/no-explicit-any */
const EPS = 1e-6;

function coords(w: number) {
  const N = uint(w);
  return { x: instanceIndex.mod(N), y: instanceIndex.div(N), ix: int(instanceIndex.mod(N)), iy: int(instanceIndex.div(N)) };
}
const cX = (x: any, w: number) => x.toFloat().max(float(0)).min(float(w - 1)).toInt();
const cY = (y: any, h: number) => y.toFloat().max(float(0)).min(float(h - 1)).toInt();

/** Pipe outflow flux (L,R,T,B). */
function flatFlux(b: StorageTexture, d: StorageTexture, sediment: StorageTexture, source: StorageTexture, fPrev: StorageTexture, fOut: StorageTexture, w: number, h: number): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const surf = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x.add(textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x);
    const emit = textureLoad(source, ivec2(ix, iy)).x;
    // Treat a spring as hydraulic head as well as incoming volume. This makes the
    // pipe solver evacuate discharge on the next flux pass instead of accumulating
    // a stationary mound until the geometric water surface becomes steep enough.
    const hc = surf(ix, iy).add(emit.mul(0.04));
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const prev = textureLoad(fPrev, ivec2(ix, iy));
    const prevL = textureLoad(fPrev, ivec2(cX(ix.sub(1), w), iy));
    const prevR = textureLoad(fPrev, ivec2(cX(ix.add(1), w), iy));
    const prevT = textureLoad(fPrev, ivec2(ix, cY(iy.add(1), h)));
    const prevB = textureLoad(fPrev, ivec2(ix, cY(iy.sub(1), h)));
    const concentration = textureLoad(sediment, ivec2(ix, iy)).x.div(max(dc, float(EPS)));
    const viscosity = float(1).add(concentration.mul(mudViscosityFactor)).min(float(8));
    const k: any = p.dt.mul(p.pipeArea).mul(p.gravity).div(p.pipeLength).div(viscosity);
    let fL = max(float(0), prev.x.mul(p.damping).add(k.mul(hc.sub(surf(ix.sub(1), iy)))));
    let fR = max(float(0), prev.y.mul(p.damping).add(k.mul(hc.sub(surf(ix.add(1), iy)))));
    let fT = max(float(0), prev.z.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.add(1))))));
    let fB = max(float(0), prev.w.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.sub(1))))));
    // A pipe cell normally forgets the direction of water entering from its
    // neighbors. At a grade break that makes discharge stop until a pressure mound
    // grows high enough to restart it. Transfer the coherent part of incoming flux
    // into the forward outlet; the volume clamp below keeps the update conservative.
    const inL = prevL.y, inR = prevR.x, inT = prevT.w, inB = prevB.z;
    const incoming = inL.add(inR).add(inT).add(inB);
    const incomingDir = vec2(inL.sub(inR), inB.sub(inT));
    const coherence = length(incomingDir).div(max(incoming, float(EPS)));
    const continuation = coherence.mul(0.52);
    fL = fL.add(max(float(0), incomingDir.x.mul(-1)).mul(continuation));
    fR = fR.add(max(float(0), incomingDir.x).mul(continuation));
    fT = fT.add(max(float(0), incomingDir.y).mul(continuation));
    fB = fB.add(max(float(0), incomingDir.y.mul(-1)).mul(continuation));
    // seal the outer boundary (water leaves via the ocean ring, not the wall).
    fL = ix.lessThan(int(1)).select(float(0), fL);
    fR = ix.greaterThan(int(w - 2)).select(float(0), fR);
    fT = iy.greaterThan(int(h - 2)).select(float(0), fT);
    fB = iy.lessThan(int(1)).select(float(0), fB);
    const sum = fL.add(fR).add(fT).add(fB);
    const l2 = p.pipeLength.mul(p.pipeLength);
    const scale = min(float(1), dc.mul(l2).div(max(sum.mul(p.dt), float(EPS))));
    textureStore(fOut, uvec2(x, y), vec4(fL.mul(scale), fR.mul(scale), fT.mul(scale), fB.mul(scale))).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Velocity (vx,vy) from flux imbalance + flow inertia. */
function flatVelocity(f: StorageTexture, d: StorageTexture, velPrev: StorageTexture, velOut: StorageTexture, w: number, h: number): CN {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const depth = textureLoad(d, ivec2(ix, iy)).x;
    const out: any = vec4(0, 0, 0, 1).toVar();
    If(depth.greaterThan(float(1e-5)), () => {
      const self = textureLoad(f, ivec2(ix, iy));
      const Lr = textureLoad(f, ivec2(cX(ix.sub(1), w), iy)).y;
      const Rl = textureLoad(f, ivec2(cX(ix.add(1), w), iy)).x;
      const Bt = textureLoad(f, ivec2(ix, cY(iy.sub(1), h))).z;
      const Tb = textureLoad(f, ivec2(ix, cY(iy.add(1), h))).w;
      const dc = max(depth, float(0.02));
      const vx = Lr.sub(self.x).add(self.y.sub(Rl)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      const vy = Bt.sub(self.w).add(self.z.sub(Tb)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      const bx = cX(ix.toFloat().sub(vx.mul(0.6)), w);
      const by = cY(iy.toFloat().sub(vy.mul(0.6)), h);
      const prev = textureLoad(velPrev, ivec2(bx, by));
      out.assign(vec4(mix(vx, prev.x, erosionUniforms.flowInertia), mix(vy, prev.y, erosionUniforms.flowInertia), 0, 1));
    });
    textureStore(velOut, uvec2(x, y), out).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Fused water update: flow + orographic rain + flow/deep-aware evap + sea-fill. */
function flatUpdate(d: StorageTexture, f: StorageTexture, b: StorageTexture, source: StorageTexture, vel: StorageTexture, dOut: StorageTexture, w: number, h: number): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const self = textureLoad(f, ivec2(ix, iy));
    const outflow = self.x.add(self.y).add(self.z).add(self.w);
    const fAt = (cx: any, cy: any) => textureLoad(f, ivec2(cX(cx, w), cY(cy, h)));
    const mT = iy.lessThan(int(h - 1)).select(float(1), float(0));
    const mB = iy.greaterThan(int(0)).select(float(1), float(0));
    const mL = ix.greaterThan(int(0)).select(float(1), float(0));
    const mR = ix.lessThan(int(w - 1)).select(float(1), float(0));
    const inflow = fAt(ix.sub(1), iy).y.mul(mL)
      .add(fAt(ix.add(1), iy).x.mul(mR))
      .add(fAt(ix, iy.add(1)).w.mul(mT))
      .add(fAt(ix, iy.sub(1)).z.mul(mB));
    const l2 = p.pipeLength.mul(p.pipeLength);
    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const sourceAt = (cx: any, cy: any) => textureLoad(source, ivec2(cX(cx, w), cY(cy, h))).x;
    const surfaceAt = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x
      .add(textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x);
    const routedShare = (sx: any, sy: any, tx: any, ty: any) => {
      const hs = surfaceAt(sx, sy);
      const sourceHere = sourceAt(sx, sy);
      const drop = (nx: any, ny: any) => {
        // A spring on a slope should choose the lowest outlet immediately rather
        // than inflate a symmetric source pond. Retain only a tiny flat-ground
        // pressure path so a source can still escape a genuinely level cell.
        const downhill = max(float(0.00002), hs.sub(surfaceAt(nx, ny)).add(float(0.00015)));
        const downhillBias = downhill.mul(downhill.add(float(0.0008)));
        // Prefer moving down the radial source gradient, out of the emitter
        // footprint. A small baseline preserves routing on a flat source plateau.
        const outward = max(float(0), sourceHere.sub(sourceAt(nx, ny))).mul(120).add(float(0.08));
        return downhillBias.mul(outward);
      };
      const l = drop(sx.sub(1), sy), r = drop(sx.add(1), sy);
      const t = drop(sx, sy.add(1)), bot = drop(sx, sy.sub(1));
      const total = l.add(r).add(t).add(bot);
      let selected: any = l;
      selected = tx.greaterThan(sx).select(r, selected);
      selected = ty.greaterThan(sy).select(t, selected);
      selected = ty.lessThan(sy).select(bot, selected);
      return sourceHere.mul(selected.div(max(total, float(EPS))));
    };
    // Route spring discharge directly into its downhill-adjacent cells. This is
    // conservative (the four shares sum to the source rate) and avoids creating
    // a circular reservoir at the emitter before the pipe pressure reacts.
    const emit = routedShare(ix.sub(1), iy, ix, iy)
      .add(routedShare(ix.add(1), iy, ix, iy))
      .add(routedShare(ix, iy.sub(1), ix, iy))
      .add(routedShare(ix, iy.add(1), ix, iy));
    const oro = smoothstep(flatSeaLevel, flatSeaLevel.add(rainHighRef), bc);
    // Weather: low-frequency animated noise so rain falls in DRIFTING patches/cells,
    // not a uniform sheet. x/y drift + an evolving z give moving weather fronts.
    const ux = ix.toFloat().div(float(w)), uy = iy.toFloat().div(float(h));
    const cloud = mx_fractal_noise_float(
      vec3(ux.mul(2.6).add(time.mul(0.02)), uy.mul(2.6).sub(time.mul(0.015)), time.mul(0.05)), 3);
    const cloudMask = clamp(cloud.mul(0.9).add(0.65), float(0), float(1.5));
    // orographic: mountains wring more rain out of the passing clouds.
    const rain = p.source.mul(mix(float(1), oro, rainOrographic)).mul(cloudMask);
    let next: any = inflow.sub(outflow).mul(p.dt).div(l2).add(dc);
    next = next.add(rain.add(emit).mul(p.dt));
    // flow + deep aware evaporation.
    const velC = textureLoad(vel, ivec2(ix, iy));
    const speed = length(vec2(velC.x, velC.y));
    const flowKeep = smoothstep(evapSpeedRef.mul(0.4), evapSpeedRef, speed).mul(evapFlowReduce);
    const deepKeep = smoothstep(evapShallowRef, evapDeepRef, next).mul(evapDeepReduce);
    const keep = max(flowKeep, deepKeep).min(float(0.97));
    next = next.mul(max(float(0), float(1).sub(p.evapProp.mul(float(1).sub(keep)).mul(p.dt))));
    // The ocean is an effectively infinite reservoir. Keep its datum at sea level
    // and relax excess river water once it reaches deeper offshore cells. Without
    // this outlet the sealed map's entire ocean slowly rises, creating artificial
    // backwater at every mouth and eventually stopping otherwise healthy rivers.
    // Sediment is not removed here; it remains available to settle and build deltas.
    const need = bc.mul(-1).add(flatSeaLevel).max(float(0));
    next = next.max(need);
    // Keep the beach, shelf, and river mouth fully hydraulic. Only the genuinely
    // deep ocean acts as the infinite-reservoir sink; relaxing at first contact
    // kills mouth momentum and encourages a sediment bar exactly at the shoreline.
    const offshore = float(1).sub(smoothstep(flatSeaLevel.sub(0.14), flatSeaLevel.sub(0.06), bc));
    const oceanRelax = smoothstep(float(0), float(0.9), p.dt.mul(8)).mul(offshore);
    next = mix(next, need, oceanRelax);
    textureStore(dOut, uvec2(x, y), vec4(max(next, float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Conservative shallow-water prototype. Stores momentum as (hu,hv), uses a
 * Rusanov finite-volume flux, and applies bed slope/friction as source terms. */
function flatMomentumUpdate(
  d: StorageTexture,
  momentum: StorageTexture,
  b: StorageTexture,
  source: StorageTexture,
  dOut: StorageTexture,
  momentumOut: StorageTexture,
  fluxOut: StorageTexture,
  w: number,
  h: number,
): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const state = (cx: any, cy: any) => {
      const depth = textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x.max(float(0));
      const mom = textureLoad(momentum, ivec2(cX(cx, w), cY(cy, h)));
      const wet = depth.greaterThan(float(1e-5)).select(float(1), float(0));
      return vec3(depth, mom.x.mul(wet), mom.y.mul(wet));
    };
    const fluxX = (u: any) => {
      const invH = float(1).div(max(u.x, float(1e-5)));
      return vec3(u.y, u.y.mul(u.y).mul(invH).add(p.gravity.mul(u.x).mul(u.x).mul(0.5)), u.y.mul(u.z).mul(invH));
    };
    const fluxY = (u: any) => {
      const invH = float(1).div(max(u.x, float(1e-5)));
      return vec3(u.z, u.z.mul(u.y).mul(invH), u.z.mul(u.z).mul(invH).add(p.gravity.mul(u.x).mul(u.x).mul(0.5)));
    };
    const rusanovX = (left: any, right: any) => {
      const hL: any = max(left.x, float(0));
      const hR: any = max(right.x, float(0));
      const aL: any = left.y.abs().div(max(hL, float(1e-5))).add(sqrt(p.gravity.mul(hL) as any));
      const aR: any = right.y.abs().div(max(hR, float(1e-5))).add(sqrt(p.gravity.mul(hR) as any));
      return fluxX(left).add(fluxX(right)).mul(0.5).sub(right.sub(left).mul(max(aL, aR)).mul(0.5));
    };
    const rusanovY = (bottom: any, top: any) => {
      const hB: any = max(bottom.x, float(0));
      const hT: any = max(top.x, float(0));
      const aB: any = bottom.z.abs().div(max(hB, float(1e-5))).add(sqrt(p.gravity.mul(hB) as any));
      const aT: any = top.z.abs().div(max(hT, float(1e-5))).add(sqrt(p.gravity.mul(hT) as any));
      return fluxY(bottom).add(fluxY(top)).mul(0.5).sub(top.sub(bottom).mul(max(aB, aT)).mul(0.5));
    };

    const c = state(ix, iy);
    const l = state(ix.sub(1), iy);
    const r = state(ix.add(1), iy);
    const bottom = state(ix, iy.sub(1));
    const top = state(ix, iy.add(1));
    const fL = rusanovX(l, c);
    const fR = rusanovX(c, r);
    const fB = rusanovY(bottom, c);
    const fT = rusanovY(c, top);
    const div = fR.sub(fL).add(fT.sub(fB));
    const next: any = c.sub(div.mul(p.dt)).toVar();

    const bed = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x;
    const bedDx = bed(ix.add(1), iy).sub(bed(ix.sub(1), iy)).mul(0.5);
    const bedDy = bed(ix, iy.add(1)).sub(bed(ix, iy.sub(1))).mul(0.5);
    next.y.assign(next.y.sub(p.dt.mul(p.gravity).mul(c.x).mul(bedDx)));
    next.z.assign(next.z.sub(p.dt.mul(p.gravity).mul(c.x).mul(bedDy)));

    const sourceAt = (cx: any, cy: any) => textureLoad(source, ivec2(cX(cx, w), cY(cy, h))).x;
    const emit = sourceAt(ix, iy);
    const oro = smoothstep(flatSeaLevel, flatSeaLevel.add(rainHighRef), bed(ix, iy));
    const ux = ix.toFloat().div(float(w)), uy = iy.toFloat().div(float(h));
    const cloud = mx_fractal_noise_float(
      vec3(ux.mul(2.6).add(time.mul(0.02)), uy.mul(2.6).sub(time.mul(0.015)), time.mul(0.05)), 3);
    const cloudMask = clamp(cloud.mul(0.9).add(0.65), float(0), float(1.5));
    const rain = p.source.mul(mix(float(1), oro, rainOrographic)).mul(cloudMask);
    const emittedDepth = emit.mul(p.dt);
    next.x.assign(next.x.add(rain.mul(p.dt)).add(emittedDepth).max(float(0)));

    // A spring injects discharge, not a stack of motionless water. Drive the new
    // volume downhill and outward across the source footprint so pressure leaves
    // the source immediately instead of first building a tall dome.
    const downhill = vec2(
      bed(ix.sub(1), iy).sub(bed(ix.add(1), iy)),
      bed(ix, iy.sub(1)).sub(bed(ix, iy.add(1))),
    );
    const outward = vec2(
      sourceAt(ix.sub(1), iy).sub(sourceAt(ix.add(1), iy)),
      sourceAt(ix, iy.sub(1)).sub(sourceAt(ix, iy.add(1))),
    );
    const downhillDir = downhill.div(max(length(downhill), float(EPS)));
    const outwardDir = outward.div(max(length(outward), float(EPS)));
    const sourceDrive = downhillDir.add(outwardDir.mul(0.65));
    const sourceDir = sourceDrive.div(max(length(sourceDrive), float(EPS)));
    const sourceMomentum = emittedDepth.mul(0.75);
    next.y.assign(next.y.add(sourceDir.x.mul(sourceMomentum)));
    next.z.assign(next.z.add(sourceDir.y.mul(sourceMomentum)));

    // Match the pipe solver's flow/depth-aware evaporation. Remove momentum with
    // evaporated mass so shallow drying does not artificially accelerate the remainder.
    const preEvapSpeed = length(vec2(next.y, next.z)).div(max(next.x, float(1e-5)));
    const flowKeep = smoothstep(evapSpeedRef.mul(0.4), evapSpeedRef, preEvapSpeed).mul(evapFlowReduce);
    const deepKeep = smoothstep(evapShallowRef, evapDeepRef, next.x).mul(evapDeepReduce);
    const keep = max(flowKeep, deepKeep).min(float(0.97));
    const evapScale = max(float(0), float(1).sub(p.evapProp.mul(float(1).sub(keep)).mul(p.dt)));
    next.x.assign(next.x.mul(evapScale));
    next.y.assign(next.y.mul(evapScale));
    next.z.assign(next.z.mul(evapScale));

    const need = flatSeaLevel.sub(bed(ix, iy)).max(float(0));
    next.x.assign(next.x.max(need));

    const wet = next.x.greaterThan(float(1e-5)).select(float(1), float(0));
    const speed = length(vec2(next.y, next.z)).div(max(next.x, float(1e-5)));
    const speedScale = min(float(1), float(4).div(max(speed, float(EPS))));
    next.y.assign(next.y.mul(float(0.992)).mul(speedScale).mul(wet));
    next.z.assign(next.z.mul(float(0.992)).mul(speedScale).mul(wet));

    // Sealed boundaries. The ocean ring remains the explicit depth reservoir.
    next.y.assign(ix.lessThan(int(1)).or(ix.greaterThan(int(w - 2))).select(float(0), next.y));
    next.z.assign(iy.lessThan(int(1)).or(iy.greaterThan(int(h - 2))).select(float(0), next.z));
    textureStore(dOut, uvec2(x, y), vec4(next.x, 0, 0, 1)).toWriteOnly();
    textureStore(momentumOut, uvec2(x, y), vec4(next.y, next.z, 0, 1)).toWriteOnly();
    textureStore(fluxOut, uvec2(x, y), vec4(
      max(float(0).sub(fL.x), float(0)),
      max(fR.x, float(0)),
      max(fT.x, float(0)),
      max(float(0).sub(fB.x), float(0)),
    )).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

function flatMomentumVelocity(momentum: StorageTexture, d: StorageTexture, velOut: StorageTexture, w: number, h: number): CN {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const depth = textureLoad(d, ivec2(ix, iy)).x;
    const mom = textureLoad(momentum, ivec2(ix, iy));
    const wet = depth.greaterThan(float(1e-5)).select(float(1), float(0));
    const velocity = vec2(mom.x, mom.y).div(max(depth, float(1e-5))).mul(wet);
    textureStore(velOut, uvec2(x, y), vec4(velocity.x, velocity.y, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Conservative upwind sediment advection using the same local Rusanov water
 * mass flux as the momentum solver. */
function flatMomentumSediment(
  d: StorageTexture,
  momentum: StorageTexture,
  sediment: StorageTexture,
  sedimentOut: StorageTexture,
  w: number,
  h: number,
): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const state = (cx: any, cy: any) => {
      const depth = textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x.max(float(0));
      const mom = textureLoad(momentum, ivec2(cX(cx, w), cY(cy, h)));
      const wet = depth.greaterThan(float(1e-5)).select(float(1), float(0));
      return vec3(depth, mom.x.mul(wet), mom.y.mul(wet));
    };
    const concentration = (cx: any, cy: any) => {
      const depth = textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x;
      const mass = textureLoad(sediment, ivec2(cX(cx, w), cY(cy, h))).x;
      return mass.div(max(depth, float(1e-5)));
    };
    const massFluxX = (left: any, right: any) => {
      const hL: any = max(left.x, float(0));
      const hR: any = max(right.x, float(0));
      const aL: any = left.y.abs().div(max(hL, float(1e-5))).add(sqrt(p.gravity.mul(hL) as any));
      const aR: any = right.y.abs().div(max(hR, float(1e-5))).add(sqrt(p.gravity.mul(hR) as any));
      return left.y.add(right.y).mul(0.5).sub(hR.sub(hL).mul(max(aL, aR)).mul(0.5));
    };
    const massFluxY = (bottom: any, top: any) => {
      const hB: any = max(bottom.x, float(0));
      const hT: any = max(top.x, float(0));
      const aB: any = bottom.z.abs().div(max(hB, float(1e-5))).add(sqrt(p.gravity.mul(hB) as any));
      const aT: any = top.z.abs().div(max(hT, float(1e-5))).add(sqrt(p.gravity.mul(hT) as any));
      return bottom.z.add(top.z).mul(0.5).sub(hT.sub(hB).mul(max(aB, aT)).mul(0.5));
    };
    const upwind = (q: any, before: any, after: any) => q.greaterThan(float(0)).select(q.mul(before), q.mul(after));

    const c = state(ix, iy);
    const l = state(ix.sub(1), iy);
    const r = state(ix.add(1), iy);
    const bottom = state(ix, iy.sub(1));
    const top = state(ix, iy.add(1));
    const qL = massFluxX(l, c);
    const qR = massFluxX(c, r);
    const qB = massFluxY(bottom, c);
    const qT = massFluxY(c, top);
    const sL = upwind(qL, concentration(ix.sub(1), iy), concentration(ix, iy));
    const sR = upwind(qR, concentration(ix, iy), concentration(ix.add(1), iy));
    const sB = upwind(qB, concentration(ix, iy.sub(1)), concentration(ix, iy));
    const sT = upwind(qT, concentration(ix, iy), concentration(ix, iy.add(1)));
    const current = textureLoad(sediment, ivec2(ix, iy)).x;
    const next = current.sub(sR.sub(sL).add(sT.sub(sB)).mul(p.dt));
    textureStore(sedimentOut, uvec2(x, y), vec4(max(next, float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Erode/deposit: shallow fast water (rivers) carves, deep still (lakes/sea) spared;
 *  incision + lateral + 3D volumetric hardness; deposition builds deltas/beds. */
function flatErosion(
  b: StorageTexture,
  loose: StorageTexture,
  s: StorageTexture,
  vel: StorageTexture,
  d: StorageTexture,
  flux: StorageTexture,
  hardness: StorageTexture,
  source: StorageTexture,
  activity: StorageTexture,
  bOut: StorageTexture,
  looseOut: StorageTexture,
  sOut: StorageTexture,
  activityOut: StorageTexture,
  w: number,
  h: number,
): CN {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const bAt = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x;
    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const lc = textureLoad(loose, ivec2(ix, iy)).x.min(bc);
    const sc = textureLoad(s, ivec2(ix, iy)).x;
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const bNew: any = bc.toVar();
    const looseNew: any = lc.toVar();
    const sNew: any = sc.toVar();
    const activityPrev = textureLoad(activity, ivec2(ix, iy));
    const erodeViz: any = activityPrev.x.mul(u.vizDecay).toVar();
    const depositViz: any = activityPrev.y.mul(u.vizDecay).toVar();
    // persistent wetness (activity.z): rises under water, decays slowly when dry so the
    // ground stays darkened for a few seconds after the water has run off.
    const wetViz: any = max(activityPrev.z.mul(u.wetDecay), smoothstep(float(0.0006), float(0.01), dc)).toVar();
    If(dc.greaterThan(float(0.0008)).or(sc.greaterThan(float(1e-5))), () => {
      const v = textureLoad(vel, ivec2(ix, iy));
      const dbx = bAt(ix.add(1), iy).sub(bAt(ix.sub(1), iy)).mul(0.5);
      const dby = bAt(ix, iy.add(1)).sub(bAt(ix, iy.sub(1))).mul(0.5);
      const tilt = length(vec2(dbx, dby)).min(float(0.5));
      const sinTilt = max(tilt, u.minSlope);
      const speed = length(vec2(v.x, v.y)).min(float(3));
      const depthSuppress = float(1).sub(smoothstep(u.erodeShallowDepth, u.erodeDeepDepth, dc));
      // Erosion is a near-bed shear effect: only near-bed flow carves, not the whole
      // column. Cap effective depth at channel scale so a DEEP lake's large dc can't
      // inflate discharge -> conc/lateral terms and let slow swirl dig its own bed +
      // build a rim dam. Rivers are shallow (dc < channelDepthRef) -> unaffected.
      const erodeDepth = dc.min(u.channelDepthRef);
      const discharge = erodeDepth.mul(speed);
      const conc = mix(float(1), smoothstep(float(0), u.channelDischarge, discharge), u.channelFocus);
      const hasWater = dc.greaterThan(float(0.0012)).select(float(1), float(0));
      const fAt = (cx: any, cy: any) => textureLoad(flux, ivec2(cX(cx, w), cY(cy, h)));
      const selfFlux = fAt(ix, iy);
      const outflow = selfFlux.x.add(selfFlux.y).add(selfFlux.z).add(selfFlux.w);
      const inflow = fAt(ix.sub(1), iy).y
        .add(fAt(ix.add(1), iy).x)
        .add(fAt(ix, iy.add(1)).w)
        .add(fAt(ix, iy.sub(1)).z);
      const throughFlow = min(inflow, outflow);
      // A newly wetted front can report high velocity while having no downstream
      // receiver yet. Eroding there digs a bowl before the river has had a chance
      // to advance. Require actual water passing through the cell and into already
      // wet terrain before allowing it to carve.
      const dAt = (cx: any, cy: any) => textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x;
      const downstreamWet = selfFlux.x.mul(dAt(ix.sub(1), iy))
        .add(selfFlux.y.mul(dAt(ix.add(1), iy)))
        .add(selfFlux.z.mul(dAt(ix, iy.add(1))))
        .add(selfFlux.w.mul(dAt(ix, iy.sub(1))))
        .div(max(outflow, float(EPS)));
      const establishedFlow = smoothstep(float(0.00015), float(0.0025), throughFlow)
        .mul(smoothstep(float(0.0005), float(0.004), downstreamWet));
      // DECOUPLED capacities (hysteresis). Erosion uses the slope-based capacity so
      // fast FLATS don't get carved (no source moat / mid-slope gouging). Deposition
      // uses a HIGHER transport capacity (flow keeps sediment alive on flats) so the
      // river carries its load further and deposits gradually instead of dumping a
      // plateau the instant the slope eases. Between the two -> sediment just rides.
      // Incision must use the REAL slope. Clamping it to minSlope let a broad,
      // slow-moving wet front excavate flat ground row by row at every grade break.
      // minSlope remains useful below for transport, so sediment can cross flats.
      const capacity = u.sedimentCapacity.mul(tilt).mul(speed).mul(hasWater).mul(conc);
      const transport = max(sinTilt, speed.mul(u.flowTransport));
      const capCarry = u.sedimentCapacity.mul(transport).mul(speed).mul(hasWater).mul(conc);
      // The actual spring footprint is infrastructure, not erodible terrain. Protect
      // only it and its immediate cells; the directed source momentum now evacuates
      // water without requiring a wide protected plateau.
      const sourceAt = (cx: any, cy: any) => textureLoad(source, ivec2(cX(cx, w), cY(cy, h))).x;
      const srcNear = max(
        sourceAt(ix, iy),
        max(
          max(sourceAt(ix.sub(1), iy), sourceAt(ix.add(1), iy)),
          max(sourceAt(ix, iy.sub(1)), sourceAt(ix, iy.add(1))),
        ),
      );
      const notSource = float(1).sub(smoothstep(float(0), float(0.003), srcNear));
      const softness = max(u.rockErodibility, lc.div(u.looseFull).min(float(1)));
      // 3D volumetric hardness at the bedrock point (world x,height,z). Blend
      // broad provinces, channel-scale structure, and weak fine variation. All
      // samples include elevation, so incision exposes new material rather than
      // repeatedly reading a fixed 2D surface mask.
      const p3 = vec3(ix.toFloat().div(float(w)), bc.mul(3), iy.toFloat().div(float(h)));
      const nBroad = mx_fractal_noise_float(p3.mul(u.hardness3dFreq), 3);
      const nMid = mx_fractal_noise_float(
        p3.mul(u.hardness3dFreq.mul(4.2)).add(vec3(7.1, -3.7, 11.3)), 2,
      );
      const nFine = mx_fractal_noise_float(
        p3.mul(u.hardness3dFreq.mul(12.5)).add(vec3(-13.7, 9.2, 5.4)), 2,
      );
      const hardValue = textureLoad(hardness, ivec2(ix, iy)).x;
      // The seed texture stores HARDNESS, not erodibility. High values must resist
      // erosion. The previous multiplication inverted that meaning and made hard
      // slopes disappear fastest.
      const materialErodibility = mix(float(1), float(0.18), smoothstep(float(0.25), float(1.85), hardValue));
      const hardnessVariation = nBroad.mul(0.55).add(nMid.mul(0.3)).add(nFine.mul(0.15));
      const volumetricErodibility = float(1).add(hardnessVariation.mul(u.hardness3dAmp))
        .clamp(float(0.35), float(1.3));
      const hard = materialErodibility.mul(volumetricErodibility);
      // Water must establish and transport before terrain visibly moves. A shared
      // 0.0006 cap let five erosion ticks per second cut a deep trench in seconds.
      // Keep incision deliberately slower than loose-sediment deposition; simSpeed
      // can accelerate both for diagnostics without changing their relationship.
      const ERODE_CAP = float(0.00022).mul(u.simSpeed);
      const DEPOSIT_CAP = float(0.00045).mul(u.simSpeed);
      const erodeGate = speed.greaterThan(u.erodeSpeedMin).select(float(1), float(0))
        .mul(notSource).mul(establishedFlow);
      const dh = vec2(dbx, dby).mul(-1).add(vec2(1e-5, 1e-5));
      const fdir = vec2(v.x, v.y).add(vec2(1e-6, 1e-6));
      const align = max(float(0), fdir.normalize().dot(dh.normalize()));
      // Prefer coherent downhill channels and introduce broad deterministic
      // susceptibility variation. This breaks sheet erosion into pioneering streams
      // without injecting temporal noise or changing water mass.
      const channelNoise = float(0.82).add(nMid.mul(0.22)).add(nFine.mul(0.08))
        .clamp(float(0.5), float(1.12));
      const coherentFlow = smoothstep(float(0.15), float(0.85), align).mul(0.82).add(0.18);
      // Ordinary bed incision is driven by shear along a bed. Near-vertical faces
      // are waterfall/cliff processes and must not be uniformly shaved by the same
      // rule; plunge-foot and weathering erosion can be modeled separately.
      const bedIncision = float(1).sub(smoothstep(float(0.16), float(0.38), tilt));
      const erodeBase = max(float(0), capacity.sub(sc)).mul(u.dissolve).mul(softness)
        .mul(hard).mul(erodeGate).mul(coherentFlow).mul(channelNoise).mul(bedIncision);
      const gentle = float(1).sub(smoothstep(float(0.05), float(0.18), tilt));
      const lateral = discharge.mul(float(1).sub(align)).mul(gentle).mul(u.lateralErosion).mul(softness).mul(hard).mul(erodeGate);
      // NO-SINK: never carve a cell below its LOWEST neighbor. A closed pit (lower
      // than all neighbors) traps water + sediment -> flow stalls -> it deposits a bar
      // it can never cut back through (the "digs in after the drop then dams itself"
      // bug). Canyons (lower than most, but OPEN downstream) are still allowed.
      const minNb = min(min(bAt(ix.add(1), iy), bAt(ix.sub(1), iy)), min(bAt(ix, iy.add(1)), bAt(ix, iy.sub(1))));
      const noSink = max(float(0), bc.sub(minNb));
      // Once a channel is materially below its banks, stop vertical incision and
      // let its discharge continue downstream. Without this brake, every tick lowers
      // the established path until it becomes an unnecessarily deep trench.
      const bankMean = bAt(ix.add(1), iy).add(bAt(ix.sub(1), iy))
        .add(bAt(ix, iy.add(1))).add(bAt(ix, iy.sub(1))).mul(0.25);
      const incisionDepth = max(float(0), bankMean.sub(bc));
      const incisionBrake = float(1).sub(smoothstep(float(0.008), float(0.035), incisionDepth));
      const erode = erodeBase.add(lateral.mul(incisionBrake))
        .mul(depthSuppress).mul(incisionBrake).mul(u.simSpeed).min(ERODE_CAP).min(noSink);
      // ANTI-DAM: never deposit enough to raise the bed above the local water
      // surface. Excess sediment stays suspended -> advects on to deeper water ->
      // spreads a submerged delta fan instead of instantly damming the channel.
      // Calm/deep water continuously drops suspended sediment. This is separate
      // from capacity-driven deposition: lakes and sheltered coastal water should
      // clarify even when they are not evaporating.
      const calm = float(1).sub(smoothstep(float(0.025), float(0.22), speed));
      // Sediment should ride through a connected river and settle where discharge
      // spreads across a lake/shelf. Previously slow local velocity near the source
      // dumped the load there even while substantial pipe flow passed through.
      // Capacity-driven deposition belongs where a transported load decelerates,
      // not simply anywhere flow is low. The old low-throughflow gate dumped the
      // load at the first ocean cell after ocean relaxation reduced its discharge.
      const deceleration = max(float(0), inflow.sub(outflow)).div(max(inflow, float(EPS)));
      const carriedLoad = smoothstep(float(0.00015), float(0.0025), inflow);
      const depositZone = carriedLoad.mul(smoothstep(float(0.04), float(0.65), deceleration));
      const settling = sc.mul(u.stillDeposit).mul(calm)
        .mul(smoothstep(float(0.02), float(0.12), dc));
      const dep = max(
        max(float(0), sc.sub(capCarry)).mul(u.deposit).mul(u.simSpeed).mul(depositZone),
        settling,
      ).min(DEPOSIT_CAP).min(dc.mul(0.22)).mul(notSource);
      bNew.assign(max(bc.sub(erode).add(dep), float(0)));
      looseNew.assign(max(lc.sub(erode), float(0)).add(dep));
      sNew.assign(max(float(0), sc.add(erode).sub(dep)).min(float(2)));
      erodeViz.assign(max(erodeViz, erode.div(ERODE_CAP)));
      depositViz.assign(max(depositViz, dep.div(DEPOSIT_CAP)));
    });
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(looseNew, 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
    textureStore(activityOut, uvec2(x, y), vec4(erodeViz, depositViz, wetViz, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Surface-preserving water settle. The erosion pass moves the bed (deposition raises,
 *  incision lowers) but leaves water depth untouched, so the free surface b+d jumps by
 *  the bed delta. The NEXT flux pass then reads that jump as hydraulic head and shoves
 *  water outward (incl. UPSTREAM) — a backfiring impulse opposite the flow, the cause of
 *  river weirdness near deltas. Suspended sediment occupies volume in the column, so
 *  settling/lifting it must NOT move the surface. Compensate d by -(bNew-bOld) to hold
 *  b+d invariant across the bed exchange. Reads OLD bed (b) + NEW bed (bNext, erosion's
 *  scratch output) before the bed copy lands. Kept a separate 1-write pass so erosion
 *  stays at the 4-storage-texture-per-stage WebGPU baseline. */
function flatSurfaceSettle(b: StorageTexture, bNext: StorageTexture, d: StorageTexture, dOut: StorageTexture, w: number, h: number): CN {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const delta = textureLoad(bNext, ivec2(ix, iy)).x.sub(textureLoad(b, ivec2(ix, iy)).x);
    textureStore(dOut, uvec2(x, y), vec4(max(dc.sub(delta), float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Conservative sediment transport: sediment rides with directional water
 *  transfer at the source cell's concentration. Closed boundaries conserve mass. */
function flatSedimentTransport(d: StorageTexture, f: StorageTexture, s: StorageTexture, sOut: StorageTexture, w: number, h: number): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const at = (tex: StorageTexture, cx: any, cy: any) => textureLoad(tex, ivec2(cX(cx, w), cY(cy, h)));
    const conc = (cx: any, cy: any) => at(s, cx, cy).x.div(max(at(d, cx, cy).x, float(EPS)));
    const selfF = at(f, ix, iy);
    const sc = at(s, ix, iy).x;
    const volumeScale = p.dt.div(p.pipeLength.mul(p.pipeLength));
    const outgoingWater = selfF.x.add(selfF.y).add(selfF.z).add(selfF.w).mul(volumeScale);
    const outgoingSediment = min(sc, outgoingWater.mul(conc(ix, iy)));

    const mL = ix.greaterThan(int(0)).select(float(1), float(0));
    const mR = ix.lessThan(int(w - 1)).select(float(1), float(0));
    const mT = iy.lessThan(int(h - 1)).select(float(1), float(0));
    const mB = iy.greaterThan(int(0)).select(float(1), float(0));
    const incomingSediment = at(f, ix.sub(1), iy).y.mul(conc(ix.sub(1), iy)).mul(mL)
      .add(at(f, ix.add(1), iy).x.mul(conc(ix.add(1), iy)).mul(mR))
      .add(at(f, ix, iy.add(1)).w.mul(conc(ix, iy.add(1))).mul(mT))
      .add(at(f, ix, iy.sub(1)).z.mul(conc(ix, iy.sub(1))).mul(mB))
      .mul(volumeScale);
    const next = max(float(0), sc.sub(outgoingSediment).add(incomingSediment));
    textureStore(sOut, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Conservative thermal slumping. Only mobile earth moves; rock stays fixed. */
function flatThermal(b: StorageTexture, loose: StorageTexture, d: StorageTexture, bOut: StorageTexture, looseOut: StorageTexture, w: number, h: number): CN {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const at = (tex: StorageTexture, cx: any, cy: any) => textureLoad(tex, ivec2(cX(cx, w), cY(cy, h))).x;
    const transfer = (sx: any, sy: any, tx: any, ty: any) => {
      const sourceHeight = at(b, sx, sy);
      const sourceLoose = at(loose, sx, sy);
      // Loose sediment (sand/silt) has a GENTLER angle of repose than bedrock and can't
      // stack into a plateau — it slumps to spread a delta fan. The more loose material
      // a cell holds, the lower its talus -> deposited deltas push outward instead of
      // piling up vertically.
      const looseFrac = sourceLoose.div(u.looseFull).min(float(1));
      // Deposited sand/silt has a much shallower angle of repose than rock.
      // Deep loose deposits approach a broad alluvial slope instead of a cliff.
      const talusEff = u.talus.mul(float(1).sub(looseFrac.mul(0.9)));
      const request = (nx: any, ny: any) => max(float(0), sourceHeight.sub(at(b, nx, ny)).sub(talusEff));
      const reqL = request(sx.sub(1), sy);
      const reqR = request(sx.add(1), sy);
      const reqT = request(sx, sy.add(1));
      const reqB = request(sx, sy.sub(1));
      const total = reqL.add(reqR).add(reqT).add(reqB);
      // bedrock channels stay suppressed underwater (don't heal flat), but LOOSE delta
      // sediment keeps slumping underwater so it fans out instead of damming a plateau.
      const wet = smoothstep(float(0), u.channelDepthRef, at(d, sx, sy));
      const wetSuppress = wet.mul(0.85).mul(float(1).sub(looseFrac.mul(0.95)));
      const looseSpread = mix(float(1), float(3.2), looseFrac);
      const rate = u.thermalRate.mul(float(1).sub(wetSuppress)).mul(looseSpread).mul(u.simSpeed).mul(0.25);
      const scale = min(rate, sourceLoose.div(max(total, float(EPS))));
      const dx = tx.sub(sx);
      const dy = ty.sub(sy);
      let requested: any = reqL;
      requested = dx.greaterThan(int(0)).select(reqR, requested);
      requested = dy.greaterThan(int(0)).select(reqT, requested);
      requested = dy.lessThan(int(0)).select(reqB, requested);
      return requested.mul(scale);
    };

    const outflow = transfer(ix, iy, ix.sub(1), iy)
      .add(transfer(ix, iy, ix.add(1), iy))
      .add(transfer(ix, iy, ix, iy.add(1)))
      .add(transfer(ix, iy, ix, iy.sub(1)));
    const inflow = transfer(ix.sub(1), iy, ix, iy)
      .add(transfer(ix.add(1), iy, ix, iy))
      .add(transfer(ix, iy.add(1), ix, iy))
      .add(transfer(ix, iy.sub(1), ix, iy));
    const delta = inflow.sub(outflow);
    const c = at(b, ix, iy);
    const lc = at(loose, ix, iy);
    textureStore(bOut, uvec2(x, y), vec4(max(c.add(delta), float(0)), 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(max(lc.add(delta), float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

export class FlatSim {
  readonly water: GridField;
  readonly flux: GridField;
  readonly momentum: GridField;
  readonly velocity: GridField;
  readonly sediment: GridField;
  readonly loose: GridField;
  readonly source: GridField;
  readonly activity: GridField; // rg: recent erosion/deposition
  erosionEnabled = false;
  momentumSubsteps = 1;
  private tickCount = 0;
  private _waterSolver: FlatWaterSolver = 'pipe';
  private readonly nodes: { fluxN: CN; fluxC: CN; velN: CN; velC: CN; sedN: CN; sFlowC: CN; updN: CN; watC: CN; momN: CN; momC: CN; momSedN: CN; momSedC: CN; momVelN: CN; momVelC: CN; eroN: CN; settleN: CN; wEC: CN; bC: CN; loC: CN; sEC: CN; actC: CN; thN: CN; bTC: CN; loTC: CN };
  private readonly srcCenter = uniform(new Vector2(0.5, 0.5));
  private readonly srcRadius = uniform(0.04);
  private readonly srcRate = uniform(0);
  private readonly srcN: CN;
  private readonly srcC: CN;

  constructor(private readonly renderer: WebGPURenderer, private readonly height: GridField, looseSeed: any, hardness: any, readonly w: number, readonly h: number) {
    this.water = new GridField(w, h);
    this.flux = new GridField(w, h, true);
    this.momentum = new GridField(w, h, true);
    this.velocity = new GridField(w, h, true);
    this.sediment = new GridField(w, h);
    this.loose = new GridField(w, h);
    this.source = new GridField(w, h);
    this.activity = new GridField(w, h, true);
    renderer.compute(buildGridFill(this.water.main, w, h, 0));
    renderer.compute(buildGridFill(this.momentum.main, w, h, 0));
    renderer.compute(buildGridFill(this.velocity.main, w, h, 0));
    renderer.compute(buildGridFill(this.sediment.main, w, h, 0));
    renderer.compute(buildGridFill(this.source.main, w, h, 0));
    renderer.compute(buildGridFill(this.activity.main, w, h, 0));
    renderer.compute(buildGridSeed(looseSeed, this.loose.main, w, h));
    const b = height.main;
    this.nodes = {
      fluxN: flatFlux(b, this.water.main, this.sediment.main, this.source.main, this.flux.main, this.flux.scratch, w, h),
      fluxC: buildGridCopy(this.flux.scratch, this.flux.main, w, h),
      velN: flatVelocity(this.flux.main, this.water.main, this.velocity.main, this.velocity.scratch, w, h),
      velC: buildGridCopy(this.velocity.scratch, this.velocity.main, w, h),
      sedN: flatSedimentTransport(this.water.main, this.flux.main, this.sediment.main, this.sediment.scratch, w, h),
      sFlowC: buildGridCopy(this.sediment.scratch, this.sediment.main, w, h),
      updN: flatUpdate(this.water.main, this.flux.main, b, this.source.main, this.velocity.main, this.water.scratch, w, h),
      watC: buildGridCopy(this.water.scratch, this.water.main, w, h),
      momN: flatMomentumUpdate(this.water.main, this.momentum.main, b, this.source.main, this.water.scratch, this.momentum.scratch, this.flux.main, w, h),
      momC: buildGridCopy(this.momentum.scratch, this.momentum.main, w, h),
      momSedN: flatMomentumSediment(this.water.main, this.momentum.main, this.sediment.main, this.sediment.scratch, w, h),
      momSedC: buildGridCopy(this.sediment.scratch, this.sediment.main, w, h),
      momVelN: flatMomentumVelocity(this.momentum.main, this.water.main, this.velocity.scratch, w, h),
      momVelC: buildGridCopy(this.velocity.scratch, this.velocity.main, w, h),
      eroN: flatErosion(b, this.loose.main, this.sediment.main, this.velocity.main, this.water.main, this.flux.main, hardness, this.source.main, this.activity.main, height.scratch, this.loose.scratch, this.sediment.scratch, this.activity.scratch, w, h),
      // Surface-preserving water settle: reads OLD bed (b) + NEW bed (height.scratch) +
      // current d, writes compensated d to water.scratch. Runs before bC lands the bed.
      settleN: flatSurfaceSettle(b, height.scratch, this.water.main, this.water.scratch, w, h),
      wEC: buildGridCopy(this.water.scratch, this.water.main, w, h),
      bC: buildGridCopy(height.scratch, b, w, h),
      loC: buildGridCopy(this.loose.scratch, this.loose.main, w, h),
      sEC: buildGridCopy(this.sediment.scratch, this.sediment.main, w, h),
      actC: buildGridCopy(this.activity.scratch, this.activity.main, w, h),
      thN: flatThermal(b, this.loose.main, this.water.main, height.scratch, this.loose.scratch, w, h),
      bTC: buildGridCopy(height.scratch, b, w, h),
      loTC: buildGridCopy(this.loose.scratch, this.loose.main, w, h),
    };
    const stamp = Fn(() => {
      const x = instanceIndex.mod(uint(w)), yy = instanceIndex.div(uint(w));
      const ux = x.toFloat().div(w), uy = yy.toFloat().div(h);
      const dist = length(vec2(ux.sub(this.srcCenter.x), uy.sub(this.srcCenter.y)));
      // A placed spring is a pressurized outlet, not rainfall over a filled disk.
      // Concentrate discharge near the footprint boundary so water crosses out of
      // the source immediately instead of accumulating among interior source cells.
      const inner = smoothstep(this.srcRadius.mul(0.28), this.srcRadius.mul(0.68), dist);
      const outer = float(1).sub(smoothstep(this.srcRadius.mul(0.72), this.srcRadius, dist));
      const wgt = inner.mul(outer);
      const cur = textureLoad(this.source.main, ivec2(int(x), int(yy))).x;
      // `rate` is total discharge, not a per-cell rate. This annular smoothstep
      // kernel integrates to roughly 0.36*pi*R² over normalized map coordinates.
      const kernelAreaCells = this.srcRadius.mul(this.srcRadius).mul(float(0.36 * Math.PI * w * h));
      const normalizedRate = this.srcRate.div(max(kernelAreaCells, float(EPS)));
      textureStore(this.source.scratch, uvec2(x, yy), vec4(max(float(0), cur.add(normalizedRate.mul(wgt))), 0, 0, 1)).toWriteOnly();
    });
    this.srcN = stamp().compute(w * h) as CN;
    this.srcC = buildGridCopy(this.source.scratch, this.source.main, w, h);
  }

  setRain(r: number) { waterUniforms.source.value = r; }
  get waterSolver(): FlatWaterSolver { return this._waterSolver; }
  set waterSolver(next: FlatWaterSolver) {
    if (next === this._waterSolver) return;
    this._waterSolver = next;
    for (const field of [this.flux, this.momentum, this.velocity]) {
      this.renderer.compute(buildGridFill(field.main, this.w, this.h, 0));
      this.renderer.compute(buildGridFill(field.scratch, this.w, this.h, 0));
    }
  }
  clearWater() {
    for (const field of [this.water, this.flux, this.momentum, this.velocity]) {
      this.renderer.compute(buildGridFill(field.main, this.w, this.h, 0));
      this.renderer.compute(buildGridFill(field.scratch, this.w, this.h, 0));
    }
  }
  clearSources() { this.renderer.compute(buildGridFill(this.source.main, this.w, this.h, 0)); }
  loadState(state: { height: any; loose: any; water: any; sediment: any; source: any }) {
    const r = this.renderer;
    this.tickCount = 0;
    for (const [seed, field] of [
      [state.height, this.height],
      [state.loose, this.loose],
      [state.water, this.water],
      [state.sediment, this.sediment],
      [state.source, this.source],
    ] as const) {
      r.compute(buildGridSeed(seed, field.main, this.w, this.h));
      r.compute(buildGridSeed(seed, field.scratch, this.w, this.h));
    }
    for (const field of [this.flux, this.momentum, this.velocity, this.activity]) {
      r.compute(buildGridFill(field.main, this.w, this.h, 0));
      r.compute(buildGridFill(field.scratch, this.w, this.h, 0));
    }
  }
  placeSource(u: number, v: number, rate: number, radius = 0.04) {
    this.srcCenter.value.set(u, v); this.srcRate.value = rate; this.srcRadius.value = radius;
    this.renderer.compute(this.srcN); this.renderer.compute(this.srcC);
  }

  tick(dt: number) {
    const r = this.renderer, n = this.nodes;
    this.tickCount++;
    if (this._waterSolver === 'momentum') {
      this.momentumSubsteps = momentumSubstepCount(dt, waterUniforms.gravity.value);
      waterUniforms.dt.value = dt / this.momentumSubsteps;
      for (let i = 0; i < this.momentumSubsteps; i++) {
        r.compute(n.momSedN); r.compute(n.momSedC);
        r.compute(n.momN); r.compute(n.watC); r.compute(n.momC);
      }
      r.compute(n.momVelN); r.compute(n.momVelC);
    } else {
      this.momentumSubsteps = 1;
      waterUniforms.dt.value = dt;
      r.compute(n.fluxN); r.compute(n.fluxC);
      r.compute(n.velN); r.compute(n.velC);
      r.compute(n.sedN); r.compute(n.sFlowC);
      r.compute(n.updN); r.compute(n.watC);
    }
    // Water must have time to establish a connected route before terrain responds.
    // Eroding every 20 Hz water tick lets the bed flatten faster than a river can
    // hydraulically adjust, especially on modest slopes.
    if (this.erosionEnabled && this.tickCount % 4 === 0) {
      r.compute(n.eroN);
      // settle reads OLD bed (still in height.main) + NEW bed (height.scratch) -> must run
      // before bC overwrites height.main; wEC then lands the compensated water depth.
      r.compute(n.settleN); r.compute(n.wEC);
      r.compute(n.bC); r.compute(n.loC); r.compute(n.sEC); r.compute(n.actC);
      r.compute(n.thN);
      // Thermal slumping changes the bed after the hydraulic settle above. Preserve
      // the free surface a second time or each slump becomes a fake pressure pulse
      // that flickers depth/turbidity and can shove water upstream.
      r.compute(n.settleN); r.compute(n.wEC);
      r.compute(n.bTC); r.compute(n.loTC);
    }
  }
}
