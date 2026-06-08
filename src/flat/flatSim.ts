// Flat-grid sim: pipe-model water + velocity + hydraulic/thermal erosion. Uniform
// cell area (depth = volume directly), edges CLAMPED + sealed (water drains into
// the ocean border, which the sea-fill pins to sea level). No wrap, no poles, no
// cos-lat — the sphere machinery is gone. Reuses the tuned uniforms.

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import { Vector2 } from 'three';
import {
  Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int,
  float, vec2, vec4, max, min, length, mix, smoothstep, If, uniform,
  mx_fractal_noise_float, vec3,
} from 'three/tsl';
import { GridField, buildGridCopy, buildGridFill, buildGridSeed } from '../sim/gridStore';
import { waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import {
  evapFlowReduce, evapSpeedRef, evapDeepReduce, evapDeepRef, evapShallowRef,
  rainOrographic, rainHighRef,
} from '../sim/gridWater';
import { flatSeaLevel } from '../tsl/flatSurface';

type CN = Parameters<WebGPURenderer['compute']>[0];
/* eslint-disable @typescript-eslint/no-explicit-any */
const EPS = 1e-6;

function coords(w: number) {
  const N = uint(w);
  return { x: instanceIndex.mod(N), y: instanceIndex.div(N), ix: int(instanceIndex.mod(N)), iy: int(instanceIndex.div(N)) };
}
const cX = (x: any, w: number) => x.toFloat().max(float(0)).min(float(w - 1)).toInt();
const cY = (y: any, h: number) => y.toFloat().max(float(0)).min(float(h - 1)).toInt();

/** Pipe outflow flux (L,R,T,B). */
function flatFlux(b: StorageTexture, d: StorageTexture, fPrev: StorageTexture, fOut: StorageTexture, w: number, h: number): CN {
  const p = waterUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const surf = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x.add(textureLoad(d, ivec2(cX(cx, w), cY(cy, h))).x);
    const hc = surf(ix, iy);
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const prev = textureLoad(fPrev, ivec2(ix, iy));
    const k: any = p.dt.mul(p.pipeArea).mul(p.gravity).div(p.pipeLength);
    let fL = max(float(0), prev.x.mul(p.damping).add(k.mul(hc.sub(surf(ix.sub(1), iy)))));
    let fR = max(float(0), prev.y.mul(p.damping).add(k.mul(hc.sub(surf(ix.add(1), iy)))));
    let fT = max(float(0), prev.z.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.add(1))))));
    let fB = max(float(0), prev.w.mul(p.damping).add(k.mul(hc.sub(surf(ix, iy.sub(1))))));
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
    const emit = textureLoad(source, ivec2(ix, iy)).x;
    const oro = smoothstep(flatSeaLevel, flatSeaLevel.add(rainHighRef), bc);
    const rain = p.source.mul(mix(float(1), oro, rainOrographic));
    let next: any = inflow.sub(outflow).mul(p.dt).div(l2).add(dc);
    next = next.add(rain.add(emit).mul(p.dt));
    // flow + deep aware evaporation.
    const velC = textureLoad(vel, ivec2(ix, iy));
    const speed = length(vec2(velC.x, velC.y));
    const flowKeep = smoothstep(evapSpeedRef.mul(0.4), evapSpeedRef, speed).mul(evapFlowReduce);
    const deepKeep = smoothstep(evapShallowRef, evapDeepRef, next).mul(evapDeepReduce);
    const keep = max(flowKeep, deepKeep).min(float(0.97));
    next = next.mul(max(float(0), float(1).sub(p.evapProp.mul(float(1).sub(keep)).mul(p.dt))));
    // sea fill: ocean (below sea level) must hold AT LEAST sea level (flat sea), but
    // do NOT pin it down — that instantly swallowed incoming river water+sediment.
    // Letting it rise lets a river plume spread across the surface (and carry its
    // sediment out) before the pipe flow + evap level it back to the sea. -> deltas.
    const need = bc.mul(-1).add(flatSeaLevel).max(float(0));
    next = next.max(need);
    textureStore(dOut, uvec2(x, y), vec4(max(next, float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

/** Erode/deposit: shallow fast water (rivers) carves, deep still (lakes/sea) spared;
 *  incision + lateral + 3D volumetric hardness; deposition builds deltas/beds. */
function flatErosion(b: StorageTexture, loose: StorageTexture, s: StorageTexture, vel: StorageTexture, d: StorageTexture, hardness: StorageTexture, source: StorageTexture, bOut: StorageTexture, looseOut: StorageTexture, sOut: StorageTexture, w: number, h: number): CN {
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
    If(dc.greaterThan(float(0.0008)).or(sc.greaterThan(float(1e-5))), () => {
      const v = textureLoad(vel, ivec2(ix, iy));
      const dbx = bAt(ix.add(1), iy).sub(bAt(ix.sub(1), iy)).mul(0.5);
      const dby = bAt(ix, iy.add(1)).sub(bAt(ix, iy.sub(1))).mul(0.5);
      const tilt = length(vec2(dbx, dby)).min(float(0.5));
      const sinTilt = max(tilt, u.minSlope);
      const speed = length(vec2(v.x, v.y)).min(float(3));
      const depthSuppress = float(1).sub(smoothstep(u.erodeShallowDepth, u.erodeDeepDepth, dc));
      const discharge = dc.mul(speed);
      const conc = mix(float(1), smoothstep(float(0), u.channelDischarge, discharge), u.channelFocus);
      const hasWater = dc.greaterThan(float(0.0012)).select(float(1), float(0));
      const capacity = u.sedimentCapacity.mul(sinTilt).mul(speed).mul(hasWater).mul(conc);
      const notSource = textureLoad(source, ivec2(ix, iy)).x.lessThan(float(0.0001)).select(float(1), float(0));
      const softness = max(u.rockErodibility, lc.div(u.looseFull).min(float(1)));
      // 3D volumetric hardness at the bedrock point (world x,height,z).
      // broad volumetric hardness (low freq -> regional hard/soft provinces, NOT a
      // per-cell spike lattice). bc weighted up so it varies with depth too.
      const p3 = vec3(ix.toFloat().div(float(w)), bc.mul(3), iy.toFloat().div(float(h)));
      const n3 = mx_fractal_noise_float(p3.mul(u.hardness3dFreq), 3);
      const hard = textureLoad(hardness, ivec2(ix, iy)).x.mul(float(1).add(n3.mul(u.hardness3dAmp)).max(float(0.05)));
      const CAP = float(0.0006).mul(u.simSpeed); // small per-step carve: flow outruns erosion
      const erodeGate = speed.greaterThan(u.erodeSpeedMin).select(float(1), float(0)).mul(notSource);
      const erodeBase = max(float(0), capacity.sub(sc)).mul(u.dissolve).mul(softness).mul(hard).mul(erodeGate);
      const dh = vec2(dbx, dby).mul(-1).add(vec2(1e-5, 1e-5));
      const fdir = vec2(v.x, v.y).add(vec2(1e-6, 1e-6));
      const align = max(float(0), fdir.normalize().dot(dh.normalize()));
      const gentle = float(1).sub(smoothstep(float(0.05), float(0.18), tilt));
      const lateral = discharge.mul(float(1).sub(align)).mul(gentle).mul(u.lateralErosion).mul(softness).mul(hard).mul(erodeGate);
      const erode = erodeBase.add(lateral).mul(depthSuppress).mul(u.simSpeed).min(CAP);
      // ANTI-DAM: never deposit enough to raise the bed above the local water
      // surface. Excess sediment stays suspended -> advects on to deeper water ->
      // spreads a submerged delta fan instead of instantly damming the channel.
      const dep = max(float(0), sc.sub(capacity)).mul(u.deposit).mul(u.simSpeed).min(CAP).min(dc.mul(0.35));
      bNew.assign(max(bc.sub(erode).add(dep), float(0)));
      looseNew.assign(max(lc.sub(erode), float(0)).add(dep));
      sNew.assign(max(float(0), sc.add(erode).sub(dep)).min(float(2)));
    });
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(looseNew, 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

function flatAdvect(vel: StorageTexture, s: StorageTexture, sOut: StorageTexture, w: number, h: number): CN {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const v = textureLoad(vel, ivec2(ix, iy));
    // step in CELLS/tick (NOT dt-scaled — dt*1/60 made it ~0.017 cells -> sediment
    // never traveled to the sea). backtrace ~ velocity cells so it actually flows.
    const step = erosionUniforms.advectScale.mul(erosionUniforms.simSpeed);
    const bx = cX(ix.toFloat().sub(v.x.mul(step)), w);
    const by = cY(iy.toFloat().sub(v.y.mul(step)), h);
    textureStore(sOut, uvec2(x, y), vec4(textureLoad(s, ivec2(bx, by)).x, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

function flatThermal(b: StorageTexture, d: StorageTexture, bOut: StorageTexture, w: number, h: number): CN {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const bAt = (cx: any, cy: any) => textureLoad(b, ivec2(cX(cx, w), cY(cy, h))).x;
    const c = textureLoad(b, ivec2(ix, iy)).x;
    const nbs = [bAt(ix.sub(1), iy), bAt(ix.add(1), iy), bAt(ix, iy.sub(1)), bAt(ix, iy.add(1))];
    let net: any = float(0);
    for (const nb of nbs) {
      net = net.sub(max(float(0), c.sub(nb).sub(u.talus)));
      net = net.add(max(float(0), nb.sub(c).sub(u.talus)));
    }
    const wet = smoothstep(float(0), u.channelDepthRef, textureLoad(d, ivec2(ix, iy)).x);
    const rate = u.thermalRate.mul(float(1).sub(wet.mul(0.85))).mul(u.simSpeed);
    textureStore(bOut, uvec2(x, y), vec4(max(c.add(net.mul(rate).mul(0.25)), float(0)), 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as CN;
}

export class FlatSim {
  readonly water: GridField;
  readonly flux: GridField;
  readonly velocity: GridField;
  readonly sediment: GridField;
  readonly loose: GridField;
  readonly source: GridField;
  erosionEnabled = false;
  private readonly nodes: { fluxN: CN; fluxC: CN; velN: CN; velC: CN; updN: CN; watC: CN; eroN: CN; bC: CN; loC: CN; sEC: CN; advN: CN; sAC: CN; thN: CN; bTC: CN };
  private readonly srcCenter = uniform(new Vector2(0.5, 0.5));
  private readonly srcRadius = uniform(0.04);
  private readonly srcRate = uniform(0);
  private readonly srcN: CN;
  private readonly srcC: CN;

  constructor(private readonly renderer: WebGPURenderer, height: GridField, looseSeed: any, hardness: any, readonly w: number, readonly h: number) {
    this.water = new GridField(w, h);
    this.flux = new GridField(w, h, true);
    this.velocity = new GridField(w, h, true);
    this.sediment = new GridField(w, h);
    this.loose = new GridField(w, h);
    this.source = new GridField(w, h);
    renderer.compute(buildGridFill(this.water.main, w, h, 0));
    renderer.compute(buildGridFill(this.velocity.main, w, h, 0));
    renderer.compute(buildGridFill(this.sediment.main, w, h, 0));
    renderer.compute(buildGridFill(this.source.main, w, h, 0));
    renderer.compute(buildGridSeed(looseSeed, this.loose.main, w, h));
    const b = height.main;
    this.nodes = {
      fluxN: flatFlux(b, this.water.main, this.flux.main, this.flux.scratch, w, h),
      fluxC: buildGridCopy(this.flux.scratch, this.flux.main, w, h),
      velN: flatVelocity(this.flux.main, this.water.main, this.velocity.main, this.velocity.scratch, w, h),
      velC: buildGridCopy(this.velocity.scratch, this.velocity.main, w, h),
      updN: flatUpdate(this.water.main, this.flux.main, b, this.source.main, this.velocity.main, this.water.scratch, w, h),
      watC: buildGridCopy(this.water.scratch, this.water.main, w, h),
      eroN: flatErosion(b, this.loose.main, this.sediment.main, this.velocity.main, this.water.main, hardness, this.source.main, height.scratch, this.loose.scratch, this.sediment.scratch, w, h),
      bC: buildGridCopy(height.scratch, b, w, h),
      loC: buildGridCopy(this.loose.scratch, this.loose.main, w, h),
      sEC: buildGridCopy(this.sediment.scratch, this.sediment.main, w, h),
      advN: flatAdvect(this.velocity.main, this.sediment.main, this.sediment.scratch, w, h),
      sAC: buildGridCopy(this.sediment.scratch, this.sediment.main, w, h),
      thN: flatThermal(b, this.water.main, height.scratch, w, h),
      bTC: buildGridCopy(height.scratch, b, w, h),
    };
    const stamp = Fn(() => {
      const x = instanceIndex.mod(uint(w)), yy = instanceIndex.div(uint(w));
      const ux = x.toFloat().div(w), uy = yy.toFloat().div(h);
      const dist = length(vec2(ux.sub(this.srcCenter.x), uy.sub(this.srcCenter.y)));
      const wgt = float(1).sub(smoothstep(float(0), this.srcRadius, dist));
      const cur = textureLoad(this.source.main, ivec2(int(x), int(yy))).x;
      textureStore(this.source.scratch, uvec2(x, yy), vec4(max(float(0), cur.add(this.srcRate.mul(wgt))), 0, 0, 1)).toWriteOnly();
    });
    this.srcN = stamp().compute(w * h) as CN;
    this.srcC = buildGridCopy(this.source.scratch, this.source.main, w, h);
  }

  setRain(r: number) { waterUniforms.source.value = r; }
  clearWater() { this.renderer.compute(buildGridFill(this.water.main, this.w, this.h, 0)); this.renderer.compute(buildGridFill(this.flux.main, this.w, this.h, 0)); this.renderer.compute(buildGridFill(this.velocity.main, this.w, this.h, 0)); }
  clearSources() { this.renderer.compute(buildGridFill(this.source.main, this.w, this.h, 0)); }
  placeSource(u: number, v: number, rate: number, radius = 0.04) {
    this.srcCenter.value.set(u, v); this.srcRate.value = rate; this.srcRadius.value = radius;
    this.renderer.compute(this.srcN); this.renderer.compute(this.srcC);
  }

  tick(dt: number) {
    waterUniforms.dt.value = dt;
    const r = this.renderer, n = this.nodes;
    r.compute(n.fluxN); r.compute(n.fluxC);
    r.compute(n.velN); r.compute(n.velC);
    r.compute(n.updN); r.compute(n.watC);
    if (this.erosionEnabled) {
      r.compute(n.eroN); r.compute(n.bC); r.compute(n.loC); r.compute(n.sEC);
      r.compute(n.advN); r.compute(n.sAC);
      r.compute(n.thN); r.compute(n.bTC);
    }
  }
}
