// Hydraulic + thermal erosion on the single equirect grid (Stage 4). Ports the
// cube-sphere erosion passes but with trivial wrap-X (longitude) / clamp-Y
// (latitude) indexing — no seamHeight, no face table. Pass order per tick:
//   velocity -> erosion (b/loose/sediment) -> advect -> thermal
// Reuses erosionUniforms (passes/erosion.ts) so the GUI tunes both paths.
// Refs: Mei et al. (pipe model), stream-power incision, talus slumping.

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import {
  Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int,
  float, vec2, vec4, max, length, If, smoothstep, mix, sin, mx_fractal_noise_float,
} from 'three/tsl';
import { erosionUniforms } from './passes/erosion';
import { poleCapCos } from './gridWater';
import { lonLatDirNode } from '../tsl/latlongNode';
import { PLANET } from '../config';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];
/* eslint-disable @typescript-eslint/no-explicit-any */
const EPS = 1e-6;
const PI = float(Math.PI);
/** 0 deep in the polar cap (frozen) -> 1 outside; matches the water freeze. */
const capOpen = (iy: any, h: number) =>
  smoothstep(poleCapCos.mul(0.5), poleCapCos, sin(iy.toFloat().div(float(h - 1)).mul(PI)));

function coords(w: number) {
  const N = uint(w);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}
const wrapXi = (x: any, w: number) => x.add(int(w)).mod(int(w));
const clampYi = (y: any, h: number) => y.toFloat().max(float(0)).min(float(h - 1)).toInt();
const wrapXf = (xf: any, w: number) => xf.add(float(w)).mod(float(w)).floor().toInt();
const clampYf = (yf: any, h: number) => yf.max(float(0)).min(float(h - 1)).floor().toInt();

/** Water velocity (vx,vy) from flux imbalance + flow inertia. */
export function gridVelocity(
  f: StorageTexture,
  d: StorageTexture,
  area: StorageTexture,
  velPrev: StorageTexture,
  velOut: StorageTexture,
  w: number,
  h: number,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const xm = wrapXi(ix.sub(1), w);
    const xp = wrapXi(ix.add(1), w);
    const ym = clampYi(iy.sub(1), h);
    const yp = clampYi(iy.add(1), h);

    const depth = textureLoad(d, ivec2(ix, iy)).x.div(max(textureLoad(area, ivec2(ix, iy)).x, float(EPS)));
    const out: any = vec4(0, 0, 0, 1).toVar();
    If(depth.greaterThan(float(1e-5)), () => {
      const self = textureLoad(f, ivec2(ix, iy));
      const Lr = textureLoad(f, ivec2(xm, iy)).y;
      const Rl = textureLoad(f, ivec2(xp, iy)).x;
      const Bt = textureLoad(f, ivec2(ix, ym)).z;
      const Tb = textureLoad(f, ivec2(ix, yp)).w;
      const dc = max(depth, float(0.02));
      const vx = Lr.sub(self.x).add(self.y.sub(Rl)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      const vy = Bt.sub(self.w).add(self.z.sub(Tb)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      // flow inertia: blend instant flux velocity with the previous velocity
      // advected from upstream -> momentum overshoots bends -> meander.
      const bx = wrapXf(ix.toFloat().sub(vx.mul(0.6)), w);
      const by = clampYf(iy.toFloat().sub(vy.mul(0.6)), h);
      const prev = textureLoad(velPrev, ivec2(bx, by));
      const ax = mix(vx, prev.x, erosionUniforms.flowInertia);
      const ay = mix(vy, prev.y, erosionUniforms.flowInertia);
      out.assign(vec4(ax, ay, 0, 1));
    });
    textureStore(velOut, uvec2(x, y), out).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}

/** Erode/deposit with material layers (loose soft, exposed rock resists),
 *  incision feedback, lateral cut-bank, strata. b/loose/sediment update. */
export function gridErosion(
  b: StorageTexture,
  loose: StorageTexture,
  s: StorageTexture,
  vel: StorageTexture,
  d: StorageTexture,
  hardness: StorageTexture,
  source: StorageTexture,
  area: StorageTexture,
  bOut: StorageTexture,
  looseOut: StorageTexture,
  sOut: StorageTexture,
  w: number,
  h: number,
): ComputeNode {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const bAt = (cx: any, cy: any) => textureLoad(b, ivec2(wrapXi(cx, w), clampYi(cy, h))).x;

    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const lc = textureLoad(loose, ivec2(ix, iy)).x.min(bc);
    const sc = textureLoad(s, ivec2(ix, iy)).x;
    const dc = textureLoad(d, ivec2(ix, iy)).x.div(max(textureLoad(area, ivec2(ix, iy)).x, float(EPS)));

    const bNew: any = bc.toVar();
    const looseNew: any = lc.toVar();
    const sNew: any = sc.toVar();
    const active = dc.greaterThan(float(0.0008)).or(sc.greaterThan(float(1e-5)));
    If(active, () => {
      const v = textureLoad(vel, ivec2(ix, iy));
      const dbx = bAt(ix.add(1), iy).sub(bAt(ix.sub(1), iy)).mul(0.5);
      const dby = bAt(ix, iy.add(1)).sub(bAt(ix, iy.sub(1))).mul(0.5);
      const tilt = length(vec2(dbx, dby)).min(float(0.5));
      const sinTilt = max(tilt, u.minSlope);
      const speed = length(vec2(v.x, v.y)).min(float(3));
      const hasWater = dc.greaterThan(float(0.0012)).select(float(1), float(0));
      // depth suppression: shallow flowing water (rivers, beaches) erodes; deep
      // still water (lakes, oceans) buries its bed -> spared. shallow->1, deep->0.
      const depthSuppress = float(1).sub(smoothstep(u.erodeShallowDepth, u.erodeDeepDepth, dc));
      const discharge = dc.mul(speed);
      const conc = mix(float(1), smoothstep(float(0), u.channelDischarge, discharge), u.channelFocus);
      const capacity = u.sedimentCapacity.mul(sinTilt).mul(speed).mul(hasWater).mul(conc);
      const notSource = textureLoad(source, ivec2(ix, iy)).x.lessThan(float(0.0001)).select(float(1), float(0));

      const softness = max(u.rockErodibility, lc.div(u.looseFull).min(float(1)));
      // VOLUMETRIC hardness: 2D province map * 3D noise at the bedrock point
      // dir*(R+b) -> erodibility varies with DEPTH, so carving exposes interlocking
      // hard/soft material instead of cutting evenly. n3 in ~[-1,1].
      const dir = lonLatDirNode(ix.toFloat().div(float(w)), iy.toFloat().div(float(h - 1)));
      const pos3d = dir.mul(float(PLANET.baseRadius).add(bc));
      const n3 = mx_fractal_noise_float(pos3d.mul(u.hardness3dFreq), 4);
      const hard = textureLoad(hardness, ivec2(ix, iy)).x
        .mul(float(1).add(n3.mul(u.hardness3dAmp)).max(float(0.05)));

      const CAP = float(0.001).mul(u.simSpeed);
      const erodeGate = speed.greaterThan(u.erodeSpeedMin).select(float(1), float(0)).mul(notSource);
      const erodeBase = max(float(0), capacity.sub(sc)).mul(u.dissolve).mul(softness).mul(hard).mul(erodeGate);
      // lateral / cut-bank: flow misaligned with downhill rams a bank.
      const dh = vec2(dbx, dby).mul(-1).add(vec2(1e-5, 1e-5));
      const fdir = vec2(v.x, v.y).add(vec2(1e-6, 1e-6));
      const align = max(float(0), fdir.normalize().dot(dh.normalize()));
      const gentle = float(1).sub(smoothstep(float(0.05), float(0.18), tilt));
      const lateral = discharge.mul(float(1).sub(align)).mul(gentle).mul(u.lateralErosion).mul(softness).mul(hard).mul(erodeGate);
      // strata: resistance as fn of elevation -> horizontal hard-rock layers.
      const raw = sin(bc.mul(u.strataFreq)).abs();
      const band = smoothstep(float(0), float(0.3), raw);
      const strataResist = mix(float(1), band, u.strataStrength);
      const erode = erodeBase.add(lateral).mul(strataResist).mul(depthSuppress).mul(capOpen(iy, h)).mul(u.simSpeed).min(CAP);
      const dep = max(float(0), sc.sub(capacity)).mul(u.deposit).mul(u.simSpeed).min(CAP);

      bNew.assign(max(bc.sub(erode).add(dep), float(0)));
      looseNew.assign(max(lc.sub(erode), float(0)).add(dep));
      sNew.assign(max(float(0), sc.add(erode).sub(dep)).min(float(2)));
    });

    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(looseNew, 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}

/** Semi-Lagrangian advection of suspended sediment along velocity. */
export function gridAdvect(
  vel: StorageTexture,
  s: StorageTexture,
  sOut: StorageTexture,
  w: number,
  h: number,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const v = textureLoad(vel, ivec2(ix, iy));
    const step = erosionUniforms.dt.mul(erosionUniforms.advectScale);
    const bx = wrapXf(ix.toFloat().sub(v.x.mul(step)), w);
    const by = clampYf(iy.toFloat().sub(v.y.mul(step)), h);
    const sVal = textureLoad(s, ivec2(bx, by)).x;
    textureStore(sOut, uvec2(x, y), vec4(sVal, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}

/** Thermal slumping: material above the talus angle slides to lower neighbors;
 *  suppressed under deep flowing water so incised channels don't heal flat. */
export function gridThermal(
  b: StorageTexture,
  d: StorageTexture,
  area: StorageTexture,
  bOut: StorageTexture,
  w: number,
  h: number,
): ComputeNode {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = coords(w);
    const bAt = (cx: any, cy: any) => textureLoad(b, ivec2(wrapXi(cx, w), clampYi(cy, h))).x;
    const c = textureLoad(b, ivec2(ix, iy)).x;
    const nbs = [bAt(ix.sub(1), iy), bAt(ix.add(1), iy), bAt(ix, iy.sub(1)), bAt(ix, iy.add(1))];
    let net: any = float(0);
    for (const nb of nbs) {
      net = net.sub(max(float(0), c.sub(nb).sub(u.talus)));
      net = net.add(max(float(0), nb.sub(c).sub(u.talus)));
    }
    const depth = textureLoad(d, ivec2(ix, iy)).x.div(max(textureLoad(area, ivec2(ix, iy)).x, float(EPS)));
    const wet = smoothstep(float(0), u.channelDepthRef, depth);
    const rate = u.thermalRate.mul(float(1).sub(wet.mul(0.85))).mul(u.simSpeed);
    // freeze the polar caps: cell area -> 0 at the pole, so any per-column height
    // change re-creates the pole-pinch spike. keep the flattened seed cap static.
    const bNew = max(c.add(net.mul(rate).mul(0.25).mul(capOpen(iy, h))), float(0));
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h) as ComputeNode;
}
