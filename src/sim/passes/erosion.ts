// Hydraulic erosion passes (M5, T14, V7). After the water depth update:
//   velocity:  water velocity from flux imbalance
//   erosion:   sediment capacity C = Kc*sin(tilt)*|v|; erode bedrock if s<C,
//              deposit if s>C (moves material between bedrock `b` and sediment `s`)
//   advect:    transport suspended sediment with the velocity (semi-Lagrangian)
// Canonical read main -> write scratch -> copy back (V2). Cross-seam handled by
// per-tick seam sync of b/s in Simulation. Ref: Mei et al.

import type { WebGPURenderer } from 'three/webgpu';
import type { StorageTexture } from 'three/webgpu';
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
  max,
  length,
  uniform,
} from 'three/tsl';
import { seamHeight, type SampleFace } from '../../tsl/surface';
import type { SeamTable } from '../../planet/seamTable';
import type { FaceName } from '../../config';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

/* eslint-disable @typescript-eslint/no-explicit-any */

export const erosionUniforms = {
  sedimentCapacity: uniform(0.4), // Kc — more carving capacity (deeper channels)
  dissolve: uniform(0.18), // Ks
  deposit: uniform(0.06), // Kd — low so rivers don't dam themselves
  minSlope: uniform(0.015),
  advectScale: uniform(1.0),
  /** min flow speed to erode; low so even slow river flow carves a channel. */
  erodeSpeedMin: uniform(0.22),
  /** erodibility of exposed hard rock (loose = 1). lower = more resistant. */
  rockErodibility: uniform(0.18),
  /** loose-layer thickness at which the surface is "fully soft". */
  looseFull: uniform(0.02),
  /** thermal slump: height diff above which material slides to lower neighbors. */
  talus: uniform(0.004),
  /** low -> channels stay sharp/narrow (less smoothing fighting channelization). */
  thermalRate: uniform(0.3),
  dt: uniform(1 / 60),
};

function xy(n: number) {
  const N = uint(n);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}
const clampI = (i: any, res: number) => i.toFloat().max(float(0)).min(float(res)).toInt();

/** Water velocity (vx,vy) from flux imbalance, written to rg of `velOut`. */
export function buildVelocity(
  f: StorageTexture,
  d: StorageTexture,
  velOut: StorageTexture,
  n: number,
): ComputeNode {
  const res = n - 1;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    // border cells have no valid intra-face neighbor across the seam (clamped
    // flux -> ~0 velocity -> they under-erode -> symmetric ridge along the seam
    // heightSeam can't flatten). Sample the velocity stencil one cell INWARD at
    // borders so the border inherits the first-interior flow (flow continues
    // across the seam). Interior cells unchanged.
    const cx = ix.toFloat().max(float(1)).min(float(res - 1)).toInt();
    const cy = iy.toFloat().max(float(1)).min(float(res - 1)).toInt();
    const xm = clampI(cx.sub(1), res);
    const xp = clampI(cx.add(1), res);
    const ym = clampI(cy.sub(1), res);
    const yp = clampI(cy.add(1), res);

    const self = textureLoad(f, ivec2(cx, cy));
    const Lr = textureLoad(f, ivec2(xm, cy)).y; // left neighbor's R (into us +x)
    const Rl = textureLoad(f, ivec2(xp, cy)).x; // right neighbor's L (into us -x)
    const Bt = textureLoad(f, ivec2(cx, ym)).z; // bottom neighbor's T (into us +y)
    const Tb = textureLoad(f, ivec2(cx, yp)).w; // top neighbor's B (into us -y)

    // divide by a MIN depth (not EPS) so thin films don't produce huge speeds.
    const dc = max(textureLoad(d, ivec2(cx, cy)).x, float(0.02));
    const vx = Lr.sub(self.x).add(self.y.sub(Rl)).mul(0.5).div(dc).max(float(-3)).min(float(3));
    const vy = Bt.sub(self.w).add(self.z.sub(Tb)).mul(0.5).div(dc).max(float(-3)).min(float(3));

    textureStore(velOut, uvec2(x, y), vec4(vx, vy, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/**
 * Erode/deposit with material layers. `b` = total height (rock + loose),
 * `loose` = thickness of soft material (soil/sand) on top of hard rock.
 * Loose erodes easily; exposed rock resists (rockErodibility). Eroded material
 * + all deposition is LOOSE -> valleys fill with soft sediment, steep eroded
 * slopes expose rock.
 */
export function buildErosion(
  b: StorageTexture,
  loose: StorageTexture,
  s: StorageTexture,
  vel: StorageTexture,
  d: StorageTexture,
  hardness: StorageTexture,
  source: StorageTexture,
  bOut: StorageTexture,
  looseOut: StorageTexture,
  sOut: StorageTexture,
  n: number,
  face: FaceName,
  table: SeamTable,
  sampleB: SampleFace,
): ComputeNode {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);

    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const lc = textureLoad(loose, ivec2(ix, iy)).x.min(bc); // loose <= total
    const sc = textureLoad(s, ivec2(ix, iy)).x;
    const dc = textureLoad(d, ivec2(ix, iy)).x;
    const v = textureLoad(vel, ivec2(ix, iy));

    // cross-seam b gradient (clamped self-self at borders would give ~0 tilt ->
    // border doesn't erode -> raised ridge along the seam, B10).
    const dbx = seamHeight(face, sampleB, table, ix.add(1), iy)
      .sub(seamHeight(face, sampleB, table, ix.sub(1), iy))
      .mul(0.5);
    const dby = seamHeight(face, sampleB, table, ix, iy.add(1))
      .sub(seamHeight(face, sampleB, table, ix, iy.sub(1)))
      .mul(0.5);
    const tilt = length(vec2(dbx, dby)).min(float(0.5));
    const sinTilt = max(tilt, u.minSlope);
    const speed = length(vec2(v.x, v.y)).min(float(3));
    const hasWater = dc.greaterThan(float(0.003)).select(float(1), float(0)); // thin steep flow still erodes
    const capacity = u.sedimentCapacity.mul(sinTilt).mul(speed).mul(hasWater);
    // don't carve a pit at a spring head (high outflow there reads as fast flow).
    const notSource = textureLoad(source, ivec2(ix, iy)).x.lessThan(float(0.0001)).select(float(1), float(0));

    // surface softness: 1 where loose is deep, rockErodibility where rock bare.
    const softness = max(u.rockErodibility, lc.div(u.looseFull).min(float(1)));
    // per-cell resistance variation -> symmetry breaking -> channels/canyons.
    const hard = textureLoad(hardness, ivec2(ix, iy)).x;

    const CAP = float(0.001); // allow deeper per-step carving -> channel feedback
    const erodeGate = speed.greaterThan(u.erodeSpeedMin).select(float(1), float(0)).mul(notSource);
    // erosion scaled by material softness * local resistance variation.
    const erode = max(float(0), capacity.sub(sc))
      .mul(u.dissolve)
      .mul(softness)
      .mul(hard)
      .mul(erodeGate)
      .min(CAP);
    const dep = max(float(0), sc.sub(capacity)).mul(u.deposit).min(CAP);

    const bNew: any = max(bc.sub(erode).add(dep), float(0));
    // loose: erosion removes loose first (then bites rock); deposition adds loose.
    const looseNew: any = max(lc.sub(erode), float(0)).add(dep);
    const sNew: any = max(float(0), sc.add(erode).sub(dep)).min(float(2));

    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(looseNew, 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Semi-Lagrangian advection of sediment along velocity (nearest backtrace).
 *  Backtrace is <1 cell/step, so a one-cell-out cross-seam sample (seamHeight)
 *  lets sediment leave across face edges instead of clamping + piling there. */
export function buildAdvect(
  vel: StorageTexture,
  sOut: StorageTexture,
  n: number,
  face: FaceName,
  table: SeamTable,
  sampleS: SampleFace,
): ComputeNode {
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const v = textureLoad(vel, ivec2(ix, iy));
    const step = erosionUniforms.dt.mul(erosionUniforms.advectScale);
    const bx = ix.toFloat().sub(v.x.mul(step)).round().toInt();
    const by = iy.toFloat().sub(v.y.mul(step)).round().toInt();
    const sVal = seamHeight(face, sampleS, table, bx, by);
    textureStore(sOut, uvec2(x, y), vec4(sVal, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/**
 * Thermal erosion / slumping (M6, T15). Material on slopes steeper than the
 * talus angle slides to lower neighbors -> smooths spikes/noise. Symmetric
 * pair exchange (conservative). Run each erosion tick.
 */
export function buildThermal(
  b: StorageTexture,
  bOut: StorageTexture,
  n: number,
  face: FaceName,
  table: SeamTable,
  sampleB: SampleFace,
): ComputeNode {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const c = textureLoad(b, ivec2(ix, iy)).x;
    // cross-seam neighbors so the border slumps consistently (no seam ridge).
    const nbs = [
      seamHeight(face, sampleB, table, ix.sub(1), iy),
      seamHeight(face, sampleB, table, ix.add(1), iy),
      seamHeight(face, sampleB, table, ix, iy.sub(1)),
      seamHeight(face, sampleB, table, ix, iy.add(1)),
    ];
    let net: any = float(0);
    for (const nb of nbs) {
      net = net.sub(max(float(0), c.sub(nb).sub(u.talus)));
      net = net.add(max(float(0), nb.sub(c).sub(u.talus)));
    }
    const bNew = max(c.add(net.mul(u.thermalRate).mul(0.25)), float(0));
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
