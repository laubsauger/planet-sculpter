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
  If,
  smoothstep,
  mix,
  sin,
  min,
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
  /** incision feedback (V37): 0 = uniform sheet erosion, 1 = erosion strongly
   *  concentrated where discharge (depth*speed) is high -> channels self-deepen
   *  & capture flow -> emergent rivers instead of wide sheet flow. */
  channelFocus: uniform(0.9),
  /** discharge (depth*speed) at which a cell counts as a full channel. */
  channelDischarge: uniform(0.015),
  /** thermal slump is suppressed in deep flowing water -> channels don't heal flat. */
  channelDepthRef: uniform(0.02),
  /** flow momentum/inertia (V38): 0 = velocity tracks instant gradient (straight
   *  channels), ->1 = flow overshoots bends -> swings into outer banks -> meander. */
  flowInertia: uniform(0.6),
  /** lateral (cut-bank) erosion: erode where flow runs ACROSS slope (rams a bank)
   *  -> channel migrates sideways. the other half of meandering. */
  lateralErosion: uniform(0.5),
  /** stratified bedrock (V39): erosion resistance varies with ELEVATION -> hard
   *  rock LAYERS. river incises until it hits a hard band, then stalls (⊥ runaway
   *  downcut/widening) -> terraced canyons. freq = layer count over height range. */
  strataFreq: uniform(55),
  strataStrength: uniform(0.75),
  /** cross-seam C1 smoothing strength in the seam band (kills the erosion crease). */
  seamSmooth: uniform(0.8),
  /** viz texture decay/tick: fresh erode/deposit streaks fade over ~tens of ticks. */
  vizDecay: uniform(0.95),
  /** still water settles suspended sediment -> muddy lakes clear, beds build (V35). */
  stillDeposit: uniform(0.04),
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
  area: StorageTexture,
  velPrev: StorageTexture,
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

    // d is VOLUME -> depth = vol/area (V31).
    const depth = textureLoad(d, ivec2(cx, cy)).x.div(
      max(textureLoad(area, ivec2(cx, cy)).x, float(1e-6)),
    );
    // active gate (T28, V33): dry cells have no flow -> write 0 + skip the 5
    // flux loads + math. most of the planet is dry -> big erosion-tick saving.
    const out: any = vec4(0, 0, 0, 1).toVar();
    If(depth.greaterThan(float(1e-5)), () => {
      const self = textureLoad(f, ivec2(cx, cy));
      const Lr = textureLoad(f, ivec2(xm, cy)).y; // left neighbor's R (into us +x)
      const Rl = textureLoad(f, ivec2(xp, cy)).x; // right neighbor's L (into us -x)
      const Bt = textureLoad(f, ivec2(cx, ym)).z; // bottom neighbor's T (into us +y)
      const Tb = textureLoad(f, ivec2(cx, yp)).w; // top neighbor's B (into us -y)
      // divide by a MIN depth (not EPS) so thin films don't produce huge speeds.
      const dc = max(depth, float(0.02));
      const vx = Lr.sub(self.x).add(self.y.sub(Rl)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      const vy = Bt.sub(self.w).add(self.z.sub(Tb)).mul(0.5).div(dc).max(float(-3)).min(float(3));
      // FLOW INERTIA (V38): blend the instantaneous flux velocity with the
      // PREVIOUS velocity advected from upstream -> the current carries momentum
      // and overshoots bends (swings into outer banks) instead of snapping to the
      // local downhill -> meandering. step ~0.6 cell upstream backtrace.
      const bx = clampI(cx.toFloat().sub(vx.mul(0.6)), res);
      const by = clampI(cy.toFloat().sub(vy.mul(0.6)), res);
      const prev = textureLoad(velPrev, ivec2(bx, by));
      const ax = mix(vx, prev.x, erosionUniforms.flowInertia);
      const ay = mix(vy, prev.y, erosionUniforms.flowInertia);
      out.assign(vec4(ax, ay, 0, 1));
    });
    textureStore(velOut, uvec2(x, y), out).toWriteOnly();
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
  viz: StorageTexture,
  vizOut: StorageTexture,
  n: number,
  face: FaceName,
  table: SeamTable,
  sampleB: SampleFace,
  area: StorageTexture,
): ComputeNode {
  const u = erosionUniforms;
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);

    const bc = textureLoad(b, ivec2(ix, iy)).x;
    const lc = textureLoad(loose, ivec2(ix, iy)).x.min(bc); // loose <= total
    const sc = textureLoad(s, ivec2(ix, iy)).x;
    // viz (r=erosion, g=deposit) decays every tick (outside the active gate so
    // old streaks fade even after the cell dries). Fresh erode/deposit max'd in.
    const vizPrev = textureLoad(viz, ivec2(ix, iy));
    const vizR: any = vizPrev.x.mul(u.vizDecay).toVar();
    const vizG: any = vizPrev.y.mul(u.vizDecay).toVar();
    // d is VOLUME -> depth = vol/area (V31).
    const dc = textureLoad(d, ivec2(ix, iy)).x.div(
      max(textureLoad(area, ivec2(ix, iy)).x, float(1e-6)),
    );
    // active gate (T28, V33): a cell with no water AND no suspended sediment has
    // nothing to erode/deposit -> passthrough + skip the heavy work (8 seamHeight
    // tilt loads, velocity, capacity). dry land / deep-ocean interior = most
    // cells -> big erosion-tick saving. wet OR sediment-bearing cells stay active
    // (no missed rivers, unlike a coarse tile mask).
    const bNew: any = bc.toVar();
    const looseNew: any = lc.toVar();
    const sNew: any = sc.toVar();
    const active = dc.greaterThan(float(0.0008)).or(sc.greaterThan(float(1e-5)));
    If(active, () => {
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
      const hasWater = dc.greaterThan(float(0.0012)).select(float(1), float(0)); // thin steep flow still erodes (low thresh)
      // INCISION FEEDBACK (V37): concentrate erosion where DISCHARGE (depth*speed,
      // i.e. stream power) is high. A cell that gathers flow erodes more -> deepens
      // -> captures more flow -> deepens more = emergent channel. channelFocus
      // dials between uniform sheet erosion (0) and strong concentration (1).
      const discharge = dc.mul(speed);
      const conc = mix(float(1), smoothstep(float(0), u.channelDischarge, discharge), u.channelFocus);
      const capacity = u.sedimentCapacity.mul(sinTilt).mul(speed).mul(hasWater).mul(conc);
      // don't carve a pit at a spring head (high outflow there reads as fast flow).
      const notSource = textureLoad(source, ivec2(ix, iy)).x.lessThan(float(0.0001)).select(float(1), float(0));

      // surface softness: 1 where loose is deep, rockErodibility where rock bare.
      const softness = max(u.rockErodibility, lc.div(u.looseFull).min(float(1)));
      // per-cell resistance variation -> symmetry breaking -> channels/canyons.
      const hard = textureLoad(hardness, ivec2(ix, iy)).x;

      const CAP = float(0.001); // allow deeper per-step carving -> channel feedback
      const erodeGate = speed.greaterThan(u.erodeSpeedMin).select(float(1), float(0)).mul(notSource);
      // erosion scaled by material softness * local resistance variation.
      const erodeBase = max(float(0), capacity.sub(sc))
        .mul(u.dissolve)
        .mul(softness)
        .mul(hard)
        .mul(erodeGate);
      // LATERAL / cut-bank erosion (V38): when the (inertial) flow runs ACROSS the
      // slope — misaligned with downhill — it's ramming a bank; erode extra so the
      // channel migrates sideways = the second half of meandering. eps-guarded
      // normalize (zero vectors -> NaN otherwise).
      const dh = vec2(dbx, dby).mul(-1).add(vec2(1e-5, 1e-5)); // downhill dir
      const fdir = vec2(v.x, v.y).add(vec2(1e-6, 1e-6));
      const align = max(float(0), fdir.normalize().dot(dh.normalize()));
      // SLOPE GATE: meandering/lateral cut only on GENTLE slopes (floodplains).
      // on steep slopes gravity dominates -> straight incision, ⊥ sideways cut.
      const gentle = float(1).sub(smoothstep(float(0.05), float(0.18), tilt));
      const lateral = discharge
        .mul(float(1).sub(align))
        .mul(gentle)
        .mul(u.lateralErosion)
        .mul(softness)
        .mul(hard)
        .mul(erodeGate);
      // STRATA (V39): resistance as a function of bedrock ELEVATION -> horizontal
      // hard-rock layers. abs(sin) dips to 0 at each hard band; resist scales BOTH
      // vertical + lateral erode so a river stalls when it cuts into hard rock
      // (curbs runaway downcutting + endless widening).
      // thin, distinct hard layers (mostly soft between) that genuinely resist.
      const raw = sin(bc.mul(u.strataFreq)).abs();
      const band = smoothstep(float(0), float(0.3), raw); // ~0 at hard layer, 1 soft
      const strataResist = mix(float(1), band, u.strataStrength); // hard band -> 1-strength
      const erode = erodeBase.add(lateral).mul(strataResist).min(CAP);
      const dep = max(float(0), sc.sub(capacity)).mul(u.deposit).min(CAP);

      bNew.assign(max(bc.sub(erode).add(dep), float(0)));
      // loose: erosion removes loose first (then bites rock); deposition adds loose.
      looseNew.assign(max(lc.sub(erode), float(0)).add(dep));
      sNew.assign(max(float(0), sc.add(erode).sub(dep)).min(float(2)));
      // record fresh activity into the viz channels (normalized by per-step CAP).
      vizR.assign(max(vizR, erode.div(CAP)));
      vizG.assign(max(vizG, dep.div(CAP)));
    });

    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
    textureStore(looseOut, uvec2(x, y), vec4(looseNew, 0, 0, 1)).toWriteOnly();
    textureStore(vizOut, uvec2(x, y), vec4(vizR, vizG, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/**
 * Cross-seam C1 continuity (the apron's effect without resizing textures).
 * Independent per-face erosion makes the bedrock SLOPE diverge at a shared edge
 * -> a crease (the persistent "seam"). heightSeam only matches the edge VALUE,
 * not the slope. Here: in a narrow band along each face edge, diffuse b toward
 * the average of its CROSS-SEAM neighbors (seamHeight, verified mapping) so the
 * surface meets smoothly across the seam. Gated to the band (interior skipped ->
 * cheap). Strongest at the edge, fading out by `band` cells.
 */
export function buildSeamSmooth(
  b: StorageTexture,
  bOut: StorageTexture,
  n: number,
  face: FaceName,
  table: SeamTable,
  sampleB: SampleFace,
): ComputeNode {
  const res = n - 1;
  const band = 4.0; // wider -> cross-seam influence diffuses further from the edge
  const fn = Fn(() => {
    const { x, y, ix, iy } = xy(n);
    const c = textureLoad(b, ivec2(ix, iy)).x;
    const bNew: any = c.toVar();
    const edist = min(
      min(ix.toFloat(), float(res).sub(ix.toFloat())),
      min(iy.toFloat(), float(res).sub(iy.toFloat())),
    );
    If(edist.lessThan(float(band)), () => {
      const avg = seamHeight(face, sampleB, table, ix.sub(1), iy)
        .add(seamHeight(face, sampleB, table, ix.add(1), iy))
        .add(seamHeight(face, sampleB, table, ix, iy.sub(1)))
        .add(seamHeight(face, sampleB, table, ix, iy.add(1)))
        .mul(0.25);
      // weight: 1 at the edge -> 0 at `band` cells in, scaled by seamSmooth.
      const w = smoothstep(float(band), float(0), edist).mul(erosionUniforms.seamSmooth);
      bNew.assign(mix(c, avg, w));
    });
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
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
  d: StorageTexture,
  area: StorageTexture,
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
    // channel persistence (V37): suppress slumping where water is deep, so an
    // incised river bed doesn't immediately heal flat by slumping its banks in.
    const depth = textureLoad(d, ivec2(ix, iy)).x.div(max(textureLoad(area, ivec2(ix, iy)).x, float(1e-6)));
    const wet = smoothstep(float(0), u.channelDepthRef, depth); // 0 dry .. 1 deep
    const rate = u.thermalRate.mul(float(1).sub(wet.mul(0.85)));
    const bNew = max(c.add(net.mul(rate).mul(0.25)), float(0));
    textureStore(bOut, uvec2(x, y), vec4(bNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}
