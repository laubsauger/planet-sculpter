// M1 static height: CPU-built per-face Float32 texture, sampled in vertex node.
// Values are a function of sphere DIRECTION (via shared faceUVToDir), so shared
// edge texels on adjacent faces carry identical heights -> crack-free (V5 geom).
// M2 converts this to a StorageTexture written by brush/sim compute.

import {
  DataTexture,
  RedFormat,
  FloatType,
  LinearFilter,
  ClampToEdgeWrapping,
} from 'three';
import { faceUVToDir, type Vec3 } from '../tsl/warp';
import type { FaceName } from '../config';

// --- tiny deterministic value-noise fbm over 3D direction ---------------------

function hash3(x: number, y: number, z: number): number {
  let h = x * 374761393 + y * 668265263 + z * 1274126177;
  h = (h ^ (h >>> 13)) >>> 0;
  h = (h * 1274126177) >>> 0;
  return (h & 0xffffff) / 0xffffff; // [0,1)
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

function valueNoise(x: number, y: number, z: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = smooth(xf), v = smooth(yf), w = smooth(zf);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const c = (dx: number, dy: number, dz: number) => hash3(xi + dx, yi + dy, zi + dz);
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
  const x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
  const x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
  const y0 = lerp(x00, x10, v);
  const y1 = lerp(x01, x11, v);
  return lerp(y0, y1, w);
}

/** clamped smoothstep on a pre-normalized t. */
function smoothStep01(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

function fbm(dir: Vec3): number {
  // Ridged multifractal: sharp mountain ridges where terrain is high, smoother
  // lowlands -> steep/interesting in parts, not uniformly smooth.
  let amp = 0.5;
  let freq = 1.6;
  let sum = 0;
  let norm = 0;
  let weight = 1;
  // 7 octaves -> more detail at finer scales (was 5); faster amplitude falloff
  // keeps ridges crisp without 1-texel spikes.
  for (let o = 0; o < 7; o++) {
    const nv = valueNoise(dir[0] * freq, dir[1] * freq, dir[2] * freq);
    let signal = 1 - Math.abs(nv * 2 - 1); // ridge: peak at nv=0.5
    signal = signal * signal * 0.6 + signal * 0.4; // softer sharpen
    signal *= weight;
    sum += signal * amp;
    norm += amp;
    weight = Math.min(1, signal * 1.2);
    amp *= 0.5;
    freq *= 2.0;
  }
  let h = sum / norm; // [0,1]
  h = Math.pow(h, 1.15);
  h = h * 0.82 + valueNoise(dir[0] * 1.8 + 50, dir[1] * 1.8, dir[2] * 1.8 - 50) * 0.18;

  // Regional ruggedness: more & wider mountain ranges (lower threshold) breaking
  // up the grasslands; plains still flatter.
  const typeN = valueNoise(dir[0] * 0.75 + 20, dir[1] * 0.75 - 15, dir[2] * 0.75 + 8);
  const rugged = smoothStep01((typeN - 0.28) / 0.45); // 0 plains .. 1 mountains
  h = h * (0.58 + 0.55 * rugged);

  // CONTINENTAL uplift: broad low-frequency landmasses ride well ABOVE sea level
  // (varied highlands/plateaus), while basins stay low -> oceans. Replaces the
  // flat near-sea-level shelf with real above-water terrain + large-scale relief.
  const cont = valueNoise(dir[0] * 0.6 + 71, dir[1] * 0.6 - 33, dir[2] * 0.6 + 12);
  const continental = smoothStep01((cont - 0.4) / 0.3); // 0 ocean basin .. 1 continent
  h = h + continental * 0.22;

  // fine surface roughness (high-freq, signed) — stronger on rugged terrain so
  // mountains read craggy, plains stay smooth. adds close-up detail.
  const fine = valueNoise(dir[0] * 11 - 17, dir[1] * 11 + 23, dir[2] * 11 - 5) * 2 - 1;
  h += fine * 0.035 * (0.35 + 0.65 * rugged);

  // PLATEAUS / mesas: regional flat-topped highlands. Subtle compression of the
  // height variation toward a raised level -> gentle tablelands (⊥ chunky).
  const plat = smoothStep01((valueNoise(dir[0] * 0.85 - 31, dir[1] * 0.85 + 14, dir[2] * 0.85 + 19) - 0.56) / 0.16);
  const onHigh = smoothStep01((h - 0.32) / 0.25); // only flatten elevated terrain
  h = h + (0.52 - h) * 0.25 * plat * onHigh;

  // RIFT valleys: thin sparse carved lows (inverted ridge noise) -> fine canyons,
  // not chunky troughs. higher freq + narrow + shallow so they read as fissures.
  const riftN = 1 - Math.abs(valueNoise(dir[0] * 4.5 + 61, dir[1] * 4.5 - 9, dir[2] * 4.5 + 40) * 2 - 1);
  const rift = smoothStep01((riftN - 0.88) / 0.07); // only sharp narrow ridge cores
  h -= rift * 0.05 * smoothStep01((h - 0.18) / 0.2); // shallow carve

  // lift the highest peaks higher (snow-capped) without raising lowlands.
  h = h + Math.max(0, h - 0.5) * 0.55;

  return Math.max(0, Math.min(1, h));
}

export interface HeightTexture {
  face: FaceName;
  texture: DataTexture;
  data: Float32Array; // n*n, row-major; texel k <-> vertex k
  n: number;
}

/**
 * Initial loose-material (soil/sand) thickness per face. Its own fbm (offset
 * direction) so the soft layer varies naturally across the surface -> uneven
 * strata instead of a flat constant. Range ~[0.004, 0.04].
 */
export function buildLooseTexture(face: FaceName, res: number): HeightTexture {
  const n = res + 1;
  const data = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      const dir = faceUVToDir(face, u, v);
      // offset + reuse fbm for an independent field.
      const t = fbm([dir[0] + 11.3, dir[1] - 7.1, dir[2] + 4.9] as Vec3);
      // varied cover: thin/bare patches AND deep soil pockets -> uneven strata.
      data[j * n + i] = 0.008 + t * t * 0.09;
    }
  }
  const texture = new DataTexture(data, n, n, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { face, texture, data, n };
}

/** n = res+1 texels per edge; texel (i,j) uses identical (u,v) as vertex (i,j). */
export function buildHeightTexture(face: FaceName, res: number): HeightTexture {
  const n = res + 1;
  const data = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      const dir = faceUVToDir(face, u, v);
      data[j * n + i] = fbm(dir);
    }
  }
  const texture = new DataTexture(data, n, n, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { face, texture, data, n };
}

/**
 * Regional rainfall map [0..1] from low-frequency noise: large wet regions
 * (where rain falls) and dry regions (deserts) -> rain in certain parts, not
 * others. Multiplied by the global rain rate (which can cycle over time).
 */
export function buildRainfallTexture(face: FaceName, res: number): HeightTexture {
  const n = res + 1;
  const data = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      const dir = faceUVToDir(face, u, v);
      // several big climate regions (low freq) + mid detail -> MANY distinct wet
      // and dry zones spread over the planet (⊥ one blob, ⊥ globally uniform).
      const a = valueNoise(dir[0] * 1.3 + 30, dir[1] * 1.3 - 12, dir[2] * 1.3 + 7);
      const b = valueNoise(dir[0] * 3.2 - 4, dir[1] * 3.2 + 9, dir[2] * 3.2 - 2);
      // gentle latitude banding (y = pole axis) so wet/dry also shifts by band.
      const lat = Math.abs(dir[1]); // 0 equator .. 1 pole
      const band = 0.5 + 0.4 * Math.cos(lat * Math.PI * 2);
      const r = a * 0.5 + b * 0.22 + band * 0.28;
      // moderate spread: full 0..1 range with distinct dry (deserts) + wet zones
      // and gradients between -> varied, not all-or-nothing.
      data[j * n + i] = Math.max(0, Math.min(1, (r - 0.28) * 1.9));
    }
  }
  const texture = new DataTexture(data, n, n, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { face, texture, data, n };
}

/**
 * Per-cell erosion-resistance multiplier (~[0.45, 1.75]) from higher-frequency
 * noise. Spatial variation breaks sheet-flow symmetry: softer cells erode
 * faster -> flow concentrates -> channels/canyons form (feedback).
 */
export function buildHardnessTexture(face: FaceName, res: number): HeightTexture {
  const n = res + 1;
  const data = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      const dir = faceUVToDir(face, u, v);
      // MULTI-SCALE erodibility (higher = softer): large resistant REGIONS that
      // rivers must wind around + mid + fine variation for channel nucleation.
      // Strong contrast so flow deflects naturally instead of shooting straight.
      const lo = valueNoise(dir[0] * 2.5 + 13, dir[1] * 2.5 - 6, dir[2] * 2.5 + 21); // big zones
      const t = valueNoise(dir[0] * 7 - 5.2, dir[1] * 7 + 3.7, dir[2] * 7 - 1.9);
      const t2 = valueNoise(dir[0] * 14 + 2, dir[1] * 14 - 8, dir[2] * 14 + 5);
      const r = lo * 0.45 + t * 0.37 + t2 * 0.18;
      data[j * n + i] = 0.12 + Math.pow(r, 1.9) * 2.5; // wide [~0.12, ~2.6], sharp contrast
    }
  }
  const texture = new DataTexture(data, n, n, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { face, texture, data, n };
}
