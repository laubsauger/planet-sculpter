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

function fbm(dir: Vec3): number {
  // Ridged multifractal: sharp mountain ridges where terrain is high, smoother
  // lowlands -> steep/interesting in parts, not uniformly smooth.
  let amp = 0.5;
  let freq = 1.6;
  let sum = 0;
  let norm = 0;
  let weight = 1;
  // fewer octaves + faster amplitude falloff -> ridges without 1-texel spikes.
  for (let o = 0; o < 5; o++) {
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
  // blend ridged with a smooth base -> interesting but not spiky/overdone.
  h = Math.pow(h, 1.15);
  h = h * 0.7 + valueNoise(dir[0] * 1.8 + 50, dir[1] * 1.8, dir[2] * 1.8 - 50) * 0.3;
  return h;
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
      // low frequency -> big climate zones.
      const a = valueNoise(dir[0] * 1.4 + 30, dir[1] * 1.4 - 12, dir[2] * 1.4 + 7);
      const b = valueNoise(dir[0] * 2.8 - 4, dir[1] * 2.8 + 9, dir[2] * 2.8 - 2);
      const t = a * 0.7 + b * 0.3;
      data[j * n + i] = Math.max(0, Math.min(1, (t - 0.32) * 2.6)); // wet zones vs dry
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
      // higher frequency -> finer resistance variation -> finer, branchier channels.
      const t = fbm([dir[0] * 5.5 - 5.2, dir[1] * 5.5 + 3.7, dir[2] * 5.5 - 1.9] as Vec3);
      data[j * n + i] = 0.4 + t * 1.5;
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
