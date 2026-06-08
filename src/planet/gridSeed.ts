// Equirectangular single-grid seed textures (W x H). lon cols wrap (u=i/W),
// lat rows span pole-to-pole (v=j/(H-1)). Reuses the shared per-direction value
// functions so the terrain matches what the cube-sphere produced — just sampled
// on the lat/long grid instead of 6 faces.

import {
  DataTexture,
  RedFormat,
  FloatType,
  LinearFilter,
  ClampToEdgeWrapping,
  RepeatWrapping,
} from 'three';
import { lonLatToDir, cellAreaAt, type Vec3 } from './latlong';
import { heightValue, looseValue, rainfallValue, hardnessValue } from './heightField';

export interface GridTexture {
  texture: DataTexture;
  data: Float32Array; // W*H row-major
  w: number;
  h: number;
}

function makeGrid(w: number, h: number, data: Float32Array): GridTexture {
  const texture = new DataTexture(data, w, h, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = RepeatWrapping; // longitude wraps
  texture.wrapT = ClampToEdgeWrapping; // latitude clamps at poles
  texture.needsUpdate = true;
  return { texture, data, w, h };
}

/** Fill a W×H grid by evaluating a per-direction value function. */
export function buildGridField(w: number, h: number, valueFn: (dir: Vec3) => number): GridTexture {
  const data = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    const v = h > 1 ? j / (h - 1) : 0.5; // pole..pole
    for (let i = 0; i < w; i++) {
      const u = i / w; // wraps
      data[j * w + i] = valueFn(lonLatToDir(u, v));
    }
  }
  return makeGrid(w, h, data);
}

export const buildGridHeight = (w: number, h: number) => buildGridField(w, h, heightValue);
export const buildGridLoose = (w: number, h: number) => buildGridField(w, h, looseValue);
export const buildGridRainfall = (w: number, h: number) => buildGridField(w, h, rainfallValue);
export const buildGridHardness = (w: number, h: number) => buildGridField(w, h, hardnessValue);

/** Blend the height grid toward a flat dome over each polar cap. At the equirect
 *  pole all longitude columns share direction (0,±1,0) but distinct per-column
 *  heights -> vertices collapse to a spiky line on the Y axis. Flattening each cap
 *  to its mean height (smooth transition at the boundary) makes a clean pole.
 *  Operates IN PLACE on a height GridTexture; mirror the cap cos with poleCapCos. */
export function flattenPoleCaps(grid: GridTexture, capCos = 0.18): void {
  const { data, w, h } = grid;
  const innerCos = capCos * 0.5;
  const cosLat = (j: number) => Math.sin((j / (h - 1)) * Math.PI);
  // per-hemisphere cap mean (north v>0.5, south v<0.5).
  let nSum = 0, nCnt = 0, sSum = 0, sCnt = 0;
  for (let j = 0; j < h; j++) {
    if (cosLat(j) >= capCos) continue;
    const north = j / (h - 1) > 0.5;
    for (let i = 0; i < w; i++) {
      const val = data[j * w + i];
      if (north) { nSum += val; nCnt++; } else { sSum += val; sCnt++; }
    }
  }
  const nMean = nCnt ? nSum / nCnt : 0;
  const sMean = sCnt ? sSum / sCnt : 0;
  for (let j = 0; j < h; j++) {
    const c = cosLat(j);
    if (c >= capCos) continue;
    const mean = j / (h - 1) > 0.5 ? nMean : sMean;
    // t: 1 at cap boundary -> 0 at deep pole (smoothstep).
    const x = Math.min(1, Math.max(0, (c - innerCos) / (capCos - innerCos)));
    const t = x * x * (3 - 2 * x);
    for (let i = 0; i < w; i++) {
      const k = j * w + i;
      data[k] = mean + (data[k] - mean) * t;
    }
  }
  grid.texture.needsUpdate = true;
}

/** Cell area ∝ cos(lat), mean-normalized to 1 (depth = vol/area, conservation). */
export function buildGridCellArea(w: number, h: number): GridTexture {
  const data = new Float32Array(w * h);
  let sum = 0;
  for (let j = 0; j < h; j++) {
    const v = h > 1 ? j / (h - 1) : 0.5;
    const a = cellAreaAt(v);
    for (let i = 0; i < w; i++) {
      data[j * w + i] = a;
      sum += a;
    }
  }
  const mean = sum / (w * h) || 1;
  for (let k = 0; k < data.length; k++) data[k] /= mean;
  return makeGrid(w, h, data);
}
