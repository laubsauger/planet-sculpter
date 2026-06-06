// Per-face cell-area field (T26, V31). Cube-sphere cells are NOT equal area:
// tan-warp reduces the ratio but ~1.4x variation remains (center vs edge/corner).
// Storing water/sediment as VOLUME and dividing by this area to get depth makes
// flux + seam exchange conserve across uneven cells (depth-store mis-conserves
// exactly at the distorted seam cells -> ridges). Area = |dP/du x dP/dv| of the
// warped direction map, NORMALIZED so the per-face mean = 1 (keeps tuned sim
// constants valid; only distorted cells shift).

import {
  DataTexture,
  RedFormat,
  FloatType,
  LinearFilter,
  ClampToEdgeWrapping,
} from 'three';
import { faceUVToDir, type Vec3 } from '../tsl/warp';
import type { FaceName } from '../config';
import type { HeightTexture } from './heightField';

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function crossLen(a: Vec3, b: Vec3): number {
  const cx = a[1] * b[2] - a[2] * b[1];
  const cy = a[2] * b[0] - a[0] * b[2];
  const cz = a[0] * b[1] - a[1] * b[0];
  return Math.hypot(cx, cy, cz);
}

/** Relative cell area per texel, mean-normalized to 1 over the face. */
export function buildCellAreaTexture(face: FaceName, res: number): HeightTexture {
  const n = res + 1;
  const data = new Float32Array(n * n);
  const h = 1 / res; // half-texel finite-difference step in (u,v)
  let sum = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      // central-difference jacobian columns dP/du, dP/dv of the warped dir map.
      const du = sub(faceUVToDir(face, u + h, v), faceUVToDir(face, u - h, v));
      const dv = sub(faceUVToDir(face, u, v + h), faceUVToDir(face, u, v - h));
      const area = crossLen(du, dv); // proportional to true cell area
      data[j * n + i] = area;
      sum += area;
    }
  }
  const mean = sum / (n * n) || 1;
  for (let k = 0; k < data.length; k++) data[k] /= mean; // mean -> 1

  const texture = new DataTexture(data, n, n, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { face, texture, data, n };
}
