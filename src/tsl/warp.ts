// Shared cube-sphere math (V1). CPU here: used by picking (T9) + tests (T13).
// TSL vertex/compute shaders mirror these exact formulas so geometry & physics
// never disagree. Keep formulas in ONE place.

import { FACES, type FaceName } from '../config';

export type Vec3 = [number, number, number];

interface FaceBasis {
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}

// Right-handed, consistent set. dirToFaceUV is the exact inverse of faceUVToDir.
export const FACE_BASES: Record<FaceName, FaceBasis> = {
  px: { forward: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] },
  nx: { forward: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] },
  py: { forward: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] },
  ny: { forward: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] },
  pz: { forward: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0] },
  nz: { forward: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] },
};

const QUARTER_PI = Math.PI / 4;

/** Tangent area-equalization warp: [-1,1] -> [-1,1], spreads center texels. */
export function warp(a: number): number {
  return Math.tan(a * QUARTER_PI);
}

/** Inverse of warp. */
export function unwarp(a: number): number {
  return Math.atan(a) / QUARTER_PI;
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** (face, u,v in [-1,1]) -> unit direction on sphere. */
export function faceUVToDir(face: FaceName, u: number, v: number): Vec3 {
  const { forward, right, up } = FACE_BASES[face];
  const uw = warp(u);
  const vw = warp(v);
  return normalize([
    forward[0] + uw * right[0] + vw * up[0],
    forward[1] + uw * right[1] + vw * up[1],
    forward[2] + uw * right[2] + vw * up[2],
  ]);
}

/** Unit direction -> (face, u,v in [-1,1]). Inverse of faceUVToDir. */
export function dirToFaceUV(dir: Vec3): { face: FaceName; u: number; v: number } {
  const ax = Math.abs(dir[0]);
  const ay = Math.abs(dir[1]);
  const az = Math.abs(dir[2]);

  let face: FaceName;
  if (ax >= ay && ax >= az) face = dir[0] >= 0 ? 'px' : 'nx';
  else if (ay >= ax && ay >= az) face = dir[1] >= 0 ? 'py' : 'ny';
  else face = dir[2] >= 0 ? 'pz' : 'nz';

  const { forward, right, up } = FACE_BASES[face];
  const f = dot(dir, forward); // > 0 on the chosen face
  const uw = dot(dir, right) / f;
  const vw = dot(dir, up) / f;
  return { face, u: unwarp(uw), v: unwarp(vw) };
}

/**
 * Project a direction onto a SPECIFIC face's (u,v), without the max-axis face
 * pick. Used for seam mapping where the neighbor face is already known but the
 * shared-edge dir is ambiguous to dirToFaceUV. dir should lie on/near `face`.
 */
export function projectDirToFace(face: FaceName, dir: Vec3): { u: number; v: number } {
  const { forward, right, up } = FACE_BASES[face];
  const f = dot(dir, forward);
  return { u: unwarp(dot(dir, right) / f), v: unwarp(dot(dir, up) / f) };
}

export { FACES };
