// Build one cube-sphere face as BufferGeometry (V1). CPU build, once.
// Vertices placed via shared faceUVToDir so adjacent faces' edge verts
// coincide exactly -> no cracks (V5).

import { BufferGeometry, BufferAttribute } from 'three';
import { faceUVToDir } from '../tsl/warp';
import type { FaceName } from '../config';

export interface FaceGeometry {
  face: FaceName;
  geometry: BufferGeometry;
  /** verts per edge = res + 1 */
  segments: number;
}

/**
 * res = grid cells per edge. Produces (res+1)^2 verts.
 * Attributes: position (on base-radius sphere), uv in [0,1], normal (radial).
 * Height displacement applied later in the vertex node (T8); base normals radial.
 */
export function buildFaceGeometry(face: FaceName, res: number, baseRadius: number): FaceGeometry {
  const n = res + 1;
  const count = n * n;
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);

  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const idx = j * n + i;
      const u = -1 + (2 * i) / res;
      const v = -1 + (2 * j) / res;
      const dir = faceUVToDir(face, u, v); // unit
      positions[idx * 3] = dir[0] * baseRadius;
      positions[idx * 3 + 1] = dir[1] * baseRadius;
      positions[idx * 3 + 2] = dir[2] * baseRadius;
      normals[idx * 3] = dir[0];
      normals[idx * 3 + 1] = dir[1];
      normals[idx * 3 + 2] = dir[2];
      uvs[idx * 2] = i / res;
      uvs[idx * 2 + 1] = j / res;
    }
  }

  const indices = new Uint32Array(res * res * 6);
  let k = 0;
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const a = j * n + i;
      const b = a + 1;
      const c = a + n;
      const d = c + 1;
      // CCW-from-outside winding: normal = u x v = +forward (outward).
      // (a,c,b)/(b,c,d) would invert -> near faces culled -> see-through.
      indices[k++] = a; indices[k++] = b; indices[k++] = d;
      indices[k++] = a; indices[k++] = d; indices[k++] = c;
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new BufferAttribute(uvs, 2));
  geometry.setIndex(new BufferAttribute(indices, 1));
  return { face, geometry, segments: n };
}
