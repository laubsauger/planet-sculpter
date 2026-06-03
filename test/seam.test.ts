// V13: seamTable 24-edge map correctness, no GPU.
// A face's boundary vertex and its mapped neighbor vertex must be the SAME
// point on the sphere -> dirs equal -> no crack after seam sync.
import { describe, it, expect } from 'vitest';
import { FACES } from '../src/config';
import {
  buildSeamTable,
  neighborTexel,
  edgeUV,
  EDGES,
  type EdgeId,
} from '../src/planet/seamTable';
import { faceUVToDir } from '../src/tsl/warp';

const RES = 32;

function texelToDir(face: (typeof FACES)[number], x: number, y: number) {
  const u = (x / RES) * 2 - 1;
  const v = (y / RES) * 2 - 1;
  return faceUVToDir(face, u, v);
}

describe('seam table (V5, V13)', () => {
  const table = buildSeamTable(RES);

  it('maps every boundary vertex to a coincident neighbor vertex', () => {
    for (const face of FACES) {
      for (const edge of EDGES) {
        const seam = table[face][edge as EdgeId];
        // sample interior of the edge (skip exact corners; 3-face ambiguity)
        for (let i = 2; i <= RES - 2; i++) {
          const [u, v] = edgeUV(edge as EdgeId, i, RES);
          const myDir = faceUVToDir(face, u, v);
          const [nx, ny] = neighborTexel(seam, i, RES);
          const nDir = texelToDir(seam.nFace, nx, ny);
          expect(nDir[0]).toBeCloseTo(myDir[0], 5);
          expect(nDir[1]).toBeCloseTo(myDir[1], 5);
          expect(nDir[2]).toBeCloseTo(myDir[2], 5);
        }
      }
    }
  });

  it('neighbor fixed coord sits on a border (0 or res)', () => {
    for (const face of FACES) {
      for (const edge of EDGES) {
        const seam = table[face][edge as EdgeId];
        expect([0, RES]).toContain(seam.nFixedVal);
      }
    }
  });

  it('seam relation is symmetric (neighbor maps back to this face)', () => {
    for (const face of FACES) {
      for (const edge of EDGES) {
        const seam = table[face][edge as EdgeId];
        // the neighbor face must list `face` as a neighbor on some edge
        const back = EDGES.map((e) => table[seam.nFace][e as EdgeId]).find(
          (s) => s.nFace === face,
        );
        expect(back).toBeDefined();
      }
    }
  });
});
