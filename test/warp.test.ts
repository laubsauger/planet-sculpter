// V13: warp dir<->(face,u,v) round-trip + seamTable correctness, no GPU.
import { describe, it, expect } from 'vitest';
import { FACES } from '../src/config';
import { faceUVToDir, dirToFaceUV, warp, unwarp, type Vec3 } from '../src/tsl/warp';

describe('warp / unwarp', () => {
  it('inverts itself', () => {
    for (const a of [-1, -0.5, -0.137, 0, 0.42, 0.9, 1]) {
      expect(unwarp(warp(a))).toBeCloseTo(a, 10);
    }
  });
  it('fixes endpoints and center', () => {
    expect(warp(0)).toBeCloseTo(0, 12);
    expect(warp(1)).toBeCloseTo(1, 12);
    expect(warp(-1)).toBeCloseTo(-1, 12);
  });
});

describe('faceUVToDir / dirToFaceUV round-trip (V1, V13)', () => {
  it('recovers face + uv for interior samples', () => {
    const samples = [-0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9];
    for (const face of FACES) {
      for (const u of samples) {
        for (const v of samples) {
          const dir = faceUVToDir(face, u, v);
          const r = dirToFaceUV(dir);
          expect(r.face).toBe(face);
          expect(r.u).toBeCloseTo(u, 6);
          expect(r.v).toBeCloseTo(v, 6);
        }
      }
    }
  });

  it('produces unit-length directions', () => {
    for (const face of FACES) {
      const d = faceUVToDir(face, 0.3, -0.7);
      const len = Math.hypot(d[0], d[1], d[2]);
      expect(len).toBeCloseTo(1, 10);
    }
  });

  it('face centers point along their forward axis', () => {
    const expected: Record<string, Vec3> = {
      px: [1, 0, 0], nx: [-1, 0, 0],
      py: [0, 1, 0], ny: [0, -1, 0],
      pz: [0, 0, 1], nz: [0, 0, -1],
    };
    for (const face of FACES) {
      const d = faceUVToDir(face, 0, 0);
      expect(d[0]).toBeCloseTo(expected[face][0], 10);
      expect(d[1]).toBeCloseTo(expected[face][1], 10);
      expect(d[2]).toBeCloseTo(expected[face][2], 10);
    }
  });
});
