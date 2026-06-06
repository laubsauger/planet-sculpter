// V31: per-face cell-area field. Mean-normalized to 1, positive, bounded
// variation (tan-warp leaves residual distortion -> not all 1). No GPU.
import { describe, it, expect } from 'vitest';
import { FACES } from '../src/config';
import { buildCellAreaTexture } from '../src/planet/cellArea';

describe('cellArea', () => {
  const res = 32;

  it('is positive + finite everywhere', () => {
    for (const face of FACES) {
      const { data } = buildCellAreaTexture(face, res);
      for (const a of data) {
        expect(Number.isFinite(a)).toBe(true);
        expect(a).toBeGreaterThan(0);
      }
    }
  });

  it('has per-face mean ~1 (mean-normalized, V31)', () => {
    for (const face of FACES) {
      const { data } = buildCellAreaTexture(face, res);
      const mean = data.reduce((s, a) => s + a, 0) / data.length;
      expect(mean).toBeCloseTo(1, 6);
    }
  });

  it('has real but bounded area variation (tan-warp residual)', () => {
    for (const face of FACES) {
      const { data } = buildCellAreaTexture(face, res);
      let lo = Infinity;
      let hi = -Infinity;
      for (const a of data) {
        lo = Math.min(lo, a);
        hi = Math.max(hi, a);
      }
      expect(hi / lo).toBeGreaterThan(1.05); // not uniform -> volume storage matters
      expect(hi / lo).toBeLessThan(2.5); // tan-warp keeps it modest
    }
  });
});
