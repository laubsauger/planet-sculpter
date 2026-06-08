import { describe, expect, it } from 'vitest';
import { buildFlatBenchmark, FLAT_BENCHMARKS } from '../src/flat/flatBenchmarks';

const data = (texture: ReturnType<typeof buildFlatBenchmark>['height']) =>
  texture.image.data as Float32Array;

describe('flat diagnostic benchmarks', () => {
  it('builds deterministic finite fields', () => {
    for (const name of FLAT_BENCHMARKS) {
      if (name === 'default') continue;
      const a = buildFlatBenchmark(name, 32, 24);
      const b = buildFlatBenchmark(name, 32, 24);
      expect([...data(a.height)]).toEqual([...data(b.height)]);
      for (const field of [a.height, a.loose, a.water, a.sediment, a.source]) {
        expect([...data(field)].every(Number.isFinite)).toBe(true);
      }
    }
  });

  it('initializes the intended forcing', () => {
    const river = buildFlatBenchmark('riverToSea', 64, 64);
    const dam = buildFlatBenchmark('damBreak', 64, 64);
    const rain = buildFlatBenchmark('rainErosion', 64, 64);
    expect(Math.max(...data(river.source))).toBeGreaterThan(0);
    expect(Math.max(...data(dam.water))).toBeGreaterThan(0);
    expect(rain.rainOn).toBe(true);
  });
});
