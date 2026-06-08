// Equirectangular mapping: lonLatToDir <-> dirToLonLat round-trip + cell area.
import { describe, it, expect } from 'vitest';
import { lonLatToDir, dirToLonLat, cellAreaAt } from '../src/planet/latlong';

describe('lat/long mapping', () => {
  it('round-trips (u,v) -> dir -> (u,v) away from the poles', () => {
    for (const u of [0.0, 0.1, 0.37, 0.5, 0.8, 0.99]) {
      for (const v of [0.15, 0.4, 0.5, 0.6, 0.85]) {
        const d = lonLatToDir(u, v);
        // dir is unit length
        expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 6);
        const r = dirToLonLat(d);
        expect(r.u).toBeCloseTo(u, 5);
        expect(r.v).toBeCloseTo(v, 5);
      }
    }
  });

  it('wrap seam: u=0 and u=1 are the same meridian', () => {
    const a = lonLatToDir(0, 0.5);
    const b = lonLatToDir(1, 0.5);
    expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeLessThan(1e-9);
  });

  it('poles: v=0/1 collapse to ±Y', () => {
    expect(lonLatToDir(0.3, 1)[1]).toBeCloseTo(1, 6); // north pole +Y
    expect(lonLatToDir(0.7, 0)[1]).toBeCloseTo(-1, 6); // south pole -Y
  });

  it('cell area ~1 at equator, ~0 at poles', () => {
    expect(cellAreaAt(0.5)).toBeCloseTo(1, 6);
    expect(cellAreaAt(0)).toBeLessThan(0.01);
    expect(cellAreaAt(1)).toBeLessThan(0.01);
  });
});
