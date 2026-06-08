import { describe, expect, it } from 'vitest';
import { MOMENTUM_MAX_SUBSTEPS, momentumSubstepCount } from '../src/flat/momentumCfl';

describe('momentum CFL substeps', () => {
  it('keeps the normal 20 Hz simulation tick to one substep', () => {
    expect(momentumSubstepCount(1 / 20, 9.81)).toBe(1);
  });

  it('splits large timesteps and respects the hard dispatch bound', () => {
    expect(momentumSubstepCount(0.1, 9.81)).toBe(2);
    expect(momentumSubstepCount(10, 9.81)).toBe(MOMENTUM_MAX_SUBSTEPS);
  });
});
