export const MOMENTUM_CFL = 0.45;
export const MOMENTUM_MAX_DEPTH = 2;
export const MOMENTUM_MAX_SPEED = 4;
export const MOMENTUM_MAX_SUBSTEPS = 8;

export function momentumSubstepCount(
  dt: number,
  gravity: number,
  maxDepth = MOMENTUM_MAX_DEPTH,
  maxSpeed = MOMENTUM_MAX_SPEED,
): number {
  const waveSpeedBound = maxSpeed + Math.sqrt(gravity * maxDepth);
  const maxDt = MOMENTUM_CFL / waveSpeedBound;
  return Math.min(MOMENTUM_MAX_SUBSTEPS, Math.max(1, Math.ceil(dt / maxDt)));
}
