// Shared scalar uniforms used across materials + the sea. Lives alone to avoid
// import cycles.
import { uniform } from 'three/tsl';
import { PLANET } from '../config';

export const heightScaleUniform = uniform(PLANET.heightScale);

/** Sea level in stored-height units [0..1] (terrain below is underwater). */
export const seaLevelUniform = uniform(PLANET.seaLevel);

/** World radius of the ocean shell = baseRadius + seaLevel*heightScale. */
export const seaRadiusUniform = uniform(PLANET.baseRadius + PLANET.seaLevel * PLANET.heightScale);

/** Keep the derived sea radius in sync when sea level changes. */
export function setSeaLevel(level: number): void {
  seaLevelUniform.value = level;
  seaRadiusUniform.value = PLANET.baseRadius + level * PLANET.heightScale;
}
