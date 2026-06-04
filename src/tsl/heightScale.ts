// Shared scalar uniforms used across materials + the sea. Lives alone to avoid
// import cycles.
import { uniform } from 'three/tsl';
import { PLANET } from '../config';

export const heightScaleUniform = uniform(PLANET.heightScale);

/** Sea level in stored-height units [0..1] (terrain below is underwater). The
 *  fluid sim fills basins up to this (no separate ocean mesh). */
export const seaLevelUniform = uniform(PLANET.seaLevel);

export function setSeaLevel(level: number): void {
  seaLevelUniform.value = level;
}
