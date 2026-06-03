// Shared vertical displacement scale (uniform), used by terrain + water
// materials and the surface-normal helper. Lives alone to avoid import cycles.
import { uniform } from 'three/tsl';
import { PLANET } from '../config';

export const heightScaleUniform = uniform(PLANET.heightScale);
