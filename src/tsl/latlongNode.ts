// TSL mirror of latlong.ts lonLatToDir — the shader's notion of a texel's
// sphere direction, matching the CPU math for geometry/picking/sim.

import { vec3, float, sin, cos } from 'three/tsl';

const TWO_PI = Math.PI * 2;

/* eslint-disable @typescript-eslint/no-explicit-any */

/** (u,v float nodes in [0,1]) -> unit direction node. u=longitude, v=latitude. */
export function lonLatDirNode(u: any, v: any): any {
  const lon = float(u).mul(TWO_PI);
  const lat = float(v).sub(0.5).mul(Math.PI);
  const cl = cos(lat);
  return vec3(cl.mul(cos(lon)), sin(lat), cl.mul(sin(lon)));
}
