// TSL mirror of warp.ts faceUVToDir (V1). Same tan-warp + face basis, so the
// shader's notion of a texel's sphere direction matches the CPU math used for
// geometry and picking. Used by the direction-space brush and (later) the sim.

import { vec3, normalize, float } from 'three/tsl';
import { FACE_BASES } from './warp';
import type { FaceName } from '../config';

const QUARTER_PI = Math.PI / 4;

/* eslint-disable @typescript-eslint/no-explicit-any */
// TSL nodes aren't cleanly typed through this DSL; use loose typing.

/** (face, u,v float nodes in [-1,1]) -> unit direction node. */
export function faceDirNode(face: FaceName, u: any, v: any): any {
  const { forward, right, up } = FACE_BASES[face];
  const uw = float(u).mul(QUARTER_PI).tan(); // tan-warp
  const vw = float(v).mul(QUARTER_PI).tan();
  const f = vec3(forward[0], forward[1], forward[2]);
  const r = vec3(right[0], right[1], right[2]);
  const up3 = vec3(up[0], up[1], up[2]);
  return normalize(f.add(r.mul(uw)).add(up3.mul(vw)));
}
