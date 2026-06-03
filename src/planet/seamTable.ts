// Cube-sphere seam map (T11, V5). For each face edge, which neighbor face +
// texel does each boundary vertex map to. Built from the shared faceUVToDir /
// dirToFaceUV math (V1) by nudging just past the edge into the neighbor's
// domain, so there is no hand-maintained 24-edge table to get wrong (V13 test).

import { FACES, type FaceName } from '../config';
import { faceUVToDir, dirToFaceUV, projectDirToFace, type Vec3 } from '../tsl/warp';

/** Edge id: 0 = x:0, 1 = x:res, 2 = y:0, 3 = y:res. */
export type EdgeId = 0 | 1 | 2 | 3;
export const EDGES: EdgeId[] = [0, 1, 2, 3];

export interface EdgeSeam {
  face: FaceName;
  edge: EdgeId;
  nFace: FaceName;
  /** neighbor fixed coordinate is X (true) or Y (false). */
  nFixedIsX: boolean;
  /** value of the neighbor fixed coordinate (0 or res) = the shared edge. */
  nFixedVal: number;
  /** neighbor fixed coord ONE cell inward (1 or res-1) = the cell beyond our edge. */
  nInwardVal: number;
  /** neighbor varying coordinate = reverse ? res - i : i. */
  varyReverse: boolean;
}

/** texel (x,y) -> (u,v) in [-1,1] for a vertex grid of `res` cells. */
function texelUV(x: number, y: number, res: number): [number, number] {
  return [(x / res) * 2 - 1, (y / res) * 2 - 1];
}

/** Boundary vertex (u,v) for face edge at varying index i. */
function edgeUV(edge: EdgeId, i: number, res: number): [number, number] {
  const t = (i / res) * 2 - 1;
  switch (edge) {
    case 0: return [-1, t]; // x = 0
    case 1: return [1, t]; // x = res
    case 2: return [t, -1]; // y = 0
    case 3: return [t, 1]; // y = res
  }
}

/** (u,v) nudged just past the edge into the neighbor's domain. */
function edgeUVOutward(edge: EdgeId, i: number, res: number): [number, number] {
  const d = 2 / res; // one texel beyond
  const t = (i / res) * 2 - 1;
  switch (edge) {
    case 0: return [-1 - d, t];
    case 1: return [1 + d, t];
    case 2: return [t, -1 - d];
    case 3: return [t, 1 + d];
  }
}

function uvToTexel(u: number, v: number, res: number): [number, number] {
  return [Math.round(((u + 1) / 2) * res), Math.round(((v + 1) / 2) * res)];
}

/** Build the seam descriptor for one (face, edge). */
function buildEdge(face: FaceName, edge: EdgeId, res: number): EdgeSeam {
  // Two interior samples to recover the linear neighbor mapping.
  const i1 = Math.round(res * 0.3);
  const i2 = Math.round(res * 0.6);

  // Identify the neighbor FACE by nudging just past the edge (max-axis pick is
  // unambiguous there). The exact edge dir itself is ambiguous to dirToFaceUV.
  const neighborFaceAt = (i: number): FaceName => {
    const [u, v] = edgeUVOutward(edge, i, res);
    const dir = faceUVToDir(face, u, v);
    return dirToFaceUV([dir[0], dir[1], dir[2]] as Vec3).face;
  };
  // Map the EXACT boundary vertex onto the (known) neighbor face.
  const probe = (i: number, nFace: FaceName): { nx: number; ny: number } => {
    const [u, v] = edgeUV(edge, i, res);
    const dir = faceUVToDir(face, u, v);
    const r = projectDirToFace(nFace, [dir[0], dir[1], dir[2]] as Vec3);
    const [nx, ny] = uvToTexel(r.u, r.v, res);
    return { nx, ny };
  };

  const nf1 = neighborFaceAt(i1);
  const nf2 = neighborFaceAt(i2);
  if (nf1 !== nf2) {
    throw new Error(`seam probe disagreed on neighbor face for ${face} edge ${edge}`);
  }
  const p1 = { nFace: nf1, ...probe(i1, nf1) };
  const p2 = { nFace: nf2, ...probe(i2, nf2) };

  // Which neighbor axis is fixed (constant across i)?
  const xFixed = p1.nx === p2.nx;
  const nFixedIsX = xFixed;
  const nFixedVal = xFixed ? p1.nx : p1.ny;

  // Direction of the varying neighbor axis vs i.
  const vary1 = xFixed ? p1.ny : p1.nx;
  const vary2 = xFixed ? p2.ny : p2.nx;
  const slope = (vary2 - vary1) / (i2 - i1);
  const varyReverse = slope < 0;

  // one cell inward from the shared edge (0 -> 1, res -> res-1).
  const nInwardVal = nFixedVal === 0 ? 1 : nFixedVal - 1;

  return { face, edge, nFace: p1.nFace, nFixedIsX, nFixedVal, nInwardVal, varyReverse };
}

/** Neighbor texel for face-edge boundary vertex at varying index i. */
export function neighborTexel(seam: EdgeSeam, i: number, res: number): [number, number] {
  const vary = seam.varyReverse ? res - i : i;
  return seam.nFixedIsX ? [seam.nFixedVal, vary] : [vary, seam.nFixedVal];
}

export type SeamTable = Record<FaceName, Record<EdgeId, EdgeSeam>>;

export function buildSeamTable(res: number): SeamTable {
  const table = {} as SeamTable;
  for (const face of FACES) {
    table[face] = {} as Record<EdgeId, EdgeSeam>;
    for (const edge of EDGES) table[face][edge] = buildEdge(face, edge, res);
  }
  return table;
}

/** Varying texel index for a boundary vertex on a given edge. */
export function edgeVaryingIndex(edge: EdgeId, x: number, y: number): number {
  return edge === 0 || edge === 1 ? y : x;
}

export { edgeUV, texelUV };
