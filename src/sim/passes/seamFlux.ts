// Cross-seam water flux (V5). Faces are sealed internally for conservation, so
// water must cross seams here: each border cell exchanges water with the
// neighbor face's adjacent (one-inward) cell, driven by the (b+d) surface-height
// difference — same gravity-to-core rule as intra-face flux, but computed from
// surface heights directly (avoids matching rotated flux components across the
// seam). Makes the water SURFACE continuous (no seam) AND lets water flow into
// depressions that straddle a face boundary. Read main -> write scratch -> copy.

import type { WebGPURenderer } from 'three/webgpu';
import {
  Fn,
  instanceIndex,
  textureLoad,
  textureStore,
  ivec2,
  uvec2,
  uint,
  int,
  float,
  vec4,
  max,
  If,
} from 'three/tsl';
import { FACES, type FaceName } from '../../config';
import { type FieldSet, buildCopyCompute } from '../fields';
import { EDGES, type SeamTable, type EdgeId } from '../../planet/seamTable';
import type { FluidUniforms } from './water';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

/* eslint-disable @typescript-eslint/no-explicit-any */

const EPS = 1e-6;

function buildSeamFlux(
  face: FaceName,
  surface: FieldSet,
  water: FieldSet,
  sediment: FieldSet,
  table: SeamTable,
  n: number,
  p: FluidUniforms,
): ComputeNode {
  const res = n - 1;
  // surface = b + depth (vol/area) precomputed -> 3 tex/face (surface,vol,sed)
  // instead of 4 (b,vol,sed,area); 4×5 cross-seam faces = 20 > 16 limit.
  const selfSurf = surface.field(face).main;
  const selfD = water.field(face).main;
  const selfS = sediment.field(face).main;
  const dOut = water.field(face).scratch;
  const sOut = sediment.field(face).scratch;
  const N = uint(n);

  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);

    // d/s are VOLUME/MASS. surface precomputed (b + vol/area); conc = mass/vol.
    const dc: any = textureLoad(selfD, ivec2(ix, iy)).x.toVar(); // volume
    const sc: any = textureLoad(selfS, ivec2(ix, iy)).x.toVar(); // mass
    const surfC = textureLoad(selfSurf, ivec2(ix, iy)).x;
    // suspended sediment concentration (per unit water) carried by the flow.
    const cC = sc.div(max(dc, float(EPS)));
    const rate = p.pipeArea.mul(p.gravity).div(p.pipeLength);
    const change: any = float(0).toVar();
    const sChange: any = float(0).toVar();

    for (const edge of EDGES) {
      const seam = table[face][edge as EdgeId];
      const nbSurf = surface.field(seam.nFace).main;
      const nbD = water.field(seam.nFace).main;
      const nbS = sediment.field(seam.nFace).main;

      const onBorder =
        edge === 0
          ? ix.lessThan(int(1))
          : edge === 1
            ? ix.greaterThan(int(res - 1))
            : edge === 2
              ? iy.lessThan(int(1))
              : iy.greaterThan(int(res - 1));
      const vary = edge === 0 || edge === 1 ? iy : ix;
      const varyI = seam.varyReverse ? int(res).sub(vary) : vary;
      const bx = seam.nFixedIsX ? int(seam.nInwardVal) : varyI;
      const by = seam.nFixedIsX ? varyI : int(seam.nInwardVal);

      const ndH = textureLoad(nbD, ivec2(bx, by)).x; // neighbor volume
      const surfN = textureLoad(nbSurf, ivec2(bx, by)).x;
      // outflow if our surface is higher; clamp so we don't drain >half of ours
      // or pull >half of the neighbor's water (volume clamps).
      const qDt = surfC.sub(surfN).mul(rate).mul(p.dt).max(ndH.mul(-0.5)).min(dc.mul(0.5));
      // sediment rides with the water: leaving water (qDt>0) carries OUR
      // concentration; arriving water (qDt<0) carries the NEIGHBOR's. Without
      // this, water crosses the seam but its sediment is left behind -> a loose/
      // sediment ridge piles along every seam (symmetric -> not removable by
      // edge-averaging). Mirror of the water exchange -> conservative.
      const cN = textureLoad(nbS, ivec2(bx, by)).x.div(max(ndH, float(EPS)));
      const carried = qDt.greaterThan(float(0)).select(cC, cN).mul(qDt);
      If(onBorder, () => {
        change.subAssign(qDt);
        sChange.subAssign(carried);
      });
    }

    const dNew = max(dc.add(change), float(0));
    const sNew = max(sc.add(sChange), float(0));
    textureStore(dOut, uvec2(x, y), vec4(dNew, 0, 0, 1)).toWriteOnly();
    textureStore(sOut, uvec2(x, y), vec4(sNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Cross-seam water + sediment exchange across all faces (two-phase: all read
 *  main -> scratch, then copy back). */
export class SeamFlux {
  private readonly flux = new Map<FaceName, ComputeNode>();
  private readonly copy: ComputeNode[] = [];

  constructor(
    surface: FieldSet,
    water: FieldSet,
    sediment: FieldSet,
    table: SeamTable,
    n: number,
    p: FluidUniforms,
  ) {
    for (const face of FACES) {
      this.flux.set(face, buildSeamFlux(face, surface, water, sediment, table, n, p));
      const w = water.field(face);
      const s = sediment.field(face);
      this.copy.push(buildCopyCompute(w.scratch, w.main, n) as ComputeNode);
      this.copy.push(buildCopyCompute(s.scratch, s.main, n) as ComputeNode);
    }
  }

  sync(renderer: WebGPURenderer): void {
    for (const node of this.flux.values()) renderer.compute(node);
    for (const node of this.copy) renderer.compute(node);
  }
}
