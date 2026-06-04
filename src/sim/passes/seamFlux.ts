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
import { type HeightFields, type FieldSet, buildCopyCompute } from '../fields';
import { EDGES, type SeamTable, type EdgeId } from '../../planet/seamTable';
import type { FluidUniforms } from './water';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

/* eslint-disable @typescript-eslint/no-explicit-any */

function buildSeamFlux(
  face: FaceName,
  height: HeightFields,
  water: FieldSet,
  table: SeamTable,
  n: number,
  p: FluidUniforms,
): ComputeNode {
  const res = n - 1;
  const selfB = height.field(face).main;
  const selfD = water.field(face).main;
  const dOut = water.field(face).scratch;
  const N = uint(n);

  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const ix = int(x);
    const iy = int(y);

    const dc: any = textureLoad(selfD, ivec2(ix, iy)).x.toVar();
    const surfC = textureLoad(selfB, ivec2(ix, iy)).x.add(dc);
    const rate = p.pipeArea.mul(p.gravity).div(p.pipeLength);
    const change: any = float(0).toVar();

    for (const edge of EDGES) {
      const seam = table[face][edge as EdgeId];
      const nbB = height.field(seam.nFace).main;
      const nbD = water.field(seam.nFace).main;

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

      const surfN = textureLoad(nbB, ivec2(bx, by)).x.add(textureLoad(nbD, ivec2(bx, by)).x);
      // outflow if our surface is higher; clamp so we don't drain >half of ours
      // or pull >half of the neighbor's water.
      const ndH = textureLoad(nbD, ivec2(bx, by)).x;
      const qDt = surfC.sub(surfN).mul(rate).mul(p.dt).max(ndH.mul(-0.5)).min(dc.mul(0.5));
      If(onBorder, () => {
        change.subAssign(qDt);
      });
    }

    const dNew = max(dc.add(change), float(0));
    textureStore(dOut, uvec2(x, y), vec4(dNew, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Cross-seam water exchange across all faces (two-phase: all read main -> scratch, then copy back). */
export class SeamFlux {
  private readonly flux = new Map<FaceName, ComputeNode>();
  private readonly copy = new Map<FaceName, ComputeNode>();

  constructor(
    height: HeightFields,
    water: FieldSet,
    table: SeamTable,
    n: number,
    p: FluidUniforms,
  ) {
    for (const face of FACES) {
      this.flux.set(face, buildSeamFlux(face, height, water, table, n, p));
      const f = water.field(face);
      this.copy.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  sync(renderer: WebGPURenderer): void {
    for (const node of this.flux.values()) renderer.compute(node);
    for (const node of this.copy.values()) renderer.compute(node);
  }
}
