// Normal baking (perf). Computes the seam-aware analytic object-space normal
// per texel into an rgba texture, ONCE per terrain/water change, so the material
// just samples it (cheap) instead of recomputing ~25 direction evals/fragment.
// This is what makes higher resolution affordable.

import type { WebGPURenderer } from 'three/webgpu';
import { Fn, instanceIndex, textureStore, uvec2, uint, int, vec4 } from 'three/tsl';
import { FACES, type FaceName } from '../../config';
import { makePosAt, objNormalAt, type SampleFace } from '../../tsl/surface';
import type { SeamTable } from '../../planet/seamTable';
import type { FieldSet } from '../fields';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

function buildNormalCompute(
  face: FaceName,
  sample: SampleFace,
  table: SeamTable,
  out: Parameters<typeof textureStore>[0],
  n: number,
): ComputeNode {
  const posAt = makePosAt(face, sample, table);
  const fn = Fn(() => {
    const x = instanceIndex.mod(uint(n));
    const y = instanceIndex.div(uint(n));
    const objN = objNormalAt(posAt, int(x), int(y));
    textureStore(out, uvec2(x, y), vec4(objN, 1)).toWriteOnly();
  });
  return fn().compute(n * n) as ComputeNode;
}

/** Bakes object-space normals for all 6 faces into `normals`. */
export class NormalBaker {
  private readonly nodes = new Map<FaceName, ComputeNode>();

  constructor(sample: SampleFace, table: SeamTable, normals: FieldSet, n: number) {
    for (const face of FACES) {
      this.nodes.set(face, buildNormalCompute(face, sample, table, normals.field(face).main, n));
    }
  }

  bake(renderer: WebGPURenderer): void {
    for (const node of this.nodes.values()) renderer.compute(node);
  }
}
