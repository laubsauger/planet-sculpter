// River-source emitter (T16). Direction-space stamp that paints a persistent
// emission RATE into a per-face source map; the fluid addSource pass injects
// that rate every tick -> a continuous spring. Reuses the brush's dir-space
// approach (paints across seams, face-culled). Re-stamping sets max (no runaway).

import type { WebGPURenderer } from 'three/webgpu';
import { Vector3 } from 'three';
/* eslint-disable @typescript-eslint/no-explicit-any */
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
  vec3,
  vec4,
  length,
  smoothstep,
  max,
  uniform,
} from 'three/tsl';
import { type FieldSet, buildCopyCompute, buildFillZero } from '../sim/fields';
import { faceDirNode } from '../tsl/warpNode';
import { FACE_BASES } from '../tsl/warp';
import { FACES, type FaceName } from '../config';

type ComputeNode = Parameters<WebGPURenderer['compute']>[0];
const FACE_CULL_DOT = 0.37;

export class EmitterTool {
  private uCenterDir = uniform(new Vector3(1, 0, 0));
  private uRadius = uniform(0.05);
  private uRate = uniform(0.05);
  private readonly stampNode = new Map<FaceName, ComputeNode>();
  private readonly copyNode = new Map<FaceName, ComputeNode>();

  constructor(
    private readonly source: FieldSet,
    private readonly n: number,
  ) {
    const res = n - 1;
    for (const face of FACES) {
      const f = source.field(face);
      const fn = Fn(() => {
        const x = instanceIndex.mod(uint(n));
        const y = instanceIndex.div(uint(n));
        const cur = textureLoad(f.main, ivec2(int(x), int(y))).x;
        const su = x.toFloat().div(res).mul(2).sub(1);
        const sv = y.toFloat().div(res).mul(2).sub(1);
        const dir = faceDirNode(face, su, sv);
        const dist = length(dir.sub(vec3(this.uCenterDir)));
        const w = float(1).sub(smoothstep(float(0), this.uRadius, dist));
        // set the emission rate (max so re-painting doesn't accumulate to infinity).
        const next: any = max(cur, this.uRate.mul(w));
        textureStore(f.scratch, uvec2(x, y), vec4(next, 0, 0, 1)).toWriteOnly();
      });
      this.stampNode.set(face, fn().compute(n * n) as ComputeNode);
      this.copyNode.set(face, buildCopyCompute(f.scratch, f.main, n) as ComputeNode);
    }
  }

  stamp(renderer: WebGPURenderer, centerDir: Vector3, rate: number, radius: number): void {
    this.uCenterDir.value.copy(centerDir).normalize();
    this.uRate.value = rate;
    this.uRadius.value = radius;
    const c = this.uCenterDir.value;
    for (const face of FACES) {
      const fwd = FACE_BASES[face].forward;
      if (fwd[0] * c.x + fwd[1] * c.y + fwd[2] * c.z < FACE_CULL_DOT) continue;
      renderer.compute(this.stampNode.get(face)!);
      renderer.compute(this.copyNode.get(face)!);
    }
  }

  clear(renderer: WebGPURenderer): void {
    for (const face of FACES) {
      renderer.compute(buildFillZero(this.source.field(face).main, this.n));
    }
  }
}
