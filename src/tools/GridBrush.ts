// Direction-space terrain brush on the single equirect height grid. Stamps a
// soft falloff (raise/lower/smooth/flatten). Reads main -> writes scratch ->
// copy back. Longitude wraps / latitude clamps for the smooth neighbor reads.

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import { Vector3 } from 'three';
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
  min,
  mix,
  uniform,
} from 'three/tsl';
import { lonLatDirNode } from '../tsl/latlongNode';
import { buildGridCopy } from '../sim/gridStore';
import type { BrushMode } from './BrushTool';

/* eslint-disable @typescript-eslint/no-explicit-any */
const MODE: Record<BrushMode, number> = { raise: 0, lower: 1, smooth: 2, flatten: 3 };

export class GridBrush {
  private readonly uCenter = uniform(new Vector3(1, 0, 0));
  private readonly uRadius = uniform(0.13);
  private readonly uStrength = uniform(0.02);
  private readonly uRate = uniform(0.4);
  private readonly uTarget = uniform(0.35);
  private readonly uMode = uniform(0);
  private readonly stampNode;
  private readonly copyNode;

  constructor(main: StorageTexture, scratch: StorageTexture, w: number, h: number) {
    const W = w;
    const H = h;
    const wrapX = (x: any) => x.add(int(W)).mod(int(W));
    const clampY = (y: any) => y.toFloat().max(float(0)).min(float(H - 1)).toInt();
    const at = (x: any, y: any) => textureLoad(main, ivec2(wrapX(x), clampY(y))).x;

    const fn = Fn(() => {
      const x = instanceIndex.mod(uint(W));
      const y = instanceIndex.div(uint(W));
      const ix = int(x);
      const iy = int(y);
      const u = x.toFloat().div(W);
      const v = y.toFloat().div(H - 1);
      const dir = lonLatDirNode(u, v);
      const dist = length(dir.sub(vec3(this.uCenter)));
      const wgt = float(1).sub(smoothstep(float(0), this.uRadius, dist));

      const cur = textureLoad(main, ivec2(ix, iy)).x;
      const avg = at(ix.sub(1), iy)
        .add(at(ix.add(1), iy))
        .add(at(ix, iy.sub(1)))
        .add(at(ix, iy.add(1)))
        .mul(0.25);

      const raise = cur.add(this.uStrength.mul(wgt));
      const lower = cur.sub(this.uStrength.mul(wgt));
      const smooth = mix(cur, avg, this.uRate.mul(wgt));
      const flatten = mix(cur, this.uTarget, this.uRate.mul(wgt));

      let out: any = raise;
      out = this.uMode.equal(float(1)).select(lower, out);
      out = this.uMode.equal(float(2)).select(smooth, out);
      out = this.uMode.equal(float(3)).select(flatten, out);
      out = max(float(0), min(float(1), out));
      textureStore(scratch, uvec2(x, y), vec4(out, 0, 0, 1)).toWriteOnly();
    });
    this.stampNode = fn().compute(W * H);
    this.copyNode = buildGridCopy(scratch, main, W, H);
  }

  stamp(
    renderer: WebGPURenderer,
    centerDir: Vector3,
    s: { mode: BrushMode; radius: number; strength: number; rate: number; target: number },
  ): void {
    this.uCenter.value.copy(centerDir).normalize();
    this.uRadius.value = s.radius;
    this.uStrength.value = s.strength;
    this.uRate.value = s.rate;
    this.uTarget.value = s.target;
    this.uMode.value = MODE[s.mode];
    renderer.compute(this.stampNode);
    renderer.compute(this.copyNode);
  }
}
