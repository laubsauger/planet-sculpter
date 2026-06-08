// Flat terrain brush: uv-space soft stamp (raise/lower/smooth/flatten). Reads main
// -> writes scratch -> copies back. Clamp-both neighbor reads (no wrap).

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
import { Vector2 } from 'three';
import {
  Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int,
  float, vec2, vec4, length, smoothstep, max, min, mix, uniform,
} from 'three/tsl';
import { buildGridCopy } from '../sim/gridStore';
import type { BrushMode } from '../tools/BrushTool';

/* eslint-disable @typescript-eslint/no-explicit-any */
const MODE: Record<BrushMode, number> = { raise: 0, lower: 1, smooth: 2, flatten: 3 };

export class FlatBrush {
  private readonly uCenter = uniform(new Vector2(0.5, 0.5));
  private readonly uRadius = uniform(0.06);
  private readonly uStrength = uniform(0.02);
  private readonly uRate = uniform(0.4);
  private readonly uTarget = uniform(0.4);
  private readonly uMode = uniform(0);
  private readonly stampNode;
  private readonly copyNode;

  constructor(main: StorageTexture, scratch: StorageTexture, w: number, h: number) {
    const cx = (x: any) => x.max(float(0)).min(float(w - 1)).toInt();
    const cy = (y: any) => y.max(float(0)).min(float(h - 1)).toInt();
    const at = (x: any, y: any) => textureLoad(main, ivec2(cx(x), cy(y))).x;
    const fn = Fn(() => {
      const x = instanceIndex.mod(uint(w)), y = instanceIndex.div(uint(w));
      const ix = int(x), iy = int(y);
      const u = x.toFloat().div(w), v = y.toFloat().div(h);
      const dist = length(vec2(u.sub(this.uCenter.x), v.sub(this.uCenter.y)));
      const wgt = float(1).sub(smoothstep(float(0), this.uRadius, dist));
      const cur = textureLoad(main, ivec2(ix, iy)).x;
      const avg = at(ix.sub(1), iy).add(at(ix.add(1), iy)).add(at(ix, iy.sub(1))).add(at(ix, iy.add(1))).mul(0.25);
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
    this.stampNode = fn().compute(w * h);
    this.copyNode = buildGridCopy(scratch, main, w, h);
  }

  stamp(renderer: WebGPURenderer, u: number, v: number, s: { mode: BrushMode; radius: number; strength: number; rate: number; target: number }): void {
    this.uCenter.value.set(u, v);
    this.uRadius.value = s.radius; this.uStrength.value = s.strength;
    this.uRate.value = s.rate; this.uTarget.value = s.target; this.uMode.value = MODE[s.mode];
    renderer.compute(this.stampNode);
    renderer.compute(this.copyNode);
  }
}
