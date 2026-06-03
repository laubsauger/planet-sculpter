// Brush GPU stamp (T10). Direction-space: brush center is a unit sphere
// direction; every face's texels compute their own direction (TSL warp, V1)
// and distance to the center, so a stroke paints seamlessly ACROSS face
// boundaries (shared-edge texels get identical falloff -> no seam crease).
// Modes: 0 raise/lower (signed strength), 1 flatten (toward target), 2 smooth
// (neighbor avg). Reads canonical `main`, writes `scratch`, copies back (V2).
// Per-face stamp + copy nodes precompiled once, reused (V9).

import type { WebGPURenderer, StorageTexture } from 'three/webgpu';
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
import { Vector3 } from 'three';
import { type FaceField, buildCopyCompute } from '../sim/fields';
import { faceDirNode } from '../tsl/warpNode';
import type { FaceName } from '../config';
// PickResult no longer needed here — brush is direction-space.

export type BrushMode = 'raise' | 'lower' | 'smooth' | 'flatten';
const MODE_CODE: Record<BrushMode, number> = { raise: 0, lower: 0, smooth: 2, flatten: 1 };

export interface BrushParams {
  mode: BrushMode;
  /** brush radius as a chord length on the unit sphere. */
  radius: number;
  /** per-stamp height delta (raise/lower) in height units. */
  strength: number;
  /** flatten/smooth blend rate 0..1. */
  rate: number;
  /** flatten target height. */
  target: number;
}

// Compute-node type taken from the renderer.compute() parameter (the Fn().compute()
// chain isn't cleanly typed through to here).
type ComputeNode = Parameters<WebGPURenderer['compute']>[0];

interface FacePrograms {
  stamp: ComputeNode; // brush: main -> scratch
  copy: ComputeNode; // scratch -> main
}

export class BrushTool {
  // Shared uniforms (one set, mutated per stamp).
  private uCenterDir = uniform(new Vector3(1, 0, 0));
  private uRadius = uniform(0.15);
  private uStrength = uniform(0);
  private uMode = uniform(0);
  private uTarget = uniform(0);
  private uRate = uniform(0.5);

  private readonly programs = new Map<FaceName, FacePrograms>();

  constructor(private readonly n: number) {}

  /** Build stamp (main->scratch) + copy (scratch->main) for a face. */
  register(face: FaceName, field: FaceField): void {
    this.programs.set(face, {
      stamp: this.makeProgram(face, field.main, field.scratch),
      copy: buildCopyCompute(field.scratch, field.main, this.n) as ComputeNode,
    });
  }

  private makeProgram(face: FaceName, readTex: StorageTexture, writeTex: StorageTexture): ComputeNode {
    const N = this.n;
    const res = this.n - 1;
    const fn = Fn(() => {
      const x = instanceIndex.mod(uint(N));
      const y = instanceIndex.div(uint(N));
      const ix = int(x);
      const iy = int(y);
      const last = float(N - 1);

      const cur = textureLoad(readTex, ivec2(ix, iy)).x;

      // texel -> (u,v) in [-1,1] -> sphere direction (V1) -> chord dist to brush.
      const fx = x.toFloat();
      const fy = y.toFloat();
      const su = fx.div(res).mul(2).sub(1);
      const sv = fy.div(res).mul(2).sub(1);
      const dir = faceDirNode(face, su, sv);
      const dist = length(dir.sub(vec3(this.uCenterDir)));
      const w = float(1).sub(smoothstep(float(0), this.uRadius, dist));

      // neighbor avg (edge-clamped) for smooth. Clamp in float, cast to int
      // (int nodes lack max/min ergonomics in the typed TSL surface).
      const xm = fx.sub(1).max(float(0)).min(last).toInt();
      const xp = fx.add(1).max(float(0)).min(last).toInt();
      const ym = fy.sub(1).max(float(0)).min(last).toInt();
      const yp = fy.add(1).max(float(0)).min(last).toInt();
      const l = textureLoad(readTex, ivec2(xm, iy)).x;
      const r = textureLoad(readTex, ivec2(xp, iy)).x;
      const d = textureLoad(readTex, ivec2(ix, ym)).x;
      const u = textureLoad(readTex, ivec2(ix, yp)).x;
      const avg = l.add(r).add(d).add(u).mul(0.25);

      const deltaRaise = this.uStrength.mul(w);
      const deltaFlatten = this.uTarget.sub(cur).mul(this.uRate).mul(w);
      const deltaSmooth = avg.sub(cur).mul(this.uRate).mul(w);

      const delta = this.uMode
        .equal(1)
        .select(deltaFlatten, this.uMode.equal(2).select(deltaSmooth, deltaRaise));

      const newH = max(cur.add(delta), float(0));
      textureStore(writeTex, uvec2(x, y), vec4(newH, 0, 0, 1)).toWriteOnly();
    });
    return fn().compute(N * N) as ComputeNode;
  }

  /**
   * Dispatch one stamp centered at sphere direction `centerDir`. Runs on ALL
   * faces so a stroke crossing a seam paints continuously. Each face's compute
   * skips texels outside the radius via the falloff (w=0 -> no change).
   */
  stamp(renderer: WebGPURenderer, centerDir: Vector3, params: BrushParams): void {
    this.uCenterDir.value.copy(centerDir).normalize();
    this.uRadius.value = params.radius;
    this.uMode.value = MODE_CODE[params.mode];
    this.uTarget.value = params.target;
    this.uRate.value = params.rate;
    // lower = negative strength.
    this.uStrength.value = params.mode === 'lower' ? -params.strength : params.strength;

    for (const prog of this.programs.values()) {
      renderer.compute(prog.stamp); // main -> scratch
      renderer.compute(prog.copy); // scratch -> main
    }
  }

  /** Warm up all programs once (V8). */
  warmup(renderer: WebGPURenderer): void {
    this.uStrength.value = 0; // no-op stamp
    for (const prog of this.programs.values()) {
      renderer.compute(prog.stamp);
      renderer.compute(prog.copy);
    }
  }
}
