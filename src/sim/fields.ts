// GPU field storage (T7). Per-face r32float StorageTexture with ping-pong
// (V2: neighbor-dependent passes read `in` / write `out` then swap).
// r32float is NOT filterable in WebGPU -> NearestFilter (grid ~1:1 with verts,
// edge texels clamp -> adjacent faces agree -> crack-free, V5 geom).

import {
  DataTexture,
  RedFormat,
  RGBAFormat,
  FloatType,
  NearestFilter,
  ClampToEdgeWrapping,
} from 'three';
import { StorageTexture } from 'three/webgpu';
import {
  Fn,
  instanceIndex,
  textureLoad,
  textureStore,
  ivec2,
  uvec2,
  uint,
  vec4,
} from 'three/tsl';
import { FACES, type FaceName } from '../config';

function makeStorage(n: number, rgba = false): StorageTexture {
  const t = new StorageTexture(n, n);
  t.format = rgba ? RGBAFormat : RedFormat;
  t.type = FloatType;
  t.magFilter = NearestFilter;
  t.minFilter = NearestFilter;
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  return t;
}

/**
 * One scalar field on one face. `main` is canonical (always sampled by the
 * material; fixed binding). `scratch` is transient. Every compute op reads
 * `main` -> writes `scratch`, then copies `scratch` -> `main` (V2: never
 * read+write the same storage texture in one pass). This keeps all texture
 * refs stable so seam passes (which read neighbor faces) and the material
 * never need rebinding.
 */
export class FaceField {
  readonly main: StorageTexture;
  readonly scratch: StorageTexture;

  constructor(readonly n: number, rgba = false) {
    this.main = makeStorage(n, rgba);
    this.scratch = makeStorage(n, rgba);
  }

  dispose(): void {
    this.main.dispose();
    this.scratch.dispose();
  }
}

/** A per-face set of one field (scalar or rgba). */
export class FieldSet {
  readonly fields = new Map<FaceName, FaceField>();
  constructor(
    readonly n: number,
    rgba = false,
  ) {
    for (const face of FACES) this.fields.set(face, new FaceField(n, rgba));
  }
  field(face: FaceName): FaceField {
    const f = this.fields.get(face);
    if (!f) throw new Error(`no field for face ${face}`);
    return f;
  }
}

/** Per-face height fields. */
export class HeightFields {
  readonly n: number;
  readonly fields = new Map<FaceName, FaceField>();

  constructor(res: number) {
    this.n = res + 1;
    for (const face of FACES) this.fields.set(face, new FaceField(this.n));
  }

  field(face: FaceName): FaceField {
    const f = this.fields.get(face);
    if (!f) throw new Error(`no field for face ${face}`);
    return f;
  }
}

/** Generic copy compute: src -> dst (full texture). */
export function buildCopyCompute(src: StorageTexture, dst: StorageTexture, n: number) {
  const N = uint(n);
  const fn = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const v = textureLoad(src, ivec2(x.toInt(), y.toInt()));
    textureStore(dst, uvec2(x, y), v).toWriteOnly();
  });
  return fn().compute(n * n);
}

/**
 * Compute node that copies a CPU seed DataTexture (RedFormat float) into a
 * storage texture. Built once per (seed,target); first compute pass of the app.
 */
export function buildSeedCompute(seedTex: DataTexture, target: StorageTexture, n: number) {
  const N = uint(n);
  const seed = Fn(() => {
    const x = instanceIndex.mod(N);
    const y = instanceIndex.div(N);
    const h = textureLoad(seedTex, ivec2(x.toInt(), y.toInt())).x;
    textureStore(target, uvec2(x, y), vec4(h, 0, 0, 1)).toWriteOnly();
  });
  return seed().compute(n * n);
}
