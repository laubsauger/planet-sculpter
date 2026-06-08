// W×H storage-texture helpers for the equirect single grid (replaces the 6-face
// FieldSet). r32float / rgba32float, NearestFilter (sim reads exact texels;
// the material does its own bilinear). Longitude wraps in-shader via mod, so the
// texture itself just clamps.

import { DataTexture, RedFormat, RGBAFormat, FloatType, NearestFilter, ClampToEdgeWrapping } from 'three';
import { StorageTexture } from 'three/webgpu';
import { Fn, instanceIndex, textureLoad, textureStore, ivec2, uvec2, uint, int, vec4 } from 'three/tsl';

export function makeGridStorage(w: number, h: number, rgba = false): StorageTexture {
  const t = new StorageTexture(w, h);
  t.format = rgba ? RGBAFormat : RedFormat;
  t.type = FloatType;
  t.magFilter = NearestFilter;
  t.minFilter = NearestFilter;
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  return t;
}

/** A grid field: main (canonical, sampled by material) + scratch (transient). */
export class GridField {
  readonly main: StorageTexture;
  readonly scratch: StorageTexture;
  constructor(
    readonly w: number,
    readonly h: number,
    rgba = false,
  ) {
    this.main = makeGridStorage(w, h, rgba);
    this.scratch = makeGridStorage(w, h, rgba);
  }
}

type ComputeBuilder = ReturnType<typeof Fn>;

function gridXY(w: number) {
  const N = uint(w);
  const x = instanceIndex.mod(N);
  const y = instanceIndex.div(N);
  return { x, y, ix: int(x), iy: int(y) };
}

/** Copy a CPU seed DataTexture (.x) into a W×H storage texture. */
export function buildGridSeed(seed: DataTexture, target: StorageTexture, w: number, h: number) {
  const fn = Fn(() => {
    const { x, y, ix, iy } = gridXY(w);
    const v = textureLoad(seed, ivec2(ix, iy)).x;
    textureStore(target, uvec2(x, y), vec4(v, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h);
}

export function buildGridCopy(src: StorageTexture, dst: StorageTexture, w: number, h: number) {
  const fn = Fn(() => {
    const { x, y, ix, iy } = gridXY(w);
    textureStore(dst, uvec2(x, y), textureLoad(src, ivec2(ix, iy))).toWriteOnly();
  });
  return fn().compute(w * h);
}

export function buildGridFill(dst: StorageTexture, w: number, h: number, value = 0) {
  const fn = Fn(() => {
    const { x, y } = gridXY(w);
    textureStore(dst, uvec2(x, y), vec4(value, 0, 0, 1)).toWriteOnly();
  });
  return fn().compute(w * h);
}

export type { ComputeBuilder };
