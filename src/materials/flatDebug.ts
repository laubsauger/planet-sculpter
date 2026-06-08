import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, vec2, vec3, float, length, min, mix, normalize, uniform,
} from 'three/tsl';
import { flatSurface, bilinear, flatGridX, flatGridY } from '../tsl/flatSurface';

export const FLAT_DEBUG_MODES = [
  'off',
  'height',
  'waterDepth',
  'flowSpeed',
  'flowDir',
  'sediment',
  'erosion',
  'deposition',
  'mobileEarth',
  'source',
  'flux',
  'activeCells',
] as const;

export const flatDebugMode = uniform(1);

/* eslint-disable @typescript-eslint/no-explicit-any */
export function makeFlatDebug(
  heightTex: Texture,
  waterTex: Texture,
  velocityTex: Texture,
  sedimentTex: Texture,
  looseTex: Texture,
  sourceTex: Texture,
  fluxTex: Texture,
  activityTex: Texture,
): MeshBasicNodeMaterial {
  const fx = uv().x.mul(flatGridX), fy = uv().y.mul(flatGridY);
  const scalar = (tex: Texture) => bilinear((c: any) => textureLoad(tex, c).x, fx, fy);
  const vector = (tex: Texture) => bilinear((c: any) => textureLoad(tex, c), fx, fy);
  const ramp = (t: any) => mix(vec3(0.03, 0.08, 0.3), vec3(1, 0.82, 0.12), min(t, float(1)));

  const surface = flatSurface((c: any) => textureLoad(heightTex, c).x, false);
  const vel = vector(velocityTex).xy;
  const activity = vector(activityTex);
  const flux = vector(fluxTex);
  const fluxMagnitude = flux.x.add(flux.y).add(flux.z).add(flux.w);
  const active = scalar(waterTex).greaterThan(float(0.0008)).or(scalar(sedimentTex).greaterThan(float(1e-5)));
  const direction = vec3(normalize(vel.add(vec2(1e-6, 0))).mul(0.5).add(0.5), float(0));
  const mode = flatDebugMode;

  let col: any = vec3(0);
  col = mode.equal(float(1)).select(ramp(surface.height), col);
  col = mode.equal(float(2)).select(ramp(scalar(waterTex).mul(8)), col);
  col = mode.equal(float(3)).select(ramp(length(vel).mul(0.7)), col);
  col = mode.equal(float(4)).select(direction, col);
  col = mode.equal(float(5)).select(ramp(scalar(sedimentTex).mul(15)), col);
  col = mode.equal(float(6)).select(ramp(activity.x), col);
  col = mode.equal(float(7)).select(ramp(activity.y), col);
  col = mode.equal(float(8)).select(ramp(scalar(looseTex).mul(35)), col);
  col = mode.equal(float(9)).select(ramp(scalar(sourceTex).mul(25)), col);
  col = mode.equal(float(10)).select(ramp(fluxMagnitude.mul(0.2)), col);
  col = mode.equal(float(11)).select(active.select(vec3(0.1, 0.9, 0.2), vec3(0.08, 0.08, 0.1)), col);

  const mat = new MeshBasicNodeMaterial({ side: DoubleSide });
  mat.positionNode = surface.position;
  mat.colorNode = col;
  return mat;
}
