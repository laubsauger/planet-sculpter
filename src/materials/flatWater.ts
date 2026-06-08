// Flat water, WORLD-space lit (camera-independent base, like the terrain). The
// turquoise depth color is ALWAYS visible from any angle; sun glint + fresnel sky
// reflection are additive bonuses, not the only thing (so it never goes dark/blank
// when you rotate). Animated wave + flow normals (calm, not noise-soup). Surface =
// bedrock + depth; shallow = see-through turquoise, deep = blue; shoreline foam.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, ivec2, uv, mix, smoothstep, max, clamp, float, vec2, vec3, length,
  normalize, dot, pow, sin, time, cameraPosition, mx_fractal_noise_float,
} from 'three/tsl';
import { flatSurface, bilinear } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU } from '../tsl/lighting';
import { FLAT } from '../config';

/* eslint-disable @typescript-eslint/no-explicit-any */

const SHALLOW = vec3(0.36, 0.78, 0.76);
const MID = vec3(0.08, 0.46, 0.62);
const DEEP = vec3(0.02, 0.15, 0.34);
const SKY_REFLECT = vec3(0.6, 0.78, 0.95);
const FOAM = vec3(0.95, 0.98, 1.0);

export function makeFlatWater(heightTex: Texture, waterTex: Texture, velTex: Texture): MeshBasicNodeMaterial {
  const W = FLAT.gridW, H = FLAT.gridH;
  const fx = uv().x.mul(W), fy = uv().y.mul(H);
  const cx = (x: any) => x.max(float(0)).min(float(W - 1)).toInt();
  const cy = (y: any) => y.max(float(0)).min(float(H - 1)).toInt();
  const coord = ivec2(cx(fx.floor()), cy(fy.floor()));

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x.add(textureLoad(waterTex, c).x), false);
  const depth = bilinear((c: any) => textureLoad(waterTex, c).x, fx, fy);
  const vel = textureLoad(velTex, coord);
  const flow = vec2(vel.x, vel.y);
  const speed = length(flow);

  // calm wave + flow normals (subtle ripple, not noise soup).
  const posXZ = vec2(s.position.x, s.position.z);
  const e = float(0.06);
  const grad = (q: any, freq: number) => {
    const nz = (p: any) => mx_fractal_noise_float(vec3(p.x, p.y, 0).mul(freq), 2);
    return vec3(nz(q.add(vec2(e, 0))).sub(nz(q.sub(vec2(e, 0)))), float(0), nz(q.add(vec2(0, e))).sub(nz(q.sub(vec2(0, e)))));
  };
  const flowR = grad(posXZ.sub(flow.mul(time.mul(0.3))), 6).mul(speed.mul(0.4).add(0.04));
  const waveA = grad(posXZ.add(vec2(time.mul(0.1), time.mul(0.07))), 1.4).mul(0.1);
  const nW: any = normalize(s.worldNormal.add(flowR).add(waveA));

  // depth color (view-INDEPENDENT base, always visible).
  let col: any = mix(SHALLOW, MID, smoothstep(0.01, 0.09, depth));
  col = mix(col, DEEP, smoothstep(0.09, 0.32, depth));

  // world-space lighting: gentle sun diffuse keeps base bright + readable any angle.
  const viewW = normalize(cameraPosition.sub(s.position));
  const ndl = max(float(0), dot(nW, sunDirUniform));
  col = col.mul(ndl.mul(0.35).add(0.75));
  // fresnel sky reflection (rim, additive bonus).
  const fres = pow(float(1).sub(max(float(0), dot(nW, viewW))), float(4));
  col = mix(col, SKY_REFLECT, fres.mul(0.45));
  // sun specular glint (Blinn-Phong, additive sparkle).
  const half = normalize(sunDirUniform.add(viewW));
  const spec = pow(max(float(0), dot(nW, half)), float(80)).mul(sunIntensityU.mul(0.5));
  col = col.add(vec3(1.0, 0.97, 0.9).mul(spec));
  // shoreline lapping foam + rapids.
  const lap = sin(posXZ.x.add(posXZ.y).mul(6).sub(time.mul(2.2))).mul(0.5).add(0.5);
  const shore = float(1).sub(smoothstep(0.004, 0.045, depth));
  const rapids = smoothstep(0.6, 1.6, speed).mul(smoothstep(0.004, 0.02, depth));
  col = mix(col, FOAM, max(shore.mul(lap.mul(0.6).add(0.2)), rapids.mul(0.7)).min(float(1)));

  const opacity = clamp(smoothstep(0.0015, 0.02, depth).mul(0.65).add(smoothstep(0.02, 0.12, depth).mul(0.34)), float(0), float(0.97));

  const mat = new MeshBasicNodeMaterial({ transparent: true, side: DoubleSide });
  mat.positionNode = s.position;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
