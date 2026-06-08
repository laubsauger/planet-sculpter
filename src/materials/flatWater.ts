// Flat water, WORLD-space lit (camera-independent base, like the terrain). The
// turquoise depth color is ALWAYS visible from any angle; sun glint + fresnel sky
// reflection are additive bonuses, not the only thing (so it never goes dark/blank
// when you rotate). Animated wave + flow normals (calm, not noise-soup). Surface =
// bedrock + depth; shallow = see-through turquoise, deep = blue; shoreline foam.

import { DoubleSide, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  textureLoad, uv, mix, smoothstep, max, clamp, float, vec2, vec3, length,
  normalize, dot, pow, sin, time, cameraPosition, mx_fractal_noise_float, uniform,
} from 'three/tsl';
import { flatSurface, bilinear, flatGridX, flatGridY } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU } from '../tsl/lighting';

/* eslint-disable @typescript-eslint/no-explicit-any */

const SHALLOW = vec3(0.36, 0.78, 0.76);
const MID = vec3(0.08, 0.46, 0.62);
const DEEP = vec3(0.02, 0.15, 0.34);
const SKY_REFLECT = vec3(0.6, 0.78, 0.95);
const FOAM = vec3(0.95, 0.98, 1.0);
const SILT = vec3(0.76, 0.68, 0.48);
export const flowBandStrength = uniform(0.52);
export const flowBandScale = uniform(9);

export function makeFlatWater(heightTex: Texture, waterTex: Texture, velTex: Texture, sedimentTex: Texture): MeshBasicNodeMaterial {
  const fx = uv().x.mul(flatGridX), fy = uv().y.mul(flatGridY);

  const s = flatSurface((c: any) => textureLoad(heightTex, c).x.add(textureLoad(waterTex, c).x), false, true);
  const depth = bilinear((c: any) => textureLoad(waterTex, c).x, fx, fy);
  const vel = bilinear((c: any) => textureLoad(velTex, c).xy, fx, fy);
  const sediment = bilinear((c: any) => textureLoad(sedimentTex, c).x, fx, fy);
  const flow = vec2(vel.x, vel.y);
  const speed = length(flow);
  const flowDir = normalize(flow.add(vec2(1e-5, 0)));

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
  const concentration = sediment.div(max(depth, float(0.003)));
  col = mix(col, SILT, smoothstep(0.04, 0.45, concentration).mul(0.78));

  // Broad flow-aligned bands remain readable from overhead and in flat light.
  // Variation is mostly across the flow, with a moving along-flow breakup.
  const acrossDir = vec2(flowDir.y.mul(-1), flowDir.x);
  const along = dot(posXZ, flowDir);
  const across = dot(posXZ, acrossDir);
  const travel = time.mul(speed.mul(2.4).add(0.8));
  const bend = sin(along.mul(2.1).sub(travel.mul(0.55))).mul(1.35)
    .add(sin(along.mul(5.7).add(across.mul(0.8)).sub(travel.mul(0.2))).mul(0.42));
  const spacing = across.mul(flowBandScale).add(sin(along.mul(0.72)).mul(1.8));
  const ridges = sin(spacing.add(bend).sub(travel)).mul(0.5).add(0.5);
  const breakup = sin(along.mul(11).sub(travel.mul(2.2))).mul(0.5).add(0.5);
  const streak = smoothstep(0.64, 0.94, ridges).mul(breakup.mul(0.45).add(0.55));
  const movingWater = smoothstep(0.015, 0.55, speed);
  const streakStrength = streak.mul(movingWater).mul(smoothstep(0.00006, 0.05, depth));
  col = mix(col, FOAM, streakStrength.mul(flowBandStrength));

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
  const shore = float(1).sub(smoothstep(0.0005, 0.025, depth));
  const rapids = smoothstep(0.08, 0.8, speed).mul(smoothstep(0.0004, 0.035, depth));
  col = mix(col, FOAM, max(shore.mul(lap.mul(0.6).add(0.2)), rapids.mul(0.72).add(streakStrength.mul(0.42))).min(float(1)));

  const depthOpacity = smoothstep(0.00008, 0.0012, depth).mul(0.62)
    .add(smoothstep(0.0012, 0.05, depth).mul(0.35));
  // Velocity-driven floor: shallow-but-fast erosive flow stays visible instead of
  // rounding to zero. Gated by a tiny depth threshold so dry cells (depth 0) with a
  // stale velocity field don't paint phantom water.
  const flowOpacity = smoothstep(0.02, 0.4, speed).mul(0.42)
    .mul(smoothstep(0.00003, 0.0006, depth));
  const opacity = clamp(max(depthOpacity, flowOpacity), float(0), float(0.97));

  const mat = new MeshBasicNodeMaterial({ transparent: true, side: DoubleSide });
  const visualLift = smoothstep(0.00008, 0.004, depth).mul(0.008);
  mat.positionNode = vec3(s.position.x, s.position.y.add(visualLift), s.position.z);
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
