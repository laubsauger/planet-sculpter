// Terrain node material (T8): vertex displacement from height texture,
// faceted flat normals via screen-space derivatives (V11), biome color graph.
// One material per face (each binds its own height texture).

import { Texture } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  texture as sampleTex,
  uv,
  positionGeometry,
  normalGeometry,
  positionWorld,
  normalFlat,
  dFdx,
  dFdy,
  cross,
  normalize,
  dot,
  abs,
  mix,
  smoothstep,
  vec3,
  float,
  uniform,
} from 'three/tsl';
import { PLANET } from '../config';

// Shared uniforms (one set, reused across all face materials).
export const heightScaleUniform = uniform(PLANET.heightScale);

const SAND = vec3(0.76, 0.70, 0.50);
const GRASS = vec3(0.28, 0.45, 0.20);
const ROCK = vec3(0.40, 0.37, 0.33);
const SNOW = vec3(0.92, 0.94, 0.97);

export interface TerrainMaterial {
  material: MeshStandardNodeMaterial;
  /** Repoint the sampled height texture after a ping-pong swap. */
  setHeightTexture(tex: Texture): void;
}

export function makeTerrainMaterial(heightTex: Texture): TerrainMaterial {
  const heightNode = sampleTex(heightTex, uv());
  const h = heightNode.x;

  // Displace base sphere position along its radial normal.
  const displaced = positionGeometry.add(normalGeometry.mul(h.mul(heightScaleUniform)));

  // Slope (scalar, camera-independent): magnitude of alignment between the
  // world-space facet normal and the radial up. abs() removes derivative
  // sign-flips. Lighting normal itself uses the built-in VIEW-space normalFlat.
  const pw = positionWorld;
  const up = normalize(pw);
  const worldFacetN = normalize(cross(dFdx(pw), dFdy(pw)));
  const slope = float(1).sub(abs(dot(worldFacetN, up)));

  let col = mix(SAND, GRASS, smoothstep(0.10, 0.18, h));
  col = mix(col, ROCK, smoothstep(0.42, 0.58, h));
  col = mix(col, ROCK, smoothstep(0.55, 0.85, slope));
  col = mix(col, SNOW, smoothstep(0.72, 0.82, h));

  const mat = new MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
  mat.positionNode = displaced;
  mat.normalNode = normalFlat; // view-space faceted normal (correct lighting space)
  mat.colorNode = col;
  return {
    material: mat,
    setHeightTexture(tex: Texture) {
      heightNode.value = tex;
    },
  };
}
