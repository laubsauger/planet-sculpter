// Water surface material (M4). Analytic seam-aware displaced surface at (b + d)
// with smooth cross-face normals. Depth-tinted, hidden where dry.

import { MeshStandardNodeMaterial } from 'three/webgpu';
import { texture as sampleTex, textureLoad, uv, mix, smoothstep, vec3 } from 'three/tsl';
import { computeSurface, type SampleFace } from '../tsl/surface';
import type { HeightFields, FieldSet } from '../sim/fields';
import type { SeamTable } from '../planet/seamTable';
import type { FaceName } from '../config';

const SHALLOW = vec3(0.30, 0.62, 0.78);
const DEEP = vec3(0.05, 0.20, 0.42);
/** below this water depth, fully transparent. */
const MIN_DEPTH = 0.004;

export function makeWaterMaterial(
  face: FaceName,
  height: HeightFields,
  water: FieldSet,
  table: SeamTable,
): MeshStandardNodeMaterial {
  // surface height = bedrock + water column (both seam-aware, texel-exact).
  const sample: SampleFace = (f, coord) =>
    textureLoad(height.field(f).main, coord).x.add(textureLoad(water.field(f).main, coord).x);
  const s = computeSurface(face, sample, table);
  const d = sampleTex(water.field(face).main, uv()).x;

  const col = mix(SHALLOW, DEEP, smoothstep(0.0, 0.15, d));
  const opacity = smoothstep(MIN_DEPTH, MIN_DEPTH * 3, d).mul(0.85);

  const mat = new MeshStandardNodeMaterial({ roughness: 0.5, metalness: 0, transparent: true });
  mat.positionNode = s.position;
  mat.normalNode = s.viewNormal;
  mat.colorNode = col;
  mat.opacityNode = opacity;
  mat.depthWrite = false;
  return mat;
}
