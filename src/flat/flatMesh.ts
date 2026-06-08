// Flat render mesh: a plane with W×H verts and uv 0..1. The vertex POSITION is
// fully overridden by the material's positionNode (flatSurface), so the plane's
// own size/orientation are irrelevant — only the uv grid + vertex count matter.

import { Mesh, PlaneGeometry } from 'three';
import type { Material } from 'three';

export function buildFlatMesh(w: number, h: number, material: Material): Mesh {
  const geo = new PlaneGeometry(1, 1, w - 1, h - 1);
  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false;
  return mesh;
}
