// Flat render mesh: a plane with W×H verts and uv 0..1. The vertex POSITION is
// fully overridden by the material's positionNode (flatSurface), so the plane's
// own size/orientation are irrelevant — only the uv grid + vertex count matter.

import { Mesh, PlaneGeometry } from 'three';
import type { Material } from 'three';

export function buildFlatMesh(w: number, h: number, material: Material): Mesh {
  const geo = new PlaneGeometry(1, 1, w - 1, h - 1);
  // `flatSurface` remaps UVs to XZ. PlaneGeometry's original +Z-facing winding
  // becomes -Y-facing after that remap, so DoubleSide PBR flips our upward normal
  // when viewed from above. Reverse every triangle once so the displaced surface
  // is genuinely upward-facing and lighting cannot change with camera side.
  const index = geo.index;
  if (!index) throw new Error('Flat mesh requires indexed PlaneGeometry');
  for (let i = 0; i < index.count; i += 3) {
    const b = index.getX(i + 1);
    index.setX(i + 1, index.getX(i + 2));
    index.setX(i + 2, b);
  }
  index.needsUpdate = true;
  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false;
  return mesh;
}
