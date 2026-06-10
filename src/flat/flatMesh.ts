// Flat render mesh: a plane with W×H verts and uv 0..1. The vertex POSITION is
// fully overridden by the material's positionNode (flatSurface), so the plane's
// own size/orientation are irrelevant — only the uv grid + vertex count matter.

import { Mesh, PlaneGeometry, BufferGeometry, Float32BufferAttribute } from 'three';
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

/** Ocean skirt: a 4-quad FRAME around the grid hole [-size/2, size/2]², reaching out to
 *  `extent` world units. Uses the SAME water material — its positionNode maps uv -> world XZ
 *  and the texture samplers CLAMP, so uv beyond [0,1] reproduces the deep border ocean at the
 *  sea surface. Gives a dynamic (swell/colour) ocean continuing seamlessly to the horizon,
 *  with NO separate static plane and NO seabed occlusion. Quads are wound upward so the
 *  water material can cull its invisible underside. */
export function buildOceanSkirt(material: Material, size: number, extent: number): Mesh {
  const half = size * 0.5;
  const toUv = (c: number) => c / size + 0.5; // inverse of position = (uv-0.5)*size
  const quads: number[][][] = [
    [[-extent, extent], [extent, extent], [extent, half], [-extent, half]],   // north
    [[-extent, -half], [extent, -half], [extent, -extent], [-extent, -extent]], // south
    [[half, half], [extent, half], [extent, -half], [half, -half]],            // east
    [[-extent, half], [-half, half], [-half, -half], [-extent, -half]],        // west
  ];
  const positions: number[] = [], uvs: number[] = [], indices: number[] = [];
  let base = 0;
  for (const q of quads) {
    for (const [x, z] of q) { positions.push(x, 0, z); uvs.push(toUv(x), toUv(z)); }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    base += 4;
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1; // behind the full-res grid water
  return mesh;
}
