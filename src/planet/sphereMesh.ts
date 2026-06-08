// Equirectangular UV-sphere geometry — ONE grid mesh for the whole planet
// (replaces the 6 cube-sphere face meshes). (lonSegs+1) x (latSegs+1) verts:
//   - u (longitude) column 0 and column lonSegs share the same meridian -> the
//     wrap seam closes exactly (duplicated coincident verts, no crack).
//   - v=0 / v=latSegs rows collapse to the poles (degenerate sliver tris there;
//     hidden by ice caps).
// Height displacement applied later in the vertex node; base normals radial.

import { BufferGeometry, BufferAttribute, Mesh, Group, type Material } from 'three';
import { lonLatToDir } from './latlong';

export interface SphereGeo {
  geometry: BufferGeometry;
  lonSegs: number;
  latSegs: number;
}

export function buildSphereGeometry(lonSegs: number, latSegs: number, baseRadius: number): SphereGeo {
  const nu = lonSegs + 1;
  const nv = latSegs + 1;
  const count = nu * nv;
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);

  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const idx = j * nu + i;
      const u = i / lonSegs;
      const v = j / latSegs;
      const dir = lonLatToDir(u, v);
      positions[idx * 3] = dir[0] * baseRadius;
      positions[idx * 3 + 1] = dir[1] * baseRadius;
      positions[idx * 3 + 2] = dir[2] * baseRadius;
      normals[idx * 3] = dir[0];
      normals[idx * 3 + 1] = dir[1];
      normals[idx * 3 + 2] = dir[2];
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  const indices = new Uint32Array(lonSegs * latSegs * 6);
  let k = 0;
  for (let j = 0; j < latSegs; j++) {
    for (let i = 0; i < lonSegs; i++) {
      const a = j * nu + i;
      const b = a + 1;
      const c = a + nu;
      const d = c + 1;
      // CCW-from-OUTSIDE for the lat/long parametrization (east×north points
      // inward here, opposite the cube faces) -> reversed winding so front faces
      // face the camera (else the planet renders see-through to its far backfaces).
      indices[k++] = a; indices[k++] = d; indices[k++] = b;
      indices[k++] = a; indices[k++] = c; indices[k++] = d;
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new BufferAttribute(uvs, 2));
  geometry.setIndex(new BufferAttribute(indices, 1));
  return { geometry, lonSegs, latSegs };
}

/** Single-mesh planet (one grid). Mirrors PlanetMesh's group/material API so the
 *  Engine can swap material (terrain/water/lava/debug) the same way. */
export class SphereMesh {
  readonly group = new Group();
  readonly mesh: Mesh;
  readonly geo: SphereGeo;

  constructor(lonSegs: number, latSegs: number, baseRadius: number, material: Material) {
    this.geo = buildSphereGeometry(lonSegs, latSegs, baseRadius);
    this.mesh = new Mesh(this.geo.geometry, material);
    this.group.add(this.mesh);
  }

  setMaterial(material: Material): void {
    this.mesh.material = material;
  }

  dispose(): void {
    this.geo.geometry.dispose();
    const m = this.mesh.material;
    if (Array.isArray(m)) m.forEach((x) => x.dispose());
    else m.dispose();
  }
}
