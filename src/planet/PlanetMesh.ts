// Assemble 6 face meshes into one planet group.
// M0: flat-shaded solid node material. M1 swaps in terrainMaterial (T8).

import { Group, Mesh, Material } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { buildFaceGeometry, type FaceGeometry } from './cubeSphere';
import { FACES, type FaceName } from '../config';

export class PlanetMesh {
  readonly group = new Group();
  readonly faces = new Map<FaceName, { mesh: Mesh; geo: FaceGeometry }>();

  constructor(res: number, baseRadius: number, material: Material) {
    for (const face of FACES) {
      const geo = buildFaceGeometry(face, res, baseRadius);
      const mesh = new Mesh(geo.geometry, material);
      mesh.name = `face-${face}`;
      this.group.add(mesh);
      this.faces.set(face, { mesh, geo });
    }
  }

  /** Replace material on all faces (M0 -> M1 terrain swap). */
  setMaterial(material: Material): void {
    for (const { mesh } of this.faces.values()) mesh.material = material;
  }

  /** Per-face material (terrain binds each face's own height texture). */
  setFaceMaterial(face: FaceName, material: Material): void {
    const entry = this.faces.get(face);
    if (!entry) throw new Error(`unknown face ${face}`);
    entry.mesh.material = material;
  }

  dispose(): void {
    for (const { mesh, geo } of this.faces.values()) {
      geo.geometry.dispose();
      const m = mesh.material;
      if (Array.isArray(m)) m.forEach((x) => x.dispose());
      else m.dispose();
    }
  }
}

/** M0 placeholder material: solid color, faceted (flatShading). */
export function makeFlatSolidMaterial(): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial({ color: 0x6b8f3a, roughness: 0.95, metalness: 0 });
  mat.flatShading = true;
  return mat;
}
