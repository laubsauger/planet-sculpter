// Mouse ray -> planet surface -> (face, texelX, texelY) (T9).
// Raycast against analytic base-radius sphere (displacement << radius, brush has
// soft falloff). Ray transformed into planet-local space so it works if the
// planet group is ever rotated. No GPU readback (V: hot loop GPU-resident).

import {
  Raycaster,
  Vector2,
  Vector3,
  Sphere,
  Matrix4,
  type Camera,
  type Object3D,
} from 'three';
import { dirToFaceUV } from '../tsl/warp';
import { PLANET, type FaceName } from '../config';

export interface PickResult {
  face: FaceName;
  /** texel coords in [0, res], integer. */
  x: number;
  y: number;
  /** planet-local unit direction of the hit (for direction-space brush). */
  dir: Vector3;
}

const raycaster = new Raycaster();
const sphere = new Sphere(new Vector3(0, 0, 0), PLANET.baseRadius);
const hitWorld = new Vector3();
const hitLocal = new Vector3();
const invMatrix = new Matrix4();

/** ndc in [-1,1]. Returns null if ray misses the planet. */
export function pickPlanet(
  ndc: Vector2,
  camera: Camera,
  planet: Object3D,
  res: number,
): PickResult | null {
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectSphere(sphere, hitWorld)) return null;

  // World hit -> planet-local direction.
  invMatrix.copy(planet.matrixWorld).invert();
  hitLocal.copy(hitWorld).applyMatrix4(invMatrix).normalize();

  const { face, u, v } = dirToFaceUV([hitLocal.x, hitLocal.y, hitLocal.z]);
  const x = Math.round(((u + 1) / 2) * res);
  const y = Math.round(((v + 1) / 2) * res);
  return {
    face,
    x: Math.min(res, Math.max(0, x)),
    y: Math.min(res, Math.max(0, y)),
    dir: hitLocal.clone(),
  };
}
