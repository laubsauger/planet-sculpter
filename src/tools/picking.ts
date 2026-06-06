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

/** ndc in [-1,1]. Returns null if ray misses the planet.
 *  If `heightAt` (dir->[0,1] height) + `heightScale` are given, the hit is
 *  refined onto the DISPLACED surface (baseR + h*scale) by re-intersecting a
 *  few times — without it the pick lands on the base sphere and a click on a
 *  mountain/valley appears offset. */
export function pickPlanet(
  ndc: Vector2,
  camera: Camera,
  planet: Object3D,
  res: number,
  heightAt?: (dir: Vector3) => number,
  heightScale = 0,
): PickResult | null {
  raycaster.setFromCamera(ndc, camera);
  sphere.radius = PLANET.baseRadius;
  if (!raycaster.ray.intersectSphere(sphere, hitWorld)) return null;

  invMatrix.copy(planet.matrixWorld).invert();
  hitLocal.copy(hitWorld).applyMatrix4(invMatrix).normalize();

  // refine onto the displaced surface (converges for the gently-displaced sphere).
  if (heightAt) {
    for (let k = 0; k < 3; k++) {
      sphere.radius = PLANET.baseRadius + heightAt(hitLocal) * heightScale;
      if (!raycaster.ray.intersectSphere(sphere, hitWorld)) break;
      hitLocal.copy(hitWorld).applyMatrix4(invMatrix).normalize();
    }
  }

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
