// Thin wrapper over three's OrbitControls for planet view (V10: input every frame).
import { Camera } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLANET } from '../config';

export class OrbitController {
  readonly controls: OrbitControls;

  constructor(camera: Camera, dom: HTMLElement) {
    this.controls = new OrbitControls(camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.6;
    this.controls.minDistance = PLANET.baseRadius * 1.2;
    this.controls.maxDistance = PLANET.baseRadius * 8;
    this.controls.enablePan = false;
  }

  update(): void {
    this.controls.update();
  }
}
