// Flat From-Dust-style engine. Crisp detail-normal terrain + pipe-model water +
// erosion on a uniform heightfield (no sphere/pole/seam machinery). Sculpt brush,
// river springs, rain, realistic water with flow normals.

import {
  Scene, PerspectiveCamera, DirectionalLight, HemisphereLight, Color, Fog,
  Vector2, Vector3, Raycaster, Plane,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
import GUI from 'lil-gui';
import { OrbitController } from '../app/OrbitController';
import { buildFlatSeed } from './flatSeed';
import { buildFlatMesh } from './flatMesh';
import { makeFlatTerrain } from '../materials/flatTerrain';
import { makeFlatWater } from '../materials/flatWater';
import { FlatBrush } from './FlatBrush';
import { FlatSim } from './flatSim';
import { Sidebar } from '../ui/Sidebar';
import { GridField, buildGridSeed } from '../sim/gridStore';
import { flatHeightScale, flatSeaLevel, detailStrength, detailFreq } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU, ambientU } from '../tsl/lighting';
import { waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import { FLAT, SIM, RENDER } from '../config';
import type { BrushMode } from '../tools/BrushTool';

export class FlatEngine {
  readonly renderer: WebGPURenderer;
  private readonly scene = new Scene();
  private readonly camera: PerspectiveCamera;
  private readonly orbit: OrbitController;
  private heightField!: GridField;
  private brush!: FlatBrush;
  private sim!: FlatSim;
  private sidebar!: Sidebar;
  private sun!: DirectionalLight;
  private sky!: HemisphereLight;
  private hud: HTMLElement | null;
  private fpsEma = RENDER.targetFps;
  private lastTime = 0;
  private simAccum = 0;
  private readonly simInterval = 1 / SIM.ticksPerSecond;
  private readonly canvas: HTMLCanvasElement;

  private readonly ndc = new Vector2();
  private readonly raycaster = new Raycaster();
  private readonly plane = new Plane(new Vector3(0, 1, 0), 0);
  private readonly hit = new Vector3();
  private sculptMode = false;
  private brushing = false;
  private riverMode = false;
  private readonly brushSettings = { mode: 'raise' as BrushMode, radius: 0.06, strength: 0.02, rate: 0.4, target: 0.4 };
  private readonly water = { rainOn: false, rainRate: SIM.rainRate };
  private readonly river = { rate: 0.012, radius: 0.018 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new WebGPURenderer({ canvas, antialias: true });
    this.scene.background = new Color(0x9ec4e0);
    this.scene.fog = new Fog(0x9ec4e0, FLAT.worldSize * 1.3, FLAT.worldSize * 3.2);
    this.camera = new PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.05, 200);
    this.camera.position.set(0, FLAT.worldSize * 0.7, FLAT.worldSize * 0.85);
    this.orbit = new OrbitController(this.camera, canvas);
    this.orbit.controls.target.set(0, 0, 0);
    this.hud = document.getElementById('hud');

    this.sun = new DirectionalLight(0xfff4e2, 3.0);
    this.sun.position.set(6, 9, 4);
    this.sky = new HemisphereLight(0xbfd8ff, 0x6b5a3f, 0.55);
    this.scene.add(this.sun, this.sky);
    // world-space lighting uniforms for the unlit terrain (camera-independent).
    sunDirUniform.value.copy(this.sun.position).normalize();
    sunIntensityU.value = 2.8;
    ambientU.value = 0.55;
    window.addEventListener('resize', this.onResize);
  }

  async init(): Promise<void> {
    await this.renderer.init();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const W = FLAT.gridW, H = FLAT.gridH;
    const seed = buildFlatSeed(W, H);
    this.heightField = new GridField(W, H);
    this.renderer.compute(buildGridSeed(seed.height.texture as never, this.heightField.main, W, H));
    this.brush = new FlatBrush(this.heightField.main, this.heightField.scratch, W, H);

    this.sim = new FlatSim(this.renderer, this.heightField, seed.loose.texture as never, seed.hardness.texture as never, W, H);

    // render mesh denser than the sim grid (bicubic smooths between texels).
    const mW = Math.round(W * FLAT.meshDetail), mH = Math.round(H * FLAT.meshDetail);
    const terrain = makeFlatTerrain(this.heightField.main, this.sim.loose.main, seed.moisture.texture, seed.hardness.texture);
    this.scene.add(buildFlatMesh(mW, mH, terrain));

    const water = makeFlatWater(this.heightField.main, this.sim.water.main, this.sim.velocity.main);
    const waterMesh = buildFlatMesh(mW, mH, water);
    waterMesh.renderOrder = 1;
    this.scene.add(waterMesh);

    this.buildGui();
    this.sidebar = new Sidebar({
      brush: this.brushSettings,
      isSculpt: () => this.sculptMode,
      setSculpt: (on) => { this.sculptMode = on; this.orbit.controls.enableRotate = !on; },
      isRiver: () => this.riverMode,
      setRiver: (on) => { this.riverMode = on; },
      isVolcano: () => false,
      setVolcano: () => {},
      isRain: () => this.water.rainOn,
      toggleRain: () => { this.water.rainOn = !this.water.rainOn; },
    });
    this.canvas.addEventListener('pointerdown', this.onDown);
    this.canvas.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    window.addEventListener('keydown', this.onKey);
    this.renderer.setAnimationLoop(this.frame);
  }

  private buildGui(): void {
    const gui = new GUI({ title: 'Planet (flat)' });
    const b = gui.addFolder('Brush');
    b.add(this.brushSettings, 'mode', ['raise', 'lower', 'smooth', 'flatten']);
    b.add(this.brushSettings, 'radius', 0.01, 0.25, 0.005);
    b.add(this.brushSettings, 'strength', 0, 0.1, 0.001);
    b.add({ sculpt: () => { this.sculptMode = !this.sculptMode; this.orbit.controls.enableRotate = !this.sculptMode; } }, 'sculpt').name('toggle sculpt [g]');
    const t = gui.addFolder('Terrain');
    t.add(flatHeightScale, 'value', 0.5, 5, 0.05).name('height scale');
    t.add(flatSeaLevel, 'value', 0, 0.7, 0.01).name('sea level');
    t.add(detailStrength, 'value', 0, 1.5, 0.02).name('detail strength');
    t.add(detailFreq, 'value', 1, 24, 0.5).name('detail freq');
    const wf = gui.addFolder('Water');
    wf.add(this.water, 'rainOn').name('rain [r]');
    wf.add(this.water, 'rainRate', 0, 0.02, 0.0005).name('rain rate');
    wf.add(waterUniforms.evapProp, 'value', 0, 0.4, 0.005).name('evaporation /s');
    wf.add({ riverTool: () => { this.riverMode = !this.riverMode; if (this.riverMode && !this.sculptMode) { this.sculptMode = true; this.orbit.controls.enableRotate = false; } } }, 'riverTool').name('river source tool');
    wf.add(this.river, 'rate', 0, 0.05, 0.001).name('river rate');
    wf.add(this.river, 'radius', 0.005, 0.06, 0.002).name('river radius');
    wf.add({ clear: () => this.sim.clearWater() }, 'clear').name('clear water');
    wf.add({ cs: () => this.sim.clearSources() }, 'cs').name('clear sources');
    const e = gui.addFolder('Erosion');
    e.add(this.sim, 'erosionEnabled').name('enabled');
    e.add(erosionUniforms.simSpeed, 'value', 1, 12, 0.5).name('sim speed');
    e.add(erosionUniforms.sedimentCapacity, 'value', 0, 1, 0.02).name('capacity');
    e.add(erosionUniforms.deposit, 'value', 0, 0.3, 0.01).name('deposit');
    e.add(erosionUniforms.talus, 'value', 0.002, 0.06, 0.002).name('talus');
  }

  /** ndc -> world plane (y=0) -> uv in [0,1]. */
  private pickUv(): { u: number; v: number } | null {
    this.raycaster.setFromCamera(this.ndc, this.camera);
    if (!this.raycaster.ray.intersectPlane(this.plane, this.hit)) return null;
    const u = this.hit.x / FLAT.worldSize + 0.5;
    const v = this.hit.z / FLAT.worldSize + 0.5;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { u, v };
  }
  private setNdc(e: PointerEvent): void {
    this.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  }

  private onDown = (e: PointerEvent): void => {
    if (!this.sculptMode || e.button !== 0) return;
    this.setNdc(e);
    const p = this.pickUv();
    if (!p) return;
    if (this.riverMode) { this.sim.placeSource(p.u, p.v, this.river.rate, this.river.radius); return; }
    this.brushing = true;
    this.brush.stamp(this.renderer, p.u, p.v, this.brushSettings);
  };
  private onMove = (e: PointerEvent): void => {
    if (!this.brushing) return;
    this.setNdc(e);
    const p = this.pickUv();
    if (p) this.brush.stamp(this.renderer, p.u, p.v, this.brushSettings);
  };
  private onUp = (): void => { this.brushing = false; };
  private onKey = (e: KeyboardEvent): void => {
    switch (e.key) {
      case 'g': this.sculptMode = !this.sculptMode; this.orbit.controls.enableRotate = !this.sculptMode; break;
      case '1': this.brushSettings.mode = 'raise'; this.riverMode = false; break;
      case '2': this.brushSettings.mode = 'lower'; this.riverMode = false; break;
      case '3': this.brushSettings.mode = 'smooth'; this.riverMode = false; break;
      case '4': this.brushSettings.mode = 'flatten'; this.riverMode = false; break;
      case 'r': this.water.rainOn = !this.water.rainOn; break;
    }
    this.sidebar?.sync();
  };

  start(): void { this.renderer.setAnimationLoop(this.frame); }

  private frame = (time: number): void => {
    if (this.lastTime === 0) this.lastTime = time;
    const dt = Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;

    this.sim.setRain(this.water.rainOn ? this.water.rainRate : 0);
    this.simAccum += dt;
    let steps = 0;
    while (this.simAccum >= this.simInterval && steps < SIM.maxStepsPerFrame) {
      this.sim.tick(this.simInterval); this.simAccum -= this.simInterval; steps++;
    }
    if (this.simAccum > this.simInterval * SIM.maxStepsPerFrame) this.simAccum = 0;

    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
    if (this.hud) {
      this.fpsEma += (1000 / Math.max(dt * 1000, 0.001) - this.fpsEma) * 0.1;
      this.hud.textContent = `fps ${this.fpsEma.toFixed(0)}  FLAT ${FLAT.gridW}x${FLAT.gridH}  [g]${this.sculptMode ? 'SCULPT' : 'orbit'} ${this.riverMode ? 'RIVER' : this.brushSettings.mode} [r]rain:${this.water.rainOn ? 'on' : 'off'}`;
    }
  };

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
