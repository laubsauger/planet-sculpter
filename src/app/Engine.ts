// Owns renderer, scene, camera, loop. Fixed-timestep sim accumulator decoupled
// from render (V10). Sim throttled below render fps. Compute pipelines warmed
// at startup (V8) — wired when sim passes land (M4+).

import {
  Scene,
  PerspectiveCamera,
  DirectionalLight,
  HemisphereLight,
  Color,
  Vector2,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitController } from './OrbitController';
import { PlanetMesh, makeFlatSolidMaterial } from '../planet/PlanetMesh';
import { buildHeightTexture } from '../planet/heightField';
import { makeTerrainMaterial, type TerrainMaterial } from '../materials/terrainMaterial';
import { HeightFields, buildSeedCompute } from '../sim/fields';
import { buildSeamTable } from '../planet/seamTable';
import { SeamSync } from '../sim/passes/seamCopy';
import { BrushTool, type BrushMode } from '../tools/BrushTool';
import { pickPlanet } from '../tools/picking';
import { PLANET, SIM, RENDER, FACES, type FaceName } from '../config';

export interface SimHooks {
  /** One fixed sim tick. No-op until M4. */
  tick(dt: number): void;
  /** Dispatch each compute pipeline once to avoid first-use hitch (V8). */
  warmup(): Promise<void>;
}

export class Engine {
  readonly renderer: WebGPURenderer;
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly orbit: OrbitController;
  planet!: PlanetMesh;
  heightFields!: HeightFields;
  brush!: BrushTool;
  seamSync!: SeamSync;
  private readonly terrainMats = new Map<FaceName, TerrainMaterial>();

  // Sculpt input state.
  private sculptMode = false;
  private brushing = false;
  private brushMode: BrushMode = 'raise';
  private readonly ndc = new Vector2();

  private sim: SimHooks | null = null;
  private simAccumulator = 0;
  private readonly simInterval = 1 / SIM.ticksPerSecond;
  private lastTime = 0;

  // fps tracking for HUD + adaptive scaling (V10).
  private frameMsEma = RENDER.frameBudgetMs;
  private fpsEma = RENDER.targetFps;
  private hud: HTMLElement | null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new WebGPURenderer({ canvas, antialias: true });
    this.scene.background = new Color(0x0b0d12);

    this.camera = new PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
    this.camera.position.set(0, PLANET.baseRadius * 1.5, PLANET.baseRadius * 3);

    this.orbit = new OrbitController(this.camera, canvas);
    this.hud = document.getElementById('hud');

    this.addLights();
    window.addEventListener('resize', this.onResize);
  }

  private addLights(): void {
    const sun = new DirectionalLight(0xfff2e0, 2.2);
    sun.position.set(3, 4, 2);
    this.scene.add(sun);
    // Fill from the opposite side so the dark hemisphere stays readable.
    const fill = new DirectionalLight(0x88a0c0, 0.6);
    fill.position.set(-3, -1.5, -2);
    this.scene.add(fill);
    const sky = new HemisphereLight(0x9fc5ff, 0x3a2f25, 0.6);
    this.scene.add(sky);
  }

  async init(): Promise<void> {
    await this.renderer.init();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.planet = new PlanetMesh(PLANET.res, PLANET.baseRadius, makeFlatSolidMaterial());
    this.scene.add(this.planet.group);

    // M2: per-face GPU height field (ping-pong), seeded from CPU fbm,
    // terrain material samples the front storage texture, brush stamps it.
    this.heightFields = new HeightFields(PLANET.res);
    this.brush = new BrushTool(this.heightFields.n);

    for (const face of FACES) {
      const seed = buildHeightTexture(face, PLANET.res); // CPU DataTexture
      const field = this.heightFields.field(face);
      // First compute pass: copy seed -> canonical storage texture.
      this.renderer.compute(buildSeedCompute(seed.texture, field.main, this.heightFields.n));
      this.brush.register(face, field);

      const tm = makeTerrainMaterial(field.main); // fixed binding (canonical)
      this.terrainMats.set(face, tm);
      this.planet.setFaceMaterial(face, tm.material);
    }

    // M3: seam sync across face edges (V5).
    this.seamSync = new SeamSync(
      this.heightFields,
      buildSeamTable(PLANET.res),
      this.heightFields.n,
    );
    this.seamSync.sync(this.renderer); // initial pass
    this.brush.warmup(this.renderer); // V8: compile pipelines up front
    this.installInput();
  }

  setSim(sim: SimHooks): void {
    this.sim = sim;
  }

  // --- sculpt input (M2; UI panel replaces this at T18) ---------------------

  private installInput(): void {
    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('keydown', this.onKeyDown);
  }

  private setNdc(e: PointerEvent): void {
    this.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  }

  private applyBrush(): void {
    const pick = pickPlanet(this.ndc, this.camera, this.planet.group, PLANET.res);
    if (!pick) return;
    this.brush.stamp(this.renderer, pick.dir, {
      mode: this.brushMode,
      radius: 0.13, // chord on unit sphere
      strength: 0.02,
      rate: 0.4,
      target: 0.35,
    });
    // Propagate edited face edges to neighbors so seams stay continuous (V5).
    this.seamSync.sync(this.renderer);
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.sculptMode || e.button !== 0) return;
    this.brushing = true;
    this.setNdc(e);
    this.applyBrush();
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.brushing) return;
    this.setNdc(e);
    this.applyBrush();
  };

  private onPointerUp = (): void => {
    this.brushing = false;
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    switch (e.key) {
      case 'g':
        this.sculptMode = !this.sculptMode;
        this.orbit.controls.enableRotate = !this.sculptMode;
        break;
      case '1': this.brushMode = 'raise'; break;
      case '2': this.brushMode = 'lower'; break;
      case '3': this.brushMode = 'smooth'; break;
      case '4': this.brushMode = 'flatten'; break;
    }
  };

  start(): void {
    this.renderer.setAnimationLoop(this.frame);
  }

  private frame = (time: number): void => {
    if (this.lastTime === 0) this.lastTime = time;
    const dt = Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;

    // Decoupled fixed-step sim (V10), capped (anti spiral-of-death).
    if (this.sim) {
      this.simAccumulator += dt;
      let steps = 0;
      while (this.simAccumulator >= this.simInterval && steps < SIM.maxStepsPerFrame) {
        this.sim.tick(SIM.dt);
        this.simAccumulator -= this.simInterval;
        steps++;
      }
      if (this.simAccumulator > this.simInterval * SIM.maxStepsPerFrame) {
        this.simAccumulator = 0; // drop backlog rather than spiral
      }
    }

    this.orbit.update();
    this.renderer.render(this.scene, this.camera);

    this.updateHud(dt);
  };

  private updateHud(dt: number): void {
    const ms = dt * 1000;
    this.frameMsEma += (ms - this.frameMsEma) * 0.1;
    this.fpsEma += (1000 / Math.max(ms, 0.001) - this.fpsEma) * 0.1;
    if (this.hud) {
      this.hud.textContent =
        `fps ${this.fpsEma.toFixed(0)}  ${this.frameMsEma.toFixed(1)}ms\n` +
        `res ${PLANET.res}  sim ${SIM.ticksPerSecond}/s\n` +
        `[g] ${this.sculptMode ? 'SCULPT' : 'orbit'}  brush:${this.brushMode} [1raise 2lower 3smooth 4flatten]`;
    }
  }

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
