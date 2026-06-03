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
import { makeTerrainMaterial } from '../materials/terrainMaterial';
import { HeightFields, buildSeedCompute } from '../sim/fields';
import { buildSeamTable, type SeamTable } from '../planet/seamTable';
import { SeamSync } from '../sim/passes/seamCopy';
import { Simulation } from '../sim/Simulation';
import { makeWaterMaterial } from '../materials/waterMaterial';
import { BrushTool } from '../tools/BrushTool';
import { pickPlanet } from '../tools/picking';
import {
  Controls,
  type BrushSettings,
  type WaterSettings,
  type ErosionSettings,
} from '../ui/Controls';
import { Sidebar } from '../ui/Sidebar';
import { PLANET, SIM, RENDER, FACES } from '../config';

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
  seamTable!: SeamTable;
  simulation!: Simulation;
  private waterPlanet!: PlanetMesh;

  // Sculpt input state.
  private sculptMode = false;
  private brushing = false;
  private brushDirty = false; // a stamp is pending for this frame
  private readonly ndc = new Vector2();

  // Live-tunable settings (driven by the lil-gui panel + hotkeys).
  readonly brushSettings: BrushSettings = {
    mode: 'raise',
    radius: 0.13,
    strength: 0.02,
    rate: 0.4,
    target: 0.35,
  };
  readonly waterSettings: WaterSettings = { rainOn: false, rainRate: SIM.rainRate };
  readonly erosionSettings: ErosionSettings = { enabled: false };
  private controls!: Controls;
  private sidebar!: Sidebar;

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
    this.seamTable = buildSeamTable(PLANET.res);

    for (const face of FACES) {
      const seed = buildHeightTexture(face, PLANET.res); // CPU DataTexture
      const field = this.heightFields.field(face);
      // First compute pass: copy seed -> canonical storage texture.
      this.renderer.compute(buildSeedCompute(seed.texture, field.main, this.heightFields.n));
      this.brush.register(face, field);

      // fixed binding (canonical main; never rebinds). Seam-aware normals.
      this.planet.setFaceMaterial(
        face,
        makeTerrainMaterial(face, this.heightFields, this.seamTable).material,
      );
    }

    // M3: seam sync across face edges (V5).
    this.seamSync = new SeamSync(this.heightFields, this.seamTable, this.heightFields.n);
    this.seamSync.sync(this.renderer); // initial pass
    this.brush.warmup(this.renderer); // V8: compile pipelines up front

    // M4: hydraulic water sim + water surface meshes.
    this.simulation = new Simulation(this.renderer, this.heightFields);
    await this.simulation.warmup();
    this.waterPlanet = new PlanetMesh(PLANET.res, PLANET.baseRadius, makeFlatSolidMaterial());
    for (const face of FACES) {
      this.waterPlanet.setFaceMaterial(
        face,
        makeWaterMaterial(face, this.heightFields, this.simulation.water, this.seamTable),
      );
    }
    this.scene.add(this.waterPlanet.group);
    this.setSim(this.simulation);

    this.controls = new Controls({
      brush: this.brushSettings,
      water: this.waterSettings,
      erosion: this.erosionSettings,
      onRainChange: () => this.applyRain(),
      onClearWater: () => this.simulation.clearWater(),
      onErosionChange: () => this.simulation.setErosion(this.erosionSettings.enabled),
    });
    this.sidebar = new Sidebar({
      brush: this.brushSettings,
      isSculpt: () => this.sculptMode,
      setSculpt: (on) => this.setSculpt(on),
      isRain: () => this.waterSettings.rainOn,
      toggleRain: () => {
        this.waterSettings.rainOn = !this.waterSettings.rainOn;
        this.applyRain();
        this.controls.gui.controllersRecursive().forEach((c) => c.updateDisplay());
      },
    });
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

  /** Applied once per frame (coalesced) while dragging — not per pointer event. */
  private applyBrush(): void {
    const pick = pickPlanet(this.ndc, this.camera, this.planet.group, PLANET.res);
    if (!pick) return;
    this.brush.stamp(this.renderer, pick.dir, this.brushSettings);
  }

  private applyRain(): void {
    this.simulation.setRain(this.waterSettings.rainOn ? this.waterSettings.rainRate : 0);
  }

  private setSculpt(on: boolean): void {
    this.sculptMode = on;
    this.orbit.controls.enableRotate = !on;
    this.sidebar?.sync();
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.sculptMode || e.button !== 0) return;
    this.brushing = true;
    this.setNdc(e);
    this.brushDirty = true;
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.brushing) return;
    this.setNdc(e);
    this.brushDirty = true; // coalesced: applied in frame()
  };

  private onPointerUp = (): void => {
    if (!this.brushing) return;
    this.brushing = false;
    // Seam-sync once at stroke end to clean up any sub-texel edge mismatch.
    this.seamSync.sync(this.renderer);
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    switch (e.key) {
      case 'g':
        this.setSculpt(!this.sculptMode);
        break;
      case '1': this.brushSettings.mode = 'raise'; this.sidebar.sync(); break;
      case '2': this.brushSettings.mode = 'lower'; this.sidebar.sync(); break;
      case '3': this.brushSettings.mode = 'smooth'; this.sidebar.sync(); break;
      case '4': this.brushSettings.mode = 'flatten'; this.sidebar.sync(); break;
      case 'r':
        this.waterSettings.rainOn = !this.waterSettings.rainOn;
        this.applyRain();
        this.sidebar.sync();
        this.controls.gui.controllersRecursive().forEach((c) => c.updateDisplay());
        break;
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

    // Coalesced brush: at most one stamp per frame regardless of how many
    // pointermove events fired -> bounded GPU work, no queue backlog.
    if (this.brushing && this.brushDirty) {
      this.applyBrush();
      this.brushDirty = false;
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
        `res ${PLANET.res}  sim ${SIM.ticksPerSecond}/s  rain:${this.waterSettings.rainOn ? 'on' : 'off'} [r]\n` +
        `[g] ${this.sculptMode ? 'SCULPT' : 'orbit'}  brush:${this.brushSettings.mode} [1raise 2lower 3smooth 4flatten]`;
    }
  }

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
