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
  Vector3,
  Mesh,
  SphereGeometry,
  RingGeometry,
  MeshBasicMaterial,
  DoubleSide,
} from 'three';
import { makeAtmosphereMaterial } from '../materials/atmosphereMaterial';
import { makeCloudMaterial } from '../materials/cloudMaterial';
import { makeRainMaterial } from '../materials/rainMaterial';
import { storminess } from '../materials/cloudMaterial';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitController } from './OrbitController';
import { PlanetMesh, makeFlatSolidMaterial } from '../planet/PlanetMesh';
import {
  buildHeightTexture,
  buildLooseTexture,
  buildHardnessTexture,
  buildRainfallTexture,
} from '../planet/heightField';
import { buildCellAreaTexture } from '../planet/cellArea';
import { dirToFaceUV } from '../tsl/warp';
import { heightScaleUniform } from '../tsl/heightScale';
import { makeTerrainMaterial } from '../materials/terrainMaterial';
import { HeightFields, FieldSet, buildSeedCompute, buildFillZero } from '../sim/fields';
import { buildSeamTable, type SeamTable } from '../planet/seamTable';
import { SeamSync, NormalSeamSync, NormalBandSmooth } from '../sim/passes/seamCopy';
import { NormalBaker } from '../sim/passes/normals';
import { Simulation } from '../sim/Simulation';
import { LavaSim } from '../sim/LavaSim';
import { makeWaterMaterial } from '../materials/waterMaterial';
import { makeLavaMaterial } from '../materials/lavaMaterial';
import { makeDebugMaterial, debugModeUniform, DEBUG_MODES } from '../materials/debugMaterial';
import { lightingSettings, sunDirUniform, sunDirection } from '../tsl/lighting';
import type { Material } from 'three';
import { textureLoad } from 'three/tsl';
import type { SampleFace } from '../tsl/surface';
import { BrushTool } from '../tools/BrushTool';
import { EmitterTool } from '../tools/Emitters';
import { pickPlanet } from '../tools/picking';
import {
  Controls,
  type BrushSettings,
  type WaterSettings,
  type ErosionSettings,
} from '../ui/Controls';
import { Sidebar } from '../ui/Sidebar';
import { PLANET, SIM, RENDER, FACES, type FaceName } from '../config';

/** RingGeometry's default facing normal (+Z); used to orient the cursor to the surface. */
const CURSOR_Z = new Vector3(0, 0, 1);

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
  emitter!: EmitterTool;
  private riverMode = false;
  // small footprint (near point source) + modest rate -> a stream, not a dome.
  readonly riverSettings = { rate: 0.04, radius: 0.003 };
  seamSync!: SeamSync;
  private normalSeam!: NormalSeamSync;
  private normalBand!: NormalBandSmooth;
  seamTable!: SeamTable;
  simulation!: Simulation;
  lavaSim!: LavaSim;
  private volcanoEmitter!: EmitterTool;
  private volcanoMode = false;
  private terrainNormals!: FieldSet;
  private terrainBaker!: NormalBaker;
  private readonly terrainMats = new Map<FaceName, Material>();
  private readonly debugMats = new Map<FaceName, Material>();
  private debugOn = false;
  private debugMode = 0;
  private readonly heightData = new Map<FaceName, Float32Array>();
  private cursor!: Mesh;

  /** CPU bedrock height [0,1] at a planet-local direction (for pick refinement +
   *  cursor). Initial fbm — stale after heavy erosion, but fixes the bulk offset. */
  private surfaceHeightAt = (dir: Vector3): number => {
    const { face, u, v } = dirToFaceUV([dir.x, dir.y, dir.z]);
    const data = this.heightData.get(face);
    if (!data) return 0;
    const n = this.heightFields.n;
    const res = PLANET.res;
    const tx = Math.min(res, Math.max(0, Math.round(((u + 1) / 2) * res)));
    const ty = Math.min(res, Math.max(0, Math.round(((v + 1) / 2) * res)));
    return data[ty * n + tx];
  };
  private simPaused = false; // 'p' — freeze sim (diagnostic: sim vs render cost)
  private bakeCounter = 0;
  private waterPlanet!: PlanetMesh;
  private lavaPlanet!: PlanetMesh;

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

  private sun!: DirectionalLight;
  private fillLight!: DirectionalLight;
  private skyLight!: HemisphereLight;

  private addLights(): void {
    this.sun = new DirectionalLight(0xfff2e0, lightingSettings.sunIntensity);
    this.scene.add(this.sun);
    // Fill from the opposite side so the dark hemisphere stays readable.
    this.fillLight = new DirectionalLight(0x88a0c0, lightingSettings.fill);
    this.scene.add(this.fillLight);
    this.skyLight = new HemisphereLight(0x9fc5ff, 0x3a2f25, lightingSettings.ambient);
    this.scene.add(this.skyLight);
    this.applyLighting();

    // atmosphere halo shell (T21): above the highest peaks, additive limb glow.
    const atmo = new Mesh(
      new SphereGeometry(PLANET.baseRadius + 0.7, 64, 48),
      makeAtmosphereMaterial(),
    );
    atmo.renderOrder = 10; // after opaque terrain + water
    this.scene.add(atmo);

    // cloud shell (T22): just above the peaks, below the atmosphere halo.
    const clouds = new Mesh(
      new SphereGeometry(PLANET.baseRadius + 0.55, 96, 64),
      makeCloudMaterial(),
    );
    clouds.renderOrder = 9; // after water, before atmosphere
    this.scene.add(clouds);

    // rain veil (T23): below clouds, above surface; visible under storm cores.
    const rain = new Mesh(
      new SphereGeometry(PLANET.baseRadius + 0.32, 96, 64),
      makeRainMaterial(),
    );
    rain.renderOrder = 8;
    this.scene.add(rain);
    // these shells run noise shaders per-fragment over the whole screen every
    // frame -> heavy. let them be toggled off for perf (key 'w' / GUI).
    this.weatherMeshes = [atmo, clouds, rain];
    this.setWeather(this.weatherSettings.enabled);
  }

  private weatherMeshes: Mesh[] = [];
  // off by default: full-screen noise shells are heavy; opt in via 'w' / GUI.
  readonly weatherSettings = { enabled: false };

  setWeather = (on: boolean): void => {
    this.weatherSettings.enabled = on;
    for (const m of this.weatherMeshes) m.visible = on;
  };

  /** Push lightingSettings to the three lights + the sun-direction uniform. */
  applyLighting = (): void => {
    const dir = sunDirection(lightingSettings); // toward sun
    this.sun.position.copy(dir).multiplyScalar(10);
    this.sun.intensity = lightingSettings.sunIntensity;
    this.fillLight.position.copy(dir).multiplyScalar(-10);
    this.fillLight.intensity = lightingSettings.fill;
    this.skyLight.intensity = lightingSettings.ambient;
    sunDirUniform.value.set(dir.x, dir.y, dir.z);
  };

  async init(): Promise<void> {
    await this.renderer.init();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.planet = new PlanetMesh(PLANET.res, PLANET.baseRadius, makeFlatSolidMaterial());
    this.scene.add(this.planet.group);

    // surface-following interaction cursor: a glowing ring laid tangent to the
    // terrain at the pick point. child of the planet group so it rides rotation.
    const cursorMat = new MeshBasicMaterial({
      color: 0x6fe8ff,
      transparent: true,
      opacity: 0.85,
      side: DoubleSide,
      depthWrite: false,
      depthTest: false, // always readable (only ever placed at the near hit)
    });
    this.cursor = new Mesh(new RingGeometry(0.78, 1.0, 48), cursorMat);
    this.cursor.renderOrder = 11;
    this.cursor.visible = false;
    this.planet.group.add(this.cursor);

    // M2: per-face GPU height field (ping-pong), seeded from CPU fbm,
    // terrain material samples the front storage texture, brush stamps it.
    this.heightFields = new HeightFields(PLANET.res);
    this.brush = new BrushTool(this.heightFields.n);
    this.seamTable = buildSeamTable(PLANET.res);
    this.terrainNormals = new FieldSet(this.heightFields.n, true);

    const sampleTerrain: SampleFace = (f, coord) =>
      textureLoad(this.heightFields.field(f).main, coord).x;

    // Sim owns the loose-material field; create it before terrain materials so
    // they can sample it for material coloring.
    this.simulation = new Simulation(this.renderer, this.heightFields);
    const n = this.heightFields.n;

    for (const face of FACES) {
      const field = this.heightFields.field(face);
      // seed bedrock height + an uneven initial loose (soil/sand) layer.
      const ht = buildHeightTexture(face, PLANET.res);
      this.heightData.set(face, ht.data); // keep CPU copy for pick refinement
      this.renderer.compute(buildSeedCompute(ht.texture, field.main, n));
      this.renderer.compute(
        buildSeedCompute(buildLooseTexture(face, PLANET.res).texture, this.simulation.loose.field(face).main, n),
      );
      this.renderer.compute(
        buildSeedCompute(buildHardnessTexture(face, PLANET.res).texture, this.simulation.hardness.field(face).main, n),
      );
      this.renderer.compute(
        buildSeedCompute(buildRainfallTexture(face, PLANET.res).texture, this.simulation.rainfall.field(face).main, n),
      );
      this.renderer.compute(
        buildSeedCompute(buildCellAreaTexture(face, PLANET.res).texture, this.simulation.cellArea.field(face).main, n),
      );
      // zero the erosion-viz field (uninitialized storage = random tint -> washed
      // out / flickering terrain before any erosion).
      this.renderer.compute(buildFillZero(this.simulation.erosionViz.field(face).main, n));
      this.brush.register(face, field);

      // cheap material: displacement + baked normal + material (rock/soil) color.
      const terrainMat = makeTerrainMaterial(
        face,
        field.main,
        this.simulation.loose.field(face).main,
        this.terrainNormals.field(face).main,
        this.simulation.erosionViz.field(face).main,
        this.simulation.rainfall.field(face).main,
        this.simulation.hardness.field(face).main,
      ).material;
      this.terrainMats.set(face, terrainMat);
      this.debugMats.set(
        face,
        makeDebugMaterial(
          face,
          field.main,
          this.terrainNormals.field(face).main,
          this.simulation.water.field(face).main,
          this.simulation.sediment.field(face).main,
          this.simulation.loose.field(face).main,
          this.simulation.velocity.field(face).main,
          this.simulation.cellArea.field(face).main,
          this.simulation.erosionViz.field(face).main,
        ),
      );
      this.planet.setFaceMaterial(face, terrainMat);
    }

    // M3: seam sync across face edges (V5).
    this.seamSync = new SeamSync(this.heightFields, this.seamTable, this.heightFields.n);
    this.seamSync.sync(this.renderer); // initial pass

    // bake terrain normals from the seeded heights (perf: off the fragment path).
    this.terrainBaker = new NormalBaker(
      sampleTerrain,
      this.seamTable,
      this.terrainNormals,
      this.heightFields.n,
    );
    this.normalSeam = new NormalSeamSync(this.terrainNormals, this.seamTable, this.heightFields.n);
    this.normalBand = new NormalBandSmooth(this.terrainNormals, this.heightFields.n);
    this.terrainBaker.bake(this.renderer);
    this.normalBand.sync(this.renderer); // smooth shading crease in the seam band
    this.normalSeam.sync(this.renderer); // then lock the exact edge cross-seam
    this.brush.warmup(this.renderer); // V8: compile pipelines up front

    await this.simulation.warmup();
    this.waterPlanet = new PlanetMesh(PLANET.res, PLANET.baseRadius, makeFlatSolidMaterial());
    for (const face of FACES) {
      this.waterPlanet.setFaceMaterial(
        face,
        makeWaterMaterial(
          face,
          this.heightFields.field(face).main,
          this.simulation.water.field(face).main,
          this.simulation.waterNormals.field(face).main,
          this.simulation.cellArea.field(face).main,
        ),
      );
    }
    this.scene.add(this.waterPlanet.group);
    this.emitter = new EmitterTool(this.simulation.waterSource, this.heightFields.n);

    // M8: lava sim + emissive lava meshes + volcano emitter.
    this.lavaSim = new LavaSim(this.renderer, this.heightFields, this.seamTable, this.simulation.cellArea);
    this.lavaPlanet = new PlanetMesh(PLANET.res, PLANET.baseRadius, makeFlatSolidMaterial());
    for (const face of FACES) {
      this.lavaPlanet.setFaceMaterial(
        face,
        makeLavaMaterial(
          face,
          this.heightFields.field(face).main,
          this.lavaSim.lava.field(face).main,
          this.lavaSim.heat.field(face).main,
          this.terrainNormals.field(face).main,
        ),
      );
    }
    this.scene.add(this.lavaPlanet.group);
    this.volcanoEmitter = new EmitterTool(this.lavaSim.lavaSource, this.heightFields.n);

    this.setSim(this.simulation);

    this.controls = new Controls({
      brush: this.brushSettings,
      water: this.waterSettings,
      river: this.riverSettings,
      erosion: this.erosionSettings,
      onRainChange: () => this.applyRain(),
      onClearWater: () => this.simulation.clearWater(),
      onClearSources: () => this.emitter.clear(this.renderer),
      onErosionChange: () => this.simulation.setErosion(this.erosionSettings.enabled),
      onLightingChange: this.applyLighting,
      weather: this.weatherSettings,
      onWeatherToggle: (on: boolean) => this.setWeather(on),
    });
    this.sidebar = new Sidebar({
      brush: this.brushSettings,
      isSculpt: () => this.sculptMode,
      setSculpt: (on) => this.setSculpt(on),
      isRiver: () => this.riverMode,
      setRiver: (on) => {
        this.riverMode = on;
        if (on) this.volcanoMode = false;
      },
      isVolcano: () => this.volcanoMode,
      setVolcano: (on) => {
        this.volcanoMode = on;
        if (on) this.riverMode = false;
      },
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
    const pick = pickPlanet(
      this.ndc,
      this.camera,
      this.planet.group,
      PLANET.res,
      this.surfaceHeightAt,
      heightScaleUniform.value,
    );
    if (!pick) return;
    if (this.riverMode) {
      // place/extend a continuous river source (persists, emits each tick).
      this.emitter.stamp(this.renderer, pick.dir, this.riverSettings.rate, this.riverSettings.radius);
      return;
    }
    if (this.volcanoMode) {
      // place a volcano vent: continuous hot lava source.
      this.volcanoEmitter.stamp(this.renderer, pick.dir, 0.05, 0.014);
      this.lavaSim.active = true;
      return;
    }
    this.brush.stamp(this.renderer, pick.dir, this.brushSettings);
    this.terrainBaker.bake(this.renderer); // height changed -> rebake normals
    this.normalBand.sync(this.renderer);
    this.normalSeam.sync(this.renderer);
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
    this.setNdc(e); // always track pointer so the cursor follows on hover
    if (this.brushing) this.brushDirty = true; // coalesced: applied in frame()
  };

  /** Position the surface cursor at the pick point (planet-local). */
  private updateCursor(): void {
    const active = this.sculptMode || this.riverMode || this.volcanoMode;
    if (!active || !this.cursor) {
      if (this.cursor) this.cursor.visible = false;
      return;
    }
    const pick = pickPlanet(
      this.ndc,
      this.camera,
      this.planet.group,
      PLANET.res,
      this.surfaceHeightAt,
      heightScaleUniform.value,
    );
    if (!pick) {
      this.cursor.visible = false;
      return;
    }
    this.cursor.visible = true;
    const r = PLANET.baseRadius + this.surfaceHeightAt(pick.dir) * heightScaleUniform.value;
    this.cursor.position.copy(pick.dir).multiplyScalar(r + 0.012);
    this.cursor.quaternion.setFromUnitVectors(CURSOR_Z, pick.dir);
    const toolR = this.riverMode
      ? this.riverSettings.radius
      : this.volcanoMode
        ? 0.014
        : this.brushSettings.radius;
    const w = Math.max(0.05, toolR * PLANET.baseRadius);
    this.cursor.scale.set(w, w, w);
  }

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
      case 'w':
        this.setWeather(!this.weatherSettings.enabled); // toggle weather shells (perf)
        this.controls.gui.controllersRecursive().forEach((c) => c.updateDisplay());
        break;
      case 'p':
        this.simPaused = !this.simPaused; // freeze sim (diagnostic)
        break;
      case 'v': {
        // cycle debug modes: off -> waterDepth -> flowSpeed -> ... -> off.
        this.debugMode = (this.debugMode + 1) % DEBUG_MODES.length;
        this.debugOn = this.debugMode !== 0;
        debugModeUniform.value = this.debugMode;
        for (const f of FACES) {
          this.planet.setFaceMaterial(f, (this.debugOn ? this.debugMats : this.terrainMats).get(f)!);
        }
        break;
      }
    }
  };

  start(): void {
    this.renderer.setAnimationLoop(this.frame);
  }

  private frame = (time: number): void => {
    if (this.lastTime === 0) this.lastTime = time;
    const dt = Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;

    // climate cycle: when rain is on, the global rate ebbs/flows slowly (wet &
    // dry seasons); combined with the regional rainfall map -> rain shifts over
    // time and falls in some regions more than others.
    // base climate rain (toggle) + storm rain: storms always add rain (localized
    // by the rainfall map) so the rain veil under storm clouds matches actual sim
    // rainfall (V26, coherent via storminess).
    // rain toggle is the master switch (so water can be fully turned off). when
    // on: climate pulse + storm contribution (localized by rainfall map).
    const pulse = 0.45 + 0.55 * Math.max(0, Math.sin(time * 0.00012));
    const rate = this.waterSettings.rainOn
      ? this.waterSettings.rainRate * pulse + storminess.value * SIM.rainRate * 1.5
      : 0;
    this.simulation.setRain(rate);

    // Decoupled fixed-step sim (V10), capped (anti spiral-of-death).
    if (this.sim && !this.simPaused) {
      this.simAccumulator += dt;
      let steps = 0;
      while (this.simAccumulator >= this.simInterval && steps < SIM.maxStepsPerFrame) {
        this.sim.tick(this.simInterval); // real-time step (was SIM.dt -> 1/3 speed)
        this.lavaSim.tick(this.simInterval);
        this.simAccumulator -= this.simInterval;
        steps++;
      }
      if (this.simAccumulator > this.simInterval * SIM.maxStepsPerFrame) {
        this.simAccumulator = 0; // drop backlog rather than spiral
      }
      // rebake terrain normals when bedrock changed (throttled: the bake is the
      // heaviest erosion-tick cost; normals lag a frame, fine for gradual change).
      if (this.simulation.terrainChanged || this.lavaSim.terrainChanged) {
        this.bakeCounter = (this.bakeCounter + 1) % 2;
        if (this.bakeCounter === 0) {
          this.terrainBaker.bake(this.renderer);
          this.normalBand.sync(this.renderer);
          this.normalSeam.sync(this.renderer);
        }
      }
    }

    // Coalesced brush: at most one stamp per frame regardless of how many
    // pointermove events fired -> bounded GPU work, no queue backlog.
    if (this.brushing && this.brushDirty) {
      this.applyBrush();
      this.brushDirty = false;
    }

    this.orbit.update();
    this.updateCursor();
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
        `res ${PLANET.res}  sim ${SIM.ticksPerSecond}/s  rain:${this.waterSettings.rainOn ? 'on' : 'off'} [r]  weather:${this.weatherSettings.enabled ? 'on' : 'off'} [w]  sim:${this.simPaused ? 'PAUSED' : 'on'} [p]\n` +
        `[g] ${this.sculptMode ? 'SCULPT' : 'orbit'}  brush:${this.brushSettings.mode} [1raise 2lower 3smooth 4flatten]  [v]debug:${DEBUG_MODES[this.debugMode]}`;
    }
  }

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
