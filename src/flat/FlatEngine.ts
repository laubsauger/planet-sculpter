// Flat From-Dust-style engine. Crisp detail-normal terrain + pipe-model water +
// erosion on a uniform heightfield (no sphere/pole/seam machinery). Sculpt brush,
// river springs, rain, realistic water with flow normals.

import {
  Scene, PerspectiveCamera, DirectionalLight, HemisphereLight, Color, Fog,
  Vector2, Vector3, Raycaster, Plane, Mesh, TimestampQuery, DataTexture, RedFormat, FloatType,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
import GUI from 'lil-gui';
import { OrbitController } from '../app/OrbitController';
import { buildFlatSeed } from './flatSeed';
import { buildFlatMesh } from './flatMesh';
import { makeFlatTerrain } from '../materials/flatTerrain';
import { flowBandScale, flowBandStrength, makeFlatWater } from '../materials/flatWater';
import { makeFlatDebug, FLAT_DEBUG_MODES, flatDebugMode } from '../materials/flatDebug';
import { FlatBrush } from './FlatBrush';
import { FlatSim, FLAT_WATER_SOLVERS } from './flatSim';
import { buildFlatBenchmark, FLAT_BENCHMARKS, type FlatBenchmark, type FlatBenchmarkData } from './flatBenchmarks';
import { Sidebar } from '../ui/Sidebar';
import { GridField, buildGridSeed } from '../sim/gridStore';
import { flatHeightScale, flatSeaLevel, detailStrength, detailFreq } from '../tsl/flatSurface';
import { sunDirUniform, sunIntensityU, ambientU } from '../tsl/lighting';
import { mudViscosityFactor, waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import { FLAT, SIM, RENDER } from '../config';
import type { BrushMode } from '../tools/BrushTool';

export class FlatEngine {
  private static readonly SCENE_BACKGROUND = 0x9ec4e0;
  private static readonly DEBUG_BACKGROUND = 0x11151c;
  readonly renderer: WebGPURenderer;
  private readonly scene = new Scene();
  private readonly camera: PerspectiveCamera;
  private readonly orbit: OrbitController;
  private heightField!: GridField;
  private brush!: FlatBrush;
  private sim!: FlatSim;
  private terrainMesh!: Mesh;
  private waterMesh!: Mesh;
  private terrainMaterial!: ReturnType<typeof makeFlatTerrain>;
  private debugMaterial!: ReturnType<typeof makeFlatDebug>;
  private defaultState!: FlatBenchmarkData;
  private sidebar!: Sidebar;
  private sun!: DirectionalLight;
  private sky!: HemisphereLight;
  private hud: HTMLElement | null;
  private fpsEma = RENDER.targetFps;
  private lastTime = 0;
  private simAccum = 0;
  private simCpuMsEma = 0;
  private gpuComputeMs = 0;
  private gpuRenderMs = 0;
  private timingPending = false;
  private lastTimingResolve = 0;
  private lastComputeCalls = 0;
  private computeCallsFrame = 0;
  private debugMode = 0;
  private readonly diagnostics = { benchmark: 'default' as FlatBenchmark };
  private snapshotText = '';
  private previousEarthMass: number | null = null;
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
  private readonly river = { rate: 1.2, radius: 0.009 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new WebGPURenderer({ canvas, antialias: true, trackTimestamp: true });
    this.scene.background = new Color(FlatEngine.SCENE_BACKGROUND);
    this.scene.fog = new Fog(FlatEngine.SCENE_BACKGROUND, FLAT.worldSize * 1.3, FLAT.worldSize * 3.2);
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
    this.defaultState = {
      height: seed.height.texture,
      loose: seed.loose.texture,
      water: this.zeroTexture(W, H),
      sediment: this.zeroTexture(W, H),
      source: this.zeroTexture(W, H),
      rainOn: false,
      erosionOn: false,
    };
    this.heightField = new GridField(W, H);
    this.renderer.compute(buildGridSeed(seed.height.texture as never, this.heightField.main, W, H));
    this.brush = new FlatBrush(this.heightField.main, this.heightField.scratch, W, H);

    this.sim = new FlatSim(this.renderer, this.heightField, seed.loose.texture as never, seed.hardness.texture as never, W, H);

    // render mesh denser than the sim grid (bicubic smooths between texels).
    const mW = Math.round(W * FLAT.meshDetail), mH = Math.round(H * FLAT.meshDetail);
    this.terrainMaterial = makeFlatTerrain(
      this.heightField.main,
      this.sim.loose.main,
      seed.moisture.texture,
      seed.hardness.texture,
      this.sim.water.main,
      this.sim.sediment.main,
      this.sim.activity.main,
    );
    this.debugMaterial = makeFlatDebug(
      this.heightField.main,
      this.sim.water.main,
      this.sim.velocity.main,
      this.sim.sediment.main,
      this.sim.loose.main,
      this.sim.source.main,
      this.sim.flux.main,
      this.sim.activity.main,
    );
    this.terrainMesh = buildFlatMesh(mW, mH, this.terrainMaterial);
    this.scene.add(this.terrainMesh);

    const water = makeFlatWater(this.heightField.main, this.sim.water.main, this.sim.velocity.main, this.sim.sediment.main);
    this.waterMesh = buildFlatMesh(mW, mH, water);
    this.waterMesh.renderOrder = 1;
    this.scene.add(this.waterMesh);

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
    wf.add(this.water, 'rainRate', 0, 0.004, 0.00005).name('rain rate');
    wf.add(this.sim, 'waterSolver', FLAT_WATER_SOLVERS).name('solver');
    wf.add(waterUniforms.evapProp, 'value', 0, 0.4, 0.005).name('evaporation /s');
    wf.add(mudViscosityFactor, 'value', 0, 10, 0.25).name('mud viscosity');
    wf.add(flowBandStrength, 'value', 0, 1, 0.02).name('flow band strength');
    wf.add(flowBandScale, 'value', 2, 20, 0.5).name('flow band scale');
    wf.add({ riverTool: () => { this.riverMode = !this.riverMode; if (this.riverMode && !this.sculptMode) { this.sculptMode = true; this.orbit.controls.enableRotate = false; } } }, 'riverTool').name('river source tool');
    wf.add(this.river, 'rate', 0, 4, 0.05).name('river discharge');
    wf.add(this.river, 'radius', 0.008, 0.08, 0.002).name('spring radius');
    wf.add({ clear: () => this.sim.clearWater() }, 'clear').name('clear water');
    wf.add({ cs: () => this.sim.clearSources() }, 'cs').name('clear sources');
    const e = gui.addFolder('Erosion');
    e.add(this.sim, 'erosionEnabled').name('enabled');
    e.add(erosionUniforms.simSpeed, 'value', 1, 12, 0.5).name('sim speed');
    e.add(erosionUniforms.sedimentCapacity, 'value', 0, 1, 0.02).name('capacity');
    e.add(erosionUniforms.deposit, 'value', 0, 0.3, 0.01).name('deposit');
    e.add(erosionUniforms.talus, 'value', 0.002, 0.06, 0.002).name('talus');
    const diagnostics = gui.addFolder('Diagnostics');
    diagnostics.add(this.diagnostics, 'benchmark', FLAT_BENCHMARKS).name('benchmark');
    diagnostics.add({ load: () => this.loadBenchmark(this.diagnostics.benchmark) }, 'load').name('load benchmark');
    diagnostics.add({ snapshot: () => { void this.snapshotConservation(); } }, 'snapshot').name('conservation snapshot');
    diagnostics.add({ cycleDebug: () => this.cycleDebug() }, 'cycleDebug').name('cycle debug [v]');
  }

  private zeroTexture(w: number, h: number) {
    const data = new Float32Array(w * h);
    const texture = new DataTexture(data, w, h, RedFormat, FloatType);
    texture.needsUpdate = true;
    return texture;
  }

  private loadBenchmark(name: FlatBenchmark): void {
    const state = name === 'default' ? this.defaultState : buildFlatBenchmark(name, FLAT.gridW, FLAT.gridH);
    this.sim.loadState(state);
    this.water.rainOn = state.rainOn;
    this.sim.erosionEnabled = state.erosionOn;
    this.simAccum = 0;
    this.snapshotText = '';
    this.previousEarthMass = null;
    this.sidebar?.sync();
  }

  private cycleDebug(): void {
    this.debugMode = (this.debugMode + 1) % FLAT_DEBUG_MODES.length;
    flatDebugMode.value = this.debugMode;
    this.terrainMesh.material = this.debugMode === 0 ? this.terrainMaterial : this.debugMaterial;
    this.waterMesh.visible = this.debugMode === 0;
    const background = this.debugMode === 0 ? FlatEngine.SCENE_BACKGROUND : FlatEngine.DEBUG_BACKGROUND;
    (this.scene.background as Color).setHex(background);
    (this.scene.fog as Fog).color.setHex(background);
  }

  private async snapshotConservation(): Promise<void> {
    type ReadbackBackend = {
      copyTextureToBuffer(texture: unknown, x: number, y: number, w: number, h: number, face: number): Promise<{ readonly length: number; readonly [index: number]: number }>;
    };
    const backend = this.renderer.backend as unknown as ReadbackBackend;
    const sum = async (field: GridField): Promise<number> => {
      const data = await backend.copyTextureToBuffer(field.main, 0, 0, FLAT.gridW, FLAT.gridH, 0);
      let total = 0;
      for (let i = 0; i < data.length; i++) total += Number(data[i]);
      return total;
    };
    const [height, loose, water, sediment] = await Promise.all([
      sum(this.heightField),
      sum(this.sim.loose),
      sum(this.sim.water),
      sum(this.sim.sediment),
    ]);
    const earth = height + sediment;
    const earthDelta = this.previousEarthMass === null ? 0 : earth - this.previousEarthMass;
    this.previousEarthMass = earth;
    this.snapshotText = `mass earth:${earth.toFixed(1)} Δ${earthDelta.toExponential(2)} loose:${loose.toFixed(1)} water:${water.toFixed(1)} sed:${sediment.toFixed(1)}`;
    console.table({ earth, earthDelta, height, loose, water, sediment });
  }

  private resolveGpuTimings(): void {
    if (this.timingPending) return;
    this.timingPending = true;
    void Promise.all([
      this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE),
      this.renderer.resolveTimestampsAsync(TimestampQuery.RENDER),
    ]).then(([compute, render]) => {
      this.gpuComputeMs = compute ?? 0;
      this.gpuRenderMs = render ?? 0;
    }).finally(() => {
      this.timingPending = false;
    });
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
      case 'v': this.cycleDebug(); break;
    }
    this.sidebar?.sync();
  };

  start(): void { this.renderer.setAnimationLoop(this.frame); }

  private frame = (time: number): void => {
    if (this.lastTime === 0) this.lastTime = time;
    const dt = Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;

    this.sim.setRain(this.water.rainOn ? this.water.rainRate : 0);
    const simStart = performance.now();
    this.simAccum += dt;
    let steps = 0;
    while (this.simAccum >= this.simInterval && steps < SIM.maxStepsPerFrame) {
      this.sim.tick(this.simInterval); this.simAccum -= this.simInterval; steps++;
    }
    if (this.simAccum > this.simInterval * SIM.maxStepsPerFrame) this.simAccum = 0;
    const simCpuMs = performance.now() - simStart;
    this.simCpuMsEma += (simCpuMs - this.simCpuMsEma) * 0.1;

    this.orbit.update();
    const computeCalls = this.renderer.info.compute.calls;
    const computeDelta = computeCalls - this.lastComputeCalls;
    if (computeDelta > 0) this.computeCallsFrame = computeDelta;
    this.lastComputeCalls = computeCalls;
    this.renderer.render(this.scene, this.camera);
    if (time - this.lastTimingResolve > 1000) {
      this.lastTimingResolve = time;
      this.resolveGpuTimings();
    }
    if (this.hud) {
      this.fpsEma += (1000 / Math.max(dt * 1000, 0.001) - this.fpsEma) * 0.1;
      this.hud.textContent =
        `fps ${this.fpsEma.toFixed(0)}  FLAT ${FLAT.gridW}x${FLAT.gridH}  cpu-sim ${this.simCpuMsEma.toFixed(2)}ms  gpu compute/render ${this.gpuComputeMs.toFixed(2)}/${this.gpuRenderMs.toFixed(2)}ms  dispatch ${this.computeCallsFrame}\n` +
        `benchmark:${this.diagnostics.benchmark}  solver:${this.sim.waterSolver}${this.sim.waterSolver === 'momentum' ? ` x${this.sim.momentumSubsteps}` : ''}  debug:${FLAT_DEBUG_MODES[this.debugMode]} [v]  [g]${this.sculptMode ? 'SCULPT' : 'orbit'} ${this.riverMode ? 'RIVER' : this.brushSettings.mode} [r]rain:${this.water.rainOn ? 'on' : 'off'}\n` +
        this.snapshotText;
    }
  };

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
