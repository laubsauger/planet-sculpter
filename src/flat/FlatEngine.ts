// Flat From-Dust-style engine. Crisp detail-normal terrain + pipe-model water +
// erosion on a uniform heightfield (no sphere/pole/seam machinery). Sculpt brush,
// river springs, rain, realistic water with flow normals.

import {
  Scene, PerspectiveCamera, DirectionalLight, HemisphereLight, AmbientLight, Color, Fog,
  Vector2, Vector3, Raycaster, Mesh, TimestampQuery, DataTexture, RedFormat, FloatType, Object3D,
  PlaneGeometry, RingGeometry, MeshBasicMaterial, DoubleSide,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
import GUI from 'lil-gui';
import { OrbitController } from '../app/OrbitController';
import { buildFlatSeed } from './flatSeed';
import { buildFlatMesh, buildOceanSkirt } from './flatMesh';
import { makeFlatTerrain, shoreWetEnabled, materialDebugGrid, contourOverlay, contourCount, causticsEnabled } from '../materials/flatTerrain';
import { flowBandScale, flowBandStrength, makeFlatWater, shoreFoamEnabled, oceanSwellEnabled } from '../materials/flatWater';
import { makeFlatDebug, FLAT_DEBUG_MODES, flatDebugMode } from '../materials/flatDebug';
import { FlatBrush } from './FlatBrush';
import { FlatSim, FLAT_WATER_SOLVERS, depositionEnabled, hydraulicErosionEnabled } from './flatSim';
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
  private sunTarget!: Object3D;
  private fill!: DirectionalLight;
  private sky!: HemisphereLight;
  private ambient!: AmbientLight;
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
  private lastRenderCalls = 0;
  private renderCallsFrame = 0;
  private lastTriangles = 0;
  private trianglesFrame = 0;
  private debugMode = 0;
  private readonly diagnostics = { benchmark: 'default' as FlatBenchmark };
  private snapshotText = '';
  private previousEarthMass: number | null = null;
  private readonly simInterval = 1 / SIM.ticksPerSecond;
  private readonly canvas: HTMLCanvasElement;

  private readonly ndc = new Vector2();
  private readonly raycaster = new Raycaster();
  private readonly hit = new Vector3();
  private sculptMode = false;
  private brushing = false;
  private riverMode = false;
  // Shadow map renders ON DEMAND: terrain geometry only changes on sim ticks /
  // brush stamps / height-scale edits, so re-rendering the 2048² map every frame
  // (full 262k-vert depth pass) was pure waste. Identical image, just not redrawn
  // when nothing moved.
  private shadowDirty = true;
  private readonly brushSettings = { mode: 'raise' as BrushMode, radius: 0.035, strength: 0.012, rate: 0.4, target: 0.4 };
  private readonly water = { rainOn: false, rainRate: SIM.rainRate };
  private readonly river = { rate: 0.5, radius: 0.009 };
  // CPU mirror of the height field for accurate ray->surface picking (throttled readback).
  private heightMirror: Float32Array | null = null;
  private mirrorPending = false;
  private lastMirrorTime = 0;
  private cursor!: Mesh;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new WebGPURenderer({ canvas, antialias: true, trackTimestamp: true });
    this.scene.background = new Color(FlatEngine.SCENE_BACKGROUND);
    this.scene.fog = new Fog(FlatEngine.SCENE_BACKGROUND, FLAT.worldSize * 1.5, FLAT.worldSize * 4.2);
    this.camera = new PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.05, 200);
    this.camera.position.set(0, FLAT.worldSize * 0.7, FLAT.worldSize * 0.85);
    this.orbit = new OrbitController(this.camera, canvas);
    this.orbit.controls.target.set(0, 0, 0);
    this.hud = document.getElementById('hud');

    this.sun = new DirectionalLight(0xfff7e8, 3.2);
    this.sun.position.set(6, 9, 4);
    this.sunTarget = new Object3D();
    this.sunTarget.position.set(0, 0, 0);
    this.sun.target = this.sunTarget;
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const shadowExtent = FLAT.worldSize * 0.7;
    this.sun.shadow.camera.left = -shadowExtent;
    this.sun.shadow.camera.right = shadowExtent;
    this.sun.shadow.camera.top = shadowExtent;
    this.sun.shadow.camera.bottom = -shadowExtent;
    this.sun.shadow.camera.near = 0.1;
    this.sun.shadow.camera.far = FLAT.worldSize * 4;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.fill = new DirectionalLight(0xd9e5ee, 0.12);
    this.fill.position.set(-5, 4, -3);
    this.sky = new HemisphereLight(0xdceeff, 0xb5ad9d, 0.72);
    this.ambient = new AmbientLight(0xffffff, 0.1);
    this.scene.add(this.sun, this.sunTarget, this.fill, this.sky, this.ambient);
    this.addOceanContinuation();
    // world-space lighting uniforms for the unlit terrain (camera-independent).
    sunDirUniform.value.copy(this.sun.position).normalize();
    sunIntensityU.value = 2.8;
    ambientU.value = 0.55;
    window.addEventListener('resize', this.onResize);
  }

  private addOceanContinuation(): void {
    // Dark ocean FLOOR far below the deepest seabed, spanning past the ocean skirt. The
    // transparent deep skirt water reveals it (so the open ocean reads dark, not sky), while
    // INSIDE the grid the opaque terrain seabed always occludes it. Because it sits well below
    // every seabed cell it can never produce the old fixed-contour "hard blue line".
    const span = FLAT.worldSize * 12;
    const floorY = -FLAT.worldSize * 0.2; // safely beneath all seabed
    const floor = new Mesh(
      new PlaneGeometry(span, span),
      new MeshBasicMaterial({ color: 0x123047, side: DoubleSide }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = floorY;
    floor.renderOrder = -3;
    floor.frustumCulled = false;
    this.scene.add(floor);
  }

  async init(): Promise<void> {
    await this.renderer.init();
    this.renderer.shadowMap.enabled = true;
    this.sun.shadow.autoUpdate = false; // re-rendered only when shadowDirty (sim tick/brush)
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
      erosionOn: true,
    };
    this.heightField = new GridField(W, H);
    this.renderer.compute(buildGridSeed(seed.height.texture as never, this.heightField.main, W, H));
    this.brush = new FlatBrush(this.heightField.main, this.heightField.scratch, W, H);
    this.heightMirror = new Float32Array((seed.height.texture.image as { data: Float32Array }).data);

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
    this.terrainMesh.castShadow = true;
    this.terrainMesh.receiveShadow = true;
    this.scene.add(this.terrainMesh);

    const water = makeFlatWater(
      this.heightField.main,
      this.sim.water.main,
      this.sim.flux.main,
      this.sim.velocity.main,
      this.sim.sediment.main,
    );
    this.waterMesh = buildFlatMesh(mW, mH, water);
    this.waterMesh.renderOrder = 1;
    this.scene.add(this.waterMesh);
    // Same water material, extended to the horizon as a frame around the grid -> seamless
    // dynamic ocean (swell/colour) continuing past the sim, no static plane, no seam.
    const skirt = buildOceanSkirt(water, FLAT.worldSize, FLAT.worldSize * 4);
    this.scene.add(skirt);

    // Brush cursor: a thin ring laid on the terrain at the picked surface point, drawn on top
    // (depthTest off) so it's always visible. Scaled to the brush radius, tinted per tool.
    this.cursor = new Mesh(
      new RingGeometry(0.82, 1.0, 56),
      new MeshBasicMaterial({ color: 0x4ad6a0, transparent: true, opacity: 0.85, side: DoubleSide, depthTest: false, depthWrite: false }),
    );
    this.cursor.rotation.x = -Math.PI / 2;
    this.cursor.renderOrder = 20;
    this.cursor.frustumCulled = false;
    this.cursor.visible = false;
    this.scene.add(this.cursor);

    this.buildGui();
    this.sidebar = new Sidebar({
      brush: this.brushSettings,
      isSculpt: () => this.sculptMode,
      setSculpt: (on) => { this.sculptMode = on; this.orbit.controls.enableRotate = !on; if (!on) this.cursor.visible = false; },
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
    b.add({ sculpt: () => { this.sculptMode = !this.sculptMode; this.orbit.controls.enableRotate = !this.sculptMode; if (!this.sculptMode) this.cursor.visible = false; } }, 'sculpt').name('toggle sculpt [g]');
    const t = gui.addFolder('Terrain');
    t.add(flatHeightScale, 'value', 0.5, 5, 0.05).name('height scale').onChange(() => { this.shadowDirty = true; });
    t.add(flatSeaLevel, 'value', 0, 0.7, 0.01).name('sea level');
    t.add(detailStrength, 'value', 0, 1.5, 0.02).name('detail strength');
    t.add(detailFreq, 'value', 1, 24, 0.5).name('detail freq');
    t.add({ matGrid: false }, 'matGrid').name('material debug grid').onChange((v: boolean) => { materialDebugGrid.value = v ? 1 : 0; });
    t.add({ contours: true }, 'contours').name('contour lines').onChange((v: boolean) => { contourOverlay.value = v ? 1 : 0; });
    t.add(contourCount, 'value', 8, 120, 1).name('contour count');
    const wf = gui.addFolder('Water');
    wf.add(this.water, 'rainOn').name('rain [r]');
    wf.add(this.water, 'rainRate', 0, 0.004, 0.00005).name('rain rate');
    wf.add(this.sim, 'waterSolver', FLAT_WATER_SOLVERS).name('solver');
    wf.add(waterUniforms.evapProp, 'value', 0, 0.4, 0.005).name('evaporation /s');
    wf.add(mudViscosityFactor, 'value', 0, 10, 0.25).name('mud viscosity');
    wf.add(flowBandStrength, 'value', 0, 1, 0.02).name('flow band strength');
    wf.add(flowBandScale, 'value', 2, 20, 0.5).name('flow band scale');
    const fx = gui.addFolder('Water FX');
    fx.add({ caustics: true }, 'caustics').onChange((v: boolean) => { causticsEnabled.value = v ? 1 : 0; });
    fx.add({ shoreFoam: true }, 'shoreFoam').name('shore foam').onChange((v: boolean) => { shoreFoamEnabled.value = v ? 1 : 0; });
    fx.add({ oceanSwell: true }, 'oceanSwell').name('ocean swell').onChange((v: boolean) => { oceanSwellEnabled.value = v ? 1 : 0; });
    fx.add({ shoreWet: true }, 'shoreWet').name('lapping wet sand').onChange((v: boolean) => { shoreWetEnabled.value = v ? 1 : 0; });
    wf.add({ riverTool: () => { this.riverMode = !this.riverMode; if (this.riverMode && !this.sculptMode) { this.sculptMode = true; this.orbit.controls.enableRotate = false; } } }, 'riverTool').name('river source tool');
    wf.add(this.river, 'rate', 0, 4, 0.05).name('river discharge');
    wf.add(this.river, 'radius', 0.008, 0.08, 0.002).name('spring radius');
    wf.add({ clear: () => this.sim.clearWater() }, 'clear').name('clear water');
    wf.add({ cs: () => this.sim.clearSources() }, 'cs').name('clear sources');
    const e = gui.addFolder('Erosion');
    e.add(this.sim, 'erosionEnabled').name('enabled').listen();
    e.add({ hydraulic: true }, 'hydraulic').name('hydraulic erosion').onChange((v: boolean) => { hydraulicErosionEnabled.value = v ? 1 : 0; });
    e.add({ deposition: true }, 'deposition').name('deposition').onChange((v: boolean) => { depositionEnabled.value = v ? 1 : 0; });
    e.add(this.sim, 'thermalEnabled').name('thermal slumping');
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
    this.shadowDirty = true;
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
    const sum = async (field: GridField, rgba = false): Promise<number> => {
      const data = await backend.copyTextureToBuffer(field.main, 0, 0, FLAT.gridW, FLAT.gridH, 0);
      let total = 0;
      for (let i = 0; i < data.length; i += rgba ? 4 : 1) total += Number(data[i]);
      return total;
    };
    const [height, loose, water, sediment] = await Promise.all([
      sum(this.heightField),
      sum(this.sim.loose, true),
      sum(this.sim.water),
      sum(this.sim.sediment, true),
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

  /** Bilinear height (0..1) from the CPU mirror at uv. */
  private sampleHeight(u: number, v: number): number {
    const m = this.heightMirror;
    if (!m) return FLAT.seaLevel;
    const W = FLAT.gridW, H = FLAT.gridH;
    const fx = Math.min(W - 1, Math.max(0, u * (W - 1)));
    const fy = Math.min(H - 1, Math.max(0, v * (H - 1)));
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
    const tx = fx - x0, ty = fy - y0;
    const h0 = m[y0 * W + x0] * (1 - tx) + m[y0 * W + x1] * tx;
    const h1 = m[y1 * W + x0] * (1 - tx) + m[y1 * W + x1] * tx;
    return h0 * (1 - ty) + h1 * ty;
  }

  /** ndc -> RAYMARCH against the displaced heightfield -> uv of the FIRST (nearest) hit. A
   *  naive "intersect horizontal plane at the sampled height" iteration converges to where the
   *  ray exits to sea level BEHIND a peak, not the near face under the cursor — so we march
   *  forward from the camera and take the first step where the ray drops below the surface,
   *  then binary-refine. Sets `this.hit` to the surface world point (used by the cursor). */
  private pickUv(): { u: number; v: number } | null {
    this.raycaster.setFromCamera(this.ndc, this.camera);
    const o = this.raycaster.ray.origin, d = this.raycaster.ray.direction;
    const scale = flatHeightScale.value, size = FLAT.worldSize;
    const terrainAt = (t: number): number => {
      const u = (o.x + d.x * t) / size + 0.5, v = (o.z + d.z * t) / size + 0.5;
      if (u < 0 || u > 1 || v < 0 || v > 1) return -Infinity; // off-grid: nothing to hit
      return this.sampleHeight(u, v) * scale;
    };
    const maxT = size * 4, steps = 192;
    let tPrev = 0, hitT = -1;
    for (let i = 1; i <= steps; i++) {
      const t = (i / steps) * maxT;
      if (o.y + d.y * t < terrainAt(t)) { // dropped below surface -> crossed between tPrev..t
        let a = tPrev, b = t;
        for (let k = 0; k < 10; k++) {
          const m = (a + b) * 0.5;
          if (o.y + d.y * m < terrainAt(m)) b = m; else a = m;
        }
        hitT = (a + b) * 0.5; break;
      }
      tPrev = t;
    }
    if (hitT < 0) return null;
    this.hit.set(o.x + d.x * hitT, o.y + d.y * hitT, o.z + d.z * hitT);
    const u = this.hit.x / size + 0.5, v = this.hit.z / size + 0.5;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { u, v };
  }

  /** Throttled async readback to keep the CPU height mirror roughly in sync with the GPU
   *  field (which erosion mutates each tick). One pixel-perfect picking source, no hot-loop. */
  private resyncHeightMirror(time: number): void {
    if (this.mirrorPending || time - this.lastMirrorTime < 120) return;
    this.mirrorPending = true;
    this.lastMirrorTime = time;
    type RB = { copyTextureToBuffer(t: unknown, x: number, y: number, w: number, h: number, f: number): Promise<{ length: number;[i: number]: number }> };
    const backend = this.renderer.backend as unknown as RB;
    backend.copyTextureToBuffer(this.heightField.main, 0, 0, FLAT.gridW, FLAT.gridH, 0)
      .then((data) => {
        if (!this.heightMirror || this.heightMirror.length !== data.length) this.heightMirror = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) this.heightMirror[i] = Number(data[i]);
        this.mirrorPending = false;
      })
      .catch(() => { this.mirrorPending = false; });
  }

  private cursorColor(): number {
    if (this.riverMode) return 0x46c8ff;
    switch (this.brushSettings.mode) {
      case 'lower': return 0xff6b6b;
      case 'smooth': return 0x6ba8ff;
      case 'flatten': return 0xffd24a;
      default: return 0x4ad6a0; // raise
    }
  }

  private updateCursor(): void {
    const r = (this.riverMode ? this.river.radius : this.brushSettings.radius) * FLAT.worldSize;
    this.cursor.position.set(this.hit.x, this.hit.y + 0.04, this.hit.z);
    this.cursor.scale.set(r, r, r);
    (this.cursor.material as MeshBasicMaterial).color.setHex(this.cursorColor());
    this.cursor.visible = true;
  }
  private setNdc(e: PointerEvent): void {
    this.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  }

  private onDown = (e: PointerEvent): void => {
    if (!this.sculptMode || e.button !== 0) return;
    this.setNdc(e);
    const p = this.pickUv();
    if (!p) return;
    this.updateCursor();
    if (this.riverMode) { this.sim.placeSource(p.u, p.v, this.river.rate, this.river.radius); return; }
    this.brushing = true;
    this.brush.stamp(this.renderer, p.u, p.v, this.brushSettings);
    this.shadowDirty = true;
  };
  private onMove = (e: PointerEvent): void => {
    if (!this.sculptMode) { this.cursor.visible = false; return; }
    this.setNdc(e);
    const p = this.pickUv(); // also sets this.hit to the surface point
    if (!p) { this.cursor.visible = false; return; }
    this.updateCursor();
    if (this.brushing) { this.brush.stamp(this.renderer, p.u, p.v, this.brushSettings); this.shadowDirty = true; }
  };
  private onUp = (): void => { this.brushing = false; };
  private onKey = (e: KeyboardEvent): void => {
    switch (e.key) {
      case 'g': this.sculptMode = !this.sculptMode; this.orbit.controls.enableRotate = !this.sculptMode; if (!this.sculptMode) this.cursor.visible = false; break;
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
    if (steps > 0 || this.shadowDirty) {
      this.sun.shadow.needsUpdate = true;
      this.shadowDirty = false;
    }
    const simCpuMs = performance.now() - simStart;
    this.simCpuMsEma += (simCpuMs - this.simCpuMsEma) * 0.1;

    this.resyncHeightMirror(time);
    this.orbit.update();
    const computeCalls = this.renderer.info.compute.calls;
    const computeDelta = computeCalls - this.lastComputeCalls;
    if (computeDelta > 0) this.computeCallsFrame = computeDelta;
    this.lastComputeCalls = computeCalls;
    this.renderer.render(this.scene, this.camera);
    const renderCalls = this.renderer.info.render.calls;
    const triangles = this.renderer.info.render.triangles;
    const renderDelta = renderCalls - this.lastRenderCalls;
    const triangleDelta = triangles - this.lastTriangles;
    if (renderDelta > 0) this.renderCallsFrame = renderDelta;
    if (triangleDelta > 0) this.trianglesFrame = triangleDelta;
    this.lastRenderCalls = renderCalls;
    this.lastTriangles = triangles;
    if (time - this.lastTimingResolve > 1000) {
      this.lastTimingResolve = time;
      this.resolveGpuTimings();
    }
    if (this.hud) {
      this.fpsEma += (1000 / Math.max(dt * 1000, 0.001) - this.fpsEma) * 0.1;
      this.hud.textContent =
        `fps ${this.fpsEma.toFixed(0)}  FLAT ${FLAT.gridW}x${FLAT.gridH}  cpu-sim ${this.simCpuMsEma.toFixed(2)}ms  gpu compute/render ${this.gpuComputeMs.toFixed(2)}/${this.gpuRenderMs.toFixed(2)}ms  dispatch ${this.computeCallsFrame}  draw ${this.renderCallsFrame}  tris ${Math.round(this.trianglesFrame / 1000)}k\n` +
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
