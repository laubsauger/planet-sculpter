// Equirectangular planet engine (pivot from cube-sphere). ONE grid, no seams.
// Stage 2/2b: seamless displaced+colored sphere + orbit + lights + sculpt brush
// + picking + cursor + minimal GUI. Water/erosion port in following stages.

import {
  Scene,
  PerspectiveCamera,
  DirectionalLight,
  HemisphereLight,
  Color,
  Vector2,
  Vector3,
  Mesh,
  RingGeometry,
  MeshBasicMaterial,
  DoubleSide,
  Raycaster,
  Sphere,
  Matrix4,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
import GUI from 'lil-gui';
import { OrbitController } from './OrbitController';
import { SphereMesh } from '../planet/sphereMesh';
import { makeGridTerrain } from '../materials/gridTerrain';
import { detailStrength, detailFreq } from '../tsl/gridSurface';
import {
  buildGridHeight,
  buildGridLoose,
  buildGridRainfall,
  buildGridHardness,
  buildGridCellArea,
  flattenPoleCaps,
} from '../planet/gridSeed';
import { GridField, buildGridSeed } from '../sim/gridStore';
import { GridSim } from '../sim/GridSim';
import { makeGridWater } from '../materials/gridWater';
import { GridBrush } from '../tools/GridBrush';
import { Sidebar } from '../ui/Sidebar';
import { waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import { evapFlowReduce, evapSpeedRef, evapDeepReduce, evapDeepRef, rainOrographic, rainHighRef } from '../sim/gridWater';
import { dirToLonLat } from '../planet/latlong';
import { lightingSettings, sunDirection, sunDirUniform, sunIntensityU, fillU, ambientU } from '../tsl/lighting';
import { heightScaleUniform } from '../tsl/heightScale';
import { PLANET, SIM, RENDER } from '../config';
import type { BrushMode } from '../tools/BrushTool';

const CURSOR_Z = new Vector3(0, 0, 1);

export class GridEngine {
  readonly renderer: WebGPURenderer;
  private readonly scene = new Scene();
  private readonly camera: PerspectiveCamera;
  private readonly orbit: OrbitController;
  private planet!: SphereMesh;
  private heightField!: GridField;
  private heightData!: Float32Array;
  private brush!: GridBrush;
  private sim!: GridSim;
  private cursor!: Mesh;
  private sidebar!: Sidebar;
  private simAccumulator = 0;
  private readonly simInterval = 1 / SIM.ticksPerSecond;
  readonly waterSettings = { rainOn: false, rainRate: SIM.rainRate };
  readonly riverSettings = { rate: 0.02, radius: 0.06 };
  private riverMode = false;
  private volcanoMode = false;
  private sun!: DirectionalLight;
  private fillLight!: DirectionalLight;
  private skyLight!: HemisphereLight;
  private hud: HTMLElement | null;
  private fpsEma = RENDER.targetFps;
  private lastTime = 0;

  private readonly ndc = new Vector2();
  private sculptMode = false;
  private brushing = false;
  private brushDirty = false;
  private readonly canvas: HTMLCanvasElement;
  readonly brushSettings = { mode: 'raise' as BrushMode, radius: 0.13, strength: 0.02, rate: 0.4, target: 0.35 };

  private readonly raycaster = new Raycaster();
  private readonly sphere = new Sphere(new Vector3(0, 0, 0), PLANET.baseRadius);
  private readonly invMatrix = new Matrix4();
  private readonly hitWorld = new Vector3();
  private readonly hitLocal = new Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new WebGPURenderer({ canvas, antialias: true });
    this.scene.background = new Color(0x0b0d12);
    this.camera = new PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
    this.camera.position.set(0, PLANET.baseRadius * 1.5, PLANET.baseRadius * 3);
    this.orbit = new OrbitController(this.camera, canvas);
    this.hud = document.getElementById('hud');

    this.sun = new DirectionalLight(0xfff2e0, lightingSettings.sunIntensity);
    this.fillLight = new DirectionalLight(0x88a0c0, lightingSettings.fill);
    this.skyLight = new HemisphereLight(0x9fc5ff, 0x3a2f25, lightingSettings.ambient);
    this.scene.add(this.sun, this.fillLight, this.skyLight);
    this.applyLighting();
    window.addEventListener('resize', this.onResize);
  }

  private applyLighting = (): void => {
    const dir = sunDirection(lightingSettings);
    this.sun.position.copy(dir).multiplyScalar(10);
    this.sun.intensity = lightingSettings.sunIntensity;
    this.fillLight.position.copy(dir).multiplyScalar(-10);
    this.fillLight.intensity = lightingSettings.fill;
    this.skyLight.intensity = lightingSettings.ambient;
    sunDirUniform.value.set(dir.x, dir.y, dir.z);
    sunIntensityU.value = lightingSettings.sunIntensity;
    fillU.value = lightingSettings.fill;
    ambientU.value = lightingSettings.ambient;
  };

  /** CPU height [0,1] at a planet-local direction (pick refine + cursor). */
  private heightAt = (dir: Vector3): number => {
    const { u, v } = dirToLonLat([dir.x, dir.y, dir.z]);
    const W = PLANET.lonRes;
    const H = PLANET.latRes;
    const tx = ((Math.round(u * W) % W) + W) % W;
    const ty = Math.min(H - 1, Math.max(0, Math.round(v * (H - 1))));
    return this.heightData[ty * W + tx];
  };

  async init(): Promise<void> {
    await this.renderer.init();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const W = PLANET.lonRes;
    const H = PLANET.latRes;
    const hSeed = buildGridHeight(W, H);
    flattenPoleCaps(hSeed, 0.3); // clean polar caps (kill the pole-pinch spike)
    this.heightData = hSeed.data;
    const loose = buildGridLoose(W, H).texture;
    const rain = buildGridRainfall(W, H).texture;
    const hard = buildGridHardness(W, H).texture;
    const area = buildGridCellArea(W, H).texture;

    // height in a storage texture so the brush can edit it.
    this.heightField = new GridField(W, H);
    this.renderer.compute(buildGridSeed(hSeed.texture, this.heightField.main, W, H));
    this.brush = new GridBrush(this.heightField.main, this.heightField.scratch, W, H);

    // water + erosion sim. owns the loose-material field (erosion mutates it),
    // so the terrain material reads sim.loose for live soil/rock changes.
    this.sim = new GridSim(this.renderer, this.heightField, rain, area, loose, hard, W, H);

    // render mesh tessellated DENSER than the sim grid so the smooth height interp
    // + detail noise produce real sub-grid geometry (⊥ collapsing onto coarse grid).
    const lonSeg = Math.round(W * RENDER.meshDetail);
    const latSeg = Math.round((H - 1) * RENDER.meshDetail);
    const mat = makeGridTerrain(this.heightField.main, this.sim.loose.main, rain, hard);
    this.planet = new SphereMesh(lonSeg, latSeg, PLANET.baseRadius, mat);
    this.scene.add(this.planet.group);

    // transparent water mesh (depth = vol/area, in-shader normal).
    const waterMesh = new SphereMesh(lonSeg, latSeg, PLANET.baseRadius, makeGridWater(this.heightField.main, this.sim.water.main, area));
    waterMesh.mesh.renderOrder = 1;
    this.scene.add(waterMesh.group);

    // surface cursor (ring on the terrain).
    const cmat = new MeshBasicMaterial({
      color: 0x6fe8ff, transparent: true, opacity: 0.85, side: DoubleSide, depthWrite: false, depthTest: false,
    });
    this.cursor = new Mesh(new RingGeometry(0.78, 1.0, 48), cmat);
    this.cursor.renderOrder = 11;
    this.cursor.visible = false;
    this.planet.group.add(this.cursor);

    this.buildGui();
    this.sidebar = new Sidebar({
      brush: this.brushSettings,
      isSculpt: () => this.sculptMode,
      setSculpt: (on) => { this.sculptMode = on; this.orbit.controls.enableRotate = !on; },
      isRiver: () => this.riverMode,
      setRiver: (on) => { this.riverMode = on; if (on) this.volcanoMode = false; },
      isVolcano: () => this.volcanoMode,
      setVolcano: (on) => { this.volcanoMode = on; if (on) this.riverMode = false; },
      isRain: () => this.waterSettings.rainOn,
      toggleRain: () => { this.waterSettings.rainOn = !this.waterSettings.rainOn; },
    });
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    this.canvas.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('keydown', this.onKeyDown);
  }

  private buildGui(): void {
    const gui = new GUI({ title: 'Planet (equirect)' });
    const b = gui.addFolder('Brush');
    b.add(this.brushSettings, 'mode', ['raise', 'lower', 'smooth', 'flatten']);
    b.add(this.brushSettings, 'radius', 0.02, 0.4, 0.005);
    b.add(this.brushSettings, 'strength', 0, 0.1, 0.001);
    b.add(this.brushSettings, 'rate', 0, 1, 0.01);
    b.add(this.brushSettings, 'target', 0, 1, 0.01);
    const l = gui.addFolder('Lighting');
    l.add(lightingSettings, 'azimuth', -Math.PI, Math.PI, 0.02).onChange(this.applyLighting);
    l.add(lightingSettings, 'elevation', -1.4, 1.4, 0.02).onChange(this.applyLighting);
    l.add(lightingSettings, 'sunIntensity', 0, 4, 0.05).onChange(this.applyLighting);
    l.add(lightingSettings, 'fill', 0, 1.5, 0.05).onChange(this.applyLighting);
    l.add(lightingSettings, 'ambient', 0, 1.5, 0.05).onChange(this.applyLighting);
    const d = gui.addFolder('Detail');
    d.add(detailStrength, 'value', 0, 0.15, 0.002).name('detail strength');
    d.add(detailFreq, 'value', 6, 80, 1).name('detail freq');
    const wf = gui.addFolder('Water');
    wf.add(this.waterSettings, 'rainOn').name('rain').onChange(() => this.sidebar?.sync());
    wf.add(this.waterSettings, 'rainRate', 0, 0.02, 0.0005).name('rain rate');
    wf.add(rainOrographic, 'value', 0, 1, 0.02).name('orographic rain');
    wf.add(rainHighRef, 'value', 0.05, 0.6, 0.01).name('oro height ref');
    wf.add(waterUniforms.evapProp, 'value', 0, 0.4, 0.005).name('evaporation /s');
    wf.add(evapFlowReduce, 'value', 0, 1, 0.02).name('flow evap reduce');
    wf.add(evapSpeedRef, 'value', 0.5, 5, 0.1).name('flow evap speed');
    wf.add(evapDeepReduce, 'value', 0, 1, 0.02).name('deep evap reduce');
    wf.add(evapDeepRef, 'value', 0.01, 0.2, 0.005).name('deep evap depth');
    wf.add(this.riverSettings, 'rate', 0, 0.06, 0.002).name('river rate');
    wf.add(this.riverSettings, 'radius', 0.02, 0.2, 0.005).name('river radius');
    wf.add({ clear: () => this.sim.clearWater() }, 'clear').name('clear water');
    wf.add({ clearSrc: () => this.sim.clearSources() }, 'clearSrc').name('clear sources');

    const e = gui.addFolder('Erosion');
    e.add(this.sim, 'erosionEnabled').name('enabled');
    e.add(erosionUniforms.simSpeed, 'value', 1, 12, 0.5).name('sim speed');
    e.add(erosionUniforms.sedimentCapacity, 'value', 0, 1, 0.02).name('capacity Kc');
    e.add(erosionUniforms.dissolve, 'value', 0, 0.5, 0.01).name('dissolve Ks');
    e.add(erosionUniforms.deposit, 'value', 0, 0.3, 0.01).name('deposit Kd');
    e.add(erosionUniforms.channelFocus, 'value', 0, 1, 0.02).name('channel focus');
    e.add(erosionUniforms.lateralErosion, 'value', 0, 1, 0.02).name('lateral cut');
    e.add(erosionUniforms.erodeDeepDepth, 'value', 0.02, 0.3, 0.005).name('deep no-erode');
    e.add(erosionUniforms.thermalRate, 'value', 0, 1, 0.02).name('thermal rate');
    e.add(erosionUniforms.talus, 'value', 0.002, 0.06, 0.002).name('talus (repose)');
    e.add(erosionUniforms.strataStrength, 'value', 0, 1, 0.02).name('strata (elev)');
    e.add(erosionUniforms.hardness3dFreq, 'value', 0.5, 10, 0.1).name('hardness 3D freq');
    e.add(erosionUniforms.hardness3dAmp, 'value', 0, 1.5, 0.02).name('hardness 3D amp');
  }

  private setNdc(e: PointerEvent): void {
    this.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  }

  /** Ray -> displaced surface direction (refined onto the height). */
  private pick(): Vector3 | null {
    this.raycaster.setFromCamera(this.ndc, this.camera);
    this.sphere.radius = PLANET.baseRadius;
    if (!this.raycaster.ray.intersectSphere(this.sphere, this.hitWorld)) return null;
    this.invMatrix.copy(this.planet.group.matrixWorld).invert();
    this.hitLocal.copy(this.hitWorld).applyMatrix4(this.invMatrix).normalize();
    for (let k = 0; k < 3; k++) {
      this.sphere.radius = PLANET.baseRadius + this.heightAt(this.hitLocal) * heightScaleUniform.value;
      if (!this.raycaster.ray.intersectSphere(this.sphere, this.hitWorld)) break;
      this.hitLocal.copy(this.hitWorld).applyMatrix4(this.invMatrix).normalize();
    }
    return this.hitLocal.clone();
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.sculptMode || e.button !== 0) return;
    this.setNdc(e);
    // river tool: click places an emitter spring (no terrain edit).
    if (this.riverMode) {
      const dir = this.pick();
      if (dir) this.sim.placeSource(dir, this.riverSettings.rate, this.riverSettings.radius);
      return;
    }
    if (this.volcanoMode) return; // lava port pending (Stage 8)
    this.brushing = true;
    this.brushDirty = true;
  };
  private onPointerMove = (e: PointerEvent): void => {
    this.setNdc(e);
    if (this.brushing) this.brushDirty = true;
  };
  private onPointerUp = (): void => {
    this.brushing = false;
  };
  private onKeyDown = (e: KeyboardEvent): void => {
    switch (e.key) {
      case 'g': this.sculptMode = !this.sculptMode; this.orbit.controls.enableRotate = !this.sculptMode; break;
      case '1': this.brushSettings.mode = 'raise'; this.riverMode = this.volcanoMode = false; break;
      case '2': this.brushSettings.mode = 'lower'; this.riverMode = this.volcanoMode = false; break;
      case '3': this.brushSettings.mode = 'smooth'; this.riverMode = this.volcanoMode = false; break;
      case '4': this.brushSettings.mode = 'flatten'; this.riverMode = this.volcanoMode = false; break;
      case 'r': this.waterSettings.rainOn = !this.waterSettings.rainOn; break;
    }
    this.sidebar?.sync();
  };

  private updateCursor(dir: Vector3 | null): void {
    if (!this.sculptMode || !dir) {
      this.cursor.visible = false;
      return;
    }
    this.cursor.visible = true;
    const r = PLANET.baseRadius + this.heightAt(dir) * heightScaleUniform.value;
    this.cursor.position.copy(dir).multiplyScalar(r + 0.012);
    this.cursor.quaternion.setFromUnitVectors(CURSOR_Z, dir);
    const w = Math.max(0.05, this.brushSettings.radius * PLANET.baseRadius);
    this.cursor.scale.set(w, w, w);
  }

  start(): void {
    this.renderer.setAnimationLoop(this.frame);
  }

  private frame = (time: number): void => {
    if (this.lastTime === 0) this.lastTime = time;
    const dt = Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;

    const dir = this.sculptMode ? this.pick() : null;
    if (this.brushing && this.brushDirty && dir) {
      this.brush.stamp(this.renderer, dir, this.brushSettings);
      this.brushDirty = false;
    }
    this.updateCursor(dir);

    // fixed-step water sim.
    this.sim.setRain(this.waterSettings.rainOn ? this.waterSettings.rainRate : 0);
    this.simAccumulator += dt;
    let steps = 0;
    while (this.simAccumulator >= this.simInterval && steps < SIM.maxStepsPerFrame) {
      this.sim.tick(this.simInterval);
      this.simAccumulator -= this.simInterval;
      steps++;
    }
    if (this.simAccumulator > this.simInterval * SIM.maxStepsPerFrame) this.simAccumulator = 0;

    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
    if (this.hud) {
      this.fpsEma += (1000 / Math.max(dt * 1000, 0.001) - this.fpsEma) * 0.1;
      this.hud.textContent =
        `fps ${this.fpsEma.toFixed(0)}  EQUIRECT ${PLANET.lonRes}x${PLANET.latRes}  ` +
        `[g] ${this.sculptMode ? 'SCULPT' : 'orbit'}  brush:${this.brushSettings.mode} [1-4]`;
    }
  };

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
}
