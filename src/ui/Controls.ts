// Dev control panel (lil-gui). Live-drives sim + brush params (partial T18).
// Decoupled from Engine via plain handles to avoid an import cycle.

import GUI from 'lil-gui';
import { waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
import { lightingSettings } from '../tsl/lighting';
import { cloudCoverage, cloudOpacity, cloudScale, windSpeed, storminess } from '../materials/cloudMaterial';
import { atmosphereStrength } from '../materials/atmosphereMaterial';
import { rainStrength } from '../materials/rainMaterial';
import { setSeaLevel } from '../tsl/heightScale';
import { PLANET } from '../config';
import type { BrushMode } from '../tools/BrushTool';

export interface BrushSettings {
  mode: BrushMode;
  radius: number;
  strength: number;
  rate: number;
  target: number;
}

export interface WaterSettings {
  rainOn: boolean;
  rainRate: number;
}

export interface ErosionSettings {
  enabled: boolean;
}

export interface RiverSettings {
  rate: number;
  radius: number;
}

export interface ControlHandles {
  brush: BrushSettings;
  water: WaterSettings;
  river: RiverSettings;
  erosion: ErosionSettings;
  onRainChange: () => void;
  onClearWater: () => void;
  onClearSources: () => void;
  onErosionChange: () => void;
  onLightingChange: () => void;
}

export class Controls {
  readonly gui = new GUI({ title: 'Planet Sculptor' });

  constructor(h: ControlHandles) {
    const brush = this.gui.addFolder('Brush');
    brush.add(h.brush, 'mode', ['raise', 'lower', 'smooth', 'flatten']);
    brush.add(h.brush, 'radius', 0.02, 0.4, 0.005);
    brush.add(h.brush, 'strength', 0, 0.1, 0.001);
    brush.add(h.brush, 'rate', 0, 1, 0.01).name('smooth/flatten rate');
    brush.add(h.brush, 'target', 0, 1, 0.01).name('flatten target');

    const water = this.gui.addFolder('Water');
    water.add({ sea: PLANET.seaLevel }, 'sea', 0, 0.6, 0.005).name('sea level').onChange(setSeaLevel);
    water.add(h.water, 'rainOn').name('rain').onChange(h.onRainChange);
    water.add(h.water, 'rainRate', 0, 0.02, 0.0005).name('rain intensity').onChange(h.onRainChange);
    water.add(waterUniforms.loss, 'value', 0, 0.01, 0.0001).name('evaporation');
    water.add(waterUniforms.pipeArea, 'value', 0.2, 6, 0.1).name('flow speed');
    water.add(waterUniforms.gravity, 'value', 1, 20, 0.5).name('gravity');
    water.add(waterUniforms.damping, 'value', 0.5, 0.99, 0.01).name('damping');
    water.add(h.river, 'rate', 0.002, 0.1, 0.002).name('river rate');
    water.add(h.river, 'radius', 0.002, 0.05, 0.001).name('river radius');
    water.add({ clear: h.onClearWater }, 'clear').name('clear water');
    water.add({ clear: h.onClearSources }, 'clear').name('clear river sources');

    const ero = this.gui.addFolder('Erosion');
    ero.add(h.erosion, 'enabled').name('enabled').onChange(h.onErosionChange);
    ero.add(erosionUniforms.sedimentCapacity, 'value', 0, 2, 0.05).name('capacity (Kc)');
    ero.add(erosionUniforms.dissolve, 'value', 0, 1, 0.02).name('dissolve (Ks)');
    ero.add(erosionUniforms.deposit, 'value', 0, 1, 0.02).name('deposit (Kd)');
    ero.add(erosionUniforms.erodeSpeedMin, 'value', 0, 2, 0.05).name('erode speed min');
    ero.add(erosionUniforms.talus, 'value', 0, 0.05, 0.001).name('talus');
    ero.add(erosionUniforms.thermalRate, 'value', 0, 1, 0.05).name('thermal rate');
    ero.add(erosionUniforms.rockErodibility, 'value', 0.02, 1, 0.02).name('rock erodibility');
    ero.add(erosionUniforms.looseFull, 'value', 0.005, 0.1, 0.005).name('loose cover depth');
    ero.add(erosionUniforms.channelFocus, 'value', 0, 1, 0.02).name('channel focus');
    ero.add(erosionUniforms.channelDischarge, 'value', 0.002, 0.06, 0.002).name('channel discharge');
    ero.add(erosionUniforms.flowInertia, 'value', 0, 0.95, 0.02).name('flow inertia (meander)');
    ero.add(erosionUniforms.lateralErosion, 'value', 0, 2, 0.05).name('lateral erosion');
    ero.add(erosionUniforms.strataFreq, 'value', 10, 120, 5).name('rock layers (freq)');
    ero.add(erosionUniforms.strataStrength, 'value', 0, 1, 0.05).name('rock layer hardness');
    ero.close();

    const light = this.gui.addFolder('Lighting');
    light.add(lightingSettings, 'azimuth', -Math.PI, Math.PI, 0.02).name('sun azimuth').onChange(h.onLightingChange);
    light.add(lightingSettings, 'elevation', -1.4, 1.4, 0.02).name('sun elevation').onChange(h.onLightingChange);
    light.add(lightingSettings, 'sunIntensity', 0, 4, 0.05).name('sun intensity').onChange(h.onLightingChange);
    light.add(lightingSettings, 'fill', 0, 1.5, 0.05).name('fill').onChange(h.onLightingChange);
    light.add(lightingSettings, 'ambient', 0, 1.5, 0.05).name('dark-side fill').onChange(h.onLightingChange);
    light.close();

    const weather = this.gui.addFolder('Weather');
    weather.add(storminess, 'value', 0, 1, 0.02).name('storminess');
    weather.add(cloudCoverage, 'value', 0, 1, 0.02).name('cloud coverage');
    weather.add(cloudOpacity, 'value', 0, 1, 0.02).name('cloud opacity');
    weather.add(cloudScale, 'value', 1, 10, 0.5).name('cloud scale');
    weather.add(windSpeed, 'value', 0, 0.03, 0.001).name('wind speed');
    weather.add(rainStrength, 'value', 0, 1, 0.02).name('rain');
    weather.add(atmosphereStrength, 'value', 0, 3, 0.05).name('atmosphere');
    weather.close();
  }

  dispose(): void {
    this.gui.destroy();
  }
}
