// Dev control panel (lil-gui). Live-drives sim + brush params (partial T18).
// Decoupled from Engine via plain handles to avoid an import cycle.

import GUI from 'lil-gui';
import { waterUniforms } from '../sim/passes/water';
import { erosionUniforms } from '../sim/passes/erosion';
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

export interface ControlHandles {
  brush: BrushSettings;
  water: WaterSettings;
  erosion: ErosionSettings;
  onRainChange: () => void;
  onClearWater: () => void;
  onErosionChange: () => void;
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
    water.add(h.water, 'rainOn').name('rain').onChange(h.onRainChange);
    water.add(h.water, 'rainRate', 0, 0.02, 0.0005).name('rain intensity').onChange(h.onRainChange);
    water.add(waterUniforms.loss, 'value', 0, 0.01, 0.0001).name('evaporation');
    water.add(waterUniforms.pipeArea, 'value', 0.2, 6, 0.1).name('flow speed');
    water.add(waterUniforms.gravity, 'value', 1, 20, 0.5).name('gravity');
    water.add({ clear: h.onClearWater }, 'clear').name('clear water');

    const ero = this.gui.addFolder('Erosion');
    ero.add(h.erosion, 'enabled').name('enabled').onChange(h.onErosionChange);
    ero.add(erosionUniforms.sedimentCapacity, 'value', 0, 2, 0.05).name('capacity (Kc)');
    ero.add(erosionUniforms.dissolve, 'value', 0, 1, 0.02).name('dissolve (Ks)');
    ero.add(erosionUniforms.deposit, 'value', 0, 1, 0.02).name('deposit (Kd)');
    ero.close();
  }

  dispose(): void {
    this.gui.destroy();
  }
}
