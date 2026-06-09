// Bootstrap: WebGPU gate (I.gate) -> engine.init (V8) -> loop.
// PIVOT: flat From-Dust-style FlatEngine replaces the sphere GridEngine (poles +
// equirect distortion not worth it; flat = all res in the patch, crisp, no seams).
import { FlatEngine } from './flat/FlatEngine';

async function main(): Promise<void> {
  const gate = document.getElementById('gate')!;
  const canvas = document.getElementById('app') as HTMLCanvasElement;

  if (!('gpu' in navigator)) {
    gate.style.display = 'flex';
    canvas.style.display = 'none';
    return;
  }

  try {
    const engine = new FlatEngine(canvas);
    await engine.init();
    engine.start();
    if (import.meta.env.DEV) {
      const { erosionUniforms } = await import('./sim/passes/erosion');
      const { waterUniforms } = await import('./sim/passes/water');
      const { shoreWetEnabled, materialDebugGrid, contourOverlay, causticsEnabled } = await import('./materials/flatTerrain');
      const fw = await import('./materials/flatWater');
      Object.assign(window as object, {
        engine, erosionUniforms, waterUniforms, shoreWetEnabled, materialDebugGrid, contourOverlay,
        causticsEnabled, shoreFoamEnabled: fw.shoreFoamEnabled, oceanSwellEnabled: fw.oceanSwellEnabled,
      });
    }
  } catch (err) {
    console.error('Init failed:', err);
    gate.style.display = 'flex';
    gate.querySelector('p')!.textContent = String(err);
    canvas.style.display = 'none';
  }
}

void main();
