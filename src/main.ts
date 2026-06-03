// Bootstrap: WebGPU gate (I.gate) -> Engine.init (V8) -> loop.
import { Engine } from './app/Engine';

async function main(): Promise<void> {
  const gate = document.getElementById('gate')!;
  const canvas = document.getElementById('app') as HTMLCanvasElement;

  if (!('gpu' in navigator)) {
    gate.style.display = 'flex';
    canvas.style.display = 'none';
    return;
  }

  try {
    const engine = new Engine(canvas);
    await engine.init();
    engine.start();
    // Sim wired at M4 (T13): engine.setSim(new Simulation(...))
  } catch (err) {
    console.error('Init failed:', err);
    gate.style.display = 'flex';
    gate.querySelector('p')!.textContent = String(err);
    canvas.style.display = 'none';
  }
}

void main();
