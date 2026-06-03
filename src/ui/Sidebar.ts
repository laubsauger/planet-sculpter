// Stylish tool sidebar (T18). Glass panel: tool selection, brush size/strength,
// sculpt + rain toggles. Drives the same settings objects as hotkeys/lil-gui;
// call sync() to reflect external (hotkey) changes.

import type { BrushMode } from '../tools/BrushTool';
import type { BrushSettings } from './Controls';

export interface SidebarHandles {
  brush: BrushSettings;
  isSculpt(): boolean;
  setSculpt(on: boolean): void;
  isRain(): boolean;
  toggleRain(): void;
}

interface ToolDef {
  mode: BrushMode;
  label: string;
  icon: string; // inline SVG path content
}

const TOOLS: ToolDef[] = [
  { mode: 'raise', label: 'Raise', icon: '<path d="M3 17 L10 6 L17 17 Z"/>' },
  { mode: 'lower', label: 'Lower', icon: '<path d="M3 6 L17 6 L10 17 Z"/>' },
  {
    mode: 'smooth',
    label: 'Smooth',
    icon: '<path d="M2 12 q4 -7 8 0 t8 0" fill="none" stroke="currentColor" stroke-width="2"/>',
  },
  { mode: 'flatten', label: 'Flatten', icon: '<rect x="3" y="9" width="14" height="3" rx="1.5"/>' },
];

const CSS = `
.ps-side{position:fixed;left:12px;top:50%;transform:translateY(-50%);width:148px;
  background:rgba(16,20,28,.72);backdrop-filter:blur(14px) saturate(1.2);
  -webkit-backdrop-filter:blur(14px) saturate(1.2);
  border:1px solid rgba(255,255,255,.09);border-radius:14px;
  box-shadow:0 10px 32px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.06);
  color:#e7ecf3;font:12px/1.35 ui-sans-serif,system-ui,sans-serif;
  padding:12px;user-select:none;z-index:10;}
.ps-side h1{font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:#7d8aa0;margin:0 0 10px;font-weight:600;}
.ps-side h2{font-size:9px;letter-spacing:.1em;text-transform:uppercase;
  color:#69748a;margin:14px 0 6px;font-weight:600;}
.ps-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.ps-tool{display:flex;flex-direction:column;align-items:center;gap:3px;
  padding:7px 4px;border-radius:9px;cursor:pointer;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
  color:#aeb8c8;transition:all .15s ease;}
.ps-tool:hover{background:rgba(255,255,255,.09);color:#e7ecf3;
  transform:translateY(-1px);}
.ps-tool svg{width:18px;height:18px;fill:currentColor;}
.ps-tool span{font-size:10px;font-weight:500;}
.ps-tool.active{background:linear-gradient(160deg,#3b82f6,#2563eb);
  border-color:rgba(255,255,255,.25);color:#fff;
  box-shadow:0 3px 11px rgba(37,99,235,.45);}
.ps-row{margin:9px 0;}
.ps-row label{display:flex;justify-content:space-between;font-size:10px;
  color:#9aa6ba;margin-bottom:4px;}
.ps-row label b{color:#dbe3ef;font-weight:600;font-variant-numeric:tabular-nums;}
.ps-side input[type=range]{width:100%;accent-color:#3b82f6;height:3px;cursor:pointer;}
.ps-toggle{display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px;border-radius:9px;cursor:pointer;margin-top:6px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
  transition:all .15s ease;}
.ps-toggle:hover{background:rgba(255,255,255,.08);}
.ps-toggle span{font-size:11px;font-weight:500;}
.ps-pill{width:32px;height:18px;border-radius:9px;background:rgba(255,255,255,.14);
  position:relative;transition:background .2s;flex:none;}
.ps-pill::after{content:'';position:absolute;top:2px;left:2px;width:14px;height:14px;
  border-radius:50%;background:#cdd6e4;transition:transform .2s;}
.ps-toggle.on .ps-pill{background:#3b82f6;}
.ps-toggle.on .ps-pill::after{transform:translateX(14px);background:#fff;}
.ps-hint{margin-top:11px;font-size:9px;color:#5d6a80;line-height:1.5;}
.ps-hint kbd{background:rgba(255,255,255,.08);border-radius:3px;padding:1px 4px;
  font:9px ui-monospace,monospace;color:#9aa6ba;}
`;

export class Sidebar {
  readonly el: HTMLDivElement;
  private toolBtns = new Map<BrushMode, HTMLButtonElement>();
  private sculptToggle!: HTMLDivElement;
  private rainToggle!: HTMLDivElement;

  constructor(private readonly h: SidebarHandles) {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    this.el = document.createElement('div');
    this.el.className = 'ps-side';
    this.el.innerHTML = `<h1>Planet Sculptor</h1>`;

    // sculpt mode toggle
    this.sculptToggle = this.makeToggle('Sculpt mode', () => {
      this.h.setSculpt(!this.h.isSculpt());
      this.sync();
    });
    this.el.appendChild(this.sculptToggle);

    // tools
    const h2t = document.createElement('h2');
    h2t.textContent = 'Terrain tools';
    this.el.appendChild(h2t);
    const grid = document.createElement('div');
    grid.className = 'ps-grid';
    for (const t of TOOLS) {
      const btn = document.createElement('button');
      btn.className = 'ps-tool';
      btn.innerHTML = `<svg viewBox="0 0 20 20">${t.icon}</svg><span>${t.label}</span>`;
      btn.onclick = () => {
        this.h.brush.mode = t.mode;
        if (!this.h.isSculpt()) {
          this.h.setSculpt(true);
        }
        this.sync();
      };
      grid.appendChild(btn);
      this.toolBtns.set(t.mode, btn);
    }
    this.el.appendChild(grid);

    // brush sliders
    const h2b = document.createElement('h2');
    h2b.textContent = 'Brush';
    this.el.appendChild(h2b);
    this.makeSlider('Size', 0.02, 0.4, 0.005, this.h.brush.radius, (v) => {
      this.h.brush.radius = v;
    });
    this.makeSlider('Strength', 0, 0.1, 0.001, this.h.brush.strength, (v) => {
      this.h.brush.strength = v;
    });

    // water
    const h2w = document.createElement('h2');
    h2w.textContent = 'Water';
    this.el.appendChild(h2w);
    this.rainToggle = this.makeToggle('Rain', () => {
      this.h.toggleRain();
      this.sync();
    });
    this.el.appendChild(this.rainToggle);

    const hint = document.createElement('div');
    hint.className = 'ps-hint';
    hint.innerHTML = `<kbd>G</kbd> sculpt/orbit &nbsp; <kbd>1</kbd>–<kbd>4</kbd> tools &nbsp; <kbd>R</kbd> rain<br>drag to sculpt · scroll to zoom`;
    this.el.appendChild(hint);

    document.body.appendChild(this.el);
    this.sync();
  }

  private makeToggle(label: string, onClick: () => void): HTMLDivElement {
    const row = document.createElement('div');
    row.className = 'ps-toggle';
    row.innerHTML = `<span>${label}</span><div class="ps-pill"></div>`;
    row.onclick = onClick;
    return row;
  }

  private makeSlider(
    label: string,
    min: number,
    max: number,
    step: number,
    value: number,
    onInput: (v: number) => void,
  ): HTMLElement {
    const row = document.createElement('div');
    row.className = 'ps-row';
    const lab = document.createElement('label');
    const valEl = document.createElement('b');
    valEl.textContent = value.toFixed(3);
    lab.append(document.createTextNode(label), valEl);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.oninput = () => {
      const v = parseFloat(input.value);
      onInput(v);
      valEl.textContent = v.toFixed(3);
    };
    row.append(lab, input);
    this.el.appendChild(row);
    return valEl;
  }

  /** Reflect external state changes (hotkeys). */
  sync(): void {
    for (const [mode, btn] of this.toolBtns) {
      btn.classList.toggle('active', this.h.isSculpt() && this.h.brush.mode === mode);
    }
    this.sculptToggle.classList.toggle('on', this.h.isSculpt());
    this.rainToggle.classList.toggle('on', this.h.isRain());
  }

  dispose(): void {
    this.el.remove();
  }
}
