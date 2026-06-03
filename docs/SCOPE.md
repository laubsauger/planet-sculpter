# Planet Sculptor — Scope

> Living doc. Updated as milestones land. Source of truth for invariants/tasks is `SPEC.md` (caveman); this is the human-readable companion.

## Vision

A browser game where you sculpt a planet — raise/lower terrain, drop river springs, place volcanos — while the planet **simulates hydraulic + thermal erosion, sediment deposition, and lava flow in real time on the GPU**. Free rotate/zoom, seamless interaction. Stylized flat-shaded cartoon look. Not photorealistic — emergent and believable.

## Confirmed decisions

| Area | Decision |
|---|---|
| Renderer | three.js `WebGPURenderer` only. No WebGL fallback. |
| Shaders | TSL — node materials **and** compute. Import from `three/webgpu` + `three/tsl`. |
| Stack | vanilla three.js + Vite + TypeScript. three ^0.184. UI: lil-gui. Tests: vitest. |
| Geometry | Cube-sphere: 6 faces, per-face height storage textures, tan-warp area equalization, displaced grid mesh. |
| Simulation | GPU pipe-model (virtual-pipes, Mei et al.) hydraulic erosion + thermal slumping + sediment transport. Lava reuses the fluid machinery. |
| Visuals | Animated water surface (ocean + rivers), biome coloring by height/slope/moisture/temp (beach/grass/rock/snow), flat-shaded low-poly, lava emissive. |
| Interaction | Orbit rotate/zoom. Brush projected onto surface, real-time while sim runs. |

## Features

- **Terrain brushes:** raise / lower / smooth / flatten under a radial brush.
- **River springs:** drop a source; water flows downhill, carves channels, forms lakes/deltas.
- **Volcanos:** vent erupts, builds a cone, lava flows downhill, raises terrain, cools to rock, glows hot.
- **Climate:** global sea-level, rainfall, temperature sliders driving erosion + biomes.
- **Debug:** field-view toggle (height/water/sediment/flux) to a quad for diagnosis.

## Performance target

**60 fps on the dev machine (Apple Silicon / Metal-backed WebGPU).**

Strategy:
- Render at 60 fps; **sim throttled to ~30 ticks/s** (compute shares the GPU queue).
- Fixed-timestep sim accumulator, capped substeps/frame (anti spiral-of-death).
- **Adaptive RES:** start 256, scale toward 512/1024 only if frame budget allows; auto-drop sim substeps or RES if frame time exceeds budget.
- GPU-resident sim — no GPU→CPU readback in the hot loop (picking is an analytic sphere raycast).
- Compute pipelines warmed once at startup; node graphs built once, never per-frame.

## Milestone roadmap

| M | Goal |
|---|---|
| M0 | WebGPU boot: cube-sphere geometry + tan-warp, orbit, flat-shaded solid. |
| M1 | Static height texture → vertex displacement + biome color. No seam cracks. |
| M2 | Brush: pick → GPU stamp into height texture, live. |
| M3 | Seam table + seamCopy pass; brush stroke crosses face edge continuously. |
| M4 | Water only: add-water + flux + depth; render water; conservation correct. |
| M5 | Hydraulic erosion: erosion/deposition + sediment advection + evaporation. |
| M6 | Thermal slumping. |
| M7 | Springs/volcano emitters + sea-level/climate. |
| M8 | Lava: flow + cool + glow. |
| M9 | Polish: water waves, biome banding, debug view, perf lock 60 fps, UI. |

## Non-goals (explicit)

- Photorealism.
- Voxels / caves / overhangs (height-field only).
- Mobile / WebGL fallback.
- Multiplayer.
- Save/load (deferred).
