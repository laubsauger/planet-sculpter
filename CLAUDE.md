# CLAUDE.md — planet-sculpter

Browser game: sculpt a planet (terrain/rivers/volcanos) while the GPU sims erosion,
water, and lava in real time. Stylized flat-shade cartoon look, 60fps target.

## Commands
- `npm run dev` — Vite dev server, open in a WebGPU-capable browser (Chrome/Edge).
- `npm test` — `vitest run` (pure-math tests, no GPU).
- `npm run typecheck` — `tsc --noEmit`. `npm run build` runs typecheck + Vite build.

## Stack
- three.js `WebGPURenderer` **only** — no WebGL fallback. WebGPU absent → "WebGPU required" gate.
- TSL for all node materials & GPU compute. Import from `three/webgpu` + `three/tsl`, **never** the `three` root for those.
- Vanilla three.js + Vite + TypeScript, no framework.

## Architecture — READ THIS FIRST
The project **pivoted from a sphere/cube-sphere planet to a flat From-Dust-style
heightfield**. Poles + equirect distortion weren't worth it; flat puts all
resolution in the patch — crisp, no seams.

- **Active engine: `src/flat/`** (`FlatEngine.ts` entry, `flatSim.ts` GPU sim,
  `flatSeed/flatMesh/FlatBrush/flatBenchmarks`). `main.ts` boots `FlatEngine`.
- Flat sim **reuses** the tuned uniforms + grid store from `src/sim/`:
  `sim/gridStore.ts`, `sim/passes/water.ts`, `sim/passes/erosion.ts`, `sim/gridWater.ts`.
  These are LIVE — touch with care.
- Active materials/tsl: `materials/flat*.ts`, `tsl/flatSurface.ts`, `tsl/lighting.ts`.
- Shared: `config.ts`, `ui/Sidebar.ts`, `app/OrbitController.ts`.

**Dead sphere-era code was deleted** (GridEngine/Engine, sphere meshes, cube-sphere,
GridSim/LavaSim/Simulation, seam passes, sphere materials, picking/Emitters/GridBrush, etc.).

**Sphere-era files still wired in (vestigial, future cleanup — don't extend):** these
stay only because live code transitively imports them, so deleting needs an edit not just
`rm`. `planet/seamTable.ts`, `tsl/surface.ts`, `tsl/warp.ts`, `tsl/warpNode.ts`,
`sim/fields.ts` (pulled via `passes/erosion`, `tools/BrushTool`); `ui/Controls.ts` +
`materials/{atmosphere,cloud,rain}Material.ts` (pulled via `ui/Sidebar`).
Many SPEC §V invariants about seams/faces/cube-sphere are HISTORICAL — flat has no seams.

**Before deleting any file, trace reachability from `main.ts` first** (`grep -rE "from '\.\."`).
Logical "deadness" ≠ import-graph deadness; verify with `npm run typecheck` + `npm test`.

## SPEC.md is the law
`SPEC.md` (caveman-encoded: §G goal, §C constraints, §I interfaces, §V invariants)
defines behavior + invariants. Read relevant §V before changing sim/render. Note many
seam/face invariants are sphere-era; the flat pivot dropped them. Keep SPEC current when
behavior changes. `docs/SCOPE.md` + `docs/from_dust_*` carry vision/roadmap.

## Sim rules (still apply, flat form)
- GPU-resident. No GPU→CPU readback in the hot loop (picking is analytic via raycast plane).
- Ping-pong (in→out then swap) for neighbor-dependent passes; never read+write one
  StorageTexture in a pass. Read `textureLoad(tex,ivec2)`, write `textureStore(...)`.
- Writable storage formats only: `r32float`, `rg32float`, `rgba32float`, `r32uint`, `rgba8unorm`.
- Build `Fn`/node graphs once, reuse — never rebuild per frame (pipeline recompile stall).
- `await renderer.init()` before any render/compute; warm each compute pipeline once at startup.
- Render every frame at 60fps; sim throttled (~30 tick/s), auto-drops substeps under budget.
- Flux must clamp so outflow ≤ available water; clamp depth ≥ 0, sediment ≥ 0. Erosion bounded.
- Edges CLAMPED + sealed; ocean border pinned to sea level (see [[water-deep-ocean-flat-pin]]).
- Evaporation subtractive, not multiplicative (else uniform rain → water planet).

## Conventions / gotchas
- TSL `If` callbacks must be block bodies; `If` is compute-only, not a material `colorNode`.
- Value-noise hashes need `Math.imul` + `>>>` or terrain collapses — verify distribution offline.
- Surface normals: view-space only in `normalNode` (world-space → lighting swims with camera).
- Verify live changes via dev server + Chrome CDP + `window.engine` texture readback
  (engine + sim uniforms exposed on `window` in DEV via `main.ts`).

## Global rules
- Do NOT add fallback logic for core functionality. Fix the real bug or throw/test. Fallbacks are an antipattern here.
- Never commit/checkout/reset git.
- Install shadcn components via CLI, never hand-write from memory.
