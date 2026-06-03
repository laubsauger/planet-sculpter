# SPEC

## §G GOAL
Browser game: sculpt planet (terrain/rivers/volcanos) while GPU sims erosion/deposit/lava real-time. Free rotate. Stylized flat-shade cartoon. 60fps.

## §C CONSTRAINTS
- three.js `WebGPURenderer` only. ⊥ WebGL fallback.
- TSL for node materials & compute. import `three/webgpu` + `three/tsl`, ⊥ `three` root.
- vanilla three.js + Vite + TypeScript.
- Geometry: cube-sphere. 6 faces, per-face height storage tex, tan-warp area-equalize.
- Sim: GPU pipe-model (Mei et al.) hydraulic + thermal erosion + sediment transport. Lava reuses fluid machinery.
- Sim GPU-resident. ⊥ GPU→CPU readback in hot loop (picking analytic).
- 60fps target on user machine (Apple Silicon / Metal). RES & sim tick adaptive.
- Writable storage formats ∈ {`r32float`,`rg32float`,`rgba32float`,`r32uint`,`rgba8unorm`} only.
- ⊥ NO-GOALS: photorealism, voxel/caves/overhangs, mobile, multiplayer, save/load (later).

## §I INTERFACES
- ui: tool select → {raise,lower,smooth,flatten,spring,volcano}. brush radius/strength sliders.
- ui: climate → sea-level, rainfall, temperature sliders.
- ui: sim params (Kc,Ks,Kd,Ke,talus) + sim-throttle + debug field view toggle (height/water/sediment/flux).
- input: drag-orbit rotate/zoom planet. brush stamp on hover+click.
- file: `docs/SCOPE.md` → vision, decisions, features, 60fps strategy, milestones, non-goals. kept current.
- gate: `navigator.gpu` absent → "WebGPU required" screen.
- run: `npm run dev` (Vite) → WebGPU-capable browser.

## §V INVARIANTS
V1: tan-warp + (face,u,v)↔dir math ∈ one shared module. vertex displace & sim neighbor lookup ! use same → geometry & physics agree.
V2: ⊥ read & write same StorageTexture in one pass. neighbor-dependent passes (addWater,waterDepth,sedimentAdvect,thermal) ! ping-pong in→out then swap.
V3: sim reads via `textureLoad(tex,ivec2)`, writes via `textureStore(tex,ivec2,val)`. ⊥ one StorageTexture bound both in one computeNode.
V4: flux pass ! scale-clamp K so outflow ≤ available water. clamp `d≥0` & `s≥0`. → no explode/negative.
V5: face tex alloc `(RES+2)²` w/ 1-texel apron. seamCopy pass each tick syncs aprons of changed fields (b,d,s min) → water/sediment cross seams & ⊥ vertex cracks.
V6: height stored as offset from base radius (`r32float`). radius added only @ vertex stage → float precision.
V7: sim pass order fixed: addWater→flux→waterDepth+velocity→erosion/deposition→sedimentAdvect→evaporation→thermal→seamCopy.
V8: `await renderer.init()` before any render/compute. warm-up each compute pipeline once @ startup → no first-use hitch.
V9: `Fn`/node graphs built once, reused. ⊥ rebuild per frame → no pipeline recompile stall.
V10: render 60fps. sim throttled (~30 tick/s). frame-time > budget → auto-drop sim substeps | RES. input/orbit every frame independent of sim.
V11: surface normals ANALYTIC from height-field gradient (finite-diff of displaced neighbor positions, `tsl/surface.ts`), view-space. continuous across faces → ⊥ "6 panels". ⊥ screen-space `normalFlat` (per-mesh → hard seam at every face boundary).
V12: lava heat→0 → fold lava depth into bedrock & stop render.
V13: pure math testable w/o GPU: `warp` dir↔(face,u,v) round-trip, `seamTable` 24-edge map correct.
V14: `material.normalNode` ! view-space (use `normalFlat`|`normalView`). ⊥ world-space normal → lighting swims w/ camera. world normal OK only for scalar slope via `abs(dot(..))`.
V15: face geom triangle winding ! CCW-from-outside (normal = u×v = +forward). inverted → near faces back-face-culled → see-through planet.
V16: input-driven GPU work ! bounded/frame. brush coalesced to ≤1 stamp/frame (⊥ per pointermove), cull faces by dot(forward,centerDir)>0.37, seam-sync on pointerup not per-move. else compute queue backlog → fps decays over stroke.
V17: evap ! subtractive (`d-=ke*dt`, ke>rain) ⊥ multiplicative. uniform rain + mult evap → uniform depth everywhere (water planet). subtractive → flat ground dries, water only in net-inflow basins.
V18: material height sampling ! texel-exact `textureLoad(ivec2(round(uv*res)))`, vertex k↔texel k. ⊥ uv+NearestFilter on (res+1) tex (drifts ±1 near edges → seam groove). offset samples cross seams via table neighbor read.

## §T TASKS
id|status|task|cites
T1|x|scaffold Vite+TS, install three(webgpu)+ui(lil-gui), npm|C
T2|x|write `docs/SCOPE.md`|I.file
T3|x|`src/config.ts` RES,radius,K consts|-
T4|x|M0 `src/tsl/warp.ts` tan-warp + (face,u,v)↔dir + round-trip test|V1,V13
T5|x|M0 `src/planet/cubeSphere.ts`+`PlanetMesh.ts` 6-face geom|V1
T6|x|M0 `src/main.ts`+`app/Engine.ts` WebGPU gate, init, RAF, orbit, flat-shade solid|V8,V10,V11,I.gate
T7|x|M2 `src/sim/fields.ts` r32float StorageTexture ping-pong mgr + seed compute (apron deferred→M3; r32float NearestFilter, not filterable)|V2,V5,V6,C
T8|x|M1 `src/materials/terrainMaterial.ts` displace + flat normals + biome graph (height src `planet/heightField.ts` CPU DataTexture, dir-based crack-free)|V6,V11
T9|x|M2 `src/tools/picking.ts` ray→sphere→(face,u,v)→texel|C
T10|x|M2 `src/tools/BrushTool.ts` GPU stamp raise/lower/smooth/flatten. DIRECTION-space (center=sphere dir, all faces stamped, `tsl/warpNode.ts` faceDirNode V1) → paints across seams, no crease|V1,V3,V9
T11|x|M3 `src/planet/seamTable.ts` 24-edge map (auto-derived, 8 tests) + `sim/passes/seamCopy.ts` (avg shared-edge texels, 3-way corners). Fields refactored to canonical `main`+`scratch` (read main→write scratch→copy back), ⊥ swap/rebind|V5,V13
T12|x|M4 `sim/passes/water.ts` (addWater+flux+depth, pipe-model Mei et al., scale-clamp V4) + `materials/waterMaterial.ts` (surface b+d, depth-tint, transparent). cross-seam flow ⊥ yet (no apron, borders=walls)→M5|V2,V3,V4,V7
T13|x|M4 `src/sim/Simulation.ts` pass order addWater→flux→depth+evap, canonical main→scratch→copy (⊥ swap)|V2,V7
T14|~|M5 `sim/passes/erosion.ts` (velocity+erode/deposit+advect) wired in Simulation behind toggle + GUI Kc/Ks/Kd. b&s seam-synced/tick. UNVERIFIED visually, needs tuning|V4,V7
T15|.|M6 `sim/passes/thermal.ts` slump past talus|V2,V7
T16|.|M7 `src/tools/{Emitters,climate}.ts` springs/volcano + sea-level/climate uniforms|I.ui
T17|.|M8 `sim/passes/lava.ts` + `materials/lavaMaterial.ts` flow+cool+glow|V7,V12
T18|~|M9 polish. UI DONE EARLY: `ui/Controls.ts` (lil-gui: brush+water params, evap/flow/gravity/rain-intensity, clear-water) + `ui/Sidebar.ts` (glass panel: tool buttons, size/strength sliders, sculpt+rain toggles). REMAIN: water waves, biome bands, debug field view, perf lock 60fps|V10,I.ui
T19|.|fps counter + per-milestone visual verify|V10

## §B BUGS
id|date|cause|fix
B1|2026-06-03|terrain `normalNode` = world-space cross(dFdx(positionWorld)..) → lighting swam w/ camera, planet looked morphing|V14 ∴ use view-space `normalFlat`; slope via `abs(dot)`
B2|2026-06-03|face index winding `a,c,b`/`b,c,d` = CW-from-outside → near faces culled → saw through front to far hemisphere (worse on vertical orbit)|V15 ∴ winding `a,b,d`/`a,d,c`
B3|2026-06-03|brush stamped all 6 faces + full seamSync per pointermove → 24 computes/event, queue backlog, fps decayed over long stroke|V16 ∴ coalesce 1/frame + face-cull + seam-sync on pointerup
B4|2026-06-03|multiplicative evap `d*=(1-ke*dt)` + uniform rain → uniform equilibrium d=rain/ke everywhere → water planet|V17 ∴ subtractive evap `d-=ke*dt`, ke>rain → flat dries, basins keep water
B5|2026-06-03|screen-space `normalFlat` computed per face-mesh → hard shading crease at every face boundary → globe looked like 6 panels|V11 ∴ analytic gradient normals (`tsl/surface.ts`), continuous across faces
B6|2026-06-03|material sampled height by uv+NearestFilter on (res+1) tex → vertex k drifts to texel k±1 near edges → geometric groove along seams|V18 ∴ sample by exact int texel `textureLoad(round(uv*res))`, seam-aware neighbor read across edges
