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
- ui: weather → cloud coverage, storminess, wind dir+speed, rain toggle (tie to climate rainfall map).
- ui: lighting → sun direction, sun intensity, ambient/fill (dark-side) level, atmosphere thickness+tint.
- ui: water → flow-viz strength + flow speed (surface shows flow direction).
- ui: debug mode selector → {none,waterDepth,flowSpeed,flowDir,sediment,erosion,deposition,soilDepth,cellArea,activeTiles} (replaces single 'v' toggle).

## §V INVARIANTS
V1: tan-warp + (face,u,v)↔dir math ∈ one shared module. vertex displace & sim neighbor lookup ! use same → geometry & physics agree.
V2: ⊥ read & write same StorageTexture in one pass. neighbor-dependent passes (addWater,waterDepth,sedimentAdvect,thermal) ! ping-pong in→out then swap.
V3: sim reads via `textureLoad(tex,ivec2)`, writes via `textureStore(tex,ivec2,val)`. ⊥ one StorageTexture bound both in one computeNode.
V4: flux pass ! scale-clamp K so outflow ≤ available water. clamp `d≥0` & `s≥0`. → no explode/negative.
V5: face tex alloc `(RES+2)²` w/ 1-texel apron. seamCopy pass each tick syncs aprons of changed fields (b,d,s min) → water/sediment cross seams & ⊥ vertex cracks.
V6: height stored as offset from base radius (`r32float`). radius added only @ vertex stage → float precision. ∴ gravity = radial-to-core implicitly: flux flows high→low (b+d) radial height. ⊥ world-axis gravity.
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
V18: material height sampling ! texel-exact `textureLoad(ivec2(round(uv*res)))`, vertex k↔texel k. ⊥ uv+NearestFilter on (res+1) tex (drifts ±1 near edges → seam groove). offset samples cross seams via table neighbor read; neighbor pos uses NEIGHBOR face dir (⊥ self-warp extrapolation) → C0 normal across seam.
V19: fluid sim ! conserve mass. face borders sealed (flux & inflow=0 at walls) — clamped neighbor read at border = self → phantom inflow → explosion. cross-face via conservative seam diffusion. total = ∫source - ∫loss.
V20: fluid solver reusable: `makeFluidUniforms()` + `buildAddSource/buildFlux/buildDepth(...,p)` parameterized. water & lava share solver; differ by consts (pipeArea=visc, loss=evap/cool, source) + terrain coupling (erode vs solidify, separate pass).
V21: erosion ! bounded: v ÷ min-depth (⊥ ÷~0) & clamp ±3; tilt≤0.5, speed≤3; gate by water d>0.004; cap erode/dep per step. else bedrock→∞.
V22: surface normals BAKED to texture via compute (`sim/passes/normals.ts` NormalBaker) on change (brush/erosion/water tick), material samples (`bakedSurface`). fragment cost res-independent. res=512. normal = cross(displaced neighbor positions), cross-seam neighbor uses NEIGHBOR face dir → seamless flat+sloped. ⊥ radial+tangent form (breaks sloped seams, B9).
V23: terrain = 2 layers: hard rock + `loose` (soil/sand) on top. b=total height (unchanged for flux/render). erosion: softness = max(rockErodibility, loose/looseFull); erode removes loose first then rock; ALL deposit → loose. loose seeded w/ own fbm (varied). color: rock(grey) where loose thin|steep, soil/grass/sand/snow where loose.
V24: lighting = directional SUN (1 dir light) + hemispheric/ambient FILL so night side never fully black. fill = artistic (gameplay readability), ⊥ physical. normals view-space (V14). day/night terminator soft, ⊥ hard black.
V25: weather layers (atmosphere, clouds, rain) = transparent shells/particles ABOVE terrain+water. `depthWrite=false`, drawn after opaque, ⊥ block picking/sim. ⊥ alter bedrock/water fields.
V26: rain spawns from STORM cloud cells at their surface lat/long (where rainfall-map × storminess high). rain falls radially inward (toward core, V6) to surface. rain VISUAL cell == fluid rain SOURCE cell → sim & visual agree (⊥ rain drawn where no water added & vice versa).
V27: flow field = sim `velocity` (cross-seam, V5) exposed to `waterMaterial`. surface normal/texture distorted ALONG flow dir → direction evident. flow sampled seamless (V5/V18) ⊥ flow seam. still water (|v|~0) → calm, ⊥ phantom motion.
V28: weather/atmosphere GPU work bounded, holds 60fps (V10). animation = time-driven uniform, graphs built once (V9). cloud/atmos cost ⊥ scale w/ sim RES.
V29: all weather fields (cloud density, storm mask) use shared planet dir math (V1) → wrap sphere seamless, continuous across faces (V29 cloud noise dir-based, ⊥ per-face uv noise → seam).
V30: suspended sediment `s` ! cross seams WITH the water that carries it (`seamFlux` transports s ∝ exchanged water: out=self conc, in=neighbor conc, conservative). else water crosses seam "clean", s left behind → loose/sed ridge piles along every seam (symmetric → ⊥ removable by edge-avg). advect backtrace also cross-seam (`seamHeight`). border velocity = first-interior (⊥ clamped~0 → under-erode ridge).
--- From Dust adoption (M11) ---
V31: water + sediment stored as VOLUME (mass), ⊥ depth. depth = vol / cellArea (precomputed `cellArea` r32float tex/face, from tan-warp jacobian). conc = sedMass / max(vol,eps). flux surface = baseR+bedrock+soil+depth. → conserves across uneven cube-sphere cells & seams (uneven-area cells made depth-store non-conservative @ borders, B11 class). sim storage stays r32float (writable-format C; ⊥ r16float).
V32: interior/border PASS SPLIT. interior dispatch range [1,res-1]² uses plain same-face neighbor sampling (⊥ seam branch). border cells (the 4 edges) handled by a separate seam pass (cross-face map). → most cells ⊥ pay seam cost (perf) & seam logic isolated. supersedes per-cell `seamHeight` select in hot passes.
V33: ACTIVE TILES. each face split into tiles (16² or 32²). CPU-side active mask; sim dispatch only over active tiles. tile active if: water present | recent brush edit nearby | incoming border flow | recent erosion/deposit. inactive tiles skipped. full-face dispatch = v1/fallback only. ⊥ silent: HUD shows active-tile count.
V34: sediment-driven VISCOSITY. effectiveFlowRate = max(minFlowRate, flowRate / (1 + conc·mudViscosityFactor)). muddy basins flow slow + accumulate (deltas, viscous lakes); clear streams flow fast. applied in flux pass.
V35: sim fields DRIVE surface look (⊥ geometry alone). erosion/deposit written to viz tex (r=erode,g=deposit), decays `*=exp(-dt·decay)`; terrain material tints fresh erode (dark streak) / deposit (light fan). flow DIRECTION = WORLD-SPACE procedural streaks (phase = dot(worldPos, flowWorld)·freq - time·speed) — seamless, ⊥ face-uv advect (face-local uv → seam). speed drives normal distortion + foam on fast flow. flowWorld from neighbor-position tangents (tangentU=posE-posW etc) → accurate @ edges.
V36: explicit DEBUG MODES (uniform-switched in material), ⊥ overload art shader: waterDepth, flowSpeed, flowDir(dir→color `vec3(dir·0.5+0.5,0)`), sediment(conc), erosion, deposition, soilDepth, cellArea, activeTiles. dir→color seamless via flowWorld.

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
T12|x|M4 `sim/passes/water.ts` pipe-model (Mei) addSource+flux+depth, scale-clamp V4, flux damping (settle), borders sealed (conserve V19). cross-seam via `passes/seamFlux.ts` (surface-diff exchange → continuous water surface + flow across faces). `materials/waterMaterial.ts` depth-tint + baked normal|V2,V3,V4,V7,V19
T13|x|M4 `src/sim/Simulation.ts` pass order addWater→flux→depth+evap, canonical main→scratch→copy (⊥ swap)|V2,V7
T14|x|M5 `sim/passes/erosion.ts` velocity+erode/deposit+advect. clamped (V21), flow-gated (only fast water erodes; slow deposits→fills pools). toggle + GUI|V4,V7,V21
T15|x|M6 thermal slump `buildThermal` in `erosion.ts` (material steeper than talus → lower neighbors, conservative). smooths spikes. runs each erosion tick|V2,V7
T16|x|M7 river springs (`tools/Emitters.ts`, sidebar River tool) + SEA LEVEL via fluid sim (`buildFluidUpdate` seaFill: deep ocean pulled to global seaLevel → flat sea, no separate mesh) + GUI slider + terrain beach/coastline + submerged darkening|I.ui
T17|x|M8 lava: `sim/LavaSim.ts` reuses fluid solver (`makeFluidUniforms` viscous, vent source) + `passes/lava.ts` cool→solidify-to-bedrock + heat. `materials/lavaMaterial.ts` ember→white-hot emissive. Volcano sidebar tool. Throttled 3rd tick (freq only, ⊥ dt-scale=unstable)|V7,V12,V20
T18|x|M9 polish: `ui/Controls.ts`+`ui/Sidebar.ts`, baked normals (V22, res 512), pass fusion (`buildFluidUpdate` ~half water dispatches), water wave shimmer (waterMaterial time), biome bands (rock/soil/sand/snow + coastline), debug field view (`materials/debugMaterial.ts`, key 'v': R=sed G=loose B=water)|V10,V22,I.ui
T19|x|fps counter in HUD (`Engine.updateHud`) + per-milestone visual verify (user-confirmed throughout)|V10
T20|.|M10 lighting: directional SUN uniform (dir+intensity) + hemispheric AMBIENT/FILL so dark side readable. tune terminator soft. GUI sun-dir/intensity/fill sliders|V14,V24,I.ui
T21|.|M10 `materials/atmosphereMaterial.ts`: back-side atmosphere shell (rim glow / cheap scatter approx) + sky tint. additive transparent, drawn after opaque, depthWrite off|V24,V25,V28
T22|.|M10 `materials/cloudMaterial.ts` + cloud shell mesh: animated stylized clouds, coverage uniform, dir-based fbm (V1) continuous across faces, wind-advected (time uniform)|V25,V28,V29
T23|.|M10 storm clouds + rain: storminess → dark towering clouds where rainfall×storm high; rain particles/shader fall radially cloud→surface @ correct cells; COUPLE to fluid rain source (same cells add water d)|V6,V25,V26,V29
T24|.|M10 flow-field water: expose sim `velocity` (cross-seam V5) to `waterMaterial`; flow-aligned normal/texture distortion (scrolling along v) → flow dir evident; seamless; calm when still|V5,V18,V27
T25|.|M10 weather UI panel (`ui/Sidebar.ts`|`Controls.ts`): cloud coverage, storminess, wind dir/speed, sun dir/intensity, dark-side fill, atmosphere thickness/tint, flow-viz strength|I.ui,V24

## §B BUGS
id|date|cause|fix
B1|2026-06-03|terrain `normalNode` = world-space cross(dFdx(positionWorld)..) → lighting swam w/ camera, planet looked morphing|V14 ∴ use view-space `normalFlat`; slope via `abs(dot)`
B2|2026-06-03|face index winding `a,c,b`/`b,c,d` = CW-from-outside → near faces culled → saw through front to far hemisphere (worse on vertical orbit)|V15 ∴ winding `a,b,d`/`a,d,c`
B3|2026-06-03|brush stamped all 6 faces + full seamSync per pointermove → 24 computes/event, queue backlog, fps decayed over long stroke|V16 ∴ coalesce 1/frame + face-cull + seam-sync on pointerup
B4|2026-06-03|multiplicative evap `d*=(1-ke*dt)` + uniform rain → uniform equilibrium d=rain/ke everywhere → water planet|V17 ∴ subtractive evap `d-=ke*dt`, ke>rain → flat dries, basins keep water
B5|2026-06-03|screen-space `normalFlat` computed per face-mesh → hard shading crease at every face boundary → globe looked like 6 panels|V11 ∴ analytic gradient normals (`tsl/surface.ts`), continuous across faces
B6|2026-06-03|material sampled height by uv+NearestFilter on (res+1) tex → vertex k drifts to texel k±1 near edges → geometric groove along seams|V18 ∴ sample by exact int texel `textureLoad(round(uv*res))`, seam-aware neighbor read across edges
B7|2026-06-04|flux/depth border neighbor reads clamped to self → each face edge cell counts own flux as inflow → phantom water, brief rain → huge oceans|V19 ∴ seal borders (flux & inflow=0 at walls), cross-face via seam diffusion
B8|2026-06-04|erosion: velocity = flux/depth w/ depth~0 → huge speed → unbounded erode/deposit → bedrock→∞ (vertices streak to infinity)|V21 ∴ clamp v (÷min-depth, ±3), clamp tilt/speed, gate by water, cap erode/dep per step
B9|2026-06-04|baked normal via radial+tangent (Dc - Tu*slope) fixed flat-water seam but per-face Tu/Tv mismatch reintroduced seam on SLOPED terrain|V22 ∴ normal = cross of displaced neighbor positions (neighbor face dir), seamless flat+sloped, ~9 faceDirNode (1 neighbor/offset)
B10|2026-06-04|erosion+thermal read clamped(self) b-neighbors at borders → tilt≈0 → border won't erode → raised RIDGE along seam (border-mask made it worse). ∴ erosion tilt + thermal use CROSS-SEAM b gradient (`seamHeight`, seam table) → border erodes like interior, no ridge. res 768.
B11|2026-06-04|erosion seam: ridge along every face edge once erosion runs (relief + shows in sed/loose/water debug). faces sealed for flux, water crosses via `seamFlux` but sediment NOT carried → water crosses seam clean, sediment left behind → piles → deposits loose ridge (symmetric → edge-avg can't fix). edge-avg of b/normal/sed all no-op. ∴ `seamFlux` carries s ∝ exchanged water (conservative) + advect cross-seam + border velocity=first-interior|V30
