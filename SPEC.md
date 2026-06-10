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
V10: render 60fps. sim 60 tick/s (1 tick/frame @ 60fps; tick-rate-safe per V42). frame-time > budget → auto-drop sim substeps | RES. input/orbit every frame independent of sim.
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
V26: storms COHERENT across visual + sim (stylized, ⊥ per-texel identical). storminess uniform scales BOTH: (a) dark dense cloud cores (cloud-noise storm mask) + rain veil falling under them (same mask, radially-inward streaks), (b) sim rain rate (Engine adds storminess·rate, localized by the rainfall map). rain veil = dir-noise; sim rain = rainfall-map zones — both rise/fall with storminess so weather reads consistent.
V27: flow field = sim `velocity` (cross-seam, V5) exposed to `waterMaterial`. surface normal/texture distorted ALONG flow dir → direction evident. flow sampled seamless (V5/V18) ⊥ flow seam. still water (|v|~0) → calm, ⊥ phantom motion.
V28: weather/atmosphere GPU work bounded, holds 60fps (V10). animation = time-driven uniform, graphs built once (V9). cloud/atmos cost ⊥ scale w/ sim RES.
V29: all weather fields (cloud density, storm mask) use shared planet dir math (V1) → wrap sphere seamless, continuous across faces (V29 cloud noise dir-based, ⊥ per-face uv noise → seam).
V30: suspended sediment `s` ! cross seams WITH the water that carries it (`seamFlux` transports s ∝ exchanged water: out=self conc, in=neighbor conc, conservative). else water crosses seam "clean", s left behind → loose/sed ridge piles along every seam (symmetric → ⊥ removable by edge-avg). advect backtrace also cross-seam (`seamHeight`). border velocity = first-interior (⊥ clamped~0 → under-erode ridge).
--- From Dust adoption (M11) ---
V31: water + sediment stored as VOLUME (mass), ⊥ depth. depth = vol / cellArea (precomputed `cellArea` r32float tex/face, from tan-warp jacobian). conc = sedMass / max(vol,eps). flux surface = baseR+bedrock+soil+depth. → conserves across uneven cube-sphere cells & seams (uneven-area cells made depth-store non-conservative @ borders, B11 class). sim storage stays r32float (writable-format C; ⊥ r16float).
V32: interior/border SPLIT. seam-aware samplers (`seamHeight`,`objNormalAt`) use `If` (real control flow), ⊥ `select` (evals BOTH branches → every cell paid seam textureLoad/faceDirNode). → interior cells skip cross-seam work; only the 4 edges pay. (dispatch-range tiling = T28 active tiles.) ⊥ regress seam continuity: If-override identical to select, just lazy.
V33: ACTIVE GATE (cell-level, realized). erosion+velocity passes early-out via `If` where cell has no water (depth<8e-4) AND no suspended sediment (<1e-5) -> passthrough, skip tilt(8 seamHeight)/velocity/capacity. dry land + deep-ocean interior = most cells. behavior-preserving (those cells already water-gated to ~0). chosen over tile-mask+indirect-dispatch: no GPU→CPU readback, no missed thin rivers (every wet cell active), no tile-border conservation risk. (tile-mask + HUD count = deferred, only if cell-gate insufficient.)
V34: sediment-driven VISCOSITY. effectiveFlowRate = max(minFlowRate, flowRate / (1 + conc·mudViscosityFactor)). muddy basins flow slow + accumulate (deltas, viscous lakes); clear streams flow fast. applied in flux pass.
V35: sim fields DRIVE surface look (⊥ geometry alone). erosion/deposit written to viz tex (r=erode,g=deposit), decays `*=exp(-dt·decay)`; terrain material tints fresh erode (dark streak) / deposit (light fan). flow DIRECTION = WORLD-SPACE procedural streaks (phase = dot(worldPos, flowWorld)·freq - time·speed) — seamless, ⊥ face-uv advect (face-local uv → seam). speed drives normal distortion + foam on fast flow. flowWorld from neighbor-position tangents (tangentU=posE-posW etc) → accurate @ edges.
V36: explicit DEBUG MODES (uniform-switched in material), ⊥ overload art shader: waterDepth, flowSpeed, flowDir(dir→color `vec3(dir·0.5+0.5,0)`), sediment(conc), erosion, deposition, soilDepth, cellArea, activeTiles. dir→color seamless via flowWorld.
V37: EMERGENT RIVERS = incision feedback, ⊥ sheet flow. erosion capacity ∝ DISCHARGE (depth·speed = stream power), concentrated by `channelFocus` (mix uniform↔smoothstep(discharge)) → cell gathering flow erodes more → deepens → captures more → self-organizing channel. thermal slump SUPPRESSED in deep water (`channelDepthRef`) so incised bed ⊥ heals flat. flux then follows the lower (b+depth) channel → closes loop. bootstraps from hardness-noise symmetry-break.
V38: MEANDERING = flow inertia + lateral erosion. velocity = mix(instant flux-vel, prev vel advected from upstream, `flowInertia`) → current carries momentum, overshoots bends into outer banks (⊥ snap to local downhill). erosion gets a LATERAL term ∝ discharge·(1−align(flowDir,downhillDir))·`lateralErosion`: flow running ACROSS slope (ramming a bank) cuts sideways → channel migrates → meander grows. both capped by CAP (bounded V21). eps-guard normalize (zero vel/grad → NaN). bootstraps from terrain irregularity.
V40: SEAM CREASE FIX = cross-seam C1 continuity (apron effect ⊥ texture resize). independent per-face erosion diverges the bedrock SLOPE at a shared edge → crease (heightSeam only matches edge VALUE not slope; parity ruled out by test/seamCoincide). `buildSeamSmooth`: in a narrow band (≤2.5 cells) along each edge, diffuse b toward avg of CROSS-SEAM neighbors (`seamHeight`, verified map), weight 1@edge→0@band, ×`seamSmooth`. two-phase (read main→scratch→copy) each erosion tick after heightSeam. gated to band (interior skipped → cheap). escalate to full ghost-cell apron only if insufficient.
V41: grid storage tex (`sim/gridStore.ts`) = LinearFilter → materials sample HARDWARE bilinear/bicubic-9tap (1 fetch ⊥ 4 loads + lerp ALU). needs WebGPU `float32-filterable` (three requests when avail; present on target Metal). sim UNAFFECTED: compute reads `textureLoad` = exact texel, bypasses sampler. supersedes T7 "r32float NearestFilter, not filterable" (sphere-era).
V42: TICK-RATE INVARIANCE. `SIM.ticksPerSecond` (now 60, was 20) free to change ONLY because: (a) dt-scaled passes (flux/rain/evap/advect/thermal-dt/momentum) auto-invariant; (b) PER-INVOCATION erosion amounts (erode/deposit caps, loose slump, still-water settling) × `erosionUniforms.tickNorm` = `SIM.tickRateRef(20)/ticksPerSecond`; (c) per-invocation EXPONENTIAL decays (activity viz/wet) rebased `pow(decay, tickNorm)` @ FlatSim ctor; (d) oceanRelax = `1-exp(-14·dt)` (rate-exact, ⊥ smoothstep(dt·8) which drifted w/ dt); (e) pipe-flux `damping` per-TICK exponential — NOT analytically rebasable: 0.65 unrebased @60 = momentum erased → water pools behind tiny bars ("friction too high"); exact per-second rebase 0.87 = chaotic ocean slosh-back. TUNED 0.78 @ 60tps (FlatSim ctor); retune on rate change. drainage test: rain-stop land water 193→23 in 30s, monotonic. erosion gate = `SIM.erosionTickInterval` (4 → 15/s @ 60; tuned @ interval 8 / 20tps = 2.5/s, ref `erosionTickIntervalRef`) — tickNorm folds BOTH tick rate + gate divisor: norm = (tickRateRef/intervalRef)/(tps/interval). NEW per-invocation rate w/o tickNorm = dynamics depend on tick rate = BUG. smaller doses more often → smooth turbidity (the 2.5Hz muddy pulse fix).
V39: MATERIAL stratification (⊥ uniform → straight rivers + runaway downcut). (a) erodibility field MULTI-SCALE: large resistant zones (low-freq) + mid + fine, sharp contrast (`heightField` hardness, range ~[0.12,2.6]) → flow winds around hard zones ⊥ shoots straight. (b) STRATA: erode × resist(b) where resist = mix(1, smoothstep(sin(b·`strataFreq`)|abs|), `strataStrength`) → thin hard ROCK LAYERS by elevation; river incises until it meets a hard band then STALLS (scales vertical+lateral) → terraced canyons, ⊥ endless deepening/widening.

## §T TASKS
id|status|task|cites
T1|x|scaffold Vite+TS, install three(webgpu)+ui(lil-gui), npm|C
T2|x|write `docs/SCOPE.md`|I.file
T3|x|`src/config.ts` RES,radius,K consts|-
T4|x|M0 `src/tsl/warp.ts` tan-warp + (face,u,v)↔dir + round-trip test|V1,V13
T5|x|M0 `src/planet/cubeSphere.ts`+`PlanetMesh.ts` 6-face geom|V1
T6|x|M0 `src/main.ts`+`app/Engine.ts` WebGPU gate, init, RAF, orbit, flat-shade solid|V8,V10,V11,I.gate
T7|x|M2 `src/sim/fields.ts` r32float StorageTexture ping-pong mgr + seed compute (apron deferred→M3; "NearestFilter, not filterable" STALE → V41: LinearFilter + float32-filterable, flat-era gridStore)|V2,V5,V6,C,V41
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
T20|x|M10 lighting `tsl/lighting.ts`: sun(az/el/intensity)+opposite fill+hemispheric ambient, GUI folder, `sunDirUniform` exposed for weather shaders, soft terminator + dark-side readable|V14,V24,I.ui
T21|x|M10 `materials/atmosphereMaterial.ts`: back-side sphere shell (baseR+0.7), additive limb Fresnel glow brighter on sun side (sunDirUniform), depthWrite off, renderOrder 10|V24,V25,V28
T22|x|M10 `materials/cloudMaterial.ts` + shell (baseR+0.55): dir-based `mx_fractal_noise_float` (seamless 3D), coverage threshold, wind drift (time), sun-lit, depthWrite off. uniforms cloudCoverage/Scale/Opacity/windDir/Speed|V25,V28,V29
T23|x|M10 storm clouds + rain: storminess darkens dense cloud cores (`stormMaskNode`) + `rainMaterial` veil shell (baseR+0.32) falls radially under storm clouds (same mask) + Engine couples storminess→sim rain rate (localized by rainfall map). storminess default 0.25|V6,V25,V26,V29
T24|x|M10 flow-field water — DONE by T31: waterMaterial samples 3x3-smoothed `velocity`, world-space streaks scroll along flowWorld, gated to shallow+fast (rivers), calm ocean. velocity runs every tick|V5,V18,V27
T25|x|M10 weather UI: Controls 'Weather' folder (storminess, cloud coverage/opacity/scale, wind speed, rain, atmosphere, water flow streaks) + 'Lighting' folder (T20). uniforms drive shaders live|I.ui,V24
T26|x|M11 `cellArea` r32float tex/face (tan-warp jacobian) + migrate water+sediment to VOLUME storage; depth=vol/area in flux/erosion/render/seamFlux. verify conservation across seams|V31,V19,V5,V30
T27|x|M11 interior/border SPLIT (shader-level): `seamHeight`+`objNormalAt` use `If` (real control flow) not `select` → interior cells skip cross-seam textureLoad/faceDirNode entirely (was: both branches always eval'd). benefits erosion-tilt(4×)/thermal(4×)/advect/normal-bake automatically. dispatch-range tiling folded into T28 (active tiles)|V32,V5,V10
T28|x|M11 ACTIVE GATE (cell-level): erosion+velocity early-out (`If`) on dry+sediment-free cells -> passthrough, skip heavy work. behavior-preserving. chosen over tile-dispatch (no readback, no missed rivers, no tile-border risk). HUD tile-count + indirect dispatch deferred|V33,V10,V16
T29|x|M11 sediment VISCOSITY in flux: k /= min(1+conc·mudViscosityFactor, 8) (water only; lava omits). `mudViscosityFactor` uniform=6|V34
T30|x|M11 erosion/deposit VIZ tex `erosionViz` (rg, decay 0.95/tick) written in buildErosion + terrain tint (erode→dark earth, deposit→pale fan) + still-water settle (slow water drops sediment → muddy lakes/deltas)|V35,V23
T31|x|M11 flow-direction viz in waterMaterial: WORLD-SPACE streaks scroll along flowWorld (dot(objPos,flowWorld)·85 - time·speed·5), seamless. flowWorld = faceTangents·vel. gated by speed+depth. `flowVizStrength` uniform. velocity now runs EVERY tick (cheap, gated) so viz live w/o erosion|V27,V35
T32|x|M11 DEBUG MODE enum: `debugMaterial` uniform-switched (`debugModeUniform`, DEBUG_MODES), 'v' cycles off→waterDepth→flowSpeed→flowDir→sediment→erosion→deposition→soilDepth→cellArea→activeGate. HUD shows mode|V36,I.ui
T33|x|river rework: incision feedback — erosion capacity ∝ discharge(depth·speed)·`channelFocus`; thermal slump suppressed in deep water (`channelDepthRef`) so channels persist; GUI channel-focus/discharge knobs. emergent channels from a source|V37,V21
T34|x|meandering: flow inertia (velocity = mix(flux,prev-upstream,`flowInertia`)) overshoots bends + lateral cut-bank erosion (∝ discharge·misalign·`lateralErosion`) migrates channel sideways. GUI inertia/lateral knobs|V38,V21
T36|x|seam crease fix: `buildSeamSmooth` cross-seam C1 diffusion in seam band (2-phase, after heightSeam, gated to band) + GUI `seam smoothing`. parity ruled out via test/seamCoincide.test.ts. fps: reverted velocity to erosion-ticks-only (was every-tick for now-default-off streaks)|V40,V10
T35|x|material stratification: multi-scale erodibility (`heightField` hardness: big resistant zones+mid+fine) → rivers wind ⊥ straight; STRATA (erode×resist(sin(b·strataFreq))) → hard rock layers stall downcut/widening → terraces. GUI strata freq/strength|V39,V21

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
