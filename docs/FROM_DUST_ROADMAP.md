# From Dust Fidelity Roadmap

## Target

Build a stylized, highly interactive natural-world simulation in which rivers,
volcanoes, deltas, vegetation, fire, and disasters emerge from interacting
material layers rather than from scripted shapes.

The active implementation is currently the flat `512 x 512` map launched by
`src/main.ts`. This is a better near-term match for From Dust's bounded scenario
maps than the older whole-planet implementations, and it avoids spending a large
part of the simulation budget on cube seams or equirectangular poles.

## What Is Publicly Known About From Dust

Confirmed by Ubisoft/GDC material and contemporary developer interviews:

- Ubisoft Montpellier called the simulation system **Galileo**.
- Galileo simulated flowing water, lava, erosion, sedimentation, vegetation,
  rock, and soil as one dynamic world.
- The implementation was explicitly optimized with 128-bit SIMD and full
  multithreading; the PS3 implementation ran entirely on SPUs.
- The world was represented as interacting layers. Rock was the foundation,
  with soil/sand and vegetation above it, while water and lava flowed over and
  interacted with those layers.
- The team deliberately wanted rivers and volcanoes to emerge from the rules
  instead of using dedicated "river" or "volcano" algorithms.
- Sediment-rich water became more viscous, lava cooled into rock, water eroded
  mobile ground, vegetation spread over suitable ground, and fire burned and
  later regrew vegetation.
- The visual result used dynamic textures for water transparency, moving lava,
  and surfaces adapting to fluids.
- Natural systems were tuned for gameplay and visual impact, not strict physical
  accuracy. Tsunamis were intentionally theatrical.
- The original game ran on Xbox 360, PS3, and Shader Model 3-era PCs and the PC
  release was capped at 30 FPS. The important lesson is disciplined simulation
  scope and data-oriented parallelism, not reproducing its old frame budget.

Not publicly confirmed:

- Exact grid resolution, cell layout, timestep, equations, terrain LOD method,
  water renderer, erosion constants, or pass schedule.
- Whether Galileo used the same virtual-pipe hydraulic erosion method currently
  used here.

## Current Architecture

### Strong foundations

- GPU-resident regular-grid simulation using WebGPU compute and storage textures.
- A flat uniform grid with no seam or pole singularities.
- Pipe-model water with bounded outflow.
- Separate water, flux, velocity, suspended sediment, loose material, source,
  and total-height fields.
- Hydraulic erosion, deposition, thermal slumping, flow inertia, lateral erosion,
  material hardness, river sources, rainfall, and sea fill.
- Terrain and water render directly from the simulation textures.
- Fixed-rate simulation separated from rendering.
- Existing older implementations contain useful working references for lava,
  erosion/deposition activity maps, and debug visualization.

### Main fidelity gaps

1. **Sediment transport is not conservative.**
   The active flat simulation semi-Lagrangian-samples sediment independently of
   water transfer. This can create or destroy sediment and breaks the key
   water-carries-earth behavior needed for convincing deltas.

2. **The material model is incomplete.**
   There is total height plus a loose-depth field, but thermal slumping moves
   total terrain, not specifically mobile soil/sand. Rock, mobile earth, and
   deposited sediment need clearer conservation rules.

3. **Water is optimized for drainage, not disasters.**
   The virtual-pipe solver is suitable for runoff and erosion, but it does not
   retain a physically meaningful horizontal momentum field. It will not produce
   convincing reflected waves, surges, or theatrical tsunamis without a more
   capable shallow-water model.

4. **Lava is not connected to the active flat game.**
   The older cube-sphere path has lava and cooling code, but the running flat
   engine exposes no volcano behavior or water-lava interaction.

5. **No ecological interaction loop exists.**
   Vegetation density, spreading, wetness, fire, burning, and regrowth are central
   to From Dust's living-world effect.

6. **Simulation fields do too little visual work.**
   The active renderer uses water depth and velocity, but not sediment turbidity,
   recent erosion/deposition, wet ground, heat, vegetation, or fire.

7. **Performance is not measured precisely.**
   The HUD reports frame-rate only. There are no per-pass GPU timings, simulation
   budgets, conservation diagnostics, or repeatable benchmark scenarios.

8. **Every active pass and brush stamp dispatches over the full map.**
   At `512 x 512`, a normal water tick performs six full-grid dispatches. Enabling
   erosion adds eight more. Brush pointer movement also stamps and copies the full
   grid for every event.

9. **The active flat path duplicated mature code instead of sharing a solver
   core.**
   There are now cube-sphere, equirectangular, and flat implementations with
   diverging behavior. Continued tuning across all three will slow progress and
   make correctness hard to establish.

## Active Water, Erosion, And Rendering Defects

This section records the current visual and simulation failures so fixes are
evaluated against the same targets instead of being lost during tuning.

### Hydraulic transport

- The production pipe solver preserves water volume but does not transfer
  downstream momentum between cells. At a steep-to-flat transition, incoming
  discharge stops and must build a pressure mound before it can leave again.
- Waterfall feet and river mouths consequently pile far above the receiving
  water surface and read as viscous material rather than water.
- The infinite-ocean level relaxation must happen offshore, not in the first
  shallow coastal cells. The shelf and mouth need to remain part of the
  hydraulic domain so discharge can fan out before excess water is removed.
- The outer ocean band is currently a sealed/clamped boundary. It reflects
  hydraulic disturbances back into the map and lets suspended sediment collect
  into a perimeter wall. It must absorb excess water, flux/momentum, and
  suspended sediment.
- The dam-break benchmark still reads as viscous stretching rather than a
  pressure-driven surge. It is the required comparison case for deciding whether
  the production end state remains an enhanced pipe solver or moves to the
  shallow-water momentum solver.

Acceptance checks:

- A waterfall entering the ocean does not maintain a large terminal mound.
- A river crossing a modest grade reduction remains visibly connected.
- Mouth discharge spreads laterally and offshore instead of stopping at the
  coastline.
- Long runs do not create inward-moving edge waves or perimeter sediment walls.
- Dam-break water accelerates and pushes downstream rather than slowly elongating.

### Sediment and erosion

- Capacity-driven deposition currently favors low-throughflow cells. This can
  dump sediment at the first ocean contact cell, build a bar, and dam the outlet.
  Deposition should respond to actual deceleration/divergence, water depth, and
  settling time.
- Erodibility variation needs broad, medium, and fine volumetric scales with
  restrained amplitudes. The goal is persistent preferred channels and branching,
  not noisy per-cell pits.
- Bed incision, bank erosion, vertical cliff weathering, and waterfall plunge
  erosion need separate gates. They are different natural processes and should
  not all respond to the same local speed/slope product.
- Terrain-height changes must preserve the free water surface and visual
  turbidity should be temporally stable enough not to flicker as the bed changes.
- River-to-ocean ingress needs a soft mixing/deceleration band. Sediment should
  feather into a shallow fan and lose momentum offshore instead of stopping at
  the mouth or travelling frictionlessly to the map edge.
- Current incision/deposition tends to mirror the initial route in straight
  lengthwise cuts and lines. Preferred erodibility routes and lateral migration
  must create restrained asymmetry and eventual meanders.

Acceptance checks:

- Delta sediment forms a shallow fan with multiple preferred routes rather than
  a uniform front or outlet dam.
- A connected river does not excavate a homogeneous row-wide trench.
- Erosion does not make water depth/color visibly pulse every erosion tick.

### Water rendering

- River flow marks must derive from directional pipe discharge, not reconstructed
  velocity plus a local downhill correction. Ambiguous confluences should show
  less directional marking rather than confidently point upstream.
- White foam belongs at breakers, rapids, plunge impacts, and strong shear. Normal
  river flow should use subtle refractive streaks instead of white contour lines.
- River water and ocean water need one continuous depth/turbidity response at
  mouths. The mouth must not look like a separate translucent sheet pasted onto
  the ocean.
- Large flat water bodies need stylized structure: broad swell, intersecting
  wave scales, restrained specular variation, and animated fake caustics visible
  on shallow seabeds.
- The nearly transparent coastal band is too wide. Transparency should tighten
  while retaining readable tropical shallows.
- Coastal breakers are too subdued and need a more visible but still
  bathymetry-following crest/foam treatment.
- The bounded simulation currently renders as a detached square tile. Render a
  non-simulated ocean continuation outside the playable bounds and let distance
  fog conceal the far extent; the simulation boundary remains explicit and
  absorbing underneath.

Acceptance checks:

- Animated river marks travel downstream in the river-to-sea benchmark.
- No static or uphill chevrons appear at confluences and obstacles.
- Clear, muddy, shallow, deep, river, and ocean water remain recognizably water.
- Ocean and lake surfaces remain interesting away from rivers and
  specular-perfect camera angles.
- Normal gameplay cameras cannot see the square simulation edge or a floating
  slab against the background.

### Terrain materials

- Terrain albedo currently reads as smooth solid-color blends. Rock needs broad
  strata/fracture structure plus finer roughness variation; sand needs subtle
  ripples/grain and wet/dry variation; grass/soil need restrained patch structure.
- Detail must be material-specific and world-space stable. One shared noise bump
  cannot make rock, sand, and soil read as distinct materials.

Acceptance checks:

- Rock faces show readable structure at medium distance without noisy silhouettes.
- Beaches and exposed sand are distinguishable from dry soil.
- Material detail remains stable under camera rotation and lighting changes.

### Dispatch cadence

The HUD reports the latest frame that performed compute work. In pipe mode a
normal water tick is 8 dispatches. Erosion runs every fourth water tick and adds
12 dispatches, so an erosion-enabled run alternating between `8` and `20` is
expected when one simulation tick lands in a render frame. Slow render frames can
contain multiple simulation ticks and report combinations such as `36`
(`8 + 8 + 20`). This is a cadence indicator, not a random workload spike. The
extra two erosion dispatches preserve the water surface after thermal slumping,
preventing a bed-height change from becoming a fake hydraulic impulse.

## Recommended Technical Direction

Keep the flat map as the canonical game path for the next milestones. Treat the
older planet paths as reference implementations until the core material system,
simulation tests, and performance budget are stable.

Use the shallow-water momentum solver as the intended production fluid core.
The pipe solver remains a useful cheap runoff/reference mode, but extending it
with pseudo-momentum creates reflected/sloshy feedback and still cannot represent
dam-break pressure, surges, or convincing obstacle interactions. The current
momentum prototype is more coherent in the dam-break comparison but still needs:

- well-balanced wet/dry and bed-step handling;
- stronger terrain routing without excessive numerical diffusion;
- the same absorbing ocean boundary as pipe mode;
- river-source, erosion, sediment, and rendering validation before becoming the
  default.

Use a layered cell state:

```txt
rock height              persistent, slowly erodible
mobile earth volume      sand/soil that slumps, erodes, and deposits
water depth or volume
water momentum x/y
suspended sediment mass
lava volume
lava heat
vegetation density/type
ground wetness
fire intensity/fuel
```

The rendered terrain height is `rock + mobileEarth`. Every transfer must have a
defined source and destination. Gameplay may accelerate transfers, but it should
not silently create or destroy material except through explicit sources, sinks,
and map-boundary rules.

## Milestones

### M0: Establish a Measurable Baseline

Status: implemented and runtime-validated on an Apple M3 Max.

- [x] Declare the flat path canonical in the architecture docs.
- [x] Add deterministic benchmark scenarios: river-to-sea, dam break, delta, rain
  erosion, and brush stress.
- [x] Add debug views for water, velocity, flux, sediment, mobile earth, erosion,
  deposition, and active cells.
- [x] Add optional, low-frequency diagnostics for total water, suspended sediment,
  and terrain material. Readback must not occur in the hot loop.
- [x] Measure CPU frame time, aggregate GPU compute/render time, and dispatch count.
- [x] Capture initial baseline measurements on a representative Apple Silicon device:
  `60 FPS render`, `30-60 Hz water`, and lower-frequency erosion/ecology.

Initial Chrome/WebGPU measurements at `512 x 512`, 20 simulation ticks/second:

- Default pipe-water map: approximately `0.26 ms` aggregate GPU compute,
  `0.48 ms` GPU render, 8 dispatches per tick.
- Delta benchmark with erosion: approximately `0.65 ms` GPU compute,
  `0.47 ms` GPU render, 16 dispatches per tick.
- Dam-break pipe solver: approximately `0.26 ms` GPU compute and 8 dispatches.
- Dam-break momentum prototype: approximately `0.16-0.36 ms` GPU compute and
  5 dispatches. The water-heavy render measured `~19-21 ms` for both solvers and
  is currently the dominant dam-break cost.
- Local-wave-speed momentum dam break: approximately `0.19 ms` GPU compute and
  5 dispatches before sediment coupling. Closed-map water drift measured about
  `0.0016%` over six seconds.
- Coupled momentum delta with sediment and erosion: approximately `0.67 ms` GPU
  compute and 15 dispatches. Earth drift measured about
  `0.00145 / 108,276` over eight seconds.

Exit criteria:

- Reproducible benchmark captures and conservation-error numbers.
- The source of a performance regression can be identified by pass.

### M1: Correct Material Transport

Status: core transport and mobile-earth rules implemented; WebGPU runtime
conservation measurements and invariant automation remain.

- [x] Replace semi-Lagrangian sediment sampling with conservative sediment transfer
  proportional to directional water outflow and local concentration.
- [x] Preserve explicit mobile earth over implicit fixed rock during transfers.
- [x] Make thermal slumping move mobile earth first; only explicit rockfall or
  weathering rules should move rock.
- [x] Add still-water settling and sediment-driven viscosity after conservative
  transport works.
- [x] Port erosion/deposition activity textures and flat debug views from the older
  path.
- [ ] Add automated GPU invariant tests for non-negativity and conservation under closed-boundary
  scenarios.

Exit criteria:

- River sediment reaches the sea without unexplained gain/loss.
- Deltas form from transported material rather than deposition clamps alone.
- Muddy basins slow naturally without stopping all flow.

### M2: Improve Water Dynamics

Status: comparative solver, sediment transport, and erosion coupling implemented;
normal river routing and surge interactions still need validation before it can
replace the pipe solver.

- [x] Prototype a conservative 2D shallow-water finite-volume solver storing
  `h`, `hu`, and `hv`, using a local-wave-speed Rusanov flux.
- [x] Expose pipe and momentum modes side-by-side in the active GUI/HUD.
- [x] Compare it against the current pipe solver in the dam-break benchmark before
  replacing anything.
- [x] Replace the prototype's fixed Rusanov wave-speed bound with local interface
  wave speeds and bounded CFL substeps without GPU readback.
- [x] Add conservative sediment transport and erosion coupling to momentum mode.
- [ ] Require stable wet/dry fronts, terrain edits during flow, dam breaks, river
  routing, and reflected surge waves.
- [ ] Add controllable tsunami/surge injection only after the underlying solver can
  propagate and interact with terrain.
- [ ] Drive foam from velocity divergence, breaking/rapid flow, and shallow fronts
  rather than depth alone.
- [x] Add view-independent, flow-aligned moving bands and sediment tint so moving
  water remains readable when specular lighting is not visible.
- [x] Normalize spring controls as total discharge and add a connected
  river-to-sea visual benchmark.

Current prototype notes:

- A closed dam-break run changed water mass by about `0.0016%` over roughly six
  seconds after local-wave-speed Rusanov fluxes replaced the fixed bound.
- Momentum sediment advection retained seeded sediment to displayed precision
  over eight seconds before erosion was enabled.
- The normal 20 Hz tick uses one momentum substep; larger external timesteps split
  according to a conservative enforced depth/velocity bound, capped at eight.
- Solver switching clears solver-specific momentum, pipe flux, and velocity state
  so stale fields cannot contaminate comparisons.
- Source rates are normalized total discharge rather than per-cell rates.
- The river-to-sea visual benchmark now starts with a shallow connected river and
  retains a normalized spring. Starting from a completely dry channel still pools
  near the spring instead of routing downstream and remains an explicit M2 solver
  defect, not a renderer issue.
- Shallow visual water uses a small render-only surface lift so thin simulated
  rivers do not disappear into the reconstructed terrain depth surface.
- The connected momentum river rendered at approximately `12.7 ms` with
  `0.49 ms` compute in the latest Chrome/WebGPU validation.

Exit criteria:

- Dam-break and tsunami scenarios visibly carry momentum and react to player-made
  barriers.
- Normal river flow remains stable and fast enough for the target budget.

### M3: Restore Lava and Cross-Material Rules

- Port lava to the canonical flat layered state.
- Store lava volume and heat; make viscosity and solidification depend on heat.
- Define water-lava exchange: cooling, steam/evaporation, and conversion to rock.
- Add volcanic source pressure/rate variation so cones and branching flows emerge
  from the solver.
- Render heat, cooling crust, emissive cracks, and steam from simulation fields.

Exit criteria:

- A volcano builds persistent rock, water changes its outcome, and lava can divert
  water without scripted terrain placement.

### M4: Add the Living-World Loop

- Add vegetation suitability from mobile-earth depth, wetness, temperature,
  slope, fire history, and nearby vegetation.
- Simulate spreading at a lower frequency than fluids.
- Add fire propagation using fuel, dryness, wind, and heat from lava.
- Let vegetation reduce loose-earth erosion and increase water retention.
- Add regrowth and a small set of special plant interactions only after the base
  loop is coherent.

Exit criteria:

- Water enables vegetation, vegetation changes erosion, lava/fire clears it, and
  the landscape recovers without scripts.

### M5: Make Interaction Match From Dust

- Add a Breath-style matter tool that removes a bounded volume of water, mobile
  earth, or lava and places exactly that volume elsewhere.
- Show held material clearly and make pickup/deposit rates predictable.
- Pick against the displaced terrain, not the fixed `y=0` plane.
- Add scenario-level disasters and objectives only after the interacting systems
  are reliable.

Exit criteria:

- The main player action is moving simulated matter, and terrain solutions emerge
  from the same rules as natural events.

### M6: Performance Scaling

- Coalesce brush input to at most one stamp per frame and dispatch only its
  bounding rectangle.
- Remove unconditional full-grid copy passes by using true ping-pong state or
  fusing passes where hazards and bindings permit.
- Run systems at justified frequencies: water/momentum fast, erosion slower,
  ecology/fire slower still.
- Add active tile masks and indirect/tiled dispatch only after profiling proves
  full-grid work is the bottleneck.
- Cull or lower the render tessellation outside the useful camera region if maps
  grow beyond the current bounded size. Geometry clipmaps are unnecessary for the
  current `512 x 512` scenario, but become appropriate for much larger maps.

Exit criteria:

- Benchmark scenarios hold the target frame budget with all core systems active.
- Quality scaling changes resolution/frequency intentionally and visibly, rather
  than silently changing simulation behavior.

## Immediate Work Order

1. Build M0 diagnostics and benchmark maps.
2. Implement conservative sediment transport and explicit mobile-earth rules.
3. Port activity/debug visualization to the flat path.
4. Prototype and benchmark the momentum-based shallow-water solver.
5. Port lava plus water-lava interaction.
6. Add vegetation/fire.
7. Add matter pickup/deposit and disasters.
8. Optimize based on measured pass costs.

Do not start with active tiles, large-map terrain LOD, or more procedural render
detail. The largest current fidelity gains come from conservative interacting
materials, momentum-capable water, and making simulation state visible.

## Sources

- GDC Vault, *Creating a High-Performance Simulation: An Interactive Dynamic
  Natural World*: https://www.gdcvault.com/play/1013667/Creating-a-High-Performance-Simulation
- ACM SIGGRAPH 2011 talk, *Developing the interactive dynamic natural world of
  From Dust*: https://doi.org/10.1145/2037826.2037855
- Game Developer, *The Core Of From Dust*:
  https://www.gamedeveloper.com/design/the-core-of-i-from-dust-i-
- Game Developer, GDC Europe coverage:
  https://www.gamedeveloper.com/design/gdc-europe-eric-chahi-talks-convergence-of-technology-and-design-in-i-project-dust-i-
- Ubisoft, official From Dust page:
  https://www.ubisoft.com/en-us/games/from-dust
- Mei, Decaudin, Hu, *Fast Hydraulic Erosion Simulation and Visualization on
  GPU*: https://www-evasion.imag.fr/Publications/2007/MDH07/FastErosion_PG07.pdf
- Losasso and Hoppe, *Terrain Rendering Using GPU-Based Geometry Clipmaps*:
  https://hhoppe.com/gpugcm.pdf
