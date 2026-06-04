Yes. Since you already use a **six-plane cube-sphere planet** in **Three.js WebGPU**, the best path is:

```txt
6 cube faces
  -> each face has simulation textures
  -> compute shaders update water / sediment / erosion
  -> render shaders visualize terrain, water depth, flow speed, flow direction, erosion, and deposition
```

This maps well to the From Dust-style approach. Their Galileo sim was described as a real-time simulator for flowing water, lava, erosion, sedimentation, vegetation, rock, and soil, optimized with SIMD and multithreading. In your case, the equivalent is GPU compute over face textures, with terrain rendered from those fields. ([GDC Vault][1])

---

# Core GPU data layout

For each of the six cube faces, allocate same-resolution WebGPU textures.

Assume:

```txt
faceResolution = 256, 512, or 1024
chunkSize = 32 or 64
```

For a first version:

```ts
type PlanetFaceSimTextures = {
  // static or slowly changing
  bedrock: GPUTexture;      // r16float or r32float

  // dynamic
  soil: GPUTexture;         // r16float or r32float
  water: GPUTexture;        // r16float or r32float
  sediment: GPUTexture;     // r16float or r32float

  // derived / transient
  flow: GPUTexture;         // rgba16float
  velocity: GPUTexture;     // rg16float
  erosionDeposit: GPUTexture; // rg16float
};
```

Recommended texture formats:

```txt
bedrock:        r16float, or r32float if precision problems appear
soil:           r16float / r32float
water:          r16float / r32float
sediment:       r16float / r32float
flow:           rgba16float
velocity:       rg16float
erosionDeposit: rg16float
```

Use ping-pong textures for dynamic fields:

```txt
soilA / soilB
waterA / waterB
sedimentA / sedimentB
```

Avoid writing and reading the same texture in one compute pass.

---

# Use volume internally, depth visually

Because a cube-sphere has slightly uneven cell areas, store fluid as **volume** if you want correctness:

```txt
waterVolume
sedimentMass
```

Then derive depth:

```ts
waterDepth = waterVolume / cellArea;
sedimentConcentration = sedimentMass / max(waterVolume, epsilon);
```

For a stylized game, storing water as depth may work, but storing volume gives better conservation across face edges and distorted cells.

You can precompute:

```txt
normal texture
cell area texture
neighbor lookup texture
```

Per face.

```ts
type StaticFaceTextures = {
  normal: GPUTexture;     // rgba16float or rgba32float
  cellArea: GPUTexture;   // r16float / r32float
  neighborMap: GPUTexture; // optional
};
```

---

# Face topology

Since you already use six planes, the key is to make neighbor access seamless.

Each cell needs four neighbor samples:

```txt
east
west
north
south
```

Within a face this is trivial.

Across face boundaries, you need a mapping:

```ts
type NeighborRef = {
  faceIndex: number;
  uv: vec2<i32>;
  rotation: 0 | 1 | 2 | 3;
};
```

For performance, do **not** branch heavily inside every shader invocation if you can avoid it.

Good approach:

```txt
Interior pass:
  process all cells except 1-pixel border with simple same-face sampling

Border pass:
  process face borders with explicit cross-face mapping
```

This avoids every cell paying for seam logic.

---

# Recommended compute pipeline

Use multiple small compute passes instead of one giant pass.

```txt
1. computeFlow
2. applyFlowAndAdvectSediment
3. computeVelocity
4. erosionDeposit
5. soilSlump
6. updateNormals / dirty derived data
```

In WebGPU terms, each pass reads stable input textures and writes output textures.

---

# Pass 1: compute water outflow

Each cell computes how much water wants to leave in four directions.

Inputs:

```txt
bedrock
soil
waterVolume
cellArea
neighbor bedrock / soil / waterVolume / area
```

Derived:

```wgsl
solidRadius = baseRadius + bedrock + soil;
waterDepth = waterVolume / cellArea;
waterSurface = solidRadius + waterDepth;
```

For each neighbor:

```wgsl
neighborSolidRadius = baseRadius + nBedrock + nSoil;
neighborDepth = nWaterVolume / nCellArea;
neighborSurface = neighborSolidRadius + neighborDepth;

heightDelta = waterSurface - neighborSurface;
```

Outflow:

```wgsl
flow = max(0.0, heightDelta) * flowRate * dt;
```

But clamp total flow:

```wgsl
sumFlow = flowE + flowW + flowN + flowS;

if (sumFlow > waterVolume) {
  scale = waterVolume / sumFlow;
  flowE *= scale;
  flowW *= scale;
  flowN *= scale;
  flowS *= scale;
}
```

Store:

```txt
flow.r = east
flow.g = west
flow.b = north
flow.a = south
```

Important: flow should be **volume per tick**, not depth.

Suggested starting constants:

```txt
flowRate = 0.25 to 2.0
minWater = 0.0001
minFlowDelta = 0.00001
```

Use several substeps if needed:

```txt
simSubsteps = 2 to 8
```

For game-feel water, substeps are often better than one aggressive pass.

---

# Pass 2: apply flow and advect sediment

For each cell:

```txt
newWater =
  oldWater
  - ownOutflow
  + neighborIncomingFlow
```

Incoming flow requires sampling neighbors’ flow textures.

Example:

```txt
incomingFromWest = westNeighbor.flowEast
incomingFromEast = eastNeighbor.flowWest
incomingFromSouth = southNeighbor.flowNorth
incomingFromNorth = northNeighbor.flowSouth
```

Sediment moves proportionally with water.

For each outgoing direction:

```wgsl
sedimentConcentration = sedimentMass / max(waterVolume, epsilon);
sedimentMoved = outgoingWater * sedimentConcentration;
```

Then:

```txt
newSediment =
  oldSediment
  - outgoingSediment
  + incomingSediment
```

You can either:

1. store sediment flow in a separate `sedimentFlow` texture, or
2. recompute sediment moved from neighbor water flow and neighbor sediment concentration during apply pass.

For v1, recompute from neighbor values.

---

# Pass 3: compute velocity texture

This is mostly for erosion and visualization.

Compute net directional flow from the flow texture:

```wgsl
vx = flowEast - flowWest;
vy = flowNorth - flowSouth;
speed = length(vec2(vx, vy)) / cellArea;
```

Store:

```txt
velocity.r = vx
velocity.g = vy
```

Since this is a cube-face local velocity, the render shader can convert it into world-space flow direction using face tangent vectors.

Precompute or derive per face:

```txt
face tangent U
face tangent V
cell normal
```

World-space flow direction:

```wgsl
flowWorld = normalize(faceTangentU * vx + faceTangentV * vy);
```

For more accurate cube-sphere tangents, derive local tangents from neighboring cell positions:

```wgsl
posE = normalE * radiusE;
posW = normalW * radiusW;
posN = normalN * radiusN;
posS = normalS * radiusS;

tangentU = normalize(posE - posW);
tangentV = normalize(posN - posS);

flowWorld = normalize(tangentU * vx + tangentV * vy);
```

This is better near face edges.

---

# Pass 4: erosion and deposition

This is the From Dust-relevant part.

From Dust is publicly described as using rules where moving water erodes terrain, rivers emerge from moving soil/water, and sediment can accumulate in lakes, increasing viscosity. ([Wikipedia][2]) We do not know their exact equations, so implement the same class of behavior: **velocity-based carrying capacity, erosion when under capacity, deposition when over capacity**.

For each cell:

```wgsl
waterDepth = waterVolume / cellArea;
speed = length(velocity) / cellArea;
slope = estimateSlopeFromNeighbors();
```

Sediment capacity:

```wgsl
capacity =
  sedimentCapacityFactor *
  waterDepth *
  speed *
  slopeFactor;
```

A practical slope factor:

```wgsl
slopeFactor = 0.25 + slope * slopeInfluence;
```

Then:

```wgsl
if sedimentMass < capacity {
  erode = min(
    soil,
    (capacity - sedimentMass) * erosionRate * dt
  );

  soil -= erode;
  sedimentMass += erode;
  erosionDebug = erode;
  depositDebug = 0.0;
} else {
  deposit = min(
    sedimentMass,
    (sedimentMass - capacity) * depositionRate * dt
  );

  soil += deposit;
  sedimentMass -= deposit;
  erosionDebug = 0.0;
  depositDebug = deposit;
}
```

Store debug:

```txt
erosionDeposit.r = erosion amount
erosionDeposit.g = deposit amount
```

Suggested starting values:

```txt
sedimentCapacityFactor = 0.5 to 4.0
erosionRate = 0.01 to 0.2
depositionRate = 0.02 to 0.5
slopeInfluence = 1.0 to 8.0
```

Use exaggeration. From Dust-style terrain changes should happen fast enough to be visible.

---

# Pass 5: soil slumping

Without this, deposited soil can create ugly spikes.

For each cell, compare solid radius with neighbors:

```wgsl
solid = bedrock + soil;
neighborSolid = nBedrock + nSoil;
delta = solid - neighborSolid;
```

If the slope exceeds the loose-material angle of repose:

```wgsl
if delta > maxStableSlope {
  move = (delta - maxStableSlope) * slumpRate * dt;
}
```

Clamp:

```wgsl
move = min(move, soil * 0.25);
```

As with water, write a transfer texture or do a ping-pong pass.

Recommended:

```txt
soilFlow.rgba = amount of soil moving east/west/north/south
applySoilFlow pass
```

Suggested values:

```txt
maxStableSlope = 0.02 to 0.2 depending on world scale
slumpRate = 0.2 to 2.0
```

This gives soil a granular “soft terrain” behavior.

---

# Dynamic surface shaders

You want shaders that show:

```txt
water depth
flow speed
flow direction
erosion
deposition
sediment concentration
```

Do this with simulation textures bound into the terrain and water materials.

## Terrain shader inputs

```txt
bedrock texture
soil texture
water texture
sediment texture
velocity texture
erosionDeposit texture
normal texture
```

Terrain vertex displacement:

```wgsl
solidRadius = baseRadius + bedrock + soil;
position = planetCenter + normal * solidRadius;
```

Terrain surface color:

```wgsl
erosion = erosionDeposit.r;
deposit = erosionDeposit.g;
sediment = sedimentMass;
wetness = smoothstep(0.0, wetThreshold, waterDepth);
```

Use these to blend:

```txt
dry soil color
wet soil color
fresh erosion highlight
fresh deposit highlight
sediment/mud color
```

You can decay erosion/deposit debug over time:

```txt
erosionDeposit *= exp(-dt * debugDecay)
```

Or keep a separate visualization accumulation texture.

---

## Water shader inputs

Water mesh uses:

```txt
water texture
velocity texture
sediment texture
normal texture
terrain height
```

Water vertex displacement:

```wgsl
waterRadius = baseRadius + bedrock + soil + waterDepth;
position = planetCenter + normal * waterRadius;
```

Alpha by depth:

```wgsl
alpha = smoothstep(minVisibleDepth, maxOpaqueDepth, waterDepth);
```

Turbidity by sediment concentration:

```wgsl
mud = sedimentMass / max(waterVolume, epsilon);
```

Flow speed:

```wgsl
speed = length(velocity);
```

Use speed for:

```txt
normal map distortion strength
foam intensity
streak brightness
surface roughness
```

---

# Flow direction visualization

For debugging and stylized rendering, you have three good options.

## Option 1: UV-advected flow texture

Keep a repeating streak/noise texture. Offset it by velocity:

```wgsl
flowUV = uv + velocity.xy * time * flowAnimScale;
streak = texture(flowNoise, flowUV);
```

This is cheap, but because velocity is face-local, it may show seams unless you handle face UV continuity.

Good for debug and stylized water.

---

## Option 2: World-space procedural flow noise

Use world position and flow direction.

```wgsl
flowWorld = normalize(tangentU * velocity.x + tangentV * velocity.y);

phase = dot(worldPosition, flowWorld) * streakFrequency - time * speed * streakSpeed;
streak = sin(phase);
```

Then sharpen:

```wgsl
streak = smoothstep(0.6, 1.0, streak);
```

This avoids relying too heavily on face UVs.

---

## Option 3: Arrow/vector debug overlay

For developer mode, render instanced arrows or line segments per Nth cell.

```txt
every 4th or 8th cell:
  position = surface position
  direction = flowWorld
  length = speed * scale
```

This is much clearer than trying to infer direction from water shading.

In Three.js, you can generate an instanced arrow mesh or use a line-list geometry updated from a GPU-readback occasionally. But avoid frequent GPU readback.

Better:

```txt
debug shader renders arrows procedurally from velocity texture
```

For each terrain fragment, draw a small arrow glyph in local cell UV.

---

# Shader debug modes

Implement explicit debug modes rather than overloading the final art shader.

```ts
enum PlanetDebugMode {
  None,
  WaterDepth,
  FlowSpeed,
  FlowDirection,
  Sediment,
  Erosion,
  Deposition,
  SoilDepth,
  CellArea,
  ActiveChunks
}
```

In shader:

```wgsl
switch debugMode:
  WaterDepth:
    color = ramp(waterDepth)
  FlowSpeed:
    color = ramp(length(velocity))
  FlowDirection:
    color = directionToColor(normalize(velocity))
  Sediment:
    color = ramp(sedimentMass / max(waterVolume, epsilon))
  Erosion:
    color = ramp(erosionDeposit.r)
  Deposition:
    color = ramp(erosionDeposit.g)
```

For direction-to-color:

```wgsl
dir = normalize(velocity.xy);
color = vec3(dir * 0.5 + 0.5, 0.0);
```

---

# Three.js WebGPU implementation structure

Assuming modern Three.js WebGPU / TSL or raw WGSL compute, organize like this:

```ts
class PlanetHydroSim {
  faces: PlanetFaceSim[];

  step(dt: number) {
    for (let i = 0; i < substeps; i++) {
      this.computeInteriorFlow(dt / substeps);
      this.computeBorderFlow(dt / substeps);

      this.applyInteriorFlow(dt / substeps);
      this.applyBorderFlow(dt / substeps);

      this.computeVelocity(dt / substeps);
      this.erodeDeposit(dt / substeps);
      this.slumpSoil(dt / substeps);

      this.swapPingPong();
    }
  }
}
```

Each face owns:

```ts
class PlanetFaceSim {
  bedrock: Texture;
  soilA: Texture;
  soilB: Texture;
  waterA: Texture;
  waterB: Texture;
  sedimentA: Texture;
  sedimentB: Texture;

  flow: Texture;
  velocity: Texture;
  erosionDeposit: Texture;

  normal: Texture;
  cellArea: Texture;
}
```

If face resolution is high, avoid simulating the whole planet every frame.

---

# Performance strategy

## 1. Active tiles only

Split each face into tiles:

```txt
tileSize = 16x16 or 32x32
```

Maintain an active tile mask.

A tile is active if:

```txt
water exists
recent terrain edit happened nearby
incoming water from neighbor tile
erosion/deposition recently occurred
```

On CPU:

```ts
type ActiveTile = {
  face: number;
  tileX: number;
  tileY: number;
};
```

Dispatch compute only over active tile ranges if your pipeline supports it conveniently.

Simpler first version:

```txt
simulate full 6 faces at 256²
move to active tiles once behavior is correct
```

At 6 × 256² = 393,216 cells, full-face compute is totally reasonable on many GPUs.

At 6 × 512² = 1,572,864 cells, still possible, but substeps and multiple passes start mattering.

At 6 × 1024² = 6,291,456 cells, you need active tiles or lower-frequency simulation.

---

## 2. Multi-resolution simulation

For stylized planets, do not necessarily simulate water at render resolution.

Example:

```txt
render terrain: 512² per face
fluid sim:      256² per face
detail normal:  procedural shader noise
```

The sim drives large-scale motion. Shader noise adds detail.

This is usually the best tradeoff.

---

## 3. Half-float where possible

Use `rgba16float`, `rg16float`, `r16float` for bandwidth.

Use `r32float` only if:

```txt
planet scale causes precision issues
water levels jitter
erosion/deposition accumulates numerical error
```

---

## 4. Avoid atomics

Do not have cells write directly into neighbors. That creates race conditions or atomics.

Use:

```txt
pass 1: each cell writes own outflows
pass 2: each cell gathers incoming neighbor outflows
```

This is GPU-friendly and deterministic enough.

---

## 5. Interior and border separation

Most cells are not on face seams.

Use fast interior passes:

```txt
dispatch only [1, resolution - 2] on each face
```

Then handle borders separately.

This gives you:

```txt
simple code for 99 percent of cells
special code only for seams
```

---

# Dynamic water viscosity from sediment

From Dust is described as having sediment accumulation in lakes that increases viscosity. ([Wikipedia][2]) You can reproduce that with sediment concentration.

```wgsl
sedimentConcentration = sedimentMass / max(waterVolume, epsilon);

viscosityMultiplier =
  1.0 + sedimentConcentration * mudViscosityFactor;

effectiveFlowRate =
  waterFlowRate / viscosityMultiplier;
```

Use this inside `computeFlow`.

This creates:

```txt
clear fast streams
muddy slow basins
sediment-rich lakes
natural delta buildup
```

Suggested values:

```txt
mudViscosityFactor = 1.0 to 20.0
```

Clamp it:

```wgsl
effectiveFlowRate = max(minFlowRate, waterFlowRate / viscosityMultiplier);
```

---

# Water source / sink handling

Even before player interaction, you need sources and sinks.

For terrain generation and gameplay:

```txt
rainfall mask
springs
ocean/basin fill
evaporation
absorbing boundary not applicable on planet
```

Since it is a closed planet, water has nowhere to leave. That means you need one or more of:

```txt
evaporation
absorption into ground
reservoir limits
temporary water events
```

Otherwise every stream eventually floods the planet.

Basic evaporation:

```wgsl
waterVolume -= evaporationRate * cellArea * dt;
waterVolume = max(0.0, waterVolume);
```

Ground absorption:

```wgsl
absorbed = min(waterVolume, absorptionRate * soilPorosity * dt);
waterVolume -= absorbed;
soilMoisture += absorbed; // optional later
```

For now, evaporation is enough.

---

# Erosion stability tricks

## Clamp erosion by water depth

Do not erode with tiny water amounts:

```wgsl
if waterDepth < minErosionWaterDepth:
  erosion = 0.0;
```

## Clamp erosion by slope or speed

```wgsl
if speed < minErosionSpeed:
  erosion = 0.0;
```

## Limit per-step terrain change

```wgsl
maxChange = maxTerrainChangePerSecond * dt;
erode = min(erode, maxChange);
deposit = min(deposit, maxChange);
```

## Smooth sediment capacity

Raw velocity can flicker. Use damped velocity:

```txt
velocity = mix(oldVelocity, newVelocity, velocitySmoothing)
```

Or use previous velocity texture.

---

# Terrain generation for erosion-ready planets

Because you use six planes, generate from normal-based 3D noise:

```ts
normal = cubeSphereNormal(face, u, v)
height = fbm3D(normal * frequency)
```

Do not generate independent 2D face noise unless you want seams.

Generate:

```txt
bedrock from low-frequency 3D FBM
soil depth from slope and basin masks
initial water from low basins
```

Suggested generation:

```ts
bedrock =
  fbm3D(normal * 1.5) * 0.5 +
  ridgedNoise3D(normal * 4.0) * 0.2;

slope = estimateSlope(bedrock);

soil =
  maxSoilDepth *
  basinMask *
  (1.0 - saturate(slope * slopeSoilLoss));

water =
  max(0, seaLevel - (bedrock + soil));
```

Then run:

```txt
100 to 1000 offline sim steps
```

for pre-erosion if you want natural drainage.

Cache the resulting textures.

---

# Practical WebGPU texture packing

A compact packing option:

```txt
terrainState.r = bedrock
terrainState.g = soil
terrainState.b = waterVolume
terrainState.a = sedimentMass

flowState.r = flowEast
flowState.g = flowWest
flowState.b = flowNorth
flowState.a = flowSouth

velocityState.r = velocityU
velocityState.g = velocityV
velocityState.b = erosion
velocityState.a = deposition
```

But for ping-ponging, separating static and dynamic textures is cleaner:

```txt
bedrock: static
stateA: rgba16float = soil, water, sediment, unused
stateB: rgba16float = soil, water, sediment, unused
flow: rgba16float
debug: rgba16float = velU, velV, erosion, deposit
```

This is probably the best compromise.

---

# Minimal WGSL-ish flow pass

```wgsl
struct Params {
  baseRadius: f32,
  dt: f32,
  flowRate: f32,
  minDelta: f32,
};

@group(0) @binding(0) var bedrockTex: texture_2d<f32>;
@group(0) @binding(1) var stateTex: texture_2d<f32>; // r soil, g waterVolume, b sediment
@group(0) @binding(2) var areaTex: texture_2d<f32>;
@group(0) @binding(3) var flowOut: texture_storage_2d<rgba16float, write>;

fn surfaceAt(p: vec2<i32>) -> f32 {
  let bedrock = textureLoad(bedrockTex, p, 0).r;
  let state = textureLoad(stateTex, p, 0);
  let soil = state.r;
  let waterVolume = state.g;
  let area = textureLoad(areaTex, p, 0).r;
  let waterDepth = waterVolume / max(area, 0.00001);
  return params.baseRadius + bedrock + soil + waterDepth;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = vec2<i32>(id.xy);

  let state = textureLoad(stateTex, p, 0);
  let waterVolume = state.g;

  if (waterVolume <= 0.00001) {
    textureStore(flowOut, p, vec4<f32>(0.0));
    return;
  }

  let s = surfaceAt(p);

  let e = max(0.0, s - surfaceAt(p + vec2<i32>(1, 0)) - params.minDelta);
  let w = max(0.0, s - surfaceAt(p + vec2<i32>(-1, 0)) - params.minDelta);
  let n = max(0.0, s - surfaceAt(p + vec2<i32>(0, 1)) - params.minDelta);
  let so = max(0.0, s - surfaceAt(p + vec2<i32>(0, -1)) - params.minDelta);

  var flow = vec4<f32>(e, w, n, so) * params.flowRate * params.dt;

  let total = flow.x + flow.y + flow.z + flow.w;

  if (total > waterVolume) {
    flow *= waterVolume / total;
  }

  textureStore(flowOut, p, flow);
}
```

This is the core. Everything else builds around it.

For border cells, replace `surfaceAt` with a function that handles cross-face lookup.

---

# Minimal erosion pass

```wgsl
@compute @workgroup_size(8, 8)
fn erodeDeposit(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = vec2<i32>(id.xy);

  var state = textureLoad(stateIn, p, 0);
  var soil = state.r;
  let waterVolume = state.g;
  var sediment = state.b;

  let area = textureLoad(areaTex, p, 0).r;
  let waterDepth = waterVolume / max(area, 0.00001);

  let vel = textureLoad(velocityTex, p, 0).rg;
  let speed = length(vel) / max(area, 0.00001);

  let slope = computeSlope(p);

  var erosion = 0.0;
  var deposition = 0.0;

  if (waterDepth > params.minErosionDepth && speed > params.minErosionSpeed) {
    let capacity =
      params.capacityFactor *
      waterDepth *
      speed *
      (0.25 + slope * params.slopeInfluence);

    if (sediment < capacity) {
      erosion = min(
        soil,
        (capacity - sediment) * params.erosionRate * params.dt
      );

      soil -= erosion;
      sediment += erosion;
    } else {
      deposition = min(
        sediment,
        (sediment - capacity) * params.depositionRate * params.dt
      );

      soil += deposition;
      sediment -= deposition;
    }
  } else {
    deposition = min(
      sediment,
      sediment * params.stillWaterDepositRate * params.dt
    );

    soil += deposition;
    sediment -= deposition;
  }

  textureStore(stateOut, p, vec4<f32>(soil, waterVolume, sediment, 0.0));
  textureStore(debugOut, p, vec4<f32>(vel.x, vel.y, erosion, deposition));
}
```

---

# Visual target

The sim should make these visible:

```txt
fast water:
  bright streaks
  stretched normals
  foam / white edges
  strong flow arrows in debug

slow muddy water:
  darker / more opaque
  lower animation speed
  higher sediment color

active erosion:
  exposed brighter/darker terrain streaks
  debug red/orange

active deposition:
  lighter sediment fans
  debug yellow/green
```

For art direction, do not rely on accurate geometry alone. Use the sim fields as **surface texture drivers**.

---

# What to tell the implementation agent now

```txt
We already use a six-plane cube-sphere planet in Three.js WebGPU. Build the terrain/fluid/erosion sim around that representation.

Use one set of GPU textures per cube face. The canonical simulation state should be texture-based, not mesh-based. Use at minimum bedrock, soil, water volume, sediment mass, flow, velocity, and erosion/deposition debug textures.

Implement the fluid sim as GPU compute over the six faces. Use a shallow-water cellular model. Compute outflow per cell into four face-local directions, clamp total outflow to conserve water mass, then apply flow in a gather pass. Separate interior cells from face-border cells so most compute work avoids expensive cross-face neighbor logic.

Water should flow from higher total water surface radius to lower total water surface radius, where surface radius equals base planet radius plus bedrock plus soil plus water depth. Since this is a cube-sphere, store water as volume and derive water depth using a precomputed cell area texture.

Implement sediment advection by moving sediment proportionally with water transfer. Then implement erosion/deposition using sediment carrying capacity based on water depth, flow speed, and slope. When sediment is below capacity, erode loose soil into sediment. When sediment exceeds capacity, deposit sediment back into soil. Add still-water deposition so lakes and basins accumulate sediment and become muddy.

Add sediment-driven viscosity. Compute sediment concentration as sediment mass divided by water volume. Reduce water flow rate as concentration increases. This reproduces the From Dust-style behavior where sediment-heavy lakes become more viscous.

Implement shader visualization modes. Terrain and water shaders should sample the sim textures. Visualize water depth, flow speed, flow direction, sediment concentration, erosion, and deposition. Use velocity to drive water streaks, normal distortion, and foam. Use erosion/deposition textures to tint terrain dynamically.

Avoid atomics. Avoid direct mesh deformation as the source of truth. Avoid particle water as the authoritative sim. Use particles only as secondary visual effects.

Start with 256² per face full-face simulation. Once stable, add active tile dispatch. Use 16x16 or 32x32 active tiles, with activation from nonzero water, recent edits, incoming border flow, and recent erosion/deposition.
```

---

# Likely best first prototype

Do this in order:

```txt
1. Static cube-sphere terrain textures:
   bedrock, soil, normal, cellArea

2. Water volume texture:
   add initial water in basins

3. GPU compute flow:
   interior pass only first
   then border pass across cube faces

4. Water shader:
   depth alpha
   speed brightness
   direction streaks

5. Sediment:
   sediment mass texture
   sediment advection with water

6. Erosion/deposition:
   capacity model
   debug overlay

7. Soil mesh displacement:
   terrain vertex shader reads updated soil texture

8. Slumping:
   loose soil relaxation pass

9. Active tiles:
   only after correctness
```

The main thing to get right early is not the visual water shader. It is this invariant:

```txt
water + sediment are conserved across all six faces except explicit sources/sinks
```

Once that is stable, the visuals can become highly stylized without breaking the simulation.

[1]: https://www.gdcvault.com/play/1013667/Creating-a-High-Performance-Simulation?utm_source=chatgpt.com "Creating a High-Performance Simulation"
[2]: https://en.wikipedia.org/wiki/From_Dust?utm_source=chatgpt.com "From Dust"
