// Global tunables. RES & sim rates adaptive at runtime (see Engine).

export const PLANET = {
  /** Base sphere radius (world units). Heights stored as offset from this (V6). */
  baseRadius: 2,
  /** Per-face grid resolution (verts/texels per edge). Normals are baked to a
   *  texture (compute) so the fragment cost is res-independent; sim cost ~res². */
  res: 768,
  /** Max vertical displacement of height=1.0 in world units. Middle ground:
   *  pronounced mountains/canyons that read well, still a clear sphere. */
  heightScale: 0.48,
  /** Sea level in stored-height units [0..1]: terrain below is underwater. */
  seaLevel: 0.14,
  /** Equirectangular grid (pivot away from cube-sphere): lon cols (wrap) × lat
   *  rows (clamp at poles). lon ~2× lat. texel(tx,ty) <-> u=tx/lonRes, v=ty/(latRes-1). */
  lonRes: 768,
  latRes: 384,
} as const;

/** Flat From-Dust-style local map (pivot away from the sphere). Uniform W×H grid
 *  -> all resolution in the visible patch, no pole/distortion. Island: terrain in
 *  the middle, ocean at the edges (water drains off into it). */
export const FLAT = {
  gridW: 512,
  gridH: 512,
  worldSize: 12, // XZ extent (world units)
  heightScale: 3.8, // max Y displacement of height=1 (steep mountains)
  seaLevel: 0.26, // normalized height of the sea surface (~80% land above)
  meshDetail: 1, // reconstruction/shading smooth the sim grid without multiplying triangles
  /** Water sheet tessellation relative to the sim grid. The liquid surface is C1-smooth
   *  (bicubic) and ALL its shading (depth color, opacity, foam, waterline) is per-PIXEL
   *  from the full-res sim textures — only the sheet's geometric elevation coarsens.
   *  0.5 measured pixel-equivalent to 1.0 (diff at/below the animation noise floor in
   *  overview, grid-edge and river close-up A/B) while cutting ~9ms GPU: the 512² sheet
   *  was ~0.7px/triangle = pure raster/quad-overdraw waste. */
  waterMeshDetail: 0.5,
} as const;

export const SIM = {
  /** Fixed sim timestep (s). */
  dt: 1 / 60,
  /** Target sim ticks/second (throttled below render; V10). 60 = one tick per
   *  rendered frame -> water/sediment fields update every frame (no 20Hz visual
   *  stepping). Per-second dynamics stay TUNED-RATE-equivalent: dt-scaled passes
   *  are invariant automatically; per-invocation erosion amounts are normalized
   *  by tickNorm = tickRateRef/ticksPerSecond (see FlatSim). */
  ticksPerSecond: 60,
  /** Tick rate the per-invocation erosion/deposit caps + decays were tuned at.
   *  Changing ticksPerSecond keeps dynamics fixed by scaling those amounts with
   *  tickRateRef/ticksPerSecond — do NOT retune constants when changing rate. */
  tickRateRef: 20,
  /** Max sim substeps per frame (anti spiral-of-death; V10). */
  maxStepsPerFrame: 4,

  // Pipe-model hydraulic constants (Mei et al.). Tuned in M5 (T14).
  gravity: 9.81,
  pipeArea: 1.0,
  pipeLength: 1.0,
  rainRate: 0.0008,
  evaporation: 0.04,
  sedimentCapacity: 0.25, // Kc
  dissolve: 0.3, // Ks
  deposit: 0.3, // Kd
  /** Thermal talus angle range (rad) by hardness 0..1. */
  talusMin: 0.45,
  talusMax: 0.9,
} as const;

export const RENDER = {
  /** Target frames/second (V10). */
  targetFps: 60,
  /** Frame-time budget (ms) before adaptive downscale. */
  frameBudgetMs: 1000 / 60,
  /** Render-mesh tessellation multiplier over the sim grid. The sim stays at
   *  lonRes×latRes but the displaced mesh is meshDetail× denser, so the smooth
   *  (Hermite) height interp + procedural detail noise become real sub-grid
   *  GEOMETRY instead of being collapsed onto the coarse grid vertices. 2 ≈ 4×
   *  the triangles; bump for more silhouette detail, drop if fps suffers. */
  meshDetail: 2,
} as const;

/** 6 cube faces, fixed index order. Used everywhere (warp, seams, textures). */
export const FACES = ['px', 'nx', 'py', 'ny', 'pz', 'nz'] as const;
export type FaceName = (typeof FACES)[number];
export const FACE_COUNT = 6;
