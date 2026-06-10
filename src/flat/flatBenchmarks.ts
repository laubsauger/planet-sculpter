import { DataTexture, FloatType, RedFormat, NearestFilter, ClampToEdgeWrapping } from 'three';

export const FLAT_BENCHMARKS = [
  'default',
  'riverToSea',
  'damBreak',
  'delta',
  'rainErosion',
  'brushStress',
  'materialTest',
] as const;

export type FlatBenchmark = (typeof FLAT_BENCHMARKS)[number];

export interface FlatBenchmarkData {
  height: DataTexture;
  loose: DataTexture;
  water: DataTexture;
  sediment: DataTexture;
  source: DataTexture;
  rainOn: boolean;
  erosionOn: boolean;
}

function texture(data: Float32Array, w: number, h: number): DataTexture {
  const out = new DataTexture(data, w, h, RedFormat, FloatType);
  out.magFilter = NearestFilter;
  out.minFilter = NearestFilter;
  out.wrapS = ClampToEdgeWrapping;
  out.wrapT = ClampToEdgeWrapping;
  out.needsUpdate = true;
  return out;
}

const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

const gaussian = (x: number, y: number, cx: number, cy: number, radius: number): number => {
  const dx = x - cx;
  const dy = y - cy;
  return Math.exp(-(dx * dx + dy * dy) / (radius * radius));
};

function normalizedGaussian(
  x: number,
  y: number,
  cx: number,
  cy: number,
  radius: number,
  w: number,
  h: number,
  totalRate: number,
): number {
  // Integral of exp(-r²/R²) over the plane is pi*R².
  return gaussian(x, y, cx, cy, radius) * totalRate / (Math.PI * radius * radius * w * h);
}

/** Deterministic diagnostic maps. These intentionally favor clear behavior over art. */
export function buildFlatBenchmark(name: Exclude<FlatBenchmark, 'default'>, w: number, h: number): FlatBenchmarkData {
  const height = new Float32Array(w * h);
  const loose = new Float32Array(w * h);
  const water = new Float32Array(w * h);
  const sediment = new Float32Array(w * h);
  const source = new Float32Array(w * h);

  for (let j = 0; j < h; j++) {
    const v = j / (h - 1);
    for (let i = 0; i < w; i++) {
      const u = i / (w - 1);
      const k = j * w + i;
      const edge = smoothstep(0, 0.08, Math.min(u, v, 1 - u, 1 - v));
      loose[k] = 0.018 * edge;

      if (name === 'materialTest') {
        // A near-flat plateau well above sea level for inspecting per-type detail normals
        // (use with the terrain material-grid debug toggle). A very gentle large-scale tilt
        // + broad swell so the directional sun rakes across the detail relief from a couple
        // of angles. Loose kept full so nothing auto-exposes to rock.
        height[k] = 0.42 + (u - 0.5) * 0.06 + Math.sin(u * Math.PI * 1.5) * Math.sin(v * Math.PI * 1.5) * 0.02;
        loose[k] = 0.022;
        continue;
      }

      if (name === 'riverToSea' || name === 'delta') {
        // Three independent rivers expose grade-break handling side by side. Their
        // progressively steeper upper catchments and different discharges make it
        // obvious when impact erosion or sluggish routing only works for one slope.
        const rivers = [
          { x: 0.23, phase: 0.4, grade: 0.38, rate: 0.55, width: 0.025 },
          { x: 0.5, phase: 2.1, grade: 0.49, rate: 0.75, width: 0.029 },
          { x: 0.77, phase: 4.4, grade: 0.61, rate: 0.95, width: 0.033 },
        ];
        let channel = 0;
        let floodplain = 0;
        let weightedGrade = 0;
        let gradeWeight = 0;
        let sourceRate = 0;
        let initialWater = 0;
        let initialSediment = 0;
        const srcV = 0.2;
        for (const river of rivers) {
          const center = river.x + Math.sin(v * Math.PI * 2.7 + river.phase) * 0.026
            + Math.sin(v * Math.PI * 6.4 + river.phase * 0.7) * 0.008;
          const localChannel = Math.exp(-((u - center) * (u - center)) / (river.width * river.width));
          const localFloodplain = Math.exp(-((u - center) * (u - center)) / (0.075 * 0.075));
          const slopeLane = Math.exp(-((u - river.x) * (u - river.x)) / (0.2 * 0.2));
          channel = Math.max(channel, localChannel);
          floodplain = Math.max(floodplain, localFloodplain);
          weightedGrade += slopeLane * river.grade;
          gradeWeight += slopeLane;

          const sourceX = river.x + Math.sin(srcV * Math.PI * 2.7 + river.phase) * 0.026
            + Math.sin(srcV * Math.PI * 6.4 + river.phase * 0.7) * 0.008;
          sourceRate += normalizedGaussian(u, v, sourceX, srcV, 0.011, w, h, river.rate);
          if (v >= srcV && v <= 0.84) initialWater += localChannel * 0.01;
          if (name === 'delta') {
            const sedimentV = srcV + 0.04;
            const sedimentX = river.x + Math.sin(sedimentV * Math.PI * 2.7 + river.phase) * 0.026
              + Math.sin(sedimentV * Math.PI * 6.4 + river.phase * 0.7) * 0.008;
            initialSediment += normalizedGaussian(u, v, sedimentX, sedimentV, 0.02, w, h, river.rate * 38);
          }
        }
        const grade = gradeWeight > 1e-5 ? weightedGrade / gradeWeight : 0.46;
        // All rivers converge on the same broad shallow receiving shelf. Only the
        // final offshore strip drops into the deep basin.
        const land = 0.69 - v * grade - channel * 0.032 - floodplain * 0.009;
        const shelf = 0.215;
        const basin = 0.16;
        const toShelf = smoothstep(0.7, 0.94, v);
        const toBasin = smoothstep(0.96, 1, v);
        const coast = land * (1 - toShelf) + shelf * toShelf;
        height[k] = Math.max(0.12, coast * (1 - toBasin) + basin * toBasin);
        source[k] = sourceRate;
        water[k] = initialWater;
        sediment[k] = initialSediment;
      } else if (name === 'damBreak') {
        const ridge = gaussian(u, v, 0.5, 0.52, 0.035);
        height[k] = 0.28 + (0.72 - v) * 0.12 + ridge * 0.28;
        water[k] = v < 0.47 && u > 0.18 && u < 0.82 ? 0.16 : 0;
      } else if (name === 'rainErosion') {
        const ridgeA = gaussian(u, v, 0.34, 0.35, 0.18);
        const ridgeB = gaussian(u, v, 0.68, 0.55, 0.15);
        height[k] = (0.2 + ridgeA * 0.45 + ridgeB * 0.35) * edge;
      } else {
        height[k] = (0.32 + Math.sin(u * Math.PI * 12) * Math.sin(v * Math.PI * 12) * 0.035) * edge;
      }
    }
  }

  return {
    height: texture(height, w, h),
    loose: texture(loose, w, h),
    water: texture(water, w, h),
    sediment: texture(sediment, w, h),
    source: texture(source, w, h),
    rainOn: name === 'rainErosion',
    erosionOn: name !== 'damBreak',
  };
}
