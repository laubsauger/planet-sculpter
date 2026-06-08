import { DataTexture, FloatType, RedFormat, NearestFilter, ClampToEdgeWrapping } from 'three';

export const FLAT_BENCHMARKS = [
  'default',
  'riverToSea',
  'damBreak',
  'delta',
  'rainErosion',
  'brushStress',
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

      if (name === 'riverToSea' || name === 'delta') {
        const center = 0.5 + Math.sin(v * Math.PI * 3.2) * 0.055 + Math.sin(v * Math.PI * 7.1) * 0.014;
        const channel = Math.exp(-((u - center) * (u - center)) / (0.032 * 0.032));
        const floodplain = Math.exp(-((u - center) * (u - center)) / (0.11 * 0.11));
        // steeper upper grade, then a SMOOTH coastal ramp into the ocean basin (not a
        // cliff) so a delta can build out on a gentle underwater slope and be visible.
        const land = 0.68 - v * 0.46 - channel * 0.04 - floodplain * 0.012;
        const basin = 0.16;
        const toOcean = smoothstep(0.7, 0.95, v);
        height[k] = Math.max(0.12, land * (1 - toOcean) + basin * toOcean);
        // source sits well below the sealed top edge — against the wall it just pools
        // upslope into the corner and evaporates instead of running downhill.
        const srcV = 0.22;
        const sourceX = 0.5 + Math.sin(srcV * Math.PI * 3.2) * 0.055 + Math.sin(srcV * Math.PI * 7.1) * 0.014;
        source[k] = normalizedGaussian(u, v, sourceX, srcV, 0.022, w, h, 4);
        // Warm-start the visual/routing benchmark with a shallow connected river.
        // Spring-only routing from a dry bed remains a separate solver acceptance case.
        if (v >= srcV && v <= 0.84) water[k] = channel * 0.012;
        if (name === 'delta') sediment[k] = normalizedGaussian(u, v, sourceX, 0.15, 0.04, w, h, 35);
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
