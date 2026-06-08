// Flat island seed fields (W×H, uniform grid). Height: 2D fbm * radial falloff so
// the centre is land and the edges drop below sea level (ocean border the water
// drains into). Plus moisture / erodibility / loose-material maps. CPU Float32
// data kept for picking. Replaces the sphere lon/lat seed.

import {
  DataTexture, RedFormat, FloatType, LinearFilter, ClampToEdgeWrapping,
} from 'three';

export interface FlatTexture {
  texture: DataTexture;
  data: Float32Array;
  w: number;
  h: number;
}

function makeTex(data: Float32Array, w: number, h: number): FlatTexture {
  const texture = new DataTexture(data, w, h, RedFormat, FloatType);
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return { texture, data, w, h };
}

// --- cheap 2D value-noise fbm ---
// Math.imul + unsigned shifts: a plain float multiply loses precision and biases
// the output LOW (~0.24 mean) -> all terrain collapsed below sea level. This is a
// proper 32-bit integer hash, ~uniform [0,1) mean 0.5.
function hash(x: number, y: number): number {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}
function vnoise(x: number, y: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x: number, y: number, oct: number, ridged = false): number {
  let s = 0, amp = 0.5, f = 1, norm = 0;
  for (let i = 0; i < oct; i++) {
    let n = vnoise(x * f, y * f);
    if (ridged) n = 1 - Math.abs(n * 2 - 1);
    s += n * amp;
    norm += amp;
    f *= 2;
    amp *= 0.5;
  }
  return s / norm;
}
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export function buildFlatSeed(w: number, h: number) {
  const height = new Float32Array(w * h);
  const moisture = new Float32Array(w * h);
  const hardness = new Float32Array(w * h);
  const loose = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    const v = j / (h - 1);
    for (let i = 0; i < w; i++) {
      const u = i / (w - 1);
      // --- structured terrain: continents, ranges, plateaus, lowlands, shores ---
      const dx = u - 0.5, dy = v - 0.5;
      const r = Math.sqrt(dx * dx + dy * dy) * 2; // 0 centre .. ~1.41 corner
      const island = smooth(1.4, 0.8, r); // mostly land (~80%), ocean ring near edges
      // OCEAN FRAME: radial falloff only hits ocean at the corners (r~1.41); the
      // mid-edges (r=1.0) stay land and touch the world wall. This box mask keys off
      // distance to the NEAREST edge so a uniform ocean ring wraps all four sides.
      // Organic coastline: warp the edge-distance with noise (broad bays/capes + finer
      // wiggle) so the shore meanders and corners round off instead of a straight square.
      // Wider band so coastal mountains slope into the sea rather than forming a wall.
      const rawEdge = Math.min(u, 1 - u, v, 1 - v); // 0 at border .. 0.5 centre
      const coastWarp = (fbm(u * 0.9 + 5, v * 0.9 + 9, 3) - 0.5) * 0.16
        + (fbm(u * 2.6 + 41, v * 2.6 + 17, 4) - 0.5) * 0.1;
      // outer hard cutoff guarantees the true world edge is always ocean even if the
      // warp pushes a cape outward; the warped band shapes the coastline inside that.
      const frame = smooth(0.03, 0.24, rawEdge + coastWarp) * smooth(0.0, 0.03, rawEdge);
      // domain warp -> organic, non-radial shapes.
      const wu = u + fbm(u * 1.5 + 1.3, v * 1.5 + 4.7, 3) * 0.3;
      const wv = v + fbm(u * 1.5 + 7.1, v * 1.5 + 2.2, 3) * 0.3;
      // value-noise fbm is LOW-CONTRAST (~0.5±0.1) -> features vanish. EXPAND hard.
      const ex = (x: number, k: number) => Math.min(1, Math.max(0, (x - 0.5) * k + 0.5));
      // continental relief: real spread of highs/lows.
      const cont = ex(fbm(wu * 2.6, wv * 2.6, 6), 2.4);
      // MOUNTAIN RANGES: ridged, cubed = sharp steep peaks, gated to mtn regions.
      const rg = fbm(wu * 3.2 + 11, wv * 3.2 + 7, 6, true);
      const region = ex(fbm(wu * 1.2 + 3, wv * 1.2 + 8, 3), 2.2);
      // squared (not cubed) + lower amp -> broader rolling ranges instead of a few
      // spiky peaks dominating; leaves more midland between valley and summit.
      const mtn = rg * rg * 0.62 * region;
      // base land 0.4 (above sea 0.3) + continental variation + mountains.
      let e = (0.4 + (cont - 0.5) * 0.6 + mtn) * island;
      // PLATEAUS: terrace some midland patches (mesas/tablelands).
      const plat = fbm(wu * 2.4 + 20, wv * 2.4 + 30, 3);
      const terr = Math.round(e * 5) / 5;
      e += (terr - e) * smooth(0.65, 0.85, plat) * 0.4;
      e *= frame; // force sub-sea-level along the world edges -> ocean ring
      height[j * w + i] = Math.min(1, Math.max(0, e));
      const mst = (fbm(u * 2.3 + 31, v * 2.3 + 19, 5) - 0.5) * 2.2 + 0.5;
      moisture[j * w + i] = Math.min(1, Math.max(0, mst));
      hardness[j * w + i] = 0.25 + fbm(u * 3.0 + 51, v * 3.0 + 67, 5) * 1.6;
      loose[j * w + i] = fbm(u * 6 + 3, v * 6 + 91, 4) * 0.022;
    }
  }
  return {
    height: makeTex(height, w, h),
    moisture: makeTex(moisture, w, h),
    hardness: makeTex(hardness, w, h),
    loose: makeTex(loose, w, h),
  };
}
