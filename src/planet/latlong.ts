// Equirectangular (lat/long) sphere mapping — single seamless grid.
// REPLACES the cube-sphere (warp.ts). One W x H grid covers the whole sphere:
//   u in [0,1] = LONGITUDE (wraps: u=0 and u=1 are the same meridian -> the only
//                seam, handled trivially by mod-wrap neighbor indexing).
//   v in [0,1] = LATITUDE (v=0 south pole, 0.5 equator, v=1 north pole; CLAMPS).
// Pole rows collapse to a point (distortion), masked with ice caps. Cell area
// ∝ cos(latitude) -> stored in cellArea so the sim conserves on the squished
// polar cells. No faces, no seam table, no cross-face anything.

export type Vec3 = [number, number, number];

const TWO_PI = Math.PI * 2;

/** (u,v) in [0,1] -> unit direction on the sphere. */
export function lonLatToDir(u: number, v: number): Vec3 {
  const lon = u * TWO_PI; // 0..2π, wraps
  const lat = (v - 0.5) * Math.PI; // -π/2..π/2
  const cl = Math.cos(lat);
  return [cl * Math.cos(lon), Math.sin(lat), cl * Math.sin(lon)];
}

/** Unit direction -> (u,v) in [0,1). Inverse of lonLatToDir. */
export function dirToLonLat(dir: Vec3): { u: number; v: number } {
  const lat = Math.asin(Math.max(-1, Math.min(1, dir[1]))); // -π/2..π/2
  let lon = Math.atan2(dir[2], dir[0]); // -π..π
  if (lon < 0) lon += TWO_PI; // 0..2π
  return { u: lon / TWO_PI, v: lat / Math.PI + 0.5 };
}

/** Relative cell area at latitude row v (∝ cos lat), for the cellArea field. */
export function cellAreaAt(v: number): number {
  const lat = (v - 0.5) * Math.PI;
  return Math.max(1e-3, Math.cos(lat)); // tiny but >0 at poles (⊥ div-by-zero)
}
