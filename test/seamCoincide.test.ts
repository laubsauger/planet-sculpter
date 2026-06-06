// Seam parity check: a face's edge boundary texel and the neighbor texel the
// seam table maps it to must be the SAME physical vertex (identical sphere
// direction). If they disagree (off-by-one / parity), edge-averaging pairs the
// WRONG texels -> invisible on smooth terrain, beaded seam under erosion.
import { describe, it, expect } from 'vitest';
import { FACES } from '../src/config';
import { faceUVToDir } from '../src/tsl/warp';
import { buildSeamTable, neighborTexel, edgeUV, texelUV, EDGES } from '../src/planet/seamTable';

describe('seam edge texels coincide', () => {
  const res = 32;
  const table = buildSeamTable(res);

  it('maps each edge texel to the coincident neighbor texel (same dir)', () => {
    let worst = 0;
    let worstInfo = '';
    for (const face of FACES) {
      for (const edge of EDGES) {
        const seam = table[face][edge];
        for (let i = 0; i <= res; i++) {
          const [u, v] = edgeUV(edge, i, res);
          const dA = faceUVToDir(face, u, v);
          const [nx, ny] = neighborTexel(seam, i, res);
          const [nu, nv] = texelUV(nx, ny, res);
          const dB = faceUVToDir(seam.nFace, nu, nv);
          const dist = Math.hypot(dA[0] - dB[0], dA[1] - dB[1], dA[2] - dB[2]);
          if (dist > worst) {
            worst = dist;
            worstInfo = `${face} edge${edge} i=${i} -> ${seam.nFace}(${nx},${ny}) dist=${dist.toFixed(4)}`;
          }
        }
      }
    }
    // a correct mapping coincides exactly (shared edge); allow tiny float noise.
    expect(worst, worstInfo).toBeLessThan(1e-4);
  });
});
