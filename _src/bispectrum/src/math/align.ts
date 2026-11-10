/**
 * Find `R` with `R . h ~ t` for band-limited `h`, `t` in real coordinates: the `MomentAligner`
 * of `degree_marching.py`, in coefficient space. The quadrupole tensor
 * `Q(f) = int f(p) (3 p p^T - I) dp` depends on degree 2 only and is equivariant,
 * `Q(R . f) = R Q(f) R^T`, so the eigenframes of `Q(h)` and `Q(t)` give four candidate rotations
 * (eigenvector signs); the best are refined by LM on a rotation vector. The residual
 * `||R . h - t|| / ||t||` is the exact L2 error on the sphere (the coordinates are isometric).
 */

import { choleskySolve, symEig3 } from './linalg';
import type { Rng } from './rng';
import {
  applyBlocks,
  det3,
  dofBlocks,
  expMap,
  mul3,
  randomRotation,
  transpose3,
  type Mat3,
} from './rotation';

const QA = Math.sqrt(Math.PI / 5);
const QB = Math.sqrt(Math.PI / 15);

/** Quadrupole tensor of the signal with real coordinates `x` (needs `lmax >= 2`). */
export function quadrupole(x: ArrayLike<number>): Mat3 {
  // Standard real SH coefficients of degree 2 (see rotation.ts for the signed permutation).
  const s0 = x[4]!;
  const s1 = -x[5]!;
  const sm1 = x[6]!;
  const s2 = x[7]!;
  const sm2 = -x[8]!;
  const xy = 6 * QB * sm2;
  const yz = 6 * QB * sm1;
  const xz = 6 * QB * s1;
  return Float64Array.of(
    -2 * QA * s0 + 6 * QB * s2,
    xy,
    xz,
    xy,
    -2 * QA * s0 - 6 * QB * s2,
    yz,
    xz,
    yz,
    4 * QA * s0,
  );
}

export interface Alignment {
  rot: Mat3;
  /** `||R . h - t|| / ||t||`. */
  rel: number;
}

function residualNorm(
  h: Float64Array,
  t: Float64Array,
  rot: Mat3,
  lmax: number,
  buf: Float64Array,
) {
  applyBlocks(dofBlocks(rot, lmax), h, buf);
  let s = 0;
  for (let i = 0; i < t.length; i++) s += (buf[i]! - t[i]!) ** 2;
  return s;
}

/** LM over `R = exp(w) R0` with a central-difference Jacobian (3 columns). */
function refine(
  h: Float64Array,
  t: Float64Array,
  rot0: Mat3,
  lmax: number,
  iters: number,
): Alignment {
  const n = t.length;
  const tn = t.reduce((a, v) => a + v * v, 0) || 1;
  const buf = new Float64Array(n);
  const rp = new Float64Array(n);
  const rm = new Float64Array(n);
  const r0 = new Float64Array(n);
  let rot = rot0;
  let cost = residualNorm(h, t, rot, lmax, buf);
  let lam = 1e-3;
  const eps = 1e-6;
  let converged = false;
  for (let it = 0; it < iters && !converged; it++) {
    applyBlocks(dofBlocks(rot, lmax), h, r0);
    for (let i = 0; i < n; i++) r0[i]! -= t[i]!;
    const jac = new Float64Array(n * 3);
    for (let k = 0; k < 3; k++) {
      const w = [0, 0, 0];
      w[k] = eps;
      applyBlocks(dofBlocks(mul3(expMap(w), rot), lmax), h, rp);
      w[k] = -eps;
      applyBlocks(dofBlocks(mul3(expMap(w), rot), lmax), h, rm);
      for (let i = 0; i < n; i++) jac[i * 3 + k] = (rp[i]! - rm[i]!) / (2 * eps);
    }
    const hh = new Float64Array(9);
    const g = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 3; a++) {
        g[a]! += jac[i * 3 + a]! * r0[i]!;
        for (let b = 0; b < 3; b++) hh[a * 3 + b]! += jac[i * 3 + a]! * jac[i * 3 + b]!;
      }
    }
    let accepted = false;
    while (lam < 1e8) {
      const a = Float64Array.from(hh);
      for (let k = 0; k < 3; k++) a[k * 4]! *= 1 + lam;
      const step = Float64Array.from(g, (v) => -v);
      if (choleskySolve(a, step, 3)) {
        const cand = mul3(expMap(step), rot);
        const c = residualNorm(h, t, cand, lmax, buf);
        if (c < cost) {
          converged = cost - c < 1e-8 * cost || c < 1e-28 * tn;
          rot = cand;
          cost = c;
          lam = Math.max(lam / 5, 1e-12);
          accepted = true;
          break;
        }
      }
      lam *= 4;
    }
    if (!accepted) break;
  }
  return { rot, rel: Math.sqrt(cost / tn) };
}

export interface AlignOptions {
  refineIters?: number;
  /** Extra random restarts when the quadrupole frames leave a residual above `restartAbove`. */
  restarts?: number;
  restartAbove?: number;
  rng?: Rng;
}

export function align(
  h: Float64Array,
  t: Float64Array,
  { refineIters = 30, restarts = 16, restartAbove = 1e-3, rng }: AlignOptions = {},
): Alignment {
  const lmax = Math.round(Math.sqrt(t.length)) - 1;
  const tn = t.reduce((a, v) => a + v * v, 0) || 1;
  const buf = new Float64Array(t.length);
  const score = (rot: Mat3): Alignment => ({
    rot,
    rel: Math.sqrt(residualNorm(h, t, rot, lmax, buf) / tn),
  });

  const candidates: Alignment[] = [];
  if (lmax >= 2) {
    const vh = symEig3(quadrupole(h)).vectors;
    const vt = symEig3(quadrupole(t)).vectors;
    const vhT = transpose3(vh);
    for (const s1 of [1, -1]) {
      for (const s2 of [1, -1]) {
        let rot = mul3(mul3(vt, Float64Array.of(s1, 0, 0, 0, s2, 0, 0, 0, 1)), vhT);
        if (det3(rot) < 0)
          rot = mul3(mul3(vt, Float64Array.of(s1, 0, 0, 0, s2, 0, 0, 0, -1)), vhT);
        candidates.push(score(rot));
      }
    }
  } else {
    candidates.push(score(Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1)));
  }
  candidates.sort((a, b) => a.rel - b.rel);
  let best = candidates[0]!;
  for (const c of candidates.slice(0, 2)) {
    const r = refine(h, t, c.rot, lmax, refineIters);
    if (r.rel < best.rel) best = r;
  }
  if (best.rel > restartAbove && rng) {
    for (let k = 0; k < restarts; k++) {
      const r = refine(h, t, randomRotation(rng), lmax, refineIters);
      if (r.rel < best.rel) best = r;
      if (best.rel < restartAbove) break;
    }
  }
  return best;
}
