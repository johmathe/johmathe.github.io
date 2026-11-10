/**
 * Real coordinates of a band-limited real signal, the `RealSHBasis` of `degree_marching.py`.
 *
 * Per degree `l`: `Re f_l0`, then `Re f_lm, Im f_lm` for `m = 1..l`, scaled by `sqrt 2`, so the
 * coordinate vector is isometric to `L2(S^2)`. Dof `(l, 0)` sits at `l^2`, `(l, m, re)` at
 * `l^2 + 2m - 1` and `(l, m, im)` at `l^2 + 2m`.
 */

import { halfSpectrum, type HalfSpectrum } from './sht';

const SQRT2 = Math.SQRT2;
const ISQRT2 = Math.SQRT1_2;

export const dim = (lmax: number): number => (lmax + 1) * (lmax + 1);

export function degreeOfDof(lmax: number): Int32Array {
  const out = new Int32Array(dim(lmax));
  for (let l = 0; l <= lmax; l++) out.fill(l, l * l, (l + 1) * (l + 1));
  return out;
}

/** Dof indices whose degree satisfies `pred`. */
export function dofsWhere(lmax: number, pred: (l: number) => boolean): Int32Array {
  const out: number[] = [];
  for (let l = 0; l <= lmax; l++) {
    if (!pred(l)) continue;
    for (let d = l * l; d < (l + 1) * (l + 1); d++) out.push(d);
  }
  return Int32Array.from(out);
}

export function toHalfSpectrum(x: ArrayLike<number>, lmax: number): HalfSpectrum {
  const c = halfSpectrum(lmax);
  const n = lmax + 1;
  for (let l = 0; l <= lmax; l++) {
    const b = l * l;
    c.re[l * n] = x[b]!;
    for (let m = 1; m <= l; m++) {
      c.re[l * n + m] = x[b + 2 * m - 1]! * ISQRT2;
      c.im[l * n + m] = x[b + 2 * m]! * ISQRT2;
    }
  }
  return c;
}

export function fromHalfSpectrum(c: HalfSpectrum, lmax: number = c.lmax): Float64Array {
  const x = new Float64Array(dim(lmax));
  const n = c.lmax + 1;
  for (let l = 0; l <= Math.min(lmax, c.lmax); l++) {
    const b = l * l;
    x[b] = c.re[l * n]!;
    for (let m = 1; m <= l; m++) {
      x[b + 2 * m - 1] = c.re[l * n + m]! * SQRT2;
      x[b + 2 * m] = c.im[l * n + m]! * SQRT2;
    }
  }
  return x;
}

/** Squared norm of each degree block, the SO(3) power spectrum `||F_l||^2`. */
export function degreePower(x: ArrayLike<number>, lmax: number): Float64Array {
  const out = new Float64Array(lmax + 1);
  for (let l = 0; l <= lmax; l++) {
    let s = 0;
    for (let d = l * l; d < (l + 1) * (l + 1); d++) s += x[d]! * x[d]!;
    out[l] = s;
  }
  return out;
}

/** `x` truncated (or zero-padded) to band-limit `lmax`. */
export function bandlimit(x: ArrayLike<number>, lmax: number): Float64Array {
  const out = new Float64Array(dim(lmax));
  const n = Math.min(out.length, x.length);
  for (let i = 0; i < n; i++) out[i] = x[i]!;
  return out;
}

/**
 * How each entry of the flat complex vector `F[l^2 + l + m]`, `-l <= m <= l`, depends on the
 * dofs: `F_j = sum_s w_s x[dof_s]` with at most two terms. Negative `m` uses
 * `F_l^{-m} = (-1)^m conj(F_l^m)`.
 */
export interface FlatMap {
  lmax: number;
  /** Two dof indices per flat entry (the second repeats the first when there is one term). */
  dof: Int32Array;
  wRe: Float64Array;
  wIm: Float64Array;
}

export function flatMap(lmax: number): FlatMap {
  const n = dim(lmax);
  const dof = new Int32Array(2 * n);
  const wRe = new Float64Array(2 * n);
  const wIm = new Float64Array(2 * n);
  for (let l = 0; l <= lmax; l++) {
    for (let m = -l; m <= l; m++) {
      const j = l * l + l + m;
      const mu = Math.abs(m);
      if (mu === 0) {
        dof[2 * j] = dof[2 * j + 1] = l * l;
        wRe[2 * j] = 1;
        continue;
      }
      const sign = m < 0 && mu % 2 === 1 ? -1 : 1;
      dof[2 * j] = l * l + 2 * mu - 1;
      dof[2 * j + 1] = l * l + 2 * mu;
      wRe[2 * j] = sign * ISQRT2;
      wIm[2 * j + 1] = (m < 0 ? -sign : sign) * ISQRT2;
    }
  }
  return { lmax, dof, wRe, wIm };
}

/** Flat complex coefficients of `x` (length `dim(lmax)` each). */
export function toFlat(
  x: ArrayLike<number>,
  fm: FlatMap,
  re: Float64Array = new Float64Array(dim(fm.lmax)),
  im: Float64Array = new Float64Array(dim(fm.lmax)),
): { re: Float64Array; im: Float64Array } {
  const n = dim(fm.lmax);
  for (let j = 0; j < n; j++) {
    const a = x[fm.dof[2 * j]!]!;
    const b = x[fm.dof[2 * j + 1]!]!;
    re[j] = fm.wRe[2 * j]! * a + fm.wRe[2 * j + 1]! * b;
    im[j] = fm.wIm[2 * j]! * a + fm.wIm[2 * j + 1]! * b;
  }
  return { re, im };
}
