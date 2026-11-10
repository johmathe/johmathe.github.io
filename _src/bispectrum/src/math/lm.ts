/**
 * Levenberg-Marquardt on `||F(x) - target|| / ||target||` over a subset of dofs and invariant
 * rows: a port of `lm_solve` in `degree_marching.py`.
 */

import type { Invariant } from './invariant';
import { choleskySolve, luSolve, normalEquations } from './linalg';

export interface LMOptions {
  iters?: number;
  lam0?: number;
  tol?: number;
}

export interface LMResult {
  x: Float64Array;
  rel: number;
  iters: number;
}

function sqnorm(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return s;
}

export function lmSolve(
  inv: Invariant,
  x0: ArrayLike<number>,
  target: Float64Array,
  dofs: Int32Array,
  rows: Int32Array,
  { iters = 400, lam0 = 1e-2, tol = 1e-13 }: LMOptions = {},
): LMResult {
  const nr = rows.length;
  const nd = dofs.length;
  const tgt = Float64Array.from(rows, (e) => target[e]!);
  const tnorm = Math.sqrt(sqnorm(tgt)) || 1;
  let x = Float64Array.from(x0);
  const val = new Float64Array(nr);
  const jac = new Float64Array(nr * nd);
  const h = new Float64Array(nd * nd);
  const g = new Float64Array(nd);
  const a = new Float64Array(nd * nd);
  const step = new Float64Array(nd);
  const diag = new Float64Array(nd);

  const resid = (xx: Float64Array, out: Float64Array): Float64Array => {
    inv.values(xx, rows, out);
    for (let i = 0; i < nr; i++) out[i]! -= tgt[i]!;
    return out;
  };

  let r: Float64Array = resid(x, new Float64Array(nr));
  let rn: Float64Array = new Float64Array(nr);
  let cost = sqnorm(r);
  let lam = lam0;
  const hist = [Math.sqrt(cost) / tnorm];

  for (let it = 0; it < iters; it++) {
    inv.jacobian(x, rows, dofs, val, jac);
    normalEquations(jac, r, nr, nd, h, g);
    let dmax = 0;
    for (let i = 0; i < nd; i++) dmax = Math.max(dmax, h[i * nd + i]!);
    for (let i = 0; i < nd; i++) diag[i] = Math.max(h[i * nd + i]!, 1e-12 * dmax);

    let accepted = false;
    while (lam < 1e14) {
      a.set(h);
      for (let i = 0; i < nd; i++) {
        a[i * nd + i]! += lam * diag[i]!;
        step[i] = -g[i]!;
      }
      let ok = choleskySolve(a, step, nd);
      if (!ok) {
        a.set(h);
        for (let i = 0; i < nd; i++) {
          a[i * nd + i]! += lam * diag[i]!;
          step[i] = -g[i]!;
        }
        ok = luSolve(a, step, nd);
      }
      if (ok) {
        const xn = Float64Array.from(x);
        for (let i = 0; i < nd; i++) xn[dofs[i]!]! += step[i]!;
        resid(xn, rn);
        const cn = sqnorm(rn);
        if (cn < cost) {
          x = xn;
          [r, rn] = [rn, r];
          cost = cn;
          lam = Math.max(lam / 5, 1e-15);
          accepted = true;
          break;
        }
      }
      lam *= 4;
    }
    hist.push(Math.sqrt(cost) / tnorm);
    if (!accepted || hist[hist.length - 1]! < tol) break;
    if (hist.length > 40 && hist[hist.length - 1]! > 0.999 * hist[hist.length - 40]!) break;
  }
  return { x, rel: hist[hist.length - 1]!, iters: hist.length - 1 };
}
