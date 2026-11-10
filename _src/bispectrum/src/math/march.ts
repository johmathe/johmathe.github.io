/**
 * Degree marching: recover a band-limited signal from its selective invariant, one degree at a
 * time. A port of `seed_branches`, `extend_branch` and `march` in `degree_marching.py`.
 *
 * 1. LM on degrees `<= seedDegree` against the entries that involve only those degrees, from
 *    random starts, until `nSeedBranches` solutions are found.
 * 2. For `l = seedDegree + 1 .. lmax`: LM on degree `l` alone against the entries whose largest
 *    degree is `l` (lower degrees frozen); with `jointPolish`, LM on all degrees `<= l`.
 * 3. LM on everything for each surviving branch.
 */

import { dofsWhere } from './basis';
import type { Invariant } from './invariant';
import { lmSolve } from './lm';
import type { Rng } from './rng';

export interface MarchConfig {
  seedDegree: number;
  nSeedInits: number;
  nSeedBranches: number;
  nExtendInits: number;
  maxSolutionsPerBranch: number;
  /** From this degree on, stop at the first consistent solution per branch (`null`: never). */
  singleSolutionFromDegree: number | null;
  beamWidth: number;
  okTol: number;
  jointPolish: boolean;
  /** A frozen per-degree solve below this goes on to the joint polish, which must reach okTol. */
  frozenAcceptTol: number;
  /** When every branch dies at a degree, retry it with this many times more inits. */
  retryFactor: number;
  lmIters: number;
  finalLmIters: number;
}

/** `MarchConfig()` of `degree_marching.py`. */
export const PAPER_CONFIG: MarchConfig = {
  seedDegree: 4,
  nSeedInits: 200,
  nSeedBranches: 2,
  nExtendInits: 40,
  maxSolutionsPerBranch: 2,
  singleSolutionFromDegree: null,
  beamWidth: 12,
  okTol: 1e-10,
  jointPolish: true,
  frozenAcceptTol: 1e-6,
  retryFactor: 4,
  lmIters: 200,
  finalLmIters: 400,
};

export type MarchEvent =
  | { kind: 'seed'; tried: number; found: number }
  | {
      kind: 'degree';
      degree: number;
      branches: Float64Array[];
      bestFrozen: number;
      bestPolished: number | null;
      retried: boolean;
    }
  | { kind: 'final'; branches: { x: Float64Array; rel: number }[] };

export interface MarchResult {
  branches: { x: Float64Array; rel: number }[];
  seedTried: number;
  nSeed: number;
  degrees: { degree: number; alive: number; bestFrozen: number; retried: boolean }[];
}

function norm(x: ArrayLike<number>, idx: Int32Array): number {
  let s = 0;
  for (let i = 0; i < idx.length; i++) s += x[idx[i]!]! ** 2;
  return Math.sqrt(s);
}

function distinct(cand: Float64Array, others: Float64Array[], mask: Int32Array, scale: number) {
  return others.every((o) => {
    let s = 0;
    for (let i = 0; i < mask.length; i++) s += (cand[mask[i]!]! - o[mask[i]!]!) ** 2;
    return Math.sqrt(s) > 1e-6 * scale;
  });
}

/** Scale `s` with `||F(s u)|| = ||target||` on `rows`; `F` is homogeneous and increasing in `s`. */
function seedScale(inv: Invariant, u: Float64Array, target: Float64Array, rows: Int32Array) {
  let goal = 0;
  for (let i = 0; i < rows.length; i++) goal += target[rows[i]!]! ** 2;
  goal = Math.sqrt(goal);
  let lo = -12;
  let hi = 12;
  const v = new Float64Array(rows.length);
  const xs = new Float64Array(u.length);
  for (let it = 0; it < 50; it++) {
    const mid = 0.5 * (lo + hi);
    const s = Math.exp(mid);
    for (let i = 0; i < u.length; i++) xs[i] = s * u[i]!;
    inv.values(xs, rows, v);
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i]! ** 2;
    if (Math.sqrt(n) < goal) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

function seedBranches(
  inv: Invariant,
  target: Float64Array,
  cfg: MarchConfig,
  rng: Rng,
  emit: (e: MarchEvent) => void,
): { found: Float64Array[]; tried: number } {
  const dofs = dofsWhere(inv.lmax, (l) => l <= cfg.seedDegree);
  const rows = inv.rowsWhere((d) => d <= cfg.seedDegree);
  const found: Float64Array[] = [];
  let tried = 0;
  while (tried < cfg.nSeedInits) {
    tried++;
    const u = new Float64Array(inv.dim);
    const noise = rng.normals(dofs.length);
    const nn = Math.hypot(...noise);
    dofs.forEach((d, i) => (u[d] = noise[i]! / nn));
    const s = seedScale(inv, u, target, rows);
    for (let i = 0; i < u.length; i++) u[i]! *= s;
    const sol = lmSolve(inv, u, target, dofs, rows, { iters: cfg.lmIters });
    if (sol.rel <= cfg.okTol) found.push(sol.x);
    emit({ kind: 'seed', tried, found: found.length });
    if (found.length >= cfg.nSeedBranches) break;
  }
  return { found, tried };
}

function extendBranch(
  inv: Invariant,
  x: Float64Array,
  target: Float64Array,
  ell: number,
  cfg: MarchConfig,
  rng: Rng,
  nInits: number,
): { sols: Float64Array[]; bestFrozen: number; bestPolished: number | null } {
  const dofs = dofsWhere(inv.lmax, (l) => l === ell);
  const rows = inv.rowsWhere((d) => d === ell);
  const dofsUpto = dofsWhere(inv.lmax, (l) => l <= ell);
  const rowsUpto = inv.rowsWhere((d) => d <= ell);
  const scale = norm(
    x,
    dofsWhere(inv.lmax, (l) => l === ell - 1),
  );
  const sols: Float64Array[] = [];
  let bestFrozen = Infinity;
  let bestPolished: number | null = null;
  for (let it = 0; it < nInits; it++) {
    const x0 = Float64Array.from(x);
    const noise = rng.normals(dofs.length);
    const nn = Math.hypot(...noise);
    dofs.forEach((d, i) => (x0[d] = (scale * noise[i]!) / nn));
    const sol = lmSolve(inv, x0, target, dofs, rows, { iters: cfg.lmIters });
    bestFrozen = Math.min(bestFrozen, sol.rel);
    if (sol.rel > cfg.frozenAcceptTol) continue;
    let cand = sol.x;
    let rel = sol.rel;
    if (cfg.jointPolish) {
      const pol = lmSolve(inv, cand, target, dofsUpto, rowsUpto, { iters: cfg.lmIters });
      bestPolished = bestPolished === null ? pol.rel : Math.min(bestPolished, pol.rel);
      cand = pol.x;
      rel = pol.rel;
    }
    if (rel > cfg.okTol) continue;
    if (distinct(cand, sols, dofsUpto, scale)) sols.push(cand);
    const single = cfg.singleSolutionFromDegree !== null && ell >= cfg.singleSolutionFromDegree;
    if (sols.length && (single || sols.length >= cfg.maxSolutionsPerBranch)) break;
  }
  return { sols, bestFrozen, bestPolished };
}

export function march(
  inv: Invariant,
  target: Float64Array,
  cfg: MarchConfig,
  rng: Rng,
  emit: (e: MarchEvent) => void = () => {},
): MarchResult {
  const { found, tried } = seedBranches(inv, target, cfg, rng, emit);
  let branches = found;
  const degrees: MarchResult['degrees'] = [
    { degree: cfg.seedDegree, alive: found.length, bestFrozen: NaN, retried: false },
  ];
  emit({
    kind: 'degree',
    degree: cfg.seedDegree,
    branches: branches.map((b) => Float64Array.from(b)),
    bestFrozen: NaN,
    bestPolished: null,
    retried: false,
  });

  for (let ell = cfg.seedDegree + 1; ell <= inv.lmax; ell++) {
    if (!branches.length) break;
    const attempts = [
      cfg.nExtendInits,
      ...(cfg.retryFactor ? [cfg.nExtendInits * cfg.retryFactor] : []),
    ];
    const upto = dofsWhere(inv.lmax, (l) => l <= ell);
    const prev = dofsWhere(inv.lmax, (l) => l === ell - 1);
    let next: Float64Array[] = [];
    let bestFrozen = Infinity;
    let bestPolished: number | null = null;
    let retried = false;
    for (const nInits of attempts) {
      next = [];
      bestFrozen = Infinity;
      bestPolished = null;
      for (const b of branches) {
        const res = extendBranch(inv, b, target, ell, cfg, rng, nInits);
        const scale = norm(b, prev);
        next.push(...res.sols.filter((s) => distinct(s, next, upto, scale)));
        bestFrozen = Math.min(bestFrozen, res.bestFrozen);
        if (res.bestPolished !== null) {
          bestPolished =
            bestPolished === null ? res.bestPolished : Math.min(bestPolished, res.bestPolished);
        }
      }
      if (next.length) break;
      retried = true;
    }
    retried = retried && next.length > 0;
    degrees.push({ degree: ell, alive: next.length, bestFrozen, retried });
    branches = next.slice(0, cfg.beamWidth);
    emit({
      kind: 'degree',
      degree: ell,
      branches: branches.map((b) => Float64Array.from(b)),
      bestFrozen,
      bestPolished,
      retried,
    });
  }

  const allDofs = Int32Array.from({ length: inv.dim }, (_, i) => i);
  const finals = branches.map((b) =>
    lmSolve(inv, b, target, allDofs, inv.allRows, { iters: cfg.finalLmIters }),
  );
  const out = finals.map((f) => ({ x: f.x, rel: f.rel }));
  emit({ kind: 'final', branches: out });
  return { branches: out, seedTried: tried, nSeed: found.length, degrees };
}
