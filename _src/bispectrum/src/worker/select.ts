/**
 * Which reconstruction to show. The choice sees only what the solver sees, the invariant: a
 * candidate's residual against the target, with solver order as the tie-break. The original
 * signal enters afterwards, to align the chosen candidate for display and to measure its error.
 */

import { align } from '../math/align';
import { bandlimit } from '../math/basis';
import type { Invariant } from '../math/invariant';
import type { Rng } from '../math/rng';

/**
 * Two accepted candidates closer than this after the best rotation between them are the same
 * signal. The margin is wide on purpose: on symmetric signals the invariant is flat to first
 * order along some directions, so matching it to `okTol` only pins the signal to about
 * `sqrt(okTol)`.
 */
export const SAME_ORBIT_REL = 1e-3;

/** `||F(x) - t|| / ||t||` on `rows`, the residual `lmSolve` minimizes. */
export function residual(
  inv: Invariant,
  x: ArrayLike<number>,
  target: Float64Array,
  rows: Int32Array,
): number {
  const v = inv.values(x, rows);
  let num = 0;
  let den = 0;
  for (let i = 0; i < rows.length; i++) {
    const t = target[rows[i]!]!;
    num += (v[i]! - t) ** 2;
    den += t * t;
  }
  return Math.sqrt(num) / (Math.sqrt(den) || 1);
}

/**
 * Candidate indices by residual, smallest first, ties in solver order. Depends on nothing but
 * `rels`, so two runs with the same invariant and seed rank the same way.
 */
export function rank(rels: ArrayLike<number>): number[] {
  return Array.from({ length: rels.length }, (_, i) => i).sort(
    (a, b) => rels[a]! - rels[b]! || a - b,
  );
}

/** The candidate to show, `-1` when there is none. */
export const choose = (rels: ArrayLike<number>): number => (rels.length ? rank(rels)[0]! : -1);

/**
 * Groups the accepted candidates (`rels[i] <= okTol`) into distinct signals by aligning each one
 * with the representatives found so far, in rank order. Returns one group index per candidate,
 * `-1` for rejected ones, and the representative of each group. Blind to the original.
 */
export function orbits(
  xs: readonly Float64Array[],
  rels: ArrayLike<number>,
  okTol: number,
  degree: number,
  rng: Rng,
): { orbit: Int32Array; reps: number[] } {
  const orbit = new Int32Array(xs.length).fill(-1);
  const reps: number[] = [];
  const h = xs.map((x) => bandlimit(x, degree));
  for (const i of rank(rels)) {
    if (!(rels[i]! <= okTol)) continue;
    const g = reps.findIndex((r) => align(h[i]!, h[r]!, { rng }).rel < SAME_ORBIT_REL);
    if (g >= 0) orbit[i] = g;
    else {
      orbit[i] = reps.length;
      reps.push(i);
    }
  }
  return { orbit, reps };
}
