import { describe, expect, it, vi } from 'vitest';

import { dim } from '../src/math/basis';
import { WIDGET_CONFIG } from '../src/math/config';
import { Invariant } from '../src/math/invariant';
import type { MarchEvent, MarchResult } from '../src/math/march';
import type { WorkerResponse } from '../src/worker/protocol';
import { runSolve } from '../src/worker/run';
import { choose, rank } from '../src/worker/select';
import { loadTables, randn } from './helpers';

const L = 6;
/** Two distinct signals, both accepted: the march is replaced by one that returns them. */
const A = randn(dim(L), 11);
const B = randn(dim(L), 12);
const RELS = [3e-12, 2e-12];

vi.mock('../src/math/march', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/math/march')>()),
  march: (
    _inv: unknown,
    _target: unknown,
    _cfg: unknown,
    _rng: unknown,
    emit: (e: MarchEvent) => void = () => {},
  ): MarchResult => {
    emit({
      kind: 'degree',
      degree: L,
      branches: [A, B],
      bestFrozen: 0,
      bestPolished: null,
      retried: false,
    });
    return {
      branches: [
        { x: A, rel: RELS[0]! },
        { x: B, rel: RELS[1]! },
      ],
      seedTried: 1,
      nSeed: 2,
      degrees: [],
    };
  },
}));

const inv = new Invariant(loadTables(), L);

function solveWith(alignTo: Float64Array) {
  const out: WorkerResponse[] = [];
  runSolve(
    inv,
    { id: 1, lmax: L, target: inv.values(A), alignTo, seed: 5, config: WIDGET_CONFIG },
    (m) => out.push(m),
    () => 0,
  );
  const degree = out.find((m) => m.type === 'degree');
  const done = out.find((m) => m.type === 'done');
  if (degree?.type !== 'degree' || done?.type !== 'done') throw new Error('missing events');
  return { degree, done };
}

describe('candidate selection', () => {
  it('ranks by residual, ties in solver order', () => {
    expect(rank([2, 1, 1, 3])).toEqual([1, 2, 0, 3]);
    expect(choose([])).toBe(-1);
    expect(choose([1e-12, 1e-12])).toBe(0);
  });

  it('does not depend on the signal used for alignment', () => {
    const toA = solveWith(A);
    const toB = solveWith(B);
    // The reference decides which candidate looks closer...
    const [a0, a1] = toA.done.branches.map((b) => b.alignedRel);
    const [b0, b1] = toB.done.branches.map((b) => b.alignedRel);
    expect(a0!).toBeLessThan(1e-9);
    expect(a1!).toBeGreaterThan(1e-3);
    expect(b1!).toBeLessThan(1e-9);
    expect(b0!).toBeGreaterThan(1e-3);
    // ...and not which one is shown, at the end or in the preview.
    expect(toA.done.best).toBe(1);
    expect(toB.done.best).toBe(1);
    expect(toA.degree.best).toBe(toB.degree.best);
    expect(toA.done.accepted && toB.done.accepted).toBe(true);
    // Both are offered as distinct signals, in the same order.
    expect(toA.done.reps).toEqual([1, 0]);
    expect(toB.done.reps).toEqual([1, 0]);
  });

  it('reports a run that misses the tolerance', () => {
    RELS[0] = RELS[1] = 1e-6;
    const { done } = solveWith(A);
    expect(done.accepted).toBe(false);
    expect(done.reps).toEqual([]);
    expect(done.branches.every((b) => b.orbit === -1)).toBe(true);
  });
});
