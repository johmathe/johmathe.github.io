import { align } from '../math/align';
import { bandlimit } from '../math/basis';
import type { Invariant } from '../math/invariant';
import { march } from '../math/march';
import { Rng } from '../math/rng';
import type { BranchSnapshot, FinalBranch, SolveRequest, WorkerResponse } from './protocol';
import { choose, orbits, residual } from './select';

const IDENTITY = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);

/** Aligns `x` to the original for display, once the choice is made. */
function aligned(x: Float64Array, alignTo: Float64Array, degree: number, rng: Rng) {
  const al = align(bandlimit(x, degree), bandlimit(alignTo, degree), { rng });
  return { rot: al.rot, alignedRel: al.rel };
}

/**
 * One solve: the march, a blind choice at every degree and at the end, then alignment of the
 * chosen candidates to `req.alignTo`. Posts the same messages as the worker.
 */
export function runSolve(
  inv: Invariant,
  req: Omit<SolveRequest, 'type'>,
  post: (m: WorkerResponse) => void,
  now: () => number = () => performance.now(),
): void {
  const t0 = now();
  const alignRng = new Rng(req.seed ^ 0x9e3779b9);
  let lastSeedPost = -Infinity;
  let lastFound = 0;
  const res = march(inv, req.target, req.config, new Rng(req.seed), (e) => {
    const ms = now() - t0;
    if (e.kind === 'seed') {
      if (ms - lastSeedPost < 50 && e.found === lastFound) return;
      lastSeedPost = ms;
      lastFound = e.found;
      post({
        type: 'seed',
        id: req.id,
        tried: e.tried,
        found: e.found,
        budget: req.config.nSeedInits,
        ms,
      });
    } else if (e.kind === 'degree') {
      const rows = inv.rowsWhere((d) => d <= e.degree);
      const rels = e.branches.map((x) => residual(inv, x, req.target, rows));
      const best = choose(rels);
      const branches: BranchSnapshot[] = e.branches.map((x, i) => ({
        x,
        featureRel: rels[i]!,
        ...(i === best
          ? aligned(x, req.alignTo, e.degree, alignRng)
          : { rot: IDENTITY, alignedRel: NaN }),
      }));
      post({
        type: 'degree',
        id: req.id,
        degree: e.degree,
        branches,
        best,
        retried: e.retried,
        ms: now() - t0,
      });
    }
  });

  const xs = res.branches.map((b) => b.x);
  const rels = res.branches.map((b) => b.rel);
  const best = choose(rels);
  const { orbit, reps } = orbits(
    xs,
    rels,
    req.config.okTol,
    req.lmax,
    new Rng(req.seed ^ 0x51ed27),
  );
  const branches: FinalBranch[] = xs.map((x, i) => ({
    x,
    featureRel: rels[i]!,
    orbit: orbit[i]!,
    ...aligned(x, req.alignTo, req.lmax, alignRng),
  }));
  post({
    type: 'done',
    id: req.id,
    branches,
    best,
    accepted: best >= 0 && rels[best]! <= req.config.okTol,
    reps,
    seedTried: res.seedTried,
    ms: now() - t0,
  });
}
