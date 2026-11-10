import { align } from '../math/align';
import { bandlimit } from '../math/basis';
import { Invariant, type InvariantTables } from '../math/invariant';
import { march } from '../math/march';
import { Rng } from '../math/rng';
import type { BranchSnapshot, SolveRequest, WorkerRequest, WorkerResponse } from './protocol';

interface Scope {
  postMessage(message: WorkerResponse): void;
  addEventListener(type: 'message', fn: (e: MessageEvent<WorkerRequest>) => void): void;
}

const scope = self as unknown as Scope;
let tables: InvariantTables | null = null;
const invariants = new Map<number, Invariant>();

function invariantAt(lmax: number): Invariant {
  let inv = invariants.get(lmax);
  if (!inv) {
    inv = new Invariant(tables!, lmax);
    invariants.set(lmax, inv);
  }
  return inv;
}

function snapshot(
  x: Float64Array,
  alignTo: Float64Array,
  degree: number,
  rng: Rng,
): BranchSnapshot {
  const h = bandlimit(x, degree);
  const al = align(h, bandlimit(alignTo, degree), { rng });
  return { x: Float64Array.from(x), rot: al.rot, alignedRel: al.rel };
}

function bestIndex(snaps: BranchSnapshot[]): number {
  let best = 0;
  snaps.forEach((s, i) => {
    if (s.alignedRel < snaps[best]!.alignedRel) best = i;
  });
  return best;
}

function solve(req: SolveRequest): void {
  const inv = invariantAt(req.lmax);
  const t0 = performance.now();
  const alignRng = new Rng(req.seed ^ 0x9e3779b9);
  let lastSeedPost = -Infinity;
  let lastFound = 0;
  const res = march(inv, req.target, req.config, new Rng(req.seed), (e) => {
    const ms = performance.now() - t0;
    if (e.kind === 'seed') {
      if (ms - lastSeedPost < 50 && e.found === lastFound) return;
      lastSeedPost = ms;
      lastFound = e.found;
      scope.postMessage({
        type: 'seed',
        id: req.id,
        tried: e.tried,
        found: e.found,
        budget: req.config.nSeedInits,
        ms,
      });
    } else if (e.kind === 'degree') {
      const branches = e.branches.map((b) => snapshot(b, req.alignTo, e.degree, alignRng));
      scope.postMessage({
        type: 'degree',
        id: req.id,
        degree: e.degree,
        branches,
        best: bestIndex(branches),
        retried: e.retried,
        ms: performance.now() - t0,
      });
    }
  });
  const branches = res.branches.map((b) => ({
    ...snapshot(b.x, req.alignTo, req.lmax, alignRng),
    featureRel: b.rel,
  }));
  // Every branch matching the invariant to okTol is a valid answer; show the best aligned one.
  const ok = (i: number): boolean => branches[i]!.featureRel <= req.config.okTol;
  const better = (i: number, j: number): boolean => {
    if (ok(i) !== ok(j)) return ok(i);
    const [a, b] = [branches[i]!, branches[j]!];
    return ok(i) ? a.alignedRel < b.alignedRel : a.featureRel < b.featureRel;
  };
  let best = 0;
  for (let i = 1; i < branches.length; i++) if (better(i, best)) best = i;
  scope.postMessage({
    type: 'done',
    id: req.id,
    branches,
    best,
    seedTried: res.seedTried,
    ms: performance.now() - t0,
  });
}

scope.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    tables = msg.tables;
    scope.postMessage({ type: 'ready' });
    return;
  }
  try {
    solve(msg);
  } catch (err) {
    scope.postMessage({ type: 'error', id: msg.id, message: String(err) });
  }
});
