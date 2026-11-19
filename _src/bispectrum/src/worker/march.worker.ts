import { Invariant, type InvariantTables } from '../math/invariant';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { runSolve } from './run';

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

scope.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    tables = msg.tables;
    scope.postMessage({ type: 'ready' });
    return;
  }
  try {
    runSolve(invariantAt(msg.lmax), msg, (m) => scope.postMessage(m));
  } catch (err) {
    scope.postMessage({ type: 'error', id: msg.id, message: String(err) });
  }
});
