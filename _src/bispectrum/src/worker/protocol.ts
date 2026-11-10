import type { InvariantTables } from '../math/invariant';
import type { MarchConfig } from '../math/march';

export interface SolveRequest {
  type: 'solve';
  id: number;
  lmax: number;
  /** The selective invariant, one real number per entry: the only input to the solver. */
  target: Float64Array;
  /** Coefficients of the original, used only to align snapshots for display. */
  alignTo: Float64Array;
  seed: number;
  config: MarchConfig;
}

export type WorkerRequest = { type: 'init'; tables: InvariantTables } | SolveRequest;

export interface BranchSnapshot {
  /** Recovered coefficients (zero above the current degree). */
  x: Float64Array;
  /** Rotation taking `x` onto the original, truncated to the same degree. */
  rot: Float64Array;
  alignedRel: number;
}

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'seed'; id: number; tried: number; found: number; budget: number; ms: number }
  | {
      type: 'degree';
      id: number;
      degree: number;
      branches: BranchSnapshot[];
      best: number;
      retried: boolean;
      ms: number;
    }
  | {
      type: 'done';
      id: number;
      branches: (BranchSnapshot & { featureRel: number })[];
      best: number;
      seedTried: number;
      ms: number;
    }
  | { type: 'error'; id: number; message: string };
