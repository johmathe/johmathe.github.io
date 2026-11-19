import type { InvariantTables } from '../math/invariant';
import type { MarchConfig } from '../math/march';

export interface SolveRequest {
  type: 'solve';
  id: number;
  lmax: number;
  /** The selective invariant, one real number per entry: the only input to the solver. */
  target: Float64Array;
  /**
   * Coefficients of the original. Used only after a candidate has been chosen, to align it for
   * display and to measure its error; never to choose.
   */
  alignTo: Float64Array;
  seed: number;
  config: MarchConfig;
}

export type WorkerRequest = { type: 'init'; tables: InvariantTables } | SolveRequest;

export interface BranchSnapshot {
  /** Recovered coefficients (zero above the current degree). */
  x: Float64Array;
  /** Residual against the invariant entries up to the current degree. */
  featureRel: number;
  /** Rotation taking `x` onto the original, truncated to the same degree (display only). */
  rot: Float64Array;
  /** Relative error of the aligned, band-limited coefficients (display only). */
  alignedRel: number;
}

export interface FinalBranch extends BranchSnapshot {
  /** Distinct-signal group among the accepted candidates, `-1` if it misses the tolerance. */
  orbit: number;
}

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'seed'; id: number; tried: number; found: number; budget: number; ms: number }
  | {
      type: 'degree';
      id: number;
      degree: number;
      branches: BranchSnapshot[];
      /** Chosen by residual alone; only this branch is aligned to the original. */
      best: number;
      retried: boolean;
      ms: number;
    }
  | {
      type: 'done';
      id: number;
      branches: FinalBranch[];
      /** Chosen by residual alone. */
      best: number;
      /** Whether `branches[best]` matches the invariant to `okTol`. */
      accepted: boolean;
      /** Representative of each distinct accepted signal, in rank order (`reps[0] === best`). */
      reps: number[];
      seedTried: number;
      ms: number;
    }
  | { type: 'error'; id: number; message: string };
