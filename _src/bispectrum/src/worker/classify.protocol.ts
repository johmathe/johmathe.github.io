import type { ClassifierManifest } from '../math/classifier';

export type ClassifyRequest =
  | { type: 'init'; manifest: ClassifierManifest; bin: ArrayBuffer }
  /** Real coordinates of the signal, band-limited to the classifier's degree. */
  | { type: 'classify'; id: number; x: Float64Array };

export interface ClassifyResult {
  type: 'result';
  id: number;
  logits: Float64Array;
  /** Synthesis plus forward pass. */
  ms: number;
}
