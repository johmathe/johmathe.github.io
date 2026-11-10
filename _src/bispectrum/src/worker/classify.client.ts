import type { ClassifierManifest } from '../math/classifier';
import type { ClassifyRequest, ClassifyResult } from './classify.protocol';

/**
 * Runs the pixel CNN in a Web Worker: a forward pass takes tens of milliseconds, too long for a
 * drag handler. While one is running only the latest request is kept, so the display lags by at
 * most one pass.
 */
export class CnnClient {
  private readonly worker: Worker;
  private busy = false;
  private pending: Float64Array | null = null;
  private nextId = 1;

  constructor(
    manifest: ClassifierManifest,
    bin: ArrayBuffer,
    private readonly onResult: (logits: Float64Array, ms: number) => void,
  ) {
    this.worker = new Worker(new URL('./classify.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.addEventListener('message', (e: MessageEvent<ClassifyResult>) => {
      this.busy = false;
      const next = this.pending;
      this.pending = null;
      // A newer request supersedes this result only once its own answer arrives.
      this.onResult(e.data.logits, e.data.ms);
      if (next) this.send(next);
    });
    this.post({ type: 'init', manifest, bin });
  }

  classify(x: Float64Array): void {
    if (this.busy) this.pending = x;
    else this.send(x);
  }

  private send(x: Float64Array): void {
    this.busy = true;
    this.post({ type: 'classify', id: this.nextId++, x });
  }

  private post(msg: ClassifyRequest): void {
    this.worker.postMessage(msg);
  }
}
