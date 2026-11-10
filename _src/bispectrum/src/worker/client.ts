import type { InvariantTables } from '../math/invariant';
import type { SolveRequest, WorkerRequest, WorkerResponse } from './protocol';

export type SolveEvent = Exclude<WorkerResponse, { type: 'ready' }>;

/**
 * Runs the march in a Web Worker. A busy worker cannot read messages, so cancelling terminates
 * it and starts a fresh one.
 */
export class SolverClient {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private nextId = 1;
  private activeId = 0;

  constructor(private readonly tables: InvariantTables) {}

  private spawn(): Promise<void> {
    const worker = new Worker(new URL('./march.worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    this.ready = new Promise((resolve) => {
      const onReady = (e: MessageEvent<WorkerResponse>): void => {
        if (e.data.type !== 'ready') return;
        worker.removeEventListener('message', onReady);
        resolve();
      };
      worker.addEventListener('message', onReady);
    });
    const init: WorkerRequest = { type: 'init', tables: this.tables };
    worker.postMessage(init);
    return this.ready;
  }

  cancel(): void {
    this.activeId = 0;
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
  }

  /** Starts a solve, cancelling any running one; `onEvent` only sees events of this solve. */
  async solve(
    req: Omit<SolveRequest, 'type' | 'id'>,
    onEvent: (e: SolveEvent) => void,
  ): Promise<void> {
    if (this.activeId) this.cancel();
    const id = this.nextId++;
    this.activeId = id;
    if (!this.worker) await this.spawn();
    else await this.ready;
    if (this.activeId !== id || !this.worker) return;
    const worker = this.worker;
    const listener = (e: MessageEvent<WorkerResponse>): void => {
      const msg = e.data;
      if (msg.type === 'ready' || msg.id !== id) return;
      if (this.activeId !== id) {
        worker.removeEventListener('message', listener);
        return;
      }
      if (msg.type === 'done' || msg.type === 'error') {
        worker.removeEventListener('message', listener);
        this.activeId = 0;
      }
      onEvent(msg);
    };
    worker.addEventListener('message', listener);
    const msg: WorkerRequest = { type: 'solve', id, ...req };
    worker.postMessage(msg);
  }

  get busy(): boolean {
    return this.activeId !== 0;
  }
}
