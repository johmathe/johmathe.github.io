/** Small 2D-canvas helpers shared by the explainer figures. */

import { BEIGE, colorAt, INK } from './colormap';

export const reducedMotion = (): boolean =>
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Canvas sized to its CSS box at the device pixel ratio; `draw` works in CSS pixels. */
export class Surface {
  readonly canvas = document.createElement('canvas');
  readonly ctx: CanvasRenderingContext2D;
  w = 0;
  h = 0;
  private readonly observer: ResizeObserver;

  constructor(
    host: HTMLElement,
    cls: string,
    private readonly draw: () => void,
  ) {
    this.canvas.className = cls;
    host.append(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    // Deferred a frame: resizing the backing store inside the callback trips the observer's
    // loop detection.
    this.observer = new ResizeObserver(() => requestAnimationFrame(() => this.render()));
    this.observer.observe(this.canvas);
  }

  render(): void {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const [W, H] = [Math.round(r.width * dpr), Math.round(r.height * dpr)];
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }
    [this.w, this.h] = [r.width, r.height];
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.draw();
  }
}

/** Eased animation that a newer one, or a drag, cancels. Jumps to the end on reduced motion. */
export class Tween {
  private frame = 0;

  run(ms: number, step: (t: number) => void, done?: () => void): void {
    this.stop();
    if (reducedMotion()) {
      step(1);
      done?.();
      return;
    }
    const t0 = performance.now();
    const tick = (now: number): void => {
      const t = Math.min(1, (now - t0) / ms);
      step(t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
      if (t < 1) this.frame = requestAnimationFrame(tick);
      else done?.();
    };
    this.frame = requestAnimationFrame(tick);
  }

  stop(): void {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
  }
}

/**
 * A block-diagonal matrix as a heatmap on the diverging map (`[-1, 1]`), zeros in beige and
 * each block framed. Blocks are square and row-major.
 */
export function drawBlockMatrix(
  s: Surface,
  blocks: Float64Array[],
  opts: { top?: number; bold?: number } = {},
): void {
  const { ctx, w, h } = s;
  const sizes = blocks.map((b) => Math.round(Math.sqrt(b.length)));
  const n = sizes.reduce((a, b) => a + b, 0);
  const top = opts.top ?? 0;
  const side = Math.min(w, h - top);
  const cell = side / n;
  const x0 = (w - side) / 2;
  ctx.fillStyle = BEIGE;
  ctx.fillRect(x0, top, side, side);
  let o = 0;
  blocks.forEach((b, i) => {
    const sz = sizes[i]!;
    for (let r = 0; r < sz; r++) {
      for (let c = 0; c < sz; c++) {
        ctx.fillStyle = colorAt(b[r * sz + c]!);
        ctx.fillRect(x0 + (o + c) * cell, top + (o + r) * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.strokeStyle = INK;
    ctx.lineWidth = i === opts.bold ? 2 : 0.75;
    ctx.strokeRect(x0 + o * cell, top + o * cell, sz * cell, sz * cell);
    o += sz;
  });
}

/** A coefficient vector as one row of cells on the diverging map, split by block. */
export function drawCells(s: Surface, values: ArrayLike<number>, sizes: number[], vmax = 1): void {
  const { ctx, w, h } = s;
  const n = values.length;
  const cell = Math.min(w / n, h);
  const x0 = (w - cell * n) / 2;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = colorAt(values[i]! / vmax);
    ctx.fillRect(x0 + i * cell, 0, cell + 0.5, cell);
  }
  let o = 0;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 0.75;
  for (const sz of sizes) {
    ctx.strokeRect(x0 + o * cell, 0.5, sz * cell, cell - 1);
    o += sz;
  }
}
