import { BEIGE, hex, INK } from './colormap';

export interface DrawPadOptions {
  size: number;
  /** Called while drawing (`final = false`, once per frame) and on release. */
  onChange: (image: Float32Array, final: boolean) => void;
}

const SIGMA = 0.85;
const RADIUS = 3;
const BG = hex(BEIGE);
const FG = hex(INK);

/** Ink-on-beige pixels of a `[0, 1]` image. */
function paintDigit(ctx: CanvasRenderingContext2D, image: Float32Array, size: number): void {
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = image[i]!;
    for (let ch = 0; ch < 3; ch++)
      img.data[4 * i + ch] = Math.round(BG[ch]! + v * (FG[ch]! - BG[ch]!));
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function digitThumb(image: Float32Array, size: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  if (ctx) paintDigit(ctx, image, size);
  return c;
}

/** A 28x28 MNIST-style pad: soft brush, values in [0, 1]. */
export class DrawPad {
  readonly element: HTMLElement;
  readonly image: Float32Array;
  private readonly canvas: HTMLCanvasElement;
  private last: [number, number] | null = null;
  private frame = 0;

  constructor(private readonly opts: DrawPadOptions) {
    const { size } = opts;
    this.image = new Float32Array(size * size);
    this.element = document.createElement('div');
    this.element.className = 'drawpad';
    this.canvas = document.createElement('canvas');
    this.canvas.width = size;
    this.canvas.height = size;
    this.canvas.setAttribute('aria-label', 'Draw a digit');
    this.element.appendChild(this.canvas);
    this.bind();
    this.paint();
  }

  clear(): void {
    this.image.fill(0);
    this.paint();
  }

  load(image: Float32Array): void {
    this.image.set(image);
    this.paint();
  }

  get empty(): boolean {
    return !this.image.some((v) => v > 0.05);
  }

  private stamp(cx: number, cy: number): void {
    const { size } = this.opts;
    const y0 = Math.max(0, Math.floor(cy - RADIUS));
    const y1 = Math.min(size - 1, Math.ceil(cy + RADIUS));
    const x0 = Math.max(0, Math.floor(cx - RADIUS));
    const x1 = Math.min(size - 1, Math.ceil(cx + RADIUS));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const a = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * SIGMA * SIGMA));
        const i = y * size + x;
        this.image[i] = Math.min(1, this.image[i]! + 0.55 * a);
      }
    }
  }

  private line(x0: number, y0: number, x1: number, y1: number): void {
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 0.35));
    for (let i = 1; i <= n; i++) this.stamp(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n);
  }

  private toGrid(e: PointerEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    const s = this.opts.size;
    return [
      ((e.clientX - r.left) / r.width) * s - 0.5,
      ((e.clientY - r.top) / r.height) * s - 0.5,
    ];
  }

  private bind(): void {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this.last = this.toGrid(e);
      this.stamp(...this.last);
      this.changed();
    });
    c.addEventListener('pointermove', (e) => {
      if (!this.last) return;
      const p = this.toGrid(e);
      this.line(...this.last, ...p);
      this.last = p;
      this.changed();
    });
    const end = (): void => {
      if (!this.last) return;
      this.last = null;
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      this.paint();
      this.opts.onChange(this.image, true);
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
  }

  private changed(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.paint();
      this.opts.onChange(this.image, false);
    });
  }

  private paint(): void {
    const ctx = this.canvas.getContext('2d');
    if (ctx) paintDigit(ctx, this.image, this.opts.size);
  }
}
