import { colorAt, RULE } from './colormap';
import { el } from './controls';

export interface CellGridOptions {
  /** Cells per row. By default the grid is twice as wide as tall. */
  cols?: number;
  /** Cell color for a value scaled to `[-1, 1]`. Defaults to the brand's diverging map. */
  color?: (t: number) => string;
  cls?: string;
  label?: string;
}

const CELL = 16;
const GAP = 2;

/**
 * A list of numbers as a grid of colored cells, in entry order: what a model reads, drawn so that
 * a list that moves and a list that holds still can be told apart at a glance.
 */
export class CellGrid {
  readonly element: HTMLCanvasElement;
  readonly cols: number;
  readonly rows: number;
  private readonly color: (t: number) => string;
  private unit = 1;
  private top = 1;

  constructor(
    private readonly n: number,
    opts: CellGridOptions = {},
  ) {
    this.cols = opts.cols ?? 2 * Math.ceil(Math.sqrt(n / 2));
    this.rows = Math.ceil(n / this.cols);
    this.color = opts.color ?? colorAt;
    this.element = el('canvas', {
      class: opts.cls ?? 'cf-input',
      attrs: { role: 'img', 'aria-label': opts.label ?? `${n} numbers as colored cells` },
    });
    this.element.width = Math.min(n, this.cols) * CELL;
    this.element.height = this.rows * CELL;
  }

  /** Fixes the color scale (asinh, as in the bispectrum map) from the signal at rest. */
  setReference(values: ArrayLike<number>): void {
    const mags = Array.from(values, Math.abs)
      .filter((v) => v > 0)
      .sort((a, b) => a - b);
    const max = mags.at(-1) ?? 1;
    this.unit = Math.max(mags[Math.floor(mags.length * 0.25)] ?? max, max * 1e-4);
    this.top = Math.asinh(max / this.unit);
  }

  set(values: ArrayLike<number> | null): void {
    const ctx = this.element.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, this.element.width, this.element.height);
    if (!values) return;
    for (let e = 0; e < this.n; e++) {
      const x = (e % this.cols) * CELL;
      const y = Math.floor(e / this.cols) * CELL;
      ctx.fillStyle = RULE;
      ctx.fillRect(x, y, CELL, CELL);
      ctx.fillStyle = this.color(Math.asinh(values[e]! / this.unit) / this.top);
      ctx.fillRect(x, y, CELL - GAP, CELL - GAP);
    }
  }
}
