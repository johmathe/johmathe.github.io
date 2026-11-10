import type { FullBispectrum } from '../math/bispectrum';
import { colorAt, INK, MUTED, ORANGE_TEXT, RULE, STONE } from './colormap';
import { el } from './controls';
import { fmt } from './strips';

export interface MapLayout {
  full: FullBispectrum;
  /** 1 for the entries the selective invariant keeps: they get an ink frame. */
  selective: Uint8Array;
  describe: (entry: number) => string;
}

interface Geometry {
  cols: number;
  rows: number;
  /** Device pixels. */
  cell: number;
  gap: number;
  line: number;
  dpr: number;
}

const PANEL_GAP = 10;
const MIN_CELL = 4;
const MAX_CELL = 14;

/**
 * The full bispectrum as small multiples: one panel per output degree `l`, holding the triangle
 * `l1 <= l2` (rows `l1`, columns `l2`) of the entries `beta(l1, l2, l)`. Two copies side by side,
 * the signal's and the reconstruction's; the latter fills in as degrees get solved.
 */
export class BispectrumMap {
  readonly element: HTMLElement;
  private readonly canvases: [HTMLCanvasElement, HTMLCanvasElement];
  private readonly reconCaption: HTMLElement;
  private readonly readout: HTMLElement;
  private layout: MapLayout | null = null;
  /** Entry at `(l, l1, l2)`, or -1. */
  private cellOf = new Int32Array(0);
  private signal: Float64Array | null = null;
  private recon: Float64Array | null = null;
  private reconUpTo = -1;
  private unit = 1;
  private top = 1;
  private geom: Geometry | null = null;
  private hover = -1;
  private frame = 0;

  constructor() {
    const half = (caption: HTMLElement): [HTMLElement, HTMLCanvasElement] => {
      const canvas = el('canvas', { class: 'map-canvas' });
      return [el('div', { class: 'rw-map-half' }, caption, canvas), canvas];
    };
    this.reconCaption = el('div', { class: 'rw-map-cap', text: 'reconstruction' });
    const [left, a] = half(el('div', { class: 'rw-map-cap', text: 'signal' }));
    const [right, b] = half(this.reconCaption);
    this.canvases = [a, b];
    this.readout = el('div', { class: 'strip-readout', text: '\u00a0' });
    this.element = el(
      'div',
      { class: 'rw-map' },
      el('div', { class: 'rw-map-grid' }, left, right),
      this.readout,
    );
    new ResizeObserver(() => this.request()).observe(this.element);
    for (const c of this.canvases) {
      c.addEventListener('pointermove', (e) => this.setHover(this.entryAt(c, e)));
      c.addEventListener('pointerleave', () => this.setHover(-1));
    }
  }

  /** Resets the values: they belong to the previous band-limit. */
  setLayout(layout: MapLayout): void {
    this.layout = layout;
    const { full } = layout;
    const c = full.lmax + 1;
    this.cellOf = new Int32Array(c * c * c).fill(-1);
    full.triples.forEach(([l1, l2, l], e) => {
      this.cellOf[(l * c + l1) * c + l2] = e;
    });
    this.signal = this.recon = null;
    this.reconUpTo = -1;
    this.hover = -1;
    this.updateReadout();
    this.request();
  }

  /** Fixes the color scale from the signal at rest, so that rotating cannot rescale it. */
  setReference(values: Float64Array): void {
    const mags = Array.from(values, Math.abs)
      .filter((v) => v > 0)
      .sort((a, b) => a - b);
    const max = mags.at(-1) ?? 1;
    // asinh(v / unit) is linear below `unit` and logarithmic above.
    this.unit = Math.max(mags[Math.floor(mags.length * 0.25)] ?? max, max * 1e-4);
    this.top = Math.asinh(max / this.unit);
    this.request();
  }

  setSignal(values: Float64Array): void {
    this.signal = values;
    this.updateReadout();
    this.request();
  }

  /** Reconstruction values; only entries whose degrees are all `<= upTo` are drawn. */
  setRecon(values: Float64Array | null, upTo: number): void {
    this.recon = values;
    this.reconUpTo = values ? upTo : -1;
    const partial = values && this.layout && upTo < this.layout.full.lmax;
    this.reconCaption.textContent = partial
      ? `reconstruction, entries with every degree ≤ ${upTo}`
      : 'reconstruction';
    this.updateReadout();
    this.request();
  }

  private request(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  private reconKnown(e: number): boolean {
    return !!this.recon && !!this.layout && this.layout.full.maxDegree[e]! <= this.reconUpTo;
  }

  private draw(): void {
    const layout = this.layout;
    const w = this.canvases[0].parentElement?.clientWidth ?? 0;
    if (!layout || w === 0) return;
    const n = layout.full.lmax + 1;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const line = Math.max(1, Math.round(dpr));
    const gap = Math.round(PANEL_GAP * dpr);
    const width = Math.round(w * dpr) - 2 * line;
    // Fewest rows of panels that keep the cells legible.
    let rows = 1;
    let cols = n;
    let cell = 1;
    for (; rows <= n; rows++) {
      cols = Math.ceil(n / rows);
      cell = Math.floor((width - (cols - 1) * gap) / (cols * n));
      if (cell >= MIN_CELL * dpr || cols === 1) break;
    }
    cell = Math.max(2, Math.min(cell, Math.round(MAX_CELL * dpr)));
    const g: Geometry = { cols, rows, cell, gap, line, dpr };
    this.geom = g;
    const panel = cell * n;
    const height = rows * panel + (rows - 1) * gap + 2 * line;
    this.canvases.forEach((canvas, which) => {
      canvas.width = Math.round(w * dpr);
      canvas.height = height;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${height / dpr}px`;
      const ctx = canvas.getContext('2d');
      if (ctx) this.paint(ctx, g, which === 0 ? this.signal : this.recon, which === 1);
    });
  }

  private origin(g: Geometry, l: number): [number, number] {
    const pitch = g.cell * (this.layout!.full.lmax + 1) + g.gap;
    return [g.line + (l % g.cols) * pitch, g.line + Math.floor(l / g.cols) * pitch];
  }

  private paint(
    ctx: CanvasRenderingContext2D,
    g: Geometry,
    values: Float64Array | null,
    isRecon: boolean,
  ): void {
    const { full, selective } = this.layout!;
    const { cell, line } = g;
    const n = full.lmax + 1;
    const shown = (e: number): boolean => !!values && (!isRecon || this.reconKnown(e));
    const at = (e: number): [number, number] => {
      const [l1, l2, l] = full.triples[e]!;
      const [px, py] = this.origin(g, l);
      return [px + l2 * cell, py + l1 * cell];
    };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Grid under every drawn cell, a dot where the reconstruction has not got to yet.
    const dot = Math.max(1, Math.round(cell / 5));
    for (let e = 0; e < full.n; e++) {
      const [x, y] = at(e);
      if (shown(e)) {
        ctx.fillStyle = RULE;
        ctx.fillRect(x - line, y - line, cell + line, cell + line);
      } else {
        ctx.fillStyle = STONE;
        const o = Math.floor((cell - line - dot) / 2);
        ctx.fillRect(x + o, y + o, dot, dot);
      }
    }
    ctx.fillStyle = INK;
    for (let e = 0; e < full.n; e++) {
      if (!selective[e] || !shown(e)) continue;
      const [x, y] = at(e);
      ctx.fillRect(x - line, y - line, cell + line, cell + line);
    }
    for (let e = 0; e < full.n; e++) {
      if (!shown(e)) continue;
      const [x, y] = at(e);
      // Entries that vanish identically hold rounding noise of either sign: draw an exact zero.
      const v = full.vanishes[e] ? 0 : values![e]!;
      ctx.fillStyle = colorAt(Math.asinh(v / this.unit) / this.top);
      ctx.fillRect(x, y, cell - line, cell - line);
    }

    ctx.font = `${10 * g.dpr}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = MUTED;
    for (let l = 0; l < n; l++) {
      const [px, py] = this.origin(g, l);
      ctx.fillText(`ℓ = ${l}`, px, py + cell * n - 2 * g.dpr);
    }

    if (this.hover >= 0) {
      const [x, y] = at(this.hover);
      const m = 2 * line;
      ctx.strokeStyle = ORANGE_TEXT;
      ctx.lineWidth = 2 * line;
      ctx.strokeRect(x - line - m, y - line - m, cell + line + 2 * m, cell + line + 2 * m);
    }
  }

  private entryAt(canvas: HTMLCanvasElement, ev: PointerEvent): number {
    const g = this.geom;
    if (!g || !this.layout) return -1;
    const n = this.layout.full.lmax + 1;
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * g.dpr - g.line;
    const y = (ev.clientY - rect.top) * g.dpr - g.line;
    const pitch = g.cell * n + g.gap;
    const col = Math.floor(x / pitch);
    const row = Math.floor(y / pitch);
    const l = row * g.cols + col;
    const l2 = Math.floor((x - col * pitch) / g.cell);
    const l1 = Math.floor((y - row * pitch) / g.cell);
    if (col < 0 || col >= g.cols || row < 0 || l >= n || l1 >= n || l2 >= n) return -1;
    return this.cellOf[(l * n + l1) * n + l2]!;
  }

  private setHover(e: number): void {
    if (e === this.hover) return;
    this.hover = e;
    this.updateReadout();
    this.request();
  }

  private updateReadout(): void {
    const e = this.hover;
    if (!this.layout || e < 0) {
      this.readout.textContent = '\u00a0';
      return;
    }
    const parts = [this.layout.describe(e)];
    // Entries that vanish identically only hold rounding noise.
    if (!this.layout.full.vanishes[e]) {
      if (this.signal) parts.push(`signal ${fmt(this.signal[e]!)}`);
      if (this.reconKnown(e)) parts.push(`reconstruction ${fmt(this.recon![e]!)}`);
    }
    this.readout.textContent = parts.join('  ·  ');
  }
}
