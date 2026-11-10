import { BEIGE, INK, MUTED, ORANGE, ORANGE_TEXT, RULE, TINTS } from './colormap';

export interface StripGroup {
  degree: number;
  /** Display positions `[start, end)`. */
  start: number;
  end: number;
}

export interface StripLayout {
  /** Entry index at each display position. */
  order: Int32Array;
  groups: StripGroup[];
  color: (entry: number) => string;
  describe: (entry: number) => string;
  /** `asinh` for signed values, `log` for positive ones. */
  scale: 'asinh' | 'log';
}

export interface StripValues {
  /** Current signal: filled bars. */
  fill: Float64Array | null;
  /** Reference signal: outlines. */
  outline: Float64Array | null;
  /** Reconstruction: dots, drawn only where `dotsUpTo` admits the entry's degree. */
  dots: Float64Array | null;
  dotsUpTo: number;
  /** Degree whose entries get the highlight band. */
  highlight: number | null;
}

const PAD_L = 4;
const PAD_R = 4;
const PAD_T = 6;
const AXIS = 16;
const GAP = 3;
const MAX_BAR = 18;

export const fmt = (v: number): string =>
  (v === 0
    ? '0'
    : Math.abs(v) >= 1e-2 && Math.abs(v) < 1e3
      ? v.toPrecision(4)
      : v.toExponential(2)
  ).replaceAll('-', '−');

/** Bar chart of one invariant or coefficient vector, grouped by degree, on a 2D canvas. */
export class Strip {
  private readonly canvas: HTMLCanvasElement;
  private readonly readout: HTMLElement;
  private readonly observer: ResizeObserver;
  private layout: StripLayout | null = null;
  private values: StripValues = {
    fill: null,
    outline: null,
    dots: null,
    dotsUpTo: -1,
    highlight: null,
  };
  private degreeOf = new Int32Array(0);
  /** Scale fixed by the reference so bars do not rescale while the signal rotates. */
  private unit = 1;
  private top = 1;
  private xs = new Float64Array(0);
  private barW = 1;
  private hover = -1;
  private frame = 0;

  constructor(
    host: HTMLElement,
    private readonly height: number,
  ) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'strip-canvas';
    this.readout = document.createElement('div');
    this.readout.className = 'strip-readout';
    this.readout.textContent = '\u00a0';
    host.append(this.canvas, this.readout);
    this.observer = new ResizeObserver(() => this.request());
    this.observer.observe(host);
    this.canvas.addEventListener('pointermove', (e) => this.onHover(e));
    this.canvas.addEventListener('pointerleave', () => {
      this.hover = -1;
      this.updateReadout();
      this.request();
    });
  }

  setLayout(layout: StripLayout): void {
    this.layout = layout;
    this.degreeOf = new Int32Array(layout.order.length);
    for (const g of layout.groups) {
      for (let p = g.start; p < g.end; p++) this.degreeOf[layout.order[p]!] = g.degree;
    }
    this.hover = -1;
    this.request();
  }

  /** Updates values; pass `rescale` when the reference changed. */
  setValues(values: Partial<StripValues>, rescale = false): void {
    this.values = { ...this.values, ...values };
    if (rescale) this.fitScale();
    this.updateReadout();
    this.request();
  }

  private fitScale(): void {
    const ref = this.values.outline ?? this.values.fill;
    if (!ref || !this.layout) return;
    const mags = Array.from(ref, Math.abs)
      .filter((v) => v > 0)
      .sort((a, b) => a - b);
    const max = mags.at(-1) ?? 1;
    if (this.layout.scale === 'log') {
      this.top = Math.log10(max) + 0.3;
      this.unit = Math.log10(mags[0] ?? max / 1e6) - 0.5;
      this.unit = Math.max(this.unit, this.top - 8);
    } else {
      // asinh(v / unit) is linear below `unit` and logarithmic above.
      this.unit = Math.max(mags[Math.floor(mags.length * 0.25)] ?? max, max * 1e-4);
      this.top = Math.asinh((max * 1.08) / this.unit);
    }
  }

  /** Signed height in [-1, 1]. */
  private y(v: number): number {
    if (this.layout?.scale === 'log') {
      if (v <= 0) return 0;
      return Math.max(0, (Math.log10(v) - this.unit) / (this.top - this.unit));
    }
    return Math.max(-1, Math.min(1, Math.asinh(v / this.unit) / this.top));
  }

  private request(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  private draw(): void {
    const layout = this.layout;
    const w = this.canvas.parentElement?.clientWidth ?? 600;
    const h = this.height;
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    const ctx = this.canvas.getContext('2d');
    if (!ctx || !layout) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const n = layout.order.length;
    const gaps = Math.max(0, layout.groups.length - 1) * GAP;
    const barW = (w - PAD_L - PAD_R - gaps) / n;
    this.barW = barW;
    this.xs = new Float64Array(n);
    let x = PAD_L;
    layout.groups.forEach((g, gi) => {
      if (gi > 0) x += GAP;
      for (let p = g.start; p < g.end; p++) {
        this.xs[p] = x;
        x += barW;
      }
    });

    const signed = layout.scale === 'asinh';
    const plotH = h - PAD_T - AXIS;
    const base = signed ? PAD_T + plotH / 2 : PAD_T + plotH;
    const amp = signed ? plotH / 2 : plotH;
    const { fill, outline, dots, dotsUpTo, highlight } = this.values;

    ctx.font = '10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const g of layout.groups) {
      const x0 = this.xs[g.start]!;
      const x1 = this.xs[g.end - 1]! + barW;
      if (g.degree === highlight) {
        ctx.fillStyle = TINTS.orangeWash.hex;
        ctx.fillRect(x0 - 1, PAD_T - 4, x1 - x0 + 2, plotH + 4);
      } else if (g.degree % 2 === 1) {
        ctx.fillStyle = TINTS.sandWash.hex;
        ctx.fillRect(x0 - 1, PAD_T - 4, x1 - x0 + 2, plotH + 4);
      }
      const label = String(g.degree);
      if (x1 - x0 >= ctx.measureText(label).width || g.degree === highlight) {
        ctx.fillStyle = g.degree === highlight ? ORANGE_TEXT : MUTED;
        ctx.fillText(label, (x0 + x1) / 2, h - AXIS + 3);
      }
    }
    ctx.fillStyle = RULE;
    ctx.fillRect(PAD_L, Math.round(base), w - PAD_L - PAD_R, 1);

    const inset = barW > 3 ? 0.18 * barW : barW > 1.6 ? 0.3 : 0;
    const bw = Math.min(MAX_BAR, Math.max(0.6, barW - 2 * inset));
    for (let p = 0; p < n; p++) {
      const e = layout.order[p]!;
      const x0 = this.xs[p]! + (barW - bw) / 2;
      if (fill) {
        const v = this.y(fill[e]!) * amp;
        ctx.fillStyle = p === this.hover ? INK : layout.color(e);
        ctx.fillRect(x0, v >= 0 ? base - v : base, bw, Math.max(Math.abs(v), 0.75));
      }
      if (outline) {
        const v = this.y(outline[e]!) * amp;
        ctx.strokeStyle = INK;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0 - 0.5, base - v);
        ctx.lineTo(x0 + bw + 0.5, base - v);
        ctx.stroke();
      }
    }
    if (dots) {
      const r = Math.max(1.4, Math.min(3, barW * 0.42));
      ctx.fillStyle = ORANGE_TEXT;
      ctx.strokeStyle = BEIGE;
      ctx.lineWidth = 0.75;
      for (let p = 0; p < n; p++) {
        const e = layout.order[p]!;
        if (this.degreeOf[e]! > dotsUpTo) continue;
        const v = this.y(dots[e]!) * amp;
        ctx.beginPath();
        ctx.arc(this.xs[p]! + barW / 2, base - v, r, 0, 2 * Math.PI);
        ctx.fill();
        if (barW > 3) ctx.stroke();
      }
    }
  }

  private onHover(e: PointerEvent): void {
    const layout = this.layout;
    if (!layout || this.xs.length === 0) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let lo = 0;
    let hi = this.xs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.xs[mid]! <= x) lo = mid;
      else hi = mid - 1;
    }
    const p = x < this.xs[0]! || x > this.xs[lo]! + this.barW + GAP ? -1 : lo;
    if (p !== this.hover) {
      this.hover = p;
      this.updateReadout();
      this.request();
    }
  }

  private updateReadout(): void {
    const layout = this.layout;
    if (!layout || this.hover < 0) {
      this.readout.textContent = '\u00a0';
      return;
    }
    const e = layout.order[this.hover]!;
    const { fill, outline, dots, dotsUpTo } = this.values;
    const parts = [layout.describe(e)];
    if (fill) parts.push(`signal ${fmt(fill[e]!)}`);
    if (outline && fill && Math.abs(outline[e]! - fill[e]!) > 1e-9 * Math.abs(outline[e]!)) {
      parts.push(`at rest ${fmt(outline[e]!)}`);
    }
    if (dots && this.degreeOf[e]! <= dotsUpTo) parts.push(`reconstruction ${fmt(dots[e]!)}`);
    this.readout.textContent = parts.join('  ·  ');
  }
}

/**
 * The two kinds of number in the descriptor, as the post names them: triple products (bispectral
 * entries, whatever rule selected them) and product energies (the paper's Clebsch-Gordan powers).
 */
export type EntryKind = 'triple' | 'energy';

export const kindOf = (tag: string): EntryKind => (tag === 'cg_power' ? 'energy' : 'triple');

export const KIND_COLORS: Record<EntryKind, string> = {
  triple: ORANGE,
  energy: TINTS.orangeLight.hex,
};

export const KIND_LABELS: Record<EntryKind, string> = {
  triple: 'triple product',
  energy: 'product energy',
};
