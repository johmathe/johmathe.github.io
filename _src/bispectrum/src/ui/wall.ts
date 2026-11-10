/**
 * Wall (`wall`): how many numbers the full bispectrum holds against the selective one, on a log
 * axis. One tab for images under shifts (`n` pixels: `n (n + 1) / 2` against `n`), with GPU
 * memory lines, and one for the sphere (counts from the library's index maps, `counts.json`).
 */

import { dataJson } from '../assets';
import type { Figure, Mount } from '../figure';
import { fullCount } from '../math/ring';
import { Surface, Tween } from './canvas';
import { INK, MUTED, ORANGE_TEXT, RULE, STONE } from './colormap';
import { el, segmented, slider, type Segmented, type Slider } from './controls';

export interface SphereCounts {
  L: number[];
  coefficients: number[];
  selective: number[];
  full: number[];
  selective_terms: number[];
  full_terms: number[];
}

export interface Counts {
  sphere: SphereCounts;
}

/** One complex64 number. */
export const BYTES_PER_NUMBER = 8;
const SIDE_STEP = 8;
const SIDE_MAX = 256;
const GPUS = [
  { gb: 24, label: '24 GB GPU' },
  { gb: 80, label: '80 GB GPU' },
];

type Tab = 'image' | 'sphere';

const int = (n: number): string => Math.round(n).toLocaleString('en-US');

export function bytes(b: number): string {
  const units = ['bytes', 'kB', 'MB', 'GB', 'TB'];
  let u = 0;
  let v = b;
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000;
    u++;
  }
  return `${u === 0 ? v : v.toPrecision(3).replace(/\.0+$/, '')} ${units[u]}`;
}

interface Series {
  label: string;
  color: string;
  dash: number[];
  width: number;
  y: (x: number) => number;
}

interface Chart {
  xs: number[];
  /** Log-scaled x axis (image side) or linear (band-limit). */
  logX: boolean;
  xTicks: number[];
  xLabel: string;
  yMax: number;
  series: Series[];
  lines: { y: number; label: string }[];
}

class WallFigure implements Figure {
  private tab: Tab = 'image';
  private side = 32;
  private L = 15;
  private readonly tween = new Tween();
  private readonly chart: Surface;
  private readonly tabs: Segmented<Tab>;
  private readonly sideSlider: Slider;
  private readonly lSlider: Slider;
  private readonly legend: HTMLElement;
  private readonly readout: HTMLElement;

  constructor(
    host: HTMLElement,
    private readonly counts: SphereCounts,
  ) {
    const root = el('div', { class: 'rw wl' });
    this.tabs = segmented(
      'Signal',
      [
        { value: 'image', label: 'an image, under shifts' },
        { value: 'sphere', label: 'a sphere, under rotations' },
      ],
      this.tab,
      (t) => this.setTab(t),
    );
    this.sideSlider = slider({
      label: 'image side',
      min: 1,
      max: SIDE_MAX / SIDE_STEP,
      value: this.side / SIDE_STEP,
      format: (v) => `${v * SIDE_STEP} × ${v * SIDE_STEP}`,
      onInput: (v) => {
        this.tween.stop();
        this.side = v * SIDE_STEP;
        this.update();
      },
    });
    this.lSlider = slider({
      label: 'band-limit L',
      min: 1,
      max: counts.L.length,
      value: this.L,
      onInput: (v) => {
        this.tween.stop();
        this.L = v;
        this.update();
      },
    });
    const panel = el('div', { class: 'rw-panel' });
    this.legend = el('div', { class: 'rw-legend' });
    panel.append(this.legend);
    this.chart = new Surface(panel, 'wl-chart', () => this.draw());
    this.readout = el('p', { class: 'rg-readout wl-readout' });
    root.append(
      el(
        'div',
        { class: 'rw-controls' },
        this.tabs.element,
        this.sideSlider.element,
        this.lSlider.element,
      ),
      panel,
      this.readout,
    );
    host.replaceChildren(root);
    this.setTab(this.tab);
  }

  apply(step: string): void {
    switch (step) {
      case 'small':
        this.setTab('image');
        this.sweep(32);
        break;
      case 'big':
        this.setTab('image');
        this.sweep(SIDE_MAX);
        break;
      case 'sphere':
        this.setTab('sphere');
        this.sweep(this.counts.L.length);
        break;
    }
  }

  private setTab(tab: Tab): void {
    this.tween.stop();
    this.tab = tab;
    this.tabs.set(tab);
    this.sideSlider.element.hidden = tab !== 'image';
    this.lSlider.element.hidden = tab !== 'sphere';
    this.legend.replaceChildren(
      ...this.model().series.map((s) => {
        const swatch = el('i', { class: 'line' });
        swatch.style.borderTop = `${s.width}px ${s.dash.length ? 'dashed' : 'solid'} ${s.color}`;
        return el('span', {}, swatch, s.label);
      }),
    );
    this.update();
  }

  /** Moves the slider of the current tab to `to`. */
  private sweep(to: number): void {
    const from = this.tab === 'image' ? this.side : this.L;
    this.tween.run(900, (t) => {
      const v = from + t * (to - from);
      if (this.tab === 'image') this.side = Math.max(1, Math.round(v / SIDE_STEP)) * SIDE_STEP;
      else this.L = Math.round(v);
      this.update();
    });
  }

  private model(): Chart {
    if (this.tab === 'image') {
      const pixels = (s: number): number => s * s;
      return {
        xs: Array.from({ length: SIDE_MAX / SIDE_STEP }, (_, i) => (i + 1) * SIDE_STEP),
        logX: true,
        xTicks: [8, 16, 32, 64, 128, 256],
        xLabel: 'image side, in pixels',
        yMax: 1e11,
        series: [
          {
            label: 'full bispectrum: every pair of frequencies',
            color: INK,
            dash: [],
            width: 1.75,
            y: (s) => fullCount(pixels(s)),
          },
          {
            label: 'selective: one product per pixel',
            color: ORANGE_TEXT,
            dash: [],
            width: 2.25,
            y: pixels,
          },
        ],
        lines: GPUS.map((g) => ({ y: (g.gb * 1e9) / BYTES_PER_NUMBER, label: g.label })),
      };
    }
    const c = this.counts;
    const at =
      (k: keyof SphereCounts) =>
      (L: number): number =>
        c[k][L - 1]!;
    return {
      xs: c.L,
      logX: false,
      xTicks: [1, 8, 16, 24, 32, 40, 48, 56, 64].filter((l) => l <= c.L.length),
      xLabel: 'band-limit L',
      yMax: 1e9,
      series: [
        { label: 'full: numbers', color: INK, dash: [], width: 1.75, y: at('full') },
        {
          label: 'full: terms',
          color: INK,
          dash: [2, 3],
          width: 1.25,
          y: at('full_terms'),
        },
        {
          label: 'selective: numbers',
          color: ORANGE_TEXT,
          dash: [],
          width: 2.25,
          y: at('selective'),
        },
        {
          label: 'selective: terms',
          color: ORANGE_TEXT,
          dash: [2, 3],
          width: 1.5,
          y: at('selective_terms'),
        },
        {
          label: 'signal itself',
          color: STONE,
          dash: [6, 3],
          width: 1.25,
          y: at('coefficients'),
        },
      ],
      lines: [],
    };
  }

  private update(): void {
    this.sideSlider.set(this.side / SIDE_STEP);
    this.lSlider.set(this.L);
    this.chart.render();
    if (this.tab === 'image') {
      const n = this.side * this.side;
      const full = fullCount(n);
      const mem = full * BYTES_PER_NUMBER;
      const [small, large] = GPUS.map((g) => int(Math.floor((g.gb * 1e9) / mem)));
      this.readout.textContent =
        `A ${this.side} × ${this.side} image has ${int(n)} pixels. Full bispectrum: ` +
        `${int(full)} numbers, ${bytes(mem)} per image at ${BYTES_PER_NUMBER} bytes each, so a ` +
        `${GPUS[0]!.gb} GB GPU holds ${small} of them and an ${GPUS[1]!.gb} GB one ${large}. ` +
        `Selective: ${int(n)} numbers, ${bytes(n * BYTES_PER_NUMBER)}, ${int(Math.floor(full / n))}× fewer.`;
    } else {
      const c = this.counts;
      const i = this.L - 1;
      const ratio = c.full_terms[i]! / c.selective_terms[i]!;
      this.readout.textContent =
        `At L = ${this.L} the signal is ${int(c.coefficients[i]!)} numbers. Full bispectrum: ` +
        `${int(c.full[i]!)} numbers from ${int(c.full_terms[i]!)} terms. Selective: ` +
        `${int(c.selective[i]!)} numbers from ${int(c.selective_terms[i]!)}, ` +
        `${ratio.toFixed(ratio < 10 ? 1 : 0)}× less work.`;
    }
  }

  private draw(): void {
    const { ctx, w, h } = this.chart;
    const m = this.model();
    const pad = { l: 34, r: m.lines.length ? 78 : 14, t: 10, b: 34 };
    const [pw, ph] = [w - pad.l - pad.r, h - pad.t - pad.b];
    const [x0, x1] = [m.xs[0]!, m.xs.at(-1)!];
    const fx = (x: number): number =>
      pad.l +
      pw * (m.logX ? Math.log(x / x0) / Math.log(x1 / x0) : (x - x0) / Math.max(1, x1 - x0));
    const top = Math.log10(m.yMax);
    const fy = (y: number): number => pad.t + ph * (1 - Math.log10(Math.max(y, 1)) / top);

    ctx.font = '11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.lineWidth = 1;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    const names: Record<number, string> = { 0: '1', 3: '10³', 6: '10⁶', 9: '10⁹' };
    for (let e = 0; e <= top; e++) {
      const y = Math.round(fy(10 ** e)) + 0.5;
      ctx.strokeStyle = RULE;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + pw, y);
      ctx.stroke();
      if (names[e]) {
        ctx.fillStyle = MUTED;
        ctx.fillText(names[e]!, pad.l - 6, y);
      }
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = MUTED;
    for (const t of m.xTicks) ctx.fillText(String(t), fx(t), pad.t + ph + 6);
    ctx.font = '11px "IBM Plex Sans", system-ui, sans-serif';
    ctx.fillText(m.xLabel, pad.l + pw / 2, pad.t + ph + 20);

    for (const line of m.lines) {
      const y = Math.round(fy(line.y)) + 0.5;
      ctx.strokeStyle = STONE;
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + pw, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = MUTED;
      ctx.fillText(line.label, pad.l + pw + 6, y);
    }

    const cur = this.tab === 'image' ? this.side : this.L;
    const cx = Math.round(fx(cur)) + 0.5;
    ctx.strokeStyle = MUTED;
    ctx.beginPath();
    ctx.moveTo(cx, pad.t);
    ctx.lineTo(cx, pad.t + ph);
    ctx.stroke();

    for (const s of m.series) {
      ctx.beginPath();
      m.xs.forEach((x, i) => {
        if (i) ctx.lineTo(fx(x), fy(s.y(x)));
        else ctx.moveTo(fx(x), fy(s.y(x)));
      });
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width;
      ctx.setLineDash(s.dash);
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(fx(cur), fy(s.y(cur)), 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = s.color;
      ctx.fill();
    }
  }
}

export const mount: Mount = async (root) =>
  new WallFigure(root, (await dataJson<Counts>('counts.json')).sphere);
