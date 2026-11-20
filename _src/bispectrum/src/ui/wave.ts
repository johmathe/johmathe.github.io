/**
 * The shift figure (`wave`), one panel: a periodic signal of four harmonics on top of its mean,
 * each harmonic with its phase dial and power bar, all nine nonzero bispectrum entries
 * (`0 <= k1 <= k2`, `k1 + k2 <= 4`), and harmonic-by-harmonic recovery up to a shift.
 *
 * `../math/ring` follows `torch.fft.fft`, and so do the figure and the prose: a shift by `s` takes
 * `c_k` to `c_k e^{-iks}`, so `φ_k = arg c_k` falls by `ks`.
 */

import type { Figure, Mount } from '../figure';
import {
  bandPairs,
  chain,
  chainPairs,
  copy,
  invertChain,
  power,
  radiusAt,
  rotate,
  scramblePhases,
  spectrum,
  triple,
  type Spectrum,
} from '../math/ring';
import { Rng } from '../math/rng';
import { INK, MUTED, ORANGE, ORANGE_TEXT, RULE, SAND, STONE, TINTS } from './colormap';
import { Surface, Tween } from './canvas';
import { button, el, slider, type Slider } from './controls';

/** Harmonics on top of the mean: coefficients `c_0..c_K`. */
const K = 4;
const TAU = 2 * Math.PI;
const SAMPLES = 240;
/** Every nonzero upper-triangle entry (`0 <= k1 <= k2`, `k1 + k2 <= K`). */
const ENTRIES: [number, number][] = bandPairs(K + 1);
const COLS = K + 1;
const ROWS = Math.floor(K / 2) + 1;
const CHAIN = chainPairs(K + 1);

const MONO = '11px "IBM Plex Mono", ui-monospace, monospace';
const LABEL_MONO = '10.5px "IBM Plex Mono", ui-monospace, monospace';
const SANS = '11px "IBM Plex Sans", system-ui, sans-serif';

const wrap = (a: number): number => ((a % TAU) + TAU) % TAU;
const deg = (a: number): number => Math.round(wrap(a) * (360 / TAU)) % 360;
const sub = (n: number): string => [...String(n)].map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');
const percent = (s: number): number => Math.round((wrap(s) / TAU) * 100) % 100;

/** A random signal: mean 1, `|c_k|` between about `0.14 / k^0.5` and `0.3 / k^0.5`. */
function randomSignal(rng: Rng): Spectrum {
  const c = spectrum(K + 1);
  c.re[0] = 1;
  for (let k = 1; k <= K; k++) {
    const len = (0.3 / k ** 0.5) * (0.45 + 0.55 * rng.uniform());
    const a = TAU * rng.uniform();
    c.re[k] = len * Math.cos(a);
    c.im[k] = len * Math.sin(a);
  }
  return c;
}

/** `φ_k = arg c_k`, which a shift by `s` lowers by `ks`. */
const phase = (c: Spectrum, k: number): number => Math.atan2(c.im[k]!, c.re[k]!);
const magnitude = (c: Spectrum, k: number): number => Math.hypot(c.re[k]!, c.im[k]!);

/** The phase of `β(k1, k2)`, `φ_k1 + φ_k2 - φ_{k1 + k2}`. */
function triplePhase(c: Spectrum, k1: number, k2: number): number {
  const [re, im] = triple(c, k1, k2);
  return Math.atan2(im, re);
}

/** Only harmonic `k` of `c`, mean removed. */
function only(c: Spectrum, k: number): Spectrum {
  const out = spectrum(K + 1);
  [out.re[k], out.im[k]] = [c.re[k]!, c.im[k]!];
  return out;
}

class WaveFigure implements Figure {
  private readonly rng = new Rng(7);
  private base: Spectrum;
  private scrambled: Spectrum | null = null;
  private shift = 0;
  private pick: [number, number] | null = null;
  /** Recovery entries applied so far, `0..K + 1`. */
  private steps = 0;
  /** The free phase, as `φ_1` of the rebuilt signal minus `φ_1` of the signal. */
  private offset = 0;
  private readonly tween = new Tween();
  private readonly signal: Surface;
  private readonly harm: Surface;
  private readonly beta: Surface;
  private readonly shiftSlider: Slider;
  private cells: { k1: number; k2: number; x: number; y: number; r: number }[] = [];

  constructor(host: HTMLElement) {
    this.base = randomSignal(this.rng);

    const stage = el('div', { class: 'rw-panel wv-stage' });
    this.signal = new Surface(stage, 'wv-signal', () => this.drawSignal());
    const split = el('div', { class: 'wv-split' });
    stage.append(split);
    this.harm = new Surface(split, 'wv-harm', () => this.drawHarmonics());
    this.beta = new Surface(split, 'wv-beta', () => this.drawBeta());
    this.beta.canvas.addEventListener('click', (e) => this.onCellClick(e));

    this.shiftSlider = slider({
      label: 'Shift s',
      min: 0,
      max: 99,
      value: 0,
      format: (v) => `${v}%`,
      onInput: (v) => {
        this.tween.stop();
        this.shift = (v * TAU) / 100;
        this.update();
      },
    });

    host.replaceChildren(
      el(
        'div',
        { class: 'rw wv' },
        stage,
        el(
          'div',
          { class: 'rw-controls wv-bar' },
          this.shiftSlider.element,
          el(
            'div',
            { class: 'rw-row wv-buttons' },
            button('new signal', () => this.newSignal(), { kind: 'quiet' }),
            button('reset', () => this.reset(), { kind: 'quiet' }),
          ),
        ),
      ),
    );
    this.attachDrag();
    this.update();
  }

  // ------------------------------------------------------------------ state

  private get shape(): Spectrum {
    return this.scrambled ?? this.base;
  }

  private get current(): Spectrum {
    return rotate(this.shape, this.shift);
  }

  /** Coefficients recovered from the first `steps` entries, at the free phase. */
  private get rebuilt(): Spectrum | null {
    if (!this.steps) return null;
    const cur = this.current;
    const known = copy(invertChain(chain(cur), phase(cur, 1) + this.offset));
    for (let k = this.steps; k <= K; k++) known.re[k] = known.im[k] = 0;
    return known;
  }

  private update(): void {
    this.signal.render();
    this.harm.render();
    this.beta.render();
    this.shiftSlider.set(percent(this.shift));
  }

  private entryNote(): string {
    if (!this.pick) return '';
    const [k1, k2] = this.pick;
    const k3 = k1 + k2;
    if (k2 === 0) return 'β(0,0) = c₀³';
    if (k1 === 0) return `β(0,${k2}) = c₀ |c${sub(k2)}|²`;
    const ph = deg(triplePhase(this.current, k1, k2));
    return `β(${k1},${k2}) = c${sub(k1)} c${sub(k2)} c̄${sub(k3)} · φ${sub(k1)} + φ${sub(k2)} − φ${sub(k3)} = ${ph}°`;
  }

  // ------------------------------------------------------------------ actions

  apply(step: string): void {
    switch (step) {
      case 'shift':
        this.scrambled = null;
        this.steps = 0;
        this.pick = null;
        this.shiftBy(TAU, 4500);
        break;
      case 'scramble':
        this.pick = null;
        this.setScrambled(true);
        break;
      case 'recover':
        this.playRebuild();
        break;
    }
  }

  private shiftBy(delta: number, ms: number): void {
    const from = this.shift;
    this.tween.run(ms, (t) => {
      this.shift = from + t * delta;
      this.update();
    });
  }

  private setScrambled(on: boolean): void {
    this.tween.stop();
    this.scrambled = on ? scramblePhases(this.base, this.rng) : null;
    this.steps = 0;
    this.update();
  }

  private newSignal(): void {
    this.tween.stop();
    this.base = randomSignal(this.rng);
    this.scrambled = null;
    this.shift = 0;
    this.steps = 0;
    this.pick = null;
    this.update();
  }

  private reset(): void {
    this.tween.stop();
    this.scrambled = null;
    this.shift = 0;
    this.steps = 0;
    this.offset = 0;
    this.pick = null;
    this.update();
  }

  /** Step harmonic by harmonic at `φ₁ = 0`, then slide `φ₁` onto the signal's. */
  private playRebuild(): void {
    this.tween.stop();
    this.steps = 0;
    this.pick = null;
    const start = -(((wrap(phase(this.current, 1)) + Math.PI) % TAU) - Math.PI);
    this.offset = start;
    this.update();
    this.tween.run(
      600 * (K + 1),
      (t) => {
        const s = Math.ceil(t * (K + 1));
        if (s !== this.steps && s >= 1) {
          this.steps = s;
          this.pick = CHAIN[s - 1]!;
          this.update();
        }
      },
      () => {
        this.steps = K + 1;
        this.pick = null;
        this.update();
        setTimeout(() => {
          this.tween.run(1800, (t) => {
            this.offset = (1 - t) * start;
            this.update();
          });
        }, 500);
      },
    );
  }

  // ------------------------------------------------------------------ input

  private attachDrag(): void {
    const c = this.signal.canvas;
    let last: number | null = null;
    c.addEventListener('pointerdown', (e) => {
      this.tween.stop();
      last = e.clientX;
      c.setPointerCapture(e.pointerId);
      c.classList.add('dragging');
    });
    c.addEventListener('pointermove', (e) => {
      if (last === null) return;
      const width = c.getBoundingClientRect().width || 1;
      this.shift += ((e.clientX - last) / width) * TAU;
      last = e.clientX;
      this.update();
    });
    const end = (): void => {
      last = null;
      c.classList.remove('dragging');
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
  }

  private onCellClick(e: MouseEvent): void {
    const rect = this.beta.canvas.getBoundingClientRect();
    const [x, y] = [e.clientX - rect.left, e.clientY - rect.top];
    const hit = this.cells.find((c) => Math.hypot(x - c.x, y - c.y) <= c.r + 4);
    if (!hit) return;
    this.tween.stop();
    const same = this.pick !== null && this.pick[0] === hit.k1 && this.pick[1] === hit.k2;
    this.steps = 0;
    this.pick = same ? null : [hit.k1, hit.k2];
    this.update();
  }

  // ------------------------------------------------------------------ drawing

  /** Largest swing of the signal about its mean, for a scale shared by every shift. */
  private get swing(): number {
    let a = 0;
    for (let k = 1; k <= K; k++) a += 2 * magnitude(this.shape, k);
    return Math.max(a, 1e-6);
  }

  private plot(
    ctx: CanvasRenderingContext2D,
    f: (x: number) => number,
    x0: number,
    x1: number,
    y: (v: number) => number,
  ): void {
    ctx.beginPath();
    for (let j = 0; j <= SAMPLES; j++) {
      const t = (TAU * j) / SAMPLES;
      const [px, py] = [x0 + ((x1 - x0) * j) / SAMPLES, y(f(t))];
      if (j) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    }
  }

  private drawSignal(): void {
    const { ctx, w, h } = this.signal;
    const [x0, x1, top, bottom] = [4, w - 4, 6, h - 18];
    const mid = (top + bottom) / 2;
    const A = this.swing;
    const c0 = this.shape.re[0]!;
    const y = (v: number): number => mid - ((v - c0) / A) * ((bottom - top) / 2);

    ctx.strokeStyle = RULE;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x0, mid);
    ctx.lineTo(x1, mid);
    ctx.stroke();
    ctx.setLineDash([]);

    if (this.scrambled) {
      const before = rotate(this.base, this.shift);
      this.plot(ctx, (t) => radiusAt(before, t), x0, x1, y);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = STONE;
      ctx.lineWidth = 1.25;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const cur = this.current;
    this.plot(ctx, (t) => radiusAt(cur, t), x0, x1, y);
    ctx.lineTo(x1, mid);
    ctx.lineTo(x0, mid);
    ctx.closePath();
    ctx.fillStyle = SAND;
    ctx.fill();
    this.plot(ctx, (t) => radiusAt(cur, t), x0, x1, y);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.75;
    ctx.stroke();

    const rb = this.rebuilt;
    if (rb) {
      this.plot(ctx, (t) => radiusAt(rb, t), x0, x1, y);
      ctx.strokeStyle = ORANGE_TEXT;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // A tick at x = s that rides with the signal, so a shift shows even on a flat stretch.
    const xs = x0 + (x1 - x0) * (wrap(this.shift) / TAU);
    ctx.beginPath();
    ctx.moveTo(xs, bottom + 1);
    ctx.lineTo(xs - 4, bottom + 8);
    ctx.lineTo(xs + 4, bottom + 8);
    ctx.closePath();
    ctx.fillStyle = INK;
    ctx.fill();

    ctx.fillStyle = MUTED;
    ctx.font = MONO;
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'left';
    ctx.fillText('0', x0, h - 1);
    ctx.textAlign = 'right';
    ctx.fillText('2π', x1, h - 1);
    ctx.font = SANS;
    ctx.textAlign = 'left';
    if (rb) ctx.fillText('orange: rebuilt from the bispectrum', x0 + 22, h - 1);
    else if (this.pick) ctx.fillText(this.entryNote(), x0 + 22, h - 1);
    else if (this.scrambled) ctx.fillText('dashed: before the scramble', x0 + 22, h - 1);
  }

  private drawHarmonics(): void {
    const { ctx, w, h } = this.harm;
    const row = h / K;
    const r = Math.min(11, row / 2 - 3);
    const gutter = 30 + 2 * r;
    const barW = 8;
    const [x0, x1] = [gutter + 6, w - barW - 12];
    const cur = this.current;
    const rb = this.rebuilt;
    const pw = power(this.shape);
    const pmax = Math.max(...Array.from(pw).slice(1));
    const amax = Math.max(...Array.from({ length: K }, (_, i) => magnitude(this.shape, i + 1)));
    const hot = new Set<number>(
      this.pick === null
        ? []
        : this.pick[0] === 0
          ? [this.pick[1]]
          : [this.pick[0], this.pick[1], this.pick[0] + this.pick[1]],
    );

    for (let k = 1; k <= K; k++) {
      const [top, mid] = [(k - 1) * row, (k - 0.5) * row];
      if (hot.has(k)) {
        ctx.fillStyle = TINTS.orangeWash.hex;
        ctx.fillRect(0, top + 1, w, row - 2);
      }
      const half = row / 2 - 3;
      const y = (v: number): number => mid - (v / (2 * amax)) * half;

      ctx.fillStyle = hot.has(k) ? ORANGE_TEXT : MUTED;
      ctx.font = MONO;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(`k=${k}`, 2, mid);

      // Dial: the phase φ_k, and the rebuilt one in orange.
      const dx = 28 + r;
      ctx.beginPath();
      ctx.arc(dx, mid, r, 0, TAU);
      ctx.fillStyle = TINTS.panel.hex;
      ctx.fill();
      ctx.strokeStyle = hot.has(k) ? ORANGE_TEXT : STONE;
      ctx.lineWidth = 1.25;
      ctx.stroke();
      const a = phase(cur, k);
      ctx.beginPath();
      ctx.moveTo(dx, mid);
      ctx.lineTo(dx + 0.82 * r * Math.cos(a), mid - 0.82 * r * Math.sin(a));
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.stroke();
      if (rb && k < this.steps) {
        const b = phase(rb, k);
        ctx.beginPath();
        ctx.arc(dx + 0.82 * r * Math.cos(b), mid - 0.82 * r * Math.sin(b), 3, 0, TAU);
        ctx.fillStyle = ORANGE;
        ctx.fill();
        ctx.strokeStyle = ORANGE_TEXT;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // The harmonic itself, 2|c_k| cos(kx + φ_k).
      ctx.strokeStyle = RULE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, mid);
      ctx.lineTo(x1, mid);
      ctx.stroke();
      const hk = only(cur, k);
      this.plot(ctx, (t) => radiusAt(hk, t), x0, x1, y);
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      if (rb && k < this.steps) {
        const hr = only(rb, k);
        this.plot(ctx, (t) => radiusAt(hr, t), x0, x1, y);
        ctx.strokeStyle = ORANGE_TEXT;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Power bar, linear, the largest harmonic full.
      const bx = w - barW - 4;
      ctx.fillStyle = TINTS.sandWash.hex;
      ctx.fillRect(bx, top + 3, barW, row - 6);
      const f = pw[k]! / pmax;
      ctx.fillStyle = INK;
      ctx.fillRect(bx, top + 3 + (row - 6) * (1 - f), barW, (row - 6) * f);
    }
  }

  private drawBeta(): void {
    const { ctx, w, h } = this.beta;
    const cur = this.current;
    const used = new Set(CHAIN.slice(0, this.steps).map(([a, b]) => `${a},${b}`));
    const [pitchX, pitchY] = [w / COLS, h / ROWS];
    const r = Math.max(8, Math.min(12, pitchX * 0.23, (pitchY - 17) / 2));
    this.cells = [];
    for (const [a, b] of ENTRIES) {
      const x = pitchX * (b + 0.5);
      const y = pitchY * a + r + 3;
      this.cells.push({ k1: a, k2: b, x, y, r });
      const key = `${a},${b}`;
      const sel = this.pick !== null && a === this.pick[0] && b === this.pick[1];
      const active = used.has(key) || sel;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fillStyle = sel ? TINTS.orangeWash.hex : TINTS.panel.hex;
      ctx.fill();
      ctx.strokeStyle = active ? ORANGE_TEXT : STONE;
      ctx.lineWidth = active ? 1.75 : 1.25;
      ctx.stroke();
      const ph = triplePhase(cur, a, b);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 0.82 * r * Math.cos(ph), y - 0.82 * r * Math.sin(ph));
      ctx.strokeStyle = ORANGE_TEXT;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.fillStyle = active ? ORANGE_TEXT : MUTED;
      ctx.font = LABEL_MONO;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`β(${a},${b})`, x, y + r + 3);
    }
  }
}

export const mount: Mount = async (root) => new WaveFigure(root);
