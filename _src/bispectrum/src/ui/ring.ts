/**
 * The ring figure (`ring`): a curve drawn by eight harmonics on top of its mean, the plane of
 * its first two phases, the phases and power spectrum of its coefficients, the triple products
 * `β(k1, k2)` with `k1 + k2 <= 8`, and the chain walked back.
 *
 * `../math/ring` follows `torch.fft.fft`, where a turn by `α` takes `c_k` to `c_k e^{-ikα}`. The
 * figure shows `φ_k = -arg c_k`, which a turn advances by `kα`, as in the prose.
 */

import type { Figure, Mount } from '../figure';
import {
  bandPairs,
  chain,
  chainPairs,
  copy,
  invertChain,
  outline,
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
const K = 8;
const TAU = 2 * Math.PI;
const OUTLINE = 360;
const PAIRS = bandPairs(K + 1);
const CHAIN = chainPairs(K + 1);
const ON_CHAIN = new Set(CHAIN.map(([a, b]) => `${a},${b}`));
/** Bottom of the log scale of `|c_k|^2`; `c_0 = 1` sets the top. */
const POWER_FLOOR = 1e-4;

const MONO = '11px "IBM Plex Mono", ui-monospace, monospace';
const SANS = '11px "IBM Plex Sans", system-ui, sans-serif';

const wrap = (a: number): number => ((a % TAU) + TAU) % TAU;
const deg = (a: number): number => Math.round(wrap(a) * (360 / TAU)) % 360;
const sub = (n: number): string => [...String(n)].map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');

/** A random curve: mean 1, `|c_k|` up to `0.2 / k^0.7`, radius kept above 0.3. */
function randomCurve(rng: Rng): Spectrum {
  for (;;) {
    const c = spectrum(K + 1);
    c.re[0] = 1;
    for (let k = 1; k <= K; k++) {
      const len = (0.2 / k ** 0.7) * (0.55 + 0.45 * rng.uniform());
      const a = TAU * rng.uniform();
      c.re[k] = len * Math.cos(a);
      c.im[k] = len * Math.sin(a);
    }
    if (Math.min(...outline(c, OUTLINE)) > 0.3) return c;
  }
}

function curvePath(ctx: CanvasRenderingContext2D, c: Spectrum, cx: number, cy: number, s: number) {
  const r = outline(c, OUTLINE);
  ctx.beginPath();
  for (let j = 0; j <= OUTLINE; j++) {
    const t = (TAU * j) / OUTLINE;
    const rr = r[j % OUTLINE]! * s;
    // Angles run counter-clockwise on screen.
    const [x, y] = [cx + rr * Math.cos(t), cy - rr * Math.sin(t)];
    if (j) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  }
  ctx.closePath();
}

/** `φ_k`, the phase a turn advances by `kα`. */
const phase = (c: Spectrum, k: number): number => -Math.atan2(c.im[k]!, c.re[k]!);

/** The phase of `β(k1, k2)`, `φ_k1 + φ_k2 - φ_{k1 + k2}`. */
function triplePhase(c: Spectrum, k1: number, k2: number): number {
  const [re, im] = triple(c, k1, k2);
  return -Math.atan2(im, re);
}

/**
 * The closed line through `(a, b)` of slope 2 on the torus `[0, 2π)²`, drawn into the square
 * at `(x0, y0)` of side `S`, `φ_2` upwards. It wraps once across and twice up.
 */
function torusLine(
  ctx: CanvasRenderingContext2D,
  a: number,
  b: number,
  x0: number,
  y0: number,
  S: number,
): void {
  const [u0, v0] = [wrap(a) / TAU, wrap(b) / TAU];
  const cuts = [0, 1, 1 - u0, (1 - v0) / 2, (2 - v0) / 2].filter((s) => s >= 0 && s <= 1);
  cuts.sort((p, q) => p - q);
  ctx.beginPath();
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [s0, s1] = [cuts[i]!, cuts[i + 1]!];
    if (s1 - s0 < 1e-9) continue;
    const mid = (s0 + s1) / 2;
    const [du, dv] = [Math.floor(u0 + mid), Math.floor(v0 + 2 * mid)];
    const at = (s: number): [number, number] => [
      x0 + S * (u0 + s - du),
      y0 + S * (1 - (v0 + 2 * s - dv)),
    ];
    ctx.moveTo(...at(s0));
    ctx.lineTo(...at(s1));
  }
}

function panel(title: string, hint: string): HTMLElement {
  return el(
    'div',
    { class: 'rw-panel' },
    el(
      'div',
      { class: 'rw-panel-head' },
      el('h4', { text: title }),
      el('span', { class: 'rw-hint', text: hint }),
    ),
  );
}

interface SpecLayout {
  height: number;
  gutter: number;
  r: number;
  row: number;
  dialY: number;
  barTop: number;
  barH: number;
  tableTop: number;
  x: (k: number) => number;
}

function specLayout(w: number): SpecLayout {
  const gutter = 62;
  const pitch = (w - gutter) / (K + 1);
  const r = Math.max(9, Math.min(17, pitch * 0.36));
  const row = 2 * r + 10;
  const dialY = 18 + row / 2;
  const barTop = 18 + row + 4;
  const barH = 34;
  const tableTop = barTop + barH + 16;
  const rows = Math.floor(K / 2) + 1;
  return {
    height: Math.ceil(tableTop + rows * row + 22),
    gutter,
    r,
    row,
    dialY,
    barTop,
    barH,
    tableTop,
    x: (k) => gutter + pitch * (k + 0.5),
  };
}

class RingFigure implements Figure {
  private readonly rng = new Rng(11);
  private base: Spectrum;
  private scrambled: Spectrum | null = null;
  private alpha = 0;
  private pick: [number, number] = [2, 3];
  /** Chain entries used so far, `0..K + 1`. */
  private steps = 0;
  /** The free phase: `φ_1` of the rebuilt curve. */
  private free = 0;
  private readonly tween = new Tween();
  private readonly signal: Surface;
  private readonly plane: Surface;
  private readonly spec: Surface;
  private readonly alphaSlider: Slider;
  private readonly stepsSlider: Slider;
  private readonly freeSlider: Slider;
  private readonly entryText: HTMLElement;
  private readonly chainText: HTMLElement;
  private cells: { k1: number; k2: number; x: number; y: number; r: number }[] = [];

  constructor(host: HTMLElement) {
    this.base = randomCurve(this.rng);

    const signalPanel = panel('Curve', 'drag to turn');
    this.signal = new Surface(signalPanel, 'rg-signal', () => this.drawSignal());
    const planePanel = panel('Phases of c₁ and c₂', 'grey: arg β(1, 1) constant');
    this.plane = new Surface(planePanel, 'rg-plane', () => this.drawPlane());

    this.alphaSlider = slider({
      label: 'turn α',
      min: 0,
      max: 359,
      value: 0,
      format: (v) => `${v}°`,
      onInput: (v) => {
        this.tween.stop();
        this.alpha = (v * TAU) / 360;
        this.update();
      },
    });
    const controls = el(
      'div',
      { class: 'rw-row' },
      this.alphaSlider.element,
      button('scramble the phases', () => this.setScrambled(true)),
      button('new curve', () => this.newCurve()),
      button('reset', () => this.reset(), { kind: 'quiet' }),
    );

    const specPanel = panel('Coefficients and triple products', 'click an entry');
    specPanel.classList.add('rg-spec-panel');
    this.spec = new Surface(specPanel, 'rg-spec', () => this.drawSpec());
    this.spec.canvas.addEventListener('click', (e) => this.onCellClick(e));
    // The table's height follows its width, so the column pitch sets the row pitch.
    new ResizeObserver(() => {
      const h = `${specLayout(this.spec.canvas.clientWidth).height}px`;
      if (this.spec.canvas.style.height !== h) this.spec.canvas.style.height = h;
    }).observe(this.spec.canvas);
    this.entryText = el('p', { class: 'rg-readout' });
    specPanel.append(this.entryText);

    this.stepsSlider = slider({
      label: 'chain entries used',
      min: 0,
      max: K + 1,
      value: 0,
      format: (v) => `${v} of ${K + 1}`,
      onInput: (v) => this.stepTo(v),
    });
    this.freeSlider = slider({
      label: 'free phase φ₁',
      min: 0,
      max: 359,
      value: 0,
      format: (v) => `${v}°`,
      onInput: (v) => {
        this.tween.stop();
        this.free = (v * TAU) / 360;
        if (this.steps < 2) this.steps = K + 1;
        this.update();
      },
    });
    this.chainText = el('p', { class: 'rw-status' });

    const root = el(
      'div',
      { class: 'rw rg' },
      el('div', { class: 'rw-stage' }, signalPanel, planePanel),
      controls,
      specPanel,
      el('div', { class: 'rw-row' }, this.stepsSlider.element, this.freeSlider.element),
      this.chainText,
    );
    host.replaceChildren(root);
    this.attachDrag();
    this.update();
  }

  // ------------------------------------------------------------------ state

  private get shape(): Spectrum {
    return this.scrambled ?? this.base;
  }

  private get current(): Spectrum {
    return rotate(this.shape, this.alpha);
  }

  /** Coefficients recovered from the first `steps` chain entries, at the free phase. */
  private get rebuilt(): Spectrum | null {
    if (!this.steps) return null;
    // `invertChain` takes `arg c_1`, the opposite of the figure's phase.
    const known = copy(invertChain(chain(this.current), -this.free));
    for (let k = this.steps; k <= K; k++) known.re[k] = known.im[k] = 0;
    return known;
  }

  private update(): void {
    this.signal.render();
    this.plane.render();
    this.spec.render();
    this.alphaSlider.set(deg(this.alpha));
    this.stepsSlider.set(this.steps);
    this.freeSlider.set(deg(this.free));
    this.entryText.textContent = this.entryLine();
    this.chainText.textContent = this.chainLine();
  }

  private entryLine(): string {
    const [k1, k2] = this.pick;
    const k3 = k1 + k2;
    if (k2 === 0) return 'β(0, 0) = c₀³, real and positive.';
    if (k1 === 0) {
      return `β(0, ${k2}) = c₀ |c${sub(k2)}|², real and positive: c₀ times the power spectrum.`;
    }
    const ph = deg(triplePhase(this.current, k1, k2));
    return (
      `β(${k1}, ${k2}) = c${sub(k1)} c${sub(k2)} c̄${sub(k3)}. Its phase is ` +
      `φ${sub(k1)} + φ${sub(k2)} − φ${sub(k3)} = ${ph}°, and a turn by α adds ` +
      `(${k1} + ${k2} − ${k3})α = 0.`
    );
  }

  private chainLine(): string {
    const s = this.steps;
    if (s === 0) return `The chain is the ${K + 1} outlined entries. None used yet.`;
    if (s === 1) return 'β(0, 0) = c₀³ gives c₀, the mean radius.';
    if (s === 2) return 'β(0, 1) = c₀ |c₁|² gives |c₁|. No entry contains φ₁, so it is free.';
    const k = s - 2;
    const line =
      `β(1, ${k}) = c₁ c${sub(k)} c̄${sub(k + 1)} gives ` +
      `c${sub(k + 1)} = conj(β(1, ${k}) / (c₁ c${sub(k)})).`;
    return s <= K
      ? line
      : `${line} All ${K + 1} coefficients are recovered, up to φ₁: every value of it gives ` +
          'the same curve, turned.';
  }

  // ------------------------------------------------------------------ actions

  apply(step: string): void {
    switch (step) {
      case 'turn':
        this.scrambled = null;
        this.steps = 0;
        this.turnBy(TAU, 4500);
        break;
      case 'scramble':
        this.setScrambled(true);
        break;
      case 'pick':
        this.pick = [2, 3];
        this.steps = 0;
        this.turnBy(TAU / 2, 2400);
        break;
      case 'rebuild':
        this.free = 0;
        this.playRebuild();
        break;
      case 'free':
        this.sweepFree();
        break;
    }
  }

  private turnBy(delta: number, ms: number): void {
    const from = this.alpha;
    this.tween.run(ms, (t) => {
      this.alpha = from + t * delta;
      this.update();
    });
  }

  private setScrambled(on: boolean): void {
    this.tween.stop();
    this.scrambled = on ? scramblePhases(this.base, this.rng) : null;
    this.steps = 0;
    this.update();
  }

  private newCurve(): void {
    this.tween.stop();
    this.base = randomCurve(this.rng);
    this.scrambled = null;
    this.alpha = 0;
    this.steps = 0;
    this.update();
  }

  private reset(): void {
    this.tween.stop();
    this.scrambled = null;
    this.alpha = 0;
    this.steps = 0;
    this.free = 0;
    this.pick = [2, 3];
    this.update();
  }

  private stepTo(s: number): void {
    this.tween.stop();
    this.steps = Math.max(0, Math.min(K + 1, s));
    if (this.steps) this.pick = CHAIN[this.steps - 1]!;
    this.update();
  }

  private playRebuild(): void {
    this.stepTo(0);
    this.tween.run(
      450 * (K + 1),
      (t) => {
        const s = Math.ceil(t * (K + 1));
        if (s !== this.steps) {
          this.steps = s;
          this.pick = CHAIN[s - 1]!;
          this.update();
        }
      },
      () => this.stepTo(K + 1),
    );
  }

  /** One full turn of the free phase, ending where the rebuilt curve lies on the signal. */
  private sweepFree(): void {
    if (this.steps <= K) this.stepTo(K + 1);
    const from = this.free;
    const d = wrap(phase(this.current, 1) - from) + TAU;
    this.tween.run(3200, (t) => {
      this.free = from + t * d;
      this.update();
    });
  }

  // ------------------------------------------------------------------ input

  private attachDrag(): void {
    const c = this.signal.canvas;
    let last: number | null = null;
    const angleOf = (e: PointerEvent): number => {
      const r = c.getBoundingClientRect();
      return Math.atan2(-(e.clientY - r.top - r.height / 2), e.clientX - r.left - r.width / 2);
    };
    c.addEventListener('pointerdown', (e) => {
      this.tween.stop();
      last = angleOf(e);
      c.setPointerCapture(e.pointerId);
      c.classList.add('dragging');
    });
    c.addEventListener('pointermove', (e) => {
      if (last === null) return;
      const a = angleOf(e);
      let d = a - last;
      if (d > Math.PI) d -= TAU;
      if (d < -Math.PI) d += TAU;
      last = a;
      this.alpha += d;
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
    const rect = this.spec.canvas.getBoundingClientRect();
    const [x, y] = [e.clientX - rect.left, e.clientY - rect.top];
    const hit = this.cells.find((c) => Math.hypot(x - c.x, y - c.y) <= c.r + 3);
    if (!hit) return;
    this.tween.stop();
    this.pick = [hit.k1, hit.k2];
    this.update();
  }

  // ------------------------------------------------------------------ drawing

  private drawSignal(): void {
    const { ctx, w, h } = this.signal;
    const [cx, cy] = [w / 2, h / 2];
    const s = (Math.min(w, h) / 2 - 10) / 1.75;

    ctx.strokeStyle = RULE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, s, 0, TAU);
    ctx.stroke();

    if (this.scrambled) {
      curvePath(ctx, rotate(this.base, this.alpha), cx, cy, s);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = STONE;
      ctx.lineWidth = 1.25;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    curvePath(ctx, this.current, cx, cy, s);
    ctx.fillStyle = SAND;
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.75;
    ctx.stroke();

    // A notch at θ = 0 that rides with the curve, so a turn shows even on a round one.
    const r0 = radiusAt(this.shape, 0) * s;
    ctx.beginPath();
    ctx.moveTo(cx + 0.82 * r0 * Math.cos(this.alpha), cy - 0.82 * r0 * Math.sin(this.alpha));
    ctx.lineTo(cx + r0 * Math.cos(this.alpha), cy - r0 * Math.sin(this.alpha));
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.stroke();

    const rb = this.rebuilt;
    if (rb) {
      curvePath(ctx, rb, cx, cy, s);
      ctx.strokeStyle = ORANGE_TEXT;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.fillStyle = MUTED;
    ctx.font = SANS;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    if (rb) ctx.fillText('orange: rebuilt from the chain', 4, 4);
    else if (this.scrambled) ctx.fillText('dashed: before the scramble', 4, 4);
  }

  private drawPlane(): void {
    const { ctx, w, h } = this.plane;
    const m = { l: 34, r: 12, t: 8, b: 30 };
    const S = Math.max(40, Math.min(w - m.l - m.r, h - m.t - m.b));
    const x0 = m.l + (w - m.l - m.r - S) / 2;
    const y0 = m.t;
    const X = (a: number): number => x0 + (S * wrap(a)) / TAU;
    const Y = (b: number): number => y0 + S * (1 - wrap(b) / TAU);

    ctx.fillStyle = TINTS.panel.hex;
    ctx.fillRect(x0, y0, S, S);

    // Level sets of 2φ₁ − φ₂, the phase of β(1, 1): lines of slope 2.
    ctx.strokeStyle = RULE;
    ctx.lineWidth = 1;
    for (let j = 0; j < 8; j++) {
      torusLine(ctx, 0, (-j * TAU) / 8, x0, y0, S);
      ctx.stroke();
    }
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, S - 1, S - 1);

    ctx.fillStyle = MUTED;
    ctx.font = MONO;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    for (const [t, label] of [
      [0, '0'],
      [0.5, 'π'],
      [1, '2π'],
    ] as const) {
      ctx.fillText(label, x0 + S * t, y0 + S + 4);
    }
    ctx.fillText('φ₁', x0 + S * 0.75, y0 + S + 16);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const [t, label] of [
      [0, '0'],
      [0.5, 'π'],
      [1, '2π'],
    ] as const) {
      ctx.fillText(label, x0 - 5, y0 + S * (1 - t));
    }
    ctx.fillText('φ₂', x0 - 5, y0 + S * 0.25);

    const dot = (a: number, b: number, fill: string, stroke: string): void => {
      ctx.beginPath();
      ctx.arc(X(a), Y(b), 4.5, 0, TAU);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.25;
      ctx.stroke();
    };

    if (this.scrambled) {
      const before = rotate(this.base, this.alpha);
      const [a, b] = [phase(before, 1), phase(before, 2)];
      torusLine(ctx, a, b, x0, y0, S);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = STONE;
      ctx.lineWidth = 1.25;
      ctx.stroke();
      ctx.setLineDash([]);
      dot(a, b, TINTS.panel.hex, STONE);
    }

    const cur = this.current;
    const [a, b] = [phase(cur, 1), phase(cur, 2)];
    torusLine(ctx, a, b, x0, y0, S);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    dot(a, b, INK, INK);

    const rb = this.rebuilt;
    if (rb && this.steps >= 3) dot(phase(rb, 1), phase(rb, 2), ORANGE, ORANGE_TEXT);
  }

  private drawSpec(): void {
    const { ctx, w } = this.spec;
    const L = specLayout(w);
    const cur = this.current;
    const pw = power(this.shape);
    const rb = this.rebuilt;
    const [k1, k2] = this.pick;
    const hot = new Set([k1, k2, k1 + k2]);
    const used = new Set(CHAIN.slice(0, this.steps).map(([a, b]) => `${a},${b}`));

    const label = (text: string, y: number): void => {
      ctx.fillStyle = MUTED;
      ctx.font = MONO;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, L.gutter - 10, y);
    };
    const dial = (x: number, y: number, fill: string, rim: string, rimWidth: number): void => {
      ctx.beginPath();
      ctx.arc(x, y, L.r, 0, TAU);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = rim;
      ctx.lineWidth = rimWidth;
      ctx.stroke();
    };
    const needle = (x: number, y: number, a: number, color: string): void => {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 0.82 * L.r * Math.cos(a), y - 0.82 * L.r * Math.sin(a));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.stroke();
    };

    // Column header: k.
    label('k', 9);
    ctx.font = MONO;
    ctx.textAlign = 'center';
    for (let k = 0; k <= K; k++) {
      ctx.fillStyle = hot.has(k) ? ORANGE_TEXT : MUTED;
      ctx.fillText(String(k), L.x(k), 9);
    }

    // Phases φ_k, with the recovered ones in orange.
    label('φₖ', L.dialY);
    for (let k = 0; k <= K; k++) {
      const x = L.x(k);
      const on = hot.has(k);
      dial(x, L.dialY, on ? TINTS.orangeWash.hex : TINTS.panel.hex, on ? ORANGE_TEXT : RULE, 1);
      needle(x, L.dialY, phase(cur, k), INK);
      ctx.beginPath();
      ctx.arc(x, L.dialY, 2.25, 0, TAU);
      ctx.fillStyle = INK;
      ctx.fill();
      if (rb && k < this.steps) {
        const a = phase(rb, k);
        ctx.beginPath();
        ctx.arc(x + 0.82 * L.r * Math.cos(a), L.dialY - 0.82 * L.r * Math.sin(a), 3.5, 0, TAU);
        ctx.fillStyle = ORANGE;
        ctx.fill();
        ctx.strokeStyle = ORANGE_TEXT;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Power spectrum |c_k|², log scale from POWER_FLOOR to 1.
    label('|cₖ|²', L.barTop + L.barH / 2 - 6);
    ctx.font = SANS;
    ctx.fillText('log', L.gutter - 10, L.barTop + L.barH / 2 + 8);
    const span = -Math.log10(POWER_FLOOR);
    const bw = Math.min(14, L.r);
    for (let k = 0; k <= K; k++) {
      const f = Math.max(0, Math.min(1, (Math.log10(pw[k]!) + span) / span));
      const x = L.x(k) - bw / 2;
      ctx.fillStyle = TINTS.sandWash.hex;
      ctx.fillRect(x, L.barTop, bw, L.barH);
      ctx.fillStyle = INK;
      ctx.fillRect(x, L.barTop + L.barH * (1 - f), bw, L.barH * f);
    }

    // β(k1, k) under column k: its phase, φ_k1 + φ_k − φ_{k1 + k}.
    this.cells = [];
    for (let a = 0; a <= Math.floor(K / 2); a++) {
      label(`β(${a}, k)`, L.tableTop + L.row * (a + 0.5));
    }
    for (const [a, b] of PAIRS) {
      const [x, y] = [L.x(b), L.tableTop + L.row * (a + 0.5)];
      this.cells.push({ k1: a, k2: b, x, y, r: L.r });
      const key = `${a},${b}`;
      const sel = a === k1 && b === k2;
      const rim = used.has(key) ? ORANGE_TEXT : ON_CHAIN.has(key) ? INK : RULE;
      dial(x, y, sel ? TINTS.orangeWash.hex : TINTS.panel.hex, rim, ON_CHAIN.has(key) ? 1.75 : 1);
      needle(x, y, triplePhase(cur, a, b), ORANGE_TEXT);
    }

    ctx.fillStyle = MUTED;
    ctx.font = SANS;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(
      `outlined: the chain, ${K + 1} entries; orange rim: used so far`,
      L.gutter,
      L.height - 6,
    );
  }
}

export const mount: Mount = async (root) => new RingFigure(root);
