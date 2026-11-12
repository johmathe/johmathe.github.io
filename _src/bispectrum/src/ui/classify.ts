import { classifierAssets, sharedAssets } from '../assets';
import type { DigitSet } from '../data';
import type { Mount } from '../figure';
import { degreePower, toHalfSpectrum } from '../math/basis';
import {
  BispectrumMlp,
  PowerMlp,
  softmax,
  type ClassifierManifest,
  type SplitAccuracy,
} from '../math/classifier';
import { Invariant, type InvariantTables } from '../math/invariant';
import { Rng } from '../math/rng';
import {
  expMap,
  identity3,
  matToQuat,
  mul3,
  quatToMat,
  randomRotation,
  rotateDofs,
  scrambleDegrees,
  type Mat3,
} from '../math/rotation';
import { SphereGrid } from '../math/sht';
import { DIGIT_PLANE, digitSignal } from '../signals';
import { CnnClient } from '../worker/classify.client';
import { inkAt, METHOD_FILLS } from './colormap';
import { button, DigitPicker, el, toggle, type Pick } from './controls';
import { DrawPad } from './drawpad';
import { CellGrid } from './cells';
import { FlatView, GridInset } from './flat';
import { SphereView } from './sphere';

export interface ClassifyAssets {
  tables: InvariantTables;
  digits: DigitSet;
  manifest: ClassifierManifest;
  mlpBin: ArrayBuffer;
  cnnBin: ArrayBuffer;
  powerBin: ArrayBuffer;
}

const DISPLAY = { nlat: 96, nlon: 192 };
const TURN_MS = 700;
/** Radians per second while spinning. */
const SPIN_SPEED = 0.55;

const percent = (v: number): string => `${(100 * v).toFixed(1)}%`;

/** States the figure caption can put the figure in, one per numbered step. */
export type ClassifyStep = 'spin' | 'power' | 'scramble' | 'six';

/** In-plane half turn: the digits sit on the north pole, which faces the reader. */
const HALF_TURN: Mat3 = Float64Array.of(-1, 0, 0, 0, -1, 0, 0, 0, 1);
const TWO = 2;
const SIX = 6;

/**
 * Ten class probabilities as bars, with the predicted digit in large type. The top bar takes the
 * model's color; a wrong answer is said in words, since color is reserved for the method.
 */
class ProbBars {
  readonly element: HTMLElement;
  private readonly digit: HTMLElement;
  private readonly confidence: HTMLElement;
  private readonly cols: HTMLElement[] = [];
  private readonly fills: HTMLElement[] = [];

  constructor(label: string, fill: string) {
    this.digit = el('b', { text: '–' });
    this.confidence = el('span', { text: '\u00a0' });
    const bars = el('div', {
      class: 'cf-bars',
      attrs: { role: 'img', 'aria-label': `Class probabilities, ${label}` },
    });
    for (let d = 0; d < 10; d++) {
      const fill = el('i');
      const col = el(
        'div',
        { class: 'cf-col' },
        el('div', { class: 'cf-track' }, fill),
        el('span', { text: String(d) }),
      );
      this.cols.push(col);
      this.fills.push(fill);
      bars.append(col);
    }
    this.element = el(
      'div',
      { class: 'cf-readout', attrs: { style: `--fill: ${fill}` } },
      el(
        'div',
        { class: 'cf-verdict', attrs: { 'aria-live': 'off' } },
        this.digit,
        this.confidence,
      ),
      bars,
    );
  }

  /** `truth` marks the true class and colors a wrong answer; `null` when it is unknown. */
  set(probs: ArrayLike<number> | null, truth: number | null): void {
    let top = -1;
    if (probs) {
      top = 0;
      for (let d = 1; d < 10; d++) if (probs[d]! > probs[top]!) top = d;
    }
    this.cols.forEach((col, d) => {
      this.fills[d]!.style.height = probs ? `${(100 * probs[d]!).toFixed(1)}%` : '0';
      col.classList.toggle('top', d === top);
      col.classList.toggle('truth', d === truth);
    });
    const wrong = top >= 0 && truth !== null && top !== truth;
    this.element.classList.toggle('wrong', wrong);
    this.digit.textContent = top >= 0 ? String(top) : '–';
    this.confidence.textContent =
      probs && top >= 0 ? `${wrong ? 'wrong · ' : ''}${Math.round(100 * probs[top]!)}%` : '\u00a0';
  }
}

/**
 * A digit on the sphere, read live by three classifiers trained on upright digits only: the
 * bispectrum MLP, which cannot see the rotation; a CNN on the equirectangular image, which sees
 * little else; and an MLP on the power spectrum, which cannot see the rotation either, nor much
 * of the digit.
 */
export class ClassifyFigure {
  private readonly lmax: number;
  private readonly analysis: SphereGrid;
  private readonly display: SphereGrid;
  private readonly invariant: Invariant;
  private readonly mlp: BispectrumMlp;
  private readonly power: PowerMlp;
  private readonly cnn: CnnClient;
  private readonly rng = new Rng(20261109);

  private readonly sphere: SphereView;
  private readonly picker: DigitPicker;
  private readonly pad: DrawPad;
  private readonly mlpBars = new ProbBars('bispectrum MLP', METHOD_FILLS.bispectrum);
  private readonly cnnBars = new ProbBars('pixel CNN', METHOD_FILLS.cnn);
  private readonly powerBars = new ProbBars('power-spectrum MLP', METHOD_FILLS.powerSpectrum);
  private readonly descriptor: CellGrid;
  private readonly powerCells: CellGrid;
  private readonly spinToggle: ReturnType<typeof toggle>;
  private readonly scrambleToggle: ReturnType<typeof toggle>;

  private lastDigit = 0;
  private truth: number | null = null;
  /** The digit at rest, or `null` for an empty drawing. */
  private digit: Float64Array | null = null;
  /** What is on the sphere at rest: the digit, or a scramble with the same power spectrum. */
  private x: Float64Array | null = null;
  private scrambled = false;
  private scrambleSeed = 0;
  private rot: Mat3 = identity3();
  private spinning = false;
  private visible = true;
  private lastTick = 0;

  constructor(
    private readonly root: HTMLElement,
    private readonly assets: ClassifyAssets,
  ) {
    const { manifest, tables, digits } = assets;
    this.lmax = manifest.lmax;
    this.invariant = new Invariant(tables, this.lmax);
    if (this.invariant.n !== manifest.n_features) {
      throw new Error(
        `the classifier expects ${manifest.n_features} invariants, the tables give ${this.invariant.n}`,
      );
    }
    this.analysis = new SphereGrid(manifest.nlat, manifest.nlon, this.lmax);
    this.display = new SphereGrid(DISPLAY.nlat, DISPLAY.nlon, this.lmax, 'centered');
    this.mlp = new BispectrumMlp(manifest.mlp, assets.mlpBin);
    this.power = new PowerMlp(manifest.power, assets.powerBin);
    this.powerCells = new CellGrid(this.lmax + 1, {
      cols: this.lmax + 1,
      color: inkAt,
      cls: 'cf-input cf-strip',
      label: `The ${this.lmax + 1} numbers of the power spectrum as shaded cells`,
    });
    this.descriptor = new CellGrid(this.invariant.n, {
      label: `The ${this.invariant.n} numbers of the descriptor as colored cells`,
    });
    this.cnn = new CnnClient(manifest, assets.cnnBin, (logits) => {
      // An answer for a drawing that has since been cleared is dropped.
      if (this.x) this.cnnBars.set(softmax(logits), this.shownTruth);
    });

    root.classList.add('rw', 'cf');
    root.replaceChildren();

    this.picker = new DigitPicker(digits, 10, (p) => this.choose(p));
    this.pad = new DrawPad({
      size: digits.size,
      onChange: (img) => this.setImage(img),
    });
    const padBox = el(
      'div',
      { class: 'rw-padbox' },
      this.pad.element,
      button(
        'clear',
        () => {
          this.pad.clear();
          this.setImage(this.pad.image);
        },
        { kind: 'quiet' },
      ),
    );
    const image = el('canvas', {
      class: 'cf-input',
      attrs: { role: 'img', 'aria-label': 'The signal as an equirectangular image' },
    });
    const host = el('div', { class: 'rw-sphere' });
    const flat = new FlatView(DIGIT_PLANE);
    flat.canvas.title = 'The same signal, unrolled onto the plane the digit was drawn on';
    this.spinToggle = toggle('spin', false, (on) => this.setSpin(on));
    this.scrambleToggle = toggle(
      'scramble',
      false,
      (on) => this.setScrambled(on),
      'Keep the power spectrum, give every degree its own random turn',
    );
    const signalPanel = el(
      'div',
      { class: 'rw-panel' },
      el(
        'div',
        { class: 'rw-panel-head' },
        el('h4', { text: 'Signal' }),
        el('span', { class: 'rw-hint', text: 'drag to rotate' }),
      ),
      el('div', { class: 'rw-sphere-wrap' }, host, padBox),
      flat.canvas,
      el(
        'div',
        { class: 'rw-row' },
        button('random rotation', () => this.turnTo(randomRotation(this.rng)), {
          kind: 'quiet',
        }),
        button('upright', () => this.turnTo(identity3()), { kind: 'quiet' }),
        this.spinToggle.element,
        this.scrambleToggle.element,
      ),
    );

    const note = (a: SplitAccuracy): string =>
      `Test accuracy: ${percent(a.test_nr)} upright, ${percent(a.test_r)} rotated.`;
    const model = (
      title: string,
      hint: string,
      input: HTMLElement,
      caption: string,
      bars: ProbBars,
      a: SplitAccuracy,
    ): HTMLElement =>
      el(
        'div',
        { class: 'rw-panel cf-model' },
        el(
          'div',
          { class: 'rw-panel-head' },
          el('h4', { text: title }),
          el('span', { class: 'rw-hint', text: hint }),
        ),
        el(
          'div',
          { class: 'cf-body' },
          el('div', { class: 'cf-reads' }, input, el('div', { class: 'cf-note', text: caption })),
          bars.element,
        ),
        el('div', { class: 'cf-note', text: note(a) }),
      );

    root.append(
      el('div', { class: 'rw-controls' }, this.picker.element),
      el(
        'div',
        { class: 'cf-stage' },
        signalPanel,
        el(
          'div',
          { class: 'cf-models' },
          model(
            'Bispectrum + MLP',
            `L = ${this.lmax}`,
            this.descriptor.element,
            `what it reads: the ${this.invariant.n}-number descriptor`,
            this.mlpBars,
            manifest.accuracy.mlp,
          ),
          model(
            'Pixels + CNN',
            `${manifest.nlat} × ${manifest.nlon} grid`,
            image,
            'what it reads: the signal as an image',
            this.cnnBars,
            manifest.accuracy.cnn,
          ),
          model(
            'Power spectrum + MLP',
            `L = ${this.lmax}`,
            this.powerCells.element,
            `what it reads: ${this.lmax + 1} numbers, one per degree`,
            this.powerBars,
            manifest.accuracy.power,
          ),
          el('div', {
            class: 'cf-note',
            text: 'All three were trained on upright digits only. The bold digit under the bars is the true class.',
          }),
        ),
      ),
    );

    this.sphere = new SphereView(host, {
      grid: this.display,
      insets: [flat, new GridInset(image, this.display)],
      onDrag: (w) => {
        this.rot = mul3(w, this.rot);
        this.refresh();
      },
    });
    new IntersectionObserver((entries) => {
      this.visible = entries.some((e) => e.isIntersecting);
    }).observe(root);

    this.choose(0);
  }

  /** Puts the figure in the state a caption step describes. */
  apply(step: ClassifyStep): void {
    switch (step) {
      case 'spin':
        this.setScrambled(false);
        this.setSpin(true);
        break;
      case 'power':
        // The 2 that the power-spectrum model takes for a 3, at every angle.
        if (this.truth !== TWO || this.root.classList.contains('drawing')) this.choose(TWO);
        this.setScrambled(false);
        this.setSpin(true);
        break;
      case 'scramble':
        // Held still, so that the only thing that changes is the scramble.
        this.setSpin(false);
        this.setScrambled(true);
        break;
      case 'six':
        this.setSpin(false);
        this.setScrambled(false);
        if (this.truth !== SIX || this.root.classList.contains('drawing')) this.choose(SIX);
        this.turnTo(HALF_TURN);
        break;
    }
  }

  private choose(p: Pick): void {
    this.picker.setActive(p);
    this.root.classList.toggle('drawing', p === 'draw');
    this.rot = identity3();
    this.scrambled = false;
    this.scrambleToggle.set(false);
    if (p === 'draw') {
      // Start from the last digit so it can be edited; "clear" gives a blank pad.
      this.pad.load(this.assets.digits.images[this.lastDigit]!);
      this.truth = null;
      this.setImage(this.pad.image);
    } else if (p !== 'earth') {
      this.lastDigit = p;
      this.truth = this.assets.digits.labels[p]!;
      this.setImage(this.assets.digits.images[p]!);
    }
  }

  private setImage(image: Float32Array): void {
    if (!image.some((v) => v > 0)) {
      this.digit = this.x = null;
      this.sphere.setSignal(null, 1);
      this.descriptor.set(null);
      this.powerCells.set(null);
      for (const bars of [this.mlpBars, this.cnnBars, this.powerBars]) bars.set(null, null);
      return;
    }
    this.digit = digitSignal(image, this.assets.digits.size, this.analysis).x;
    this.descriptor.setReference(this.invariant.values(this.digit));
    this.powerCells.setReference(degreePower(this.digit, this.lmax));
    this.show();
  }

  /** Puts the digit, or a scramble of it, on the sphere. The color scales stay the digit's. */
  private show(): void {
    if (!this.digit) return;
    this.x = this.scrambled ? scrambleDegrees(this.digit, new Rng(this.scrambleSeed)) : this.digit;
    const peak = this.display
      .synthesize(toHalfSpectrum(this.digit, this.lmax))
      .reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    this.sphere.setSignal(this.x, peak || 1);
    this.refresh();
  }

  private setScrambled(on: boolean): void {
    this.scrambleToggle.set(on);
    if (on === this.scrambled) return;
    this.scrambled = on;
    if (on) this.scrambleSeed++;
    this.show();
  }

  /** A scramble is no digit, so no answer about it is right or wrong. */
  private get shownTruth(): number | null {
    return this.scrambled ? null : this.truth;
  }

  /** Reclassifies the signal under the current rotation. */
  private refresh(animateMs = 0): void {
    this.sphere.setOrientation(this.rot, animateMs);
    if (!this.x) return;
    const rotated = rotateDofs(this.x, this.rot);
    // Recomputed from the rotated coefficients on every frame: it is the map that does not move.
    const values = this.invariant.values(rotated);
    this.descriptor.set(values);
    this.mlpBars.set(softmax(this.mlp.logits(values)), this.shownTruth);
    const power = degreePower(rotated, this.lmax);
    this.powerCells.set(power);
    this.powerBars.set(softmax(this.power.logits(power)), this.shownTruth);
    this.cnn.classify(rotated);
  }

  private turnTo(r: Mat3): void {
    this.rot = r;
    this.refresh(TURN_MS);
  }

  private setSpin(on: boolean): void {
    this.spinToggle.set(on);
    if (on === this.spinning) return;
    this.spinning = on;
    if (!on) return;
    this.lastTick = performance.now();
    requestAnimationFrame(this.tick);
  }

  private readonly tick = (now: number): void => {
    if (!this.spinning) return;
    const dt = Math.min(0.05, (now - this.lastTick) / 1000);
    this.lastTick = now;
    if (this.visible && this.x) {
      // The axis wanders so that the digit visits every orientation.
      const a = now / 6000;
      const axis = [Math.cos(a), 0.8 * Math.sin(a), 0.6 * Math.sin(1.7 * a)];
      const norm = Math.hypot(...axis);
      const step = expMap(axis.map((v) => (v / norm) * SPIN_SPEED * dt));
      // Through a quaternion, so that thousands of products stay a rotation.
      this.rot = quatToMat(matToQuat(mul3(step, this.rot)));
      this.refresh();
    }
    requestAnimationFrame(this.tick);
  };
}

export const mount: Mount = async (root) => {
  const [{ tables, digits }, weights] = await Promise.all([sharedAssets(), classifierAssets()]);
  const figure = new ClassifyFigure(root, { tables, digits, ...weights });
  return { apply: (step) => figure.apply(step as ClassifyStep) };
};
