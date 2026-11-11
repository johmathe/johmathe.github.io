import { sharedAssets } from '../assets';
import type { DigitSet } from '../data';
import type { Mount } from '../figure';
import { bandlimit, degreePower, dim, toHalfSpectrum } from '../math/basis';
import { FullBispectrum } from '../math/bispectrum';
import { THOROUGH_CONFIG, WIDGET_CONFIG } from '../math/config';
import { Invariant, type InvariantTables } from '../math/invariant';
import type { MarchConfig } from '../math/march';
import { Rng } from '../math/rng';
import {
  identity3,
  mul3,
  randomRotation,
  rotateDofs,
  scrambleDegrees,
  transpose3,
  type Mat3,
} from '../math/rotation';
import { SphereGrid } from '../math/sht';
import { DIGIT_PLANE, digitSignal, loadEarth, type Signal } from '../signals';
import { SolverClient, type SolveEvent } from '../worker/client';
import type { BranchSnapshot } from '../worker/protocol';
import { BispectrumMap } from './bispectrum2d';
import { ASH, cssGradient, MUTED } from './colormap';
import {
  button,
  DigitPicker,
  el,
  segmented,
  select,
  slider,
  toggle,
  type Pick,
  type Segmented,
} from './controls';
import { DrawPad } from './drawpad';
import { FlatView } from './flat';
import { SphereView } from './sphere';
import {
  KIND_COLORS,
  KIND_LABELS,
  kindOf,
  Strip,
  type EntryKind,
  type StripGroup,
  type StripLayout,
} from './strips';

export interface WidgetAssets {
  tables: InvariantTables;
  digits: DigitSet;
  earth: Signal;
}

type Mode = 'fast' | 'thorough';
/**
 * `playground` keeps what it takes to try a signal of one's own: the picker, the two spheres, the
 * descriptor and the button. The rest of the widget still runs, off the page.
 */
export type Preset = 'full' | 'playground';
type InvView = 'bars' | 'map';

const CONFIGS: Record<Mode, MarchConfig> = { fast: WIDGET_CONFIG, thorough: THOROUGH_CONFIG };
const DEFAULT_LMAX = 12;
/** The solver is checked on every bundled signal up to here; the tables go higher for the hero. */
const MAX_LMAX = 12;
const MIN_LMAX = WIDGET_CONFIG.seedDegree;
/** The dataset grid of `experiments/spherical_mnist`; analysis runs on it. */
const ANALYSIS = { nlat: 64, nlon: 128 };
const DISPLAY = { nlat: 96, nlon: 192 };
const ALIGN_ANIM_MS = 700;
/**
 * A branch that matches the invariant but stays farther than this from every rotation of the
 * target would be a second orbit with the same invariant. The margin is wide on purpose: on
 * symmetric signals the invariant is flat to first order along some directions, so matching it to
 * `okTol` only pins the signal to about `sqrt(okTol)`.
 */
const COLLISION_REL = 1e-3;

interface Snapshot {
  degree: number;
  branches: BranchSnapshot[];
  best: number;
}

interface Solve {
  run: number;
  lmax: number;
  config: MarchConfig;
  /** User rotation when the solve started: snapshots are aligned to the signal at that time. */
  rotAtSolve: Mat3;
  snaps: Map<number, Snapshot>;
  phase: 'seeding' | 'marching' | 'done' | 'failed' | 'stopped' | 'error';
  seedTried: number;
  seedFound: number;
  budget: number;
  ms: number;
  featureRel: number | null;
  /** Largest aligned error among the branches that match the invariant. */
  worstAligned: number;
  message: string;
}

const RETRY_HINT =
  'Hit Reconstruct for other random starts. Very symmetric signals (a dot, a centred ring) are the hard case and may keep failing.';

/** States the figure caption can put the widget in, one per numbered step. */
export type ReconStep = 'turn' | 'scramble' | 'rebuild' | 'map';

const sci = (v: number): string => (v === 0 ? '0' : v.toExponential(1).replace('e-', 'e−'));
const plural = (n: number, noun: string): string => `${n} ${noun}${n === 1 ? '' : 's'}`;
const SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉';
/** `n` in Unicode subscript digits, for plain-text labels. */
const sub = (n: number): string => String(n).replace(/\d/g, (d) => SUBSCRIPTS[Number(d)]!);
const secs = (ms: number): string =>
  ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;

function degreeGroups(degrees: Int32Array | number[], order: Int32Array): StripGroup[] {
  const groups: StripGroup[] = [];
  order.forEach((e, p) => {
    const d = degrees[e]!;
    const g = groups.at(-1);
    if (g && g.degree === d) g.end = p + 1;
    else groups.push({ degree: d, start: p, end: p + 1 });
  });
  return groups;
}

function describeDof(i: number): string {
  const l = Math.floor(Math.sqrt(i));
  const r = i - l * l;
  if (r === 0) return `a(ℓ=${l}, m=0)`;
  const m = Math.ceil(r / 2);
  return `√2 ${r % 2 ? 'Re' : 'Im'} a(ℓ=${l}, m=${m})`;
}

export class ReconWidget {
  private readonly analysis: SphereGrid;
  private readonly display: SphereGrid;
  private readonly maxLmax: number;
  private readonly invariants = new Map<number, Invariant>();
  private readonly bispectra = new Map<number, FullBispectrum>();
  private readonly solver: SolverClient;
  private readonly rng = new Rng(20260509);

  private readonly original: SphereView;
  private readonly recon: SphereView;
  private readonly origFlat = new FlatView(DIGIT_PLANE);
  private readonly reconFlat = new FlatView(DIGIT_PLANE);
  private readonly coeffStrip: Strip;
  private readonly invStrip: Strip;
  private readonly powerStrip: Strip;
  private readonly map = new BispectrumMap();
  private readonly invHost: HTMLElement;
  private readonly barLegend: HTMLElement;
  private readonly mapLegend: HTMLElement;
  private readonly picker: DigitPicker;
  private readonly pad: DrawPad;
  private readonly padBox: HTMLElement;
  private readonly lmaxSlider: ReturnType<typeof slider>;
  private readonly degreeSlider: ReturnType<typeof slider>;
  private readonly alignToggle: ReturnType<typeof toggle>;
  private readonly scrambleToggle: ReturnType<typeof toggle>;
  private readonly runButton: HTMLButtonElement;
  private readonly replayButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly invTitle: HTMLElement;
  private readonly reconNote: HTMLElement;
  private readonly viewSwitch: Segmented<InvView>;

  private pick: Pick = 0;
  private lastDigit = 0;
  private lmax = DEFAULT_LMAX;
  private mode: Mode = 'fast';
  private scramble = false;
  private scrambleSeed = 1;
  private userRot: Mat3 = identity3();
  private alignOn = true;
  private viewRot: Mat3 = identity3();
  private invView: InvView = 'bars';
  /** Band-limit the 2D map is laid out for; it is only kept up to date while visible. */
  private mapLmax = -1;

  /** Analysis coefficients of the lifted image, up to the table band-limit. */
  private xFull: Float64Array;
  /** The digit at rest, band-limited to `lmax`. */
  private xRest: Float64Array;
  /** What is shown at rest: `xRest`, or its degree-scrambled twin. */
  private signal: Float64Array;
  /** `signal` under the user rotation: this is what the solver gets the invariant of. */
  private xCur: Float64Array;
  private vmax = 1;

  private solve: Solve | null = null;
  private runCount = 0;
  private shownDegree: number | null = null;
  private follow = true;
  private solveTimer = 0;
  private replayTimer = 0;

  constructor(
    private readonly root: HTMLElement,
    private readonly assets: WidgetAssets,
    preset: Preset = 'full',
  ) {
    const full = preset === 'full';
    this.maxLmax = Math.min(assets.tables.lmax, MAX_LMAX);
    this.analysis = new SphereGrid(ANALYSIS.nlat, ANALYSIS.nlon, this.maxLmax);
    this.display = new SphereGrid(DISPLAY.nlat, DISPLAY.nlon, this.maxLmax, 'centered');
    this.solver = new SolverClient(assets.tables);
    this.xFull = new Float64Array(dim(this.maxLmax));
    this.xRest = this.signal = this.xCur = new Float64Array(dim(this.lmax));

    root.classList.add('rw');
    root.replaceChildren();

    // Controls.
    this.picker = new DigitPicker(assets.digits, 10, (p) => this.choose(p), { earth: true });
    this.lmaxSlider = slider({
      label: 'band-limit L',
      min: MIN_LMAX,
      max: this.maxLmax,
      value: this.lmax,
      onInput: (v) => this.setLmax(v),
    });
    const modeSelect = select<Mode>(
      'solver',
      [
        { value: 'fast', label: 'fast' },
        { value: 'thorough', label: "thorough (the paper's settings)" },
      ],
      this.mode,
      (m) => {
        this.mode = m;
        this.scheduleSolve(0);
      },
    );

    // Original panel.
    this.pad = new DrawPad({
      size: assets.digits.size,
      onChange: (img, final) => {
        this.setImage(img);
        if (final) this.scheduleSolve(0);
        else this.cancelSolve('drawing…');
      },
    });
    this.padBox = el(
      'div',
      { class: 'rw-padbox' },
      this.pad.element,
      button(
        'clear',
        () => {
          this.pad.clear();
          this.setImage(this.pad.image);
          this.cancelSolve('Draw a digit on the pad.');
        },
        { kind: 'quiet' },
      ),
    );
    const origHost = el('div', { class: 'rw-sphere' });
    this.scrambleToggle = toggle(
      'same power spectrum',
      false,
      (on) => this.setScramble(on),
      'Rotate every degree by its own random rotation: same power spectrum, different signal',
    );
    const origPanel = el(
      'div',
      { class: 'rw-panel' },
      el(
        'div',
        { class: 'rw-panel-head' },
        el('h4', { text: 'Signal' }),
        el('span', { class: 'rw-hint', text: 'drag to rotate' }),
      ),
      el('div', { class: 'rw-sphere-wrap' }, origHost, this.padBox),
      this.origFlat.canvas,
      el(
        'div',
        { class: 'rw-row' },
        button('random rotation', () => this.rotateTo(randomRotation(this.rng)), {
          kind: 'quiet',
        }),
        button('reset', () => this.rotateTo(identity3()), { kind: 'quiet' }),
        full && this.scrambleToggle.element,
      ),
    );

    // Reconstruction panel.
    const reconHost = el('div', { class: 'rw-sphere' });
    this.alignToggle = toggle('align to signal', true, (on) => this.setAlign(on));
    this.reconNote = el('span', { class: 'rw-hint', text: 'from the descriptor alone' });
    const reconPanel = el(
      'div',
      { class: 'rw-panel' },
      el('div', { class: 'rw-panel-head' }, el('h4', { text: 'Reconstruction' }), this.reconNote),
      el('div', { class: 'rw-sphere-wrap' }, reconHost),
      this.reconFlat.canvas,
      el('div', { class: 'rw-row' }, this.alignToggle.element),
    );

    const colorbar = el(
      'div',
      { class: 'rw-colorbar' },
      el('span', { text: '−' }),
      el('div', { class: 'rw-colorbar-ramp', attrs: { style: `background: ${cssGradient()}` } }),
      el('span', { text: '+' }),
    );

    // March controls.
    this.runButton = button('Reconstruct', () => this.toggleRun(), { kind: 'primary' });
    this.replayButton = button('replay', () => this.replay(), { kind: 'quiet' });
    this.degreeSlider = slider({
      label: 'degree ≤',
      min: WIDGET_CONFIG.seedDegree,
      max: this.lmax,
      value: this.lmax,
      onInput: (v) => this.scrub(v),
    });
    this.degreeSlider.input.disabled = true;
    this.status = el('div', { class: 'rw-status', attrs: { 'aria-live': 'polite' } });

    // Strips.
    const coeffHost = el('div', { class: 'rw-strip' });
    this.invHost = el('div', { class: 'rw-strip' });
    const invHost = this.invHost;
    const powerHost = el('div', { class: 'rw-strip rw-strip-small' });
    this.invTitle = el('h5');
    const legend = (items: [string, string, string][]): HTMLElement =>
      el(
        'div',
        { class: 'rw-legend' },
        ...items.map(([cls, color, label]) =>
          el('span', {}, el('i', { class: cls, attrs: { style: `--c: ${color}` } }), label),
        ),
      );
    this.barLegend = legend(
      (['triple', 'energy'] as EntryKind[]).map((k) => ['bar', KIND_COLORS[k], KIND_LABELS[k]]),
    );
    this.mapLegend = legend([
      ['cell kept', '', 'kept in the descriptor'],
      ['cell', '', 'the rest'],
      ['pending', '', 'not rebuilt yet'],
    ]);
    this.mapLegend.hidden = true;
    this.map.element.hidden = true;
    this.viewSwitch = segmented<InvView>(
      'Descriptor view',
      [
        { value: 'bars', label: 'bars', title: 'Descriptor as bars' },
        {
          value: 'map',
          label: '2D',
          title: 'Every triple product, one (ℓ₁, ℓ₂) map per degree ℓ',
        },
      ],
      this.invView,
      (v) => this.setInvView(v),
    );
    const marks = legend([
      ['bar', ASH, 'signal'],
      ['line', '', 'signal at rest'],
      ['dot', '', 'reconstruction'],
    ]);

    root.append(
      el(
        'div',
        { class: 'rw-controls' },
        this.picker.element,
        full &&
          el('div', { class: 'rw-controls-right' }, this.lmaxSlider.element, modeSelect.element),
      ),
      el('div', { class: 'rw-stage' }, origPanel, reconPanel),
      colorbar,
      el(
        'div',
        { class: 'rw-march' },
        this.runButton,
        full && this.degreeSlider.element,
        full && this.replayButton,
        this.status,
      ),
      el(
        'div',
        { class: 'rw-strips' },
        el(
          'div',
          { class: 'rw-strip-head' },
          this.invTitle,
          el(
            'div',
            { class: 'rw-strip-tools' },
            this.barLegend,
            full ? this.mapLegend : legend([['dot', '', 'reconstruction']]),
            full && this.viewSwitch.element,
          ),
        ),
        invHost,
        full && this.map.element,
        full &&
          el(
            'div',
            { class: 'rw-strip-head' },
            el('h5', { text: 'Coefficients, grouped by degree ℓ' }),
            marks,
          ),
        full && coeffHost,
        full &&
          el(
            'div',
            { class: 'rw-strip-head' },
            el('h5', {}, 'Power spectrum ‖F', el('sub', { text: 'ℓ' }), '‖² (log scale)'),
          ),
        full && powerHost,
      ),
    );

    this.original = new SphereView(origHost, {
      grid: this.display,
      insets: [this.origFlat],
      onDrag: (w) => {
        this.userRot = mul3(w, this.userRot);
        this.refreshRotation();
      },
    });
    this.recon = new SphereView(reconHost, {
      grid: this.display,
      insets: [this.reconFlat],
      onDrag: (w) => {
        if (this.alignOn) {
          this.viewRot = this.reconRot();
          this.alignOn = false;
          this.alignToggle.set(false);
        }
        this.viewRot = mul3(w, this.viewRot);
        this.orientRecon();
      },
    });
    this.invStrip = new Strip(invHost, 132);
    this.coeffStrip = new Strip(coeffHost, 104);
    this.powerStrip = new Strip(powerHost, 72);

    this.updateLayouts();
    this.choose(full ? 0 : 'draw');
  }

  /** Puts the widget in the state a caption step describes. */
  apply(step: ReconStep): void {
    switch (step) {
      case 'turn':
        this.setScramble(false);
        this.rotateTo(randomRotation(this.rng));
        break;
      case 'scramble':
        this.setScramble(true);
        break;
      case 'rebuild':
        this.setScramble(false);
        this.setAlign(false);
        this.alignToggle.set(false);
        this.scheduleSolve(0);
        break;
      case 'map':
        this.viewSwitch.set('map');
        this.setInvView('map');
        if (!this.solve) this.scheduleSolve(0);
        break;
    }
  }

  // ---------------------------------------------------------------- signal

  /** Shows the digit (`off`) or a fresh same-power-spectrum twin of it (`on`). */
  private setScramble(on: boolean): void {
    if (!on && !this.scramble) return;
    this.scramble = on;
    this.scrambleToggle.set(on);
    if (on) this.scrambleSeed++;
    this.refreshSignal();
    this.scheduleSolve(0);
  }

  private invariant(lmax: number): Invariant {
    let inv = this.invariants.get(lmax);
    if (!inv) {
      inv = new Invariant(this.assets.tables, lmax);
      this.invariants.set(lmax, inv);
    }
    return inv;
  }

  private bispectrum(lmax: number): FullBispectrum {
    let full = this.bispectra.get(lmax);
    if (!full) {
      full = new FullBispectrum(lmax);
      this.bispectra.set(lmax, full);
    }
    return full;
  }

  private choose(p: Pick): void {
    this.pick = p;
    this.picker.setActive(p);
    this.root.classList.toggle('drawing', p === 'draw');
    this.userRot = identity3();
    if (p === 'earth') {
      this.setSignal(this.assets.earth);
    } else if (p === 'draw') {
      // Start from the last digit so it can be edited; "clear" gives a blank pad.
      this.pad.load(this.assets.digits.images[this.lastDigit]!);
      this.setImage(this.pad.image);
    } else {
      this.lastDigit = p;
      this.setImage(this.assets.digits.images[p]!);
    }
    this.scheduleSolve(0);
  }

  private setImage(image: Float32Array): void {
    this.setSignal(digitSignal(image, this.assets.digits.size, this.analysis));
  }

  private setSignal(signal: Signal): void {
    this.xFull = bandlimit(signal.x, this.maxLmax);
    this.origFlat.setProjection(signal.flat);
    this.reconFlat.setProjection(signal.flat);
    this.refreshSignal();
  }

  private setLmax(lmax: number): void {
    if (lmax === this.lmax) return;
    this.lmax = lmax;
    this.updateLayouts();
    this.refreshSignal();
    this.scheduleSolve(250);
  }

  private refreshSignal(): void {
    this.xRest = bandlimit(this.xFull, this.lmax);
    this.signal = this.scramble
      ? scrambleDegrees(this.xRest, new Rng(this.scrambleSeed))
      : this.xRest;
    const peak = (x: Float64Array): number =>
      this.display
        .synthesize(toHalfSpectrum(x, this.lmax))
        .reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    this.vmax = Math.max(peak(this.xRest), this.scramble ? peak(this.signal) : 0) || 1;
    this.original.setSignal(this.signal, this.vmax);
    const inv = this.invariant(this.lmax);
    this.invStrip.setValues({ outline: inv.values(this.xRest) }, true);
    this.coeffStrip.setValues({ outline: this.xRest }, true);
    this.powerStrip.setValues({ outline: degreePower(this.xRest, this.lmax) }, true);
    this.syncMap({ scale: true });
    this.refreshRotation();
  }

  private refreshRotation(animateMs = 0): void {
    this.xCur = rotateDofs(this.signal, this.userRot);
    this.original.setOrientation(this.userRot, animateMs);
    const inv = this.invariant(this.lmax);
    this.invStrip.setValues({ fill: inv.values(this.xCur) });
    this.coeffStrip.setValues({ fill: this.xCur });
    this.powerStrip.setValues({ fill: degreePower(this.xCur, this.lmax) });
    this.syncMap({ signal: true });
    if (this.alignOn) this.orientRecon(animateMs);
  }

  private rotateTo(r: Mat3): void {
    this.userRot = r;
    this.refreshRotation(ALIGN_ANIM_MS);
  }

  private updateLayouts(): void {
    const inv = this.invariant(this.lmax);
    const order = Int32Array.from({ length: inv.n }, (_, i) => i).sort(
      (a, b) =>
        inv.rowDegree[a]! - inv.rowDegree[b]! || inv.isPower[a]! - inv.isPower[b]! || a - b,
    );
    const invLayout: StripLayout = {
      order,
      groups: degreeGroups(inv.rowDegree, order),
      color: (e) => KIND_COLORS[kindOf(inv.tags[e]!)],
      describe: (e) => {
        const [a, b, c] = inv.triples[e]!;
        return inv.isPower[e]
          ? `product energy: the degree-${c} part of degrees ${a} × ${b}`
          : `triple product of degrees ${a}, ${b}, ${c}${inv.imagPart[e] ? ' (imaginary part)' : ''}`;
      },
      scale: 'asinh',
    };
    this.invStrip.setLayout(invLayout);
    this.renderInvTitle();

    const n = dim(this.lmax);
    const coeffOrder = Int32Array.from({ length: n }, (_, i) => i);
    const dofDegree = Int32Array.from(coeffOrder, (i) => Math.floor(Math.sqrt(i)));
    this.coeffStrip.setLayout({
      order: coeffOrder,
      groups: degreeGroups(dofDegree, coeffOrder),
      color: () => ASH,
      describe: describeDof,
      scale: 'asinh',
    });

    const powerOrder = Int32Array.from({ length: this.lmax + 1 }, (_, i) => i);
    this.powerStrip.setLayout({
      order: powerOrder,
      groups: degreeGroups(powerOrder, powerOrder),
      color: () => MUTED,
      describe: (l) => `‖F${sub(l)}‖²`,
      scale: 'log',
    });
    this.degreeSlider.set(Math.min(this.shownDegree ?? this.lmax, this.lmax), this.lmax);
  }

  // ---------------------------------------------------------------- invariant views

  private renderInvTitle(): void {
    const inv = this.invariant(this.lmax);
    if (this.invView === 'bars') {
      this.invTitle.textContent = `Descriptor: ${inv.n} numbers for ${dim(this.lmax)} coefficients, grouped by the degree they pin down`;
      return;
    }
    const powers = inv.isPower.reduce((s, p) => s + p, 0);
    const full = this.bispectrum(this.lmax);
    this.invTitle.textContent = `All ${full.n} triple products, one panel per degree ℓ (ℓ₁ down, ℓ₂ across). The descriptor keeps the ${inv.n - powers} outlined, plus ${powers} product energies`;
  }

  private setInvView(view: InvView): void {
    this.invView = view;
    const map = view === 'map';
    this.invHost.hidden = this.barLegend.hidden = map;
    this.map.element.hidden = this.mapLegend.hidden = !map;
    this.renderInvTitle();
    this.syncMap({ scale: true, signal: true, recon: true });
  }

  /** Lays the 2D map out for the current band-limit: which entries are selective, and labels. */
  private layoutMap(): void {
    const inv = this.invariant(this.lmax);
    const full = this.bispectrum(this.lmax);
    const index = new Map(full.triples.map((t, e) => [t.join(','), e]));
    const kept = new Set<number>();
    const power = new Set<number>();
    inv.triples.forEach((t, i) => {
      const e = index.get(t.join(','));
      if (e === undefined) return;
      if (inv.isPower[i]) power.add(e);
      else kept.add(e);
    });
    this.map.setLayout({
      full,
      selective: Uint8Array.from({ length: full.n }, (_, e) => (kept.has(e) ? 1 : 0)),
      describe: (e) => {
        const [a, b, c] = full.triples[e]!;
        const parts = [
          `triple product of degrees ${a}, ${b}, ${c}${full.imagPart[e] ? ' (imaginary part)' : ''}`,
          kept.has(e) ? 'kept in the descriptor' : 'not in the descriptor',
        ];
        if (power.has(e)) parts.push(`its product energy is kept`);
        if (full.vanishes[e]) parts.push('zero on every real signal');
        return parts.join('  ·  ');
      },
    });
    this.mapLmax = this.lmax;
  }

  /** Pushes the requested parts of the state to the 2D map, if it is the view on screen. */
  private syncMap(what: { scale?: boolean; signal?: boolean; recon?: boolean }): void {
    if (this.invView !== 'map') return;
    if (this.mapLmax !== this.lmax) this.layoutMap();
    const full = this.bispectrum(this.lmax);
    if (what.scale) this.map.setReference(full.values(this.signal));
    if (what.signal && this.xCur.length === dim(this.lmax)) {
      this.map.setSignal(full.values(this.xCur));
    }
    if (what.recon) {
      const b = this.current();
      const ok = b && this.solve?.lmax === this.lmax && b.x.length === dim(this.lmax);
      this.map.setRecon(ok ? full.values(b.x) : null, this.shownDegree ?? -1);
    }
  }

  // ---------------------------------------------------------------- solving

  private scheduleSolve(delayMs: number): void {
    clearTimeout(this.solveTimer);
    this.cancelSolve('');
    this.solveTimer = window.setTimeout(() => void this.startSolve(), delayMs);
  }

  private cancelSolve(message: string): void {
    clearTimeout(this.replayTimer);
    if (this.solver.busy) this.solver.cancel();
    this.solve = null;
    this.shownDegree = null;
    this.follow = true;
    this.degreeSlider.input.disabled = true;
    this.replayButton.disabled = true;
    this.runButton.textContent = 'Reconstruct';
    this.reconNote.textContent = 'from the descriptor alone';
    this.showSnapshot();
    this.status.textContent = message;
  }

  private toggleRun(): void {
    if (this.solver.busy) {
      this.solver.cancel();
      if (this.solve) this.solve.phase = 'stopped';
      this.follow = false;
      this.runButton.textContent = 'Reconstruct';
      this.syncScrubber();
      this.renderStatus();
      return;
    }
    this.scheduleSolve(0);
  }

  private async startSolve(): Promise<void> {
    if (this.pick === 'draw' && this.pad.empty) {
      this.cancelSolve('Draw a digit on the pad.');
      return;
    }
    const lmax = this.lmax;
    const inv = this.invariant(lmax);
    const config = CONFIGS[this.mode];
    const run = ++this.runCount;
    const solve: Solve = {
      run,
      lmax,
      config,
      rotAtSolve: Float64Array.from(this.userRot),
      snaps: new Map(),
      phase: 'seeding',
      seedTried: 0,
      seedFound: 0,
      budget: config.nSeedInits,
      ms: 0,
      featureRel: null,
      worstAligned: 0,
      message: '',
    };
    this.solve = solve;
    this.runButton.textContent = 'Stop';
    this.degreeSlider.set(config.seedDegree, lmax);
    this.renderStatus();
    await this.solver.solve(
      {
        lmax,
        target: inv.values(this.xCur),
        alignTo: Float64Array.from(this.xCur),
        seed: 7919 * run + 17,
        config,
      },
      (e) => this.onSolveEvent(solve, e),
    );
  }

  private onSolveEvent(solve: Solve, e: SolveEvent): void {
    if (this.solve !== solve) return;
    solve.ms = e.type === 'error' ? solve.ms : e.ms;
    switch (e.type) {
      case 'seed':
        solve.seedTried = e.tried;
        solve.seedFound = e.found;
        break;
      case 'degree':
        solve.phase = 'marching';
        solve.snaps.set(e.degree, { degree: e.degree, branches: e.branches, best: e.best });
        if (e.branches.length && this.follow) this.showDegree(e.degree);
        break;
      case 'done': {
        const ok = e.branches.filter((b) => b.featureRel <= solve.config.okTol);
        solve.featureRel = e.branches[e.best]?.featureRel ?? null;
        solve.worstAligned = Math.max(0, ...ok.map((b) => b.alignedRel));
        solve.phase = ok.length ? 'done' : 'failed';
        if (e.branches.length) {
          solve.snaps.set(solve.lmax, { degree: solve.lmax, branches: e.branches, best: e.best });
          if (this.follow) this.showDegree(solve.lmax);
        }
        this.runButton.textContent = 'Reconstruct';
        break;
      }
      case 'error':
        solve.phase = 'error';
        solve.message = e.message;
        this.runButton.textContent = 'Reconstruct';
        break;
    }
    this.syncScrubber();
    this.renderStatus();
  }

  /** The scrubber needs two solved degrees to move between; replay waits for the solve to end. */
  private syncScrubber(): void {
    const several = this.solvedDegrees().length > 1;
    const running = this.solve?.phase === 'seeding' || this.solve?.phase === 'marching';
    this.degreeSlider.input.disabled = !several;
    this.replayButton.disabled = !several || running;
  }

  private renderStatus(): void {
    const s = this.solve;
    if (!s) return;
    const { seedDegree: seedDeg, nSeedBranches } = s.config;
    const last = Math.max(-1, ...s.snaps.keys());
    const shown = this.shownDegree !== null ? s.snaps.get(this.shownDegree) : undefined;
    const b = shown?.branches[shown.best];
    const err = b ? ` Error after alignment (ℓ ≤ ${shown!.degree}): ${sci(b.alignedRel)}.` : '';
    let text = '';
    switch (s.phase) {
      case 'seeding':
        text = `Guessing degrees ℓ ≤ ${seedDeg} from random starts: ${s.seedTried} tried, ${s.seedFound} of ${nSeedBranches} found · ${secs(s.ms)}`;
        break;
      case 'marching': {
        const snap = s.snaps.get(last);
        text = `Solved degree ${last} of ${s.lmax} · ${plural(snap?.branches.length ?? 0, 'candidate')} left · ${secs(s.ms)}.${err}`;
        if (snap && snap.branches.length === 0)
          text = `No candidate survived degree ${last}. ${RETRY_HINT}`;
        break;
      }
      case 'done':
        text =
          s.worstAligned > COLLISION_REL
            ? `A candidate matches the descriptor to ${sci(s.featureRel ?? NaN)} but stays ${sci(s.worstAligned)} away from every rotation of the signal. That would be a second signal with the same descriptor. If it reproduces, please tell us.`
            : `Recovered ${dim(s.lmax)} coefficients from the ${this.invariant(s.lmax).n} numbers of the descriptor in ${secs(s.ms)} (${plural(s.seedTried, 'random start')}). Descriptor mismatch ${sci(s.featureRel ?? NaN)}.${err}`;
        break;
      case 'stopped':
        text = last < 0 ? 'Stopped.' : `Stopped after degree ${last} of ${s.lmax}.${err}`;
        break;
      case 'failed':
        text =
          s.seedFound === 0
            ? `No consistent guess in ${s.seedTried} random starts. ${RETRY_HINT}`
            : `No candidate survived degree ${last}. ${RETRY_HINT}`;
        break;
      case 'error':
        text = s.message;
        break;
    }
    this.status.textContent = text;
    this.root.classList.toggle('solving', s.phase === 'seeding' || s.phase === 'marching');
  }

  // ---------------------------------------------------------------- display

  /** Degrees with at least one live branch, ascending. */
  private solvedDegrees(): number[] {
    const s = this.solve;
    if (!s) return [];
    return [...s.snaps.values()]
      .filter((snap) => snap.branches.length)
      .map((snap) => snap.degree)
      .sort((a, b) => a - b);
  }

  private scrub(degree: number): void {
    const solved = this.solvedDegrees();
    const below = solved.filter((d) => d <= degree);
    if (!below.length) return;
    const d = below.at(-1)!;
    const running = this.solve?.phase === 'seeding' || this.solve?.phase === 'marching';
    this.follow = running && d === solved.at(-1);
    clearTimeout(this.replayTimer);
    this.showDegree(d);
    this.renderStatus();
  }

  private showDegree(d: number): void {
    this.shownDegree = d;
    this.degreeSlider.set(d);
    this.showSnapshot();
  }

  private replay(): void {
    const s = this.solve;
    if (!s) return;
    const degrees = this.solvedDegrees();
    clearTimeout(this.replayTimer);
    this.follow = false;
    let i = 0;
    const step = (): void => {
      if (this.solve !== s || i >= degrees.length) return;
      this.showDegree(degrees[i++]!);
      this.renderStatus();
      this.replayTimer = window.setTimeout(step, 450);
    };
    step();
  }

  private current(): BranchSnapshot | null {
    const s = this.solve;
    if (!s || this.shownDegree === null) return null;
    const snap = s.snaps.get(this.shownDegree);
    return snap?.branches[snap.best] ?? null;
  }

  /** Rotation that displays the current branch on top of the signal as it is now oriented. */
  private reconRot(): Mat3 {
    const b = this.current();
    if (!b || !this.solve) return this.viewRot;
    return mul3(this.userRot, mul3(transpose3(this.solve.rotAtSolve), b.rot));
  }

  private setAlign(on: boolean): void {
    if (on === this.alignOn) return;
    this.alignOn = on;
    if (!on) this.viewRot = identity3();
    this.orientRecon(ALIGN_ANIM_MS);
  }

  /** New texture for the shown snapshot; strips follow. */
  private showSnapshot(): void {
    const b = this.current();
    if (!b) {
      this.recon.setSignal(null, this.vmax);
      this.invStrip.setValues({ dots: null, highlight: null });
      this.coeffStrip.setValues({ dots: null, highlight: null });
      this.powerStrip.setValues({ dots: null, highlight: null });
      this.syncMap({ recon: true });
      return;
    }
    this.recon.setSignal(b.x, this.vmax);
    const d = this.shownDegree!;
    const lmax = this.solve!.lmax;
    this.reconNote.textContent = d < lmax ? `degrees ℓ ≤ ${d}` : 'from the descriptor alone';
    // Mark the degree just solved, or the one scrubbed to; nothing once the march is complete.
    const highlight = d < lmax || this.solve!.phase === 'marching' ? d : null;
    this.invStrip.setValues({ dots: this.invariant(lmax).values(b.x), dotsUpTo: d, highlight });
    this.powerStrip.setValues({ dots: degreePower(b.x, lmax), dotsUpTo: d, highlight });
    this.coeffStrip.setValues({ highlight, dotsUpTo: d });
    this.syncMap({ recon: true });
    this.orientRecon();
  }

  /** Orientation of the reconstruction (aligned or free) and its coefficient dots. */
  private orientRecon(animateMs = 0): void {
    const b = this.current();
    const rot = this.alignOn ? this.reconRot() : this.viewRot;
    this.recon.setOrientation(rot, animateMs);
    this.coeffStrip.setValues({ dots: b ? rotateDofs(b.x, rot) : null });
  }
}

async function build(root: HTMLElement, preset: Preset): Promise<ReconWidget> {
  const [{ tables, digits }, earth] = await Promise.all([sharedAssets(), loadEarth()]);
  return new ReconWidget(root, { tables, digits, earth }, preset);
}

export const mount: Mount = async (root) => {
  const widget = await build(root, 'full');
  return { apply: (step) => widget.apply(step as ReconStep) };
};

export const mountPlayground: Mount = async (root) => {
  await build(root, 'playground');
  return {};
};
