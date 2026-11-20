import { sharedAssets } from '../assets';
import type { DigitSet } from '../data';
import type { Mount } from '../figure';
import { bandlimit, degreePower, dim, toHalfSpectrum } from '../math/basis';
import { WIDGET_CONFIG } from '../math/config';
import { Invariant, type InvariantTables } from '../math/invariant';
import { Rng } from '../math/rng';
import {
  expMap,
  identity3,
  matToQuat,
  mul3,
  quatToMat,
  rotateDofs,
  scrambleDegrees,
  slerp,
  type Mat3,
} from '../math/rotation';
import { SphereGrid } from '../math/sht';
import { digitSignal, EARTH_VIEW, facing, loadEarth, worldMap, type Signal } from '../signals';
import { SolverClient, type SolveEvent } from '../worker/client';
import type { FinalBranch } from '../worker/protocol';
import { reducedMotion } from './canvas';
import { greyAt, inkAt } from './colormap';
import { button, DigitPicker, el, toggle, type Pick } from './controls';
import { CellGrid } from './cells';
import { DrawPad } from './drawpad';
import { FlatView } from './flat';
import { SphereView } from './sphere';

export interface HeroAssets {
  tables: InvariantTables;
  earth: Signal;
  /** With `data-picker` on the root: digits to pick from, beside the Earth and a draw pad. */
  digits?: DigitSet;
}

const DISPLAY = { nlat: 96, nlon: 192 };
/** The dataset grid of `experiments/spherical_mnist`, which digits and drawings are lifted on. */
const ANALYSIS = { nlat: 64, nlon: 128 };
/** Digits and drawings run at the widget's band-limit, where the solver is checked on them. */
const DIGIT_L = 12;
/** The world map under the globe in the talk deck: one pixel per degree. */
const MAP = { width: 360, height: 180 };
/** Radians per second. */
const SPIN_SPEED = 0.4;
const LINE_MS = 1100;
/** The north pole of the Earth at rest. */
const POLE = ((p: Mat3): number[] => [p[2]!, p[5]!, p[8]!])(
  facing(EARTH_VIEW.lat, EARTH_VIEW.lon),
);

const ease = (t: number): number => t * t * (3 - 2 * t);
const same = (a: Mat3, b: Mat3): boolean => a.every((v, i) => v === b[i]);
const sci = (v: number): string => v.toExponential(1).replace('e-', 'e−');

/** States the talk deck can put the figure in. */
export type HeroStep = 'north' | 'scramble' | 'rebuild' | 'line' | 'free';

interface Rebuilt {
  x: Float64Array;
  /** Rotation that takes the rebuilt signal onto the Earth at rest. */
  toEarth: Mat3;
  /** Relative error of the aligned, band-limited coefficients against the Earth's. */
  rel: number;
  seconds: number;
}

/** The accepted candidates of the last solve, one per distinct signal, in rank order. */
interface Candidates {
  branches: FinalBranch[];
  reps: number[];
  seconds: number;
}

/**
 * The opening figure. The Earth turns; its coefficients move, and two lists of invariant
 * features don't. Each list is put to the same test: redraw the globe from it.
 *
 * With `data-deck` on its root (the talk deck), each list is named and the globe has its world map
 * beneath it.
 */
export class HeroFigure {
  private lmax: number;
  private invariant: Invariant;
  private x: Float64Array;
  private vmax = 1;
  private readonly tables: InvariantTables;
  private readonly invariants = new Map<number, Invariant>();
  private readonly grid: SphereGrid;
  private readonly solver: SolverClient;
  private readonly rng = new Rng(20261109);
  /**
   * Seeds the random rotation of each degree in the power spectrum's attempt. Only "another"
   * draws a new one, so that a drawing in progress moves the attempt instead of rerolling it.
   */
  private scrambleSeed = 20261110;

  private readonly earth: SphereView;
  private readonly mush: SphereView;
  private readonly redrawn: SphereView;
  private cells!: { all: CellGrid; power: CellGrid; descriptor: CellGrid };
  private readonly cellHosts = { all: el('div'), power: el('div'), descriptor: el('div') };
  private readonly counts = { all: el('span'), power: el('span'), descriptor: el('span') };
  private readonly allWhat = el('p', { class: 'hr-what' });
  private readonly descriptorWhat = el('p', { class: 'hr-what' });
  private readonly title = el('h4', { text: 'Earth, roughly' });
  private readonly root: HTMLElement;
  private readonly earthSignal: Signal;
  private readonly digits: DigitSet | null;
  private analysis: SphereGrid | null = null;
  private pad: DrawPad | null = null;
  private picker: DigitPicker | null = null;
  private pick: Pick = 'earth';
  private lastDigit = 0;
  private readonly spinToggle: ReturnType<typeof toggle>;
  private readonly lineButton: HTMLButtonElement;
  private readonly againButton: HTMLButtonElement;
  private readonly candidateButton: HTMLButtonElement;
  private readonly status: HTMLElement;

  private rot: Mat3 = identity3();
  private spinning = !reducedMotion();
  private dragging = false;
  private visible = true;
  private dirty = true;
  private lastTick = 0;

  private runs = 0;
  private rebuilt: Rebuilt | null = null;
  private candidates: Candidates | null = null;
  private candidate = 0;
  private lined = false;
  /** A `line` step that came while the solver was still running. */
  private lineWhenDone = false;
  /** The redrawn globe's own pose while it is not lined up with the Earth. */
  private own: Mat3 = identity3();
  private shown: Mat3 = identity3();
  private lineFrom: Mat3 = identity3();
  private lineT0 = -Infinity;

  constructor(root: HTMLElement, assets: HeroAssets) {
    const { earth, tables } = assets;
    this.root = root;
    this.tables = tables;
    this.earthSignal = earth;
    this.digits = root.dataset.picker !== undefined ? (assets.digits ?? null) : null;
    this.x = earth.x;
    this.lmax = Math.round(Math.sqrt(this.x.length)) - 1;
    this.invariant = this.invariantAt(this.lmax);
    this.solver = new SolverClient(tables);
    this.grid = new SphereGrid(DISPLAY.nlat, DISPLAY.nlon, this.lmax, 'centered');
    const grid = this.grid;
    const n = dim(this.lmax);

    const deck = root.dataset.deck !== undefined;
    root.classList.add('rw', 'hr');
    root.replaceChildren();

    const earthHost = el('div', { class: 'rw-sphere' });
    const mushHost = el('div', { class: 'rw-sphere' });
    const redrawnHost = el('div', { class: 'rw-sphere' });
    this.spinToggle = toggle('spin', this.spinning, (on) => {
      this.spinning = on;
    });
    this.lineButton = button('line it up', () => this.setLined(!this.lined), { kind: 'quiet' });
    this.againButton = button('again', () => void this.rebuild(), {
      kind: 'quiet',
      title: 'Rebuild from other random starts',
    });
    this.candidateButton = button('candidate', () => this.showCandidate(this.candidate + 1), {
      kind: 'quiet',
      title: 'Another signal with the same features',
    });
    this.candidateButton.hidden = true;
    this.status = el('div', { class: 'hr-status', attrs: { 'aria-live': 'polite' } });

    const lead = (
      name: string,
      count: HTMLElement,
      noun: string,
      what: HTMLElement,
    ): HTMLElement =>
      el(
        'div',
        { class: 'hr-lead' },
        ...(deck ? [el('p', { class: 'hr-name', text: name })] : []),
        el('div', { class: 'hr-head' }, count, el('span', { class: 'hr-noun', text: noun })),
        what,
      );
    for (const c of Object.values(this.counts)) c.className = 'hr-count';
    for (const h of Object.values(this.cellHosts)) h.className = 'hr-cell-host';
    const redraw = (host: HTMLElement, title: string, ...controls: HTMLElement[]): HTMLElement =>
      el(
        'div',
        { class: 'hr-redraw' },
        host,
        el('div', { text: title }),
        el('div', { class: 'rw-row' }, ...controls),
      );

    const padBox = this.digits ? this.makePad(this.digits) : null;
    if (this.digits) {
      this.picker = new DigitPicker(this.digits, 10, (p) => this.choose(p), { earth: true });
      this.picker.setActive(this.pick);
      root.append(el('div', { class: 'rw-controls hr-picker' }, this.picker.element));
    }

    root.append(
      el(
        'div',
        { class: 'hr-stage' },
        el(
          'div',
          { class: 'rw-panel hr-earth' },
          el(
            'div',
            { class: 'rw-panel-head' },
            this.title,
            el('span', { class: 'rw-hint', text: 'drag to turn it' }),
          ),
          padBox ? el('div', { class: 'rw-sphere-wrap' }, earthHost, padBox) : earthHost,
          ...(deck ? [this.map(n)] : []),
          el(
            'div',
            { class: 'rw-row' },
            this.spinToggle.element,
            button('north up', () => this.northUp(), { kind: 'quiet' }),
          ),
          el(
            'div',
            { class: 'hr-list' },
            lead('Coefficients', this.counts.all, 'coefficients', this.allWhat),
            this.cellHosts.all,
          ),
        ),
        el(
          'div',
          { class: 'hr-rows' },
          el(
            'div',
            { class: 'rw-panel hr-row' },
            el(
              'div',
              {},
              lead(
                'Power spectrum',
                this.counts.power,
                'features',
                el('p', {
                  class: 'hr-what',
                  text: 'The power spectrum, invariant to rotation: how much detail the signal has at each scale.',
                }),
              ),
              this.cellHosts.power,
            ),
            redraw(
              mushHost,
              'reconstruction attempt from the power spectrum',
              button('another', () => this.scramble(), { kind: 'quiet' }),
            ),
          ),
          el(
            'div',
            { class: 'rw-panel hr-row ours' },
            el(
              'div',
              {},
              lead(
                'Selective bispectrum',
                this.counts.descriptor,
                'features',
                this.descriptorWhat,
              ),
              this.cellHosts.descriptor,
              this.status,
            ),
            redraw(
              redrawnHost,
              'reconstruction from the bispectrum',
              this.lineButton,
              this.againButton,
              this.candidateButton,
            ),
          ),
        ),
      ),
    );

    const turn = (w: Mat3): void => {
      this.dragging = true;
      this.rot = mul3(w, this.rot);
      this.dirty = true;
    };
    const release = (): void => {
      this.dragging = false;
    };
    this.earth = new SphereView(earthHost, { grid, onDrag: turn, onDragEnd: release });
    this.mush = new SphereView(mushHost, { grid, onDrag: turn, onDragEnd: release });
    this.redrawn = new SphereView(redrawnHost, {
      grid,
      onDrag: (w) => {
        // Lined up, the two globes are one object; otherwise this one turns by itself.
        if (this.lined) turn(w);
        else this.own = mul3(w, this.own);
      },
      onDragEnd: release,
    });
    this.load(earth);
    new IntersectionObserver((entries) => {
      this.visible = entries.some((e) => e.isIntersecting);
    }).observe(root);

    void this.rebuild();
    requestAnimationFrame(this.tick);
  }

  private invariantAt(lmax: number): Invariant {
    let inv = this.invariants.get(lmax);
    if (!inv) {
      inv = new Invariant(this.tables, lmax);
      this.invariants.set(lmax, inv);
    }
    return inv;
  }

  /** Shows `signal` at rest, with lists, color scales and texts for its band-limit. */
  private load(signal: Signal, x: Float64Array = signal.x): void {
    this.x = x;
    this.lmax = Math.round(Math.sqrt(x.length)) - 1;
    this.invariant = this.invariantAt(this.lmax);
    this.vmax =
      this.grid
        .synthesize(toHalfSpectrum(x, this.lmax))
        .reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;

    const n = dim(this.lmax);
    // The same cells per row in every list, so that a list's area is its length.
    const cols = 2 * (this.lmax + 1);
    const list = (count: number, color?: (t: number) => string): CellGrid => {
      const f = new CellGrid(count, {
        cols,
        cls: 'hr-cells',
        label: `${count} numbers as colored cells`,
        ...(color ? { color } : {}),
      });
      f.element.style.width = `${(100 * Math.min(count, cols)) / cols}%`;
      return f;
    };
    this.cells = {
      all: list(n, greyAt),
      power: list(this.lmax + 1, inkAt),
      descriptor: list(this.invariant.n),
    };
    this.cells.all.setReference(x);
    this.cells.power.setReference(degreePower(x, this.lmax));
    this.cells.descriptor.setReference(this.invariant.values(x));
    // Painted before they replace the old canvases, which a pen stroke does many times a second.
    this.refresh();
    for (const k of ['all', 'power', 'descriptor'] as const) {
      this.cellHosts[k].replaceChildren(this.cells[k].element);
    }
    this.counts.all.textContent = String(n);
    this.counts.power.textContent = String(this.lmax + 1);
    this.counts.descriptor.textContent = String(this.invariant.n);
    const earth = signal.kind === 'earth';
    this.allWhat.textContent = `${earth ? 'The globe' : 'The digit'} in spherical harmonics. Turn it and they change.`;
    this.descriptorWhat.textContent = `A selective bispectrum, also invariant: sums of products of three or four of the ${n} coefficients.`;
    this.title.textContent = earth
      ? 'Earth, roughly'
      : this.pick === 'draw'
        ? 'Your drawing'
        : 'Digit';

    this.earth.setSignal(x, this.vmax);
    this.showScramble();
    this.dirty = true;
  }

  /** The draw pad, over the sphere while "draw" is picked; it starts from the last digit. */
  private makePad(digits: DigitSet): HTMLElement {
    const pad = new DrawPad({
      size: digits.size,
      onChange: (img, final) => {
        this.loadImage(img);
        if (final) {
          this.lineWhenDone = true;
          void this.rebuild();
        } else this.stopSolve('Drawing…');
      },
    });
    this.pad = pad;
    return el(
      'div',
      { class: 'rw-padbox' },
      pad.element,
      button(
        'clear',
        () => {
          pad.clear();
          this.loadImage(pad.image);
          this.stopSolve('The pad is empty.');
        },
        { kind: 'quiet' },
      ),
    );
  }

  private loadImage(image: Float32Array): void {
    const digits = this.digits!;
    this.analysis ??= new SphereGrid(ANALYSIS.nlat, ANALYSIS.nlon, DIGIT_L);
    const signal = digitSignal(image, digits.size, this.analysis);
    this.load(signal, bandlimit(signal.x, DIGIT_L));
  }

  private choose(p: Pick): void {
    this.pick = p;
    this.picker?.setActive(p);
    this.root.classList.toggle('drawing', p === 'draw');
    this.rot = identity3();
    if (p === 'earth') {
      this.load(this.earthSignal);
    } else if (p === 'draw') {
      this.pad!.load(this.digits!.images[this.lastDigit]!);
      this.loadImage(this.pad!.image);
      // Hold the sphere still under the pen.
      this.spinning = false;
      this.spinToggle.set(false);
    } else {
      this.lastDigit = p;
      this.loadImage(this.digits!.images[p]!);
    }
    this.lineWhenDone = true;
    void this.rebuild();
  }

  /** Drops the running solve and blanks the reconstruction, with `message` as the status. */
  private stopSolve(message: string): void {
    this.runs++;
    if (this.solver.busy) this.solver.cancel();
    this.rebuilt = null;
    this.candidates = null;
    this.candidateButton.hidden = true;
    this.lined = false;
    this.lineButton.disabled = true;
    this.againButton.disabled = false;
    this.redrawn.setSignal(null, this.vmax);
    this.status.textContent = message;
  }

  apply(step: HeroStep): void {
    switch (step) {
      case 'north':
        this.northUp();
        break;
      case 'scramble':
        this.scramble();
        break;
      case 'rebuild':
        void this.rebuild();
        break;
      case 'line':
        if (this.rebuilt) this.setLined(true);
        else this.lineWhenDone = true;
        break;
      case 'free':
        this.lineWhenDone = false;
        this.setLined(false);
        break;
    }
  }

  /** The Earth at rest as an equirectangular map, Greenwich in the middle, at the globe's band-limit. */
  private map(n: number): HTMLElement {
    const view = new FlatView(
      worldMap(MAP.width, MAP.height, facing(EARTH_VIEW.lat, EARTH_VIEW.lon)),
      'hr-map-view',
    );
    view.paint(this.x, this.vmax);
    return el(
      'div',
      { class: 'hr-map' },
      view.canvas,
      el('p', { text: `The same ${n} coefficients, as a map` }),
    );
  }

  private northUp(): void {
    this.rot = identity3();
    this.dirty = true;
  }

  /** Another globe with the same amount of detail at every scale. */
  private scramble(): void {
    this.scrambleSeed = Math.floor(this.rng.uniform() * 2 ** 32);
    this.showScramble();
  }

  private showScramble(): void {
    this.mush.setSignal(scrambleDegrees(this.x, new Rng(this.scrambleSeed)), this.vmax);
  }

  private setLined(on: boolean): void {
    if (!this.rebuilt) return;
    this.lined = on;
    this.lineFrom = this.shown;
    this.lineT0 = performance.now();
    if (!on) this.own = identity3();
    this.lineButton.textContent = on ? 'let it go' : 'line it up';
    this.renderStatus();
  }

  /** Hands the solver the descriptor, and nothing else, and shows what comes back. */
  private async rebuild(): Promise<void> {
    if (this.pick === 'draw' && this.pad?.empty) {
      this.stopSolve('The pad is empty.');
      return;
    }
    const run = ++this.runs;
    this.rebuilt = null;
    this.candidates = null;
    this.candidateButton.hidden = true;
    this.lined = false;
    this.own = identity3();
    this.lineT0 = -Infinity;
    this.lineButton.textContent = 'line it up';
    this.lineButton.disabled = this.againButton.disabled = true;
    this.redrawn.setSignal(null, this.vmax);
    this.status.textContent = `Reconstructing from the ${this.invariant.n} features…`;
    await this.solver.solve(
      {
        lmax: this.lmax,
        target: this.invariant.values(this.x),
        alignTo: Float64Array.from(this.x),
        seed: 7919 * run + 101 + Math.floor(1e6 * Math.random()),
        config: WIDGET_CONFIG,
      },
      (e) => this.onSolve(run, e),
    );
  }

  private onSolve(run: number, e: SolveEvent): void {
    if (run !== this.runs) return;
    switch (e.type) {
      case 'seed':
        return;
      case 'degree': {
        // The globe sharpens as the solver adds one level of detail at a time.
        const b = e.branches[e.best];
        if (b) this.redrawn.setSignal(b.x, this.vmax);
        return;
      }
      case 'done': {
        this.againButton.disabled = false;
        const b = e.branches[e.best];
        if (!b) {
          this.status.textContent =
            'No candidate survived from these random starts. “again” reruns the solver from new ones.';
          return;
        }
        if (!e.accepted) {
          this.redrawn.setSignal(b.x, this.vmax);
          this.status.textContent = `Missed the tolerance: the closest candidate matches the features to ${sci(b.featureRel)}, against ${sci(WIDGET_CONFIG.okTol)} required. “again” reruns the solver from new random starts.`;
          return;
        }
        this.candidates = { branches: e.branches, reps: e.reps, seconds: e.ms / 1000 };
        this.candidateButton.hidden = e.reps.length < 2;
        this.showCandidate(0);
        // The first reconstruction, on page load, lines itself up; "again" leaves it in its own pose.
        if (run === 1 || this.lineWhenDone) {
          this.lineWhenDone = false;
          this.setLined(true);
        } else this.renderStatus();
        return;
      }
      case 'error':
        this.againButton.disabled = false;
        this.status.textContent = `The solver stopped: ${e.message}`;
    }
  }

  /** Shows accepted candidate `k` (mod their count); the choice of the first one was blind. */
  private showCandidate(k: number): void {
    const c = this.candidates;
    if (!c) return;
    this.candidate = k % c.reps.length;
    const b = c.branches[c.reps[this.candidate]!]!;
    this.rebuilt = { x: b.x, toEarth: b.rot, rel: b.alignedRel, seconds: c.seconds };
    this.redrawn.setSignal(b.x, this.vmax);
    this.lineButton.disabled = false;
    this.candidateButton.textContent = `candidate ${this.candidate + 1} of ${c.reps.length}`;
    this.renderStatus();
  }

  private renderStatus(): void {
    const r = this.rebuilt;
    if (!r) return;
    const n = this.candidates?.reps.length ?? 1;
    const others = n > 1 ? ` ${n} distinct signals match the features.` : '';
    this.status.textContent = this.lined
      ? `Lined up, the rebuilt coefficients differ from the original's by ${sci(r.rel)} (relative error).${others}`
      : `Reconstructed in ${r.seconds.toFixed(1)} s from these features alone, in a pose of its own.${others}`;
  }

  /** Recomputes the three lists from the turned globe: only the first one moves. */
  private refresh(): void {
    const turned = rotateDofs(this.x, this.rot);
    this.cells.all.set(turned);
    this.cells.power.set(degreePower(turned, this.lmax));
    this.cells.descriptor.set(this.invariant.values(turned));
    this.earth.setOrientation(this.rot);
    this.mush.setOrientation(this.rot);
  }

  private poseRedrawn(now: number): void {
    const target = this.lined && this.rebuilt ? mul3(this.rot, this.rebuilt.toEarth) : this.own;
    const t = reducedMotion() ? 1 : (now - this.lineT0) / LINE_MS;
    const pose = t < 1 ? slerp(this.lineFrom, target, ease(Math.max(0, t))) : target;
    if (same(pose, this.shown)) return;
    this.shown = pose;
    this.redrawn.setOrientation(pose);
  }

  private readonly tick = (now: number): void => {
    requestAnimationFrame(this.tick);
    const dt = Math.min(0.05, (now - this.lastTick) / 1000);
    this.lastTick = now;
    if (!this.visible) return;
    if (this.spinning && !this.dragging) {
      // About the Earth's own axis, west to east, wherever a drag has left it; a digit turns
      // about its own vertical.
      const axis = this.pick === 'earth' ? POLE : [0, 1, 0];
      const step = expMap(axis.map((v) => v * SPIN_SPEED * dt));
      // Through a quaternion, so that thousands of products stay a rotation.
      this.rot = quatToMat(matToQuat(mul3(this.rot, step)));
      this.dirty = true;
      // Not lined up, the rebuilt globe turns too, about the vertical axis of the view.
      if (!this.lined)
        this.own = quatToMat(matToQuat(mul3(expMap([0, SPIN_SPEED * dt, 0]), this.own)));
    }
    if (this.dirty) this.refresh();
    this.dirty = false;
    this.poseRedrawn(now);
  };
}

export const mount: Mount = async (root) => {
  const [{ tables, digits }, earth] = await Promise.all([sharedAssets(), loadEarth()]);
  const figure = new HeroFigure(root, { tables, earth, digits });
  return { apply: (step) => figure.apply(step as HeroStep) };
};
