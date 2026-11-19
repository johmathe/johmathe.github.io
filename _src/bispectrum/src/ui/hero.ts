import { sharedAssets } from '../assets';
import type { Mount } from '../figure';
import { degreePower, dim, toHalfSpectrum } from '../math/basis';
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
import { EARTH_VIEW, facing, loadEarth, worldMap, type Signal } from '../signals';
import { SolverClient, type SolveEvent } from '../worker/client';
import type { FinalBranch } from '../worker/protocol';
import { reducedMotion } from './canvas';
import { greyAt, inkAt } from './colormap';
import { button, el, toggle } from './controls';
import { CellGrid } from './cells';
import { FlatView } from './flat';
import { SphereView } from './sphere';

export interface HeroAssets {
  tables: InvariantTables;
  earth: Signal;
}

const DISPLAY = { nlat: 96, nlon: 192 };
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
  private readonly lmax: number;
  private readonly invariant: Invariant;
  private readonly x: Float64Array;
  private readonly vmax: number;
  private readonly solver: SolverClient;
  private readonly rng = new Rng(20261109);

  private readonly earth: SphereView;
  private readonly mush: SphereView;
  private readonly redrawn: SphereView;
  private readonly cells: { all: CellGrid; power: CellGrid; descriptor: CellGrid };
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
    this.x = earth.x;
    this.lmax = Math.round(Math.sqrt(this.x.length)) - 1;
    this.invariant = new Invariant(tables, this.lmax);
    this.solver = new SolverClient(tables);
    const grid = new SphereGrid(DISPLAY.nlat, DISPLAY.nlon, this.lmax, 'centered');
    this.vmax =
      grid
        .synthesize(toHalfSpectrum(this.x, this.lmax))
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
    this.cells.all.setReference(this.x);
    this.cells.power.setReference(degreePower(this.x, this.lmax));
    this.cells.descriptor.setReference(this.invariant.values(this.x));

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

    const lead = (name: string, count: number, noun: string, what: string): HTMLElement =>
      el(
        'div',
        { class: 'hr-lead' },
        ...(deck ? [el('p', { class: 'hr-name', text: name })] : []),
        el(
          'div',
          { class: 'hr-head' },
          el('span', { class: 'hr-count', text: String(count) }),
          el('span', { class: 'hr-noun', text: noun }),
        ),
        el('p', { class: 'hr-what', text: what }),
      );
    const redraw = (host: HTMLElement, title: string, ...controls: HTMLElement[]): HTMLElement =>
      el(
        'div',
        { class: 'hr-redraw' },
        host,
        el('div', { text: title }),
        el('div', { class: 'rw-row' }, ...controls),
      );

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
            el('h4', { text: 'Earth, roughly' }),
            el('span', { class: 'rw-hint', text: 'drag to turn it' }),
          ),
          earthHost,
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
            lead(
              'Coefficients',
              n,
              'coefficients',
              'The globe in spherical harmonics. Turn it and they change.',
            ),
            this.cells.all.element,
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
                this.lmax + 1,
                'features',
                'The power spectrum, invariant to rotation: how much detail the globe has at each scale.',
              ),
              this.cells.power.element,
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
                this.invariant.n,
                'features',
                `A selective bispectrum, also invariant: sums of products of three or four of the ${n} coefficients.`,
              ),
              this.cells.descriptor.element,
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
    this.earth.setSignal(this.x, this.vmax);
    this.scramble();
    new IntersectionObserver((entries) => {
      this.visible = entries.some((e) => e.isIntersecting);
    }).observe(root);

    void this.rebuild();
    requestAnimationFrame(this.tick);
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
    this.mush.setSignal(scrambleDegrees(this.x, this.rng), this.vmax);
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
    this.status.textContent = `Reconstructing the globe from the ${this.invariant.n} features…`;
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
    const others = n > 1 ? ` ${n} distinct globes match the features.` : '';
    this.status.textContent = this.lined
      ? `Lined up, the rebuilt coefficients differ from the rough Earth's by ${sci(r.rel)} (relative error).${others}`
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
      // About the Earth's own axis, west to east, wherever a drag has left it.
      const step = expMap(POLE.map((v) => v * SPIN_SPEED * dt));
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
  const [{ tables }, earth] = await Promise.all([sharedAssets(), loadEarth()]);
  const figure = new HeroFigure(root, { tables, earth });
  return { apply: (step) => figure.apply(step as HeroStep) };
};
