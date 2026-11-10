/**
 * Bridge (`bridge`): one wave on the ring and one spherical harmonic on the sphere, turned
 * together, with each turn written as the matrix that acts on the coefficients. Both matrices stay
 * block-diagonal: blocks 1, 2, 2, ... on the ring, 1, 3, 5, ... on the sphere.
 */

import type { Figure, Mount } from '../figure';
import { dim, toHalfSpectrum } from '../math/basis';
import { dofBlocks, expMap, identity3, mul3, rotateDofs, type Mat3 } from '../math/rotation';
import { SphereGrid } from '../math/sht';
import { drawBlockMatrix, drawCells, Surface, Tween } from './canvas';
import { colorAt, INK, MUTED, RULE } from './colormap';
import { button, el, segmented, slider, type Segmented, type Slider } from './controls';
import { SphereView } from './sphere';

/** Highest wave on the ring and highest degree on the sphere. */
const L = 4;
const TAU = 2 * Math.PI;
const TILT = (() => {
  const a = [0.85, 0.3, 0.45];
  const n = Math.hypot(...a);
  return a.map((v) => v / n);
})();

type Axis = 'pole' | 'tilted';

/**
 * The ring's turn on the coordinates `(c_0, Re c_1, Im c_1, ...)` of the prose's `c_k`:
 * `c_k -> c_k e^{ikα}`.
 */
function ringBlocks(alpha: number): Float64Array[] {
  const out = [Float64Array.of(1)];
  for (let k = 1; k <= L; k++) {
    const [c, s] = [Math.cos(k * alpha), Math.sin(k * alpha)];
    out.push(Float64Array.of(c, -s, s, c));
  }
  return out;
}

function apply(blocks: Float64Array[], x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  let o = 0;
  for (const b of blocks) {
    const sz = Math.round(Math.sqrt(b.length));
    for (let i = 0; i < sz; i++) {
      let s = 0;
      for (let j = 0; j < sz; j++) s += b[i * sz + j]! * x[o + j]!;
      out[o + i] = s;
    }
    o += sz;
  }
  return out;
}

class BridgeFigure implements Figure {
  private k = 2;
  private alpha = 0;
  private axis: Axis = 'pole';
  /** Turns the reader added by dragging the sphere. */
  private dragged: Mat3 = identity3();
  private readonly tween = new Tween();
  private readonly ring: Surface;
  private readonly ringMatrix: Surface;
  private readonly ringCells: Surface;
  private readonly sphere: SphereView;
  private readonly sphereMatrix: Surface;
  private readonly sphereCells: Surface;
  private readonly grid = new SphereGrid(96, 192, L, 'centered');
  private readonly waveSwitch: Segmented<string>;
  private readonly axisSwitch: Segmented<Axis>;
  private readonly turn: Slider;
  private readonly readout: HTMLElement;

  constructor(host: HTMLElement) {
    const root = el('div', { class: 'rw br' });
    this.waveSwitch = segmented(
      'Wraps',
      Array.from({ length: L }, (_, i) => ({ value: String(i + 1), label: String(i + 1) })),
      String(this.k),
      (v) => this.setWave(Number(v)),
    );
    this.axisSwitch = segmented(
      'Sphere axis',
      [
        { value: 'pole', label: 'about the pole' },
        { value: 'tilted', label: 'about a tilted axis' },
      ],
      this.axis,
      (v) => this.setAxis(v),
    );
    this.turn = slider({
      label: 'turn',
      min: 0,
      max: 360,
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
      { class: 'rw-controls' },
      el('div', { class: 'rw-row' }, el('span', { text: 'wraps' }), this.waveSwitch.element),
      this.turn.element,
      el('div', { class: 'rw-row' }, this.axisSwitch.element),
      button('spin once', () => this.spin()),
    );

    const ringPanel = el('div', { class: 'rw-panel' }, head('Ring', 'frequency k is one wave'));
    this.ring = new Surface(ringPanel, 'br-view', () => this.drawRing());
    ringPanel.append(el('h5', { class: 'br-label', text: 'Its numbers' }));
    this.ringCells = new Surface(ringPanel, 'br-cells', () => this.drawRingCells());
    ringPanel.append(el('h5', { class: 'br-label', text: 'Turn, as a matrix on them' }));
    this.ringMatrix = new Surface(ringPanel, 'br-matrix', () =>
      drawBlockMatrix(this.ringMatrix, ringBlocks(this.alpha), { bold: this.k }),
    );
    ringPanel.append(el('p', { class: 'br-sizes', text: 'blocks: 1, 2, 2, 2, 2' }));

    const spherePanel = el(
      'div',
      { class: 'rw-panel' },
      head('Sphere', 'drag to turn it any way'),
    );
    const wrap = el('div', { class: 'br-sphere' });
    spherePanel.append(wrap);
    this.sphere = new SphereView(wrap, {
      grid: this.grid,
      onDrag: (w) => {
        this.tween.stop();
        this.dragged = mul3(w, this.dragged);
        this.update();
      },
    });
    spherePanel.append(el('h5', { class: 'br-label', text: 'Its numbers' }));
    this.sphereCells = new Surface(spherePanel, 'br-cells', () => this.drawSphereCells());
    spherePanel.append(el('h5', { class: 'br-label', text: 'Turn, as a matrix on them' }));
    this.sphereMatrix = new Surface(spherePanel, 'br-matrix', () =>
      drawBlockMatrix(this.sphereMatrix, dofBlocks(this.rotation, L), { bold: this.k }),
    );
    spherePanel.append(el('p', { class: 'br-sizes', text: 'blocks: 1, 3, 5, 7, 9' }));

    this.readout = el('p', { class: 'rg-readout' });
    root.append(controls, el('div', { class: 'rw-stage' }, ringPanel, spherePanel), this.readout);
    host.replaceChildren(root);
    this.setWave(this.k);
  }

  private get rotation(): Mat3 {
    const w = this.axis === 'pole' ? [0, 0, this.alpha] : TILT.map((v) => v * this.alpha);
    return mul3(this.dragged, expMap(w));
  }

  /** Degree `k`'s harmonic that wraps `k` times around the pole: `Re Y_kk`. */
  private get harmonic(): Float64Array {
    const x = new Float64Array(dim(L));
    x[this.k * this.k + 2 * this.k - 1] = 1;
    return x;
  }

  private get ringCoords(): Float64Array {
    const x = new Float64Array(1 + 2 * L);
    x[2 * this.k - 1] = 1;
    return apply(ringBlocks(this.alpha), x);
  }

  apply(step: string): void {
    switch (step) {
      case 'turn':
        this.dragged = identity3();
        this.setAxis('pole');
        this.spin();
        break;
      case 'tilt':
        this.dragged = identity3();
        this.setAxis('tilted');
        this.spin();
        break;
      case 'degree':
        this.setWave(this.k === L ? 1 : this.k + 1);
        this.spin();
        break;
    }
  }

  private setWave(k: number): void {
    this.k = k;
    this.waveSwitch.set(String(k));
    const x = this.harmonic;
    const field = this.grid.synthesize(toHalfSpectrum(x, L));
    this.sphere.setSignal(x, Math.max(...Array.from(field, Math.abs)));
    this.update();
  }

  private setAxis(axis: Axis): void {
    this.axis = axis;
    this.axisSwitch.set(axis);
    this.update();
  }

  private spin(): void {
    const from = this.alpha;
    this.tween.run(4000, (t) => {
      this.alpha = from + t * TAU;
      this.update();
    });
  }

  private update(): void {
    const deg = Math.round((((this.alpha % TAU) + TAU) % TAU) * (360 / TAU)) % 360;
    this.turn.set(deg);
    this.sphere.setOrientation(this.rotation);
    for (const s of [this.ring, this.ringCells, this.ringMatrix]) s.render();
    for (const s of [this.sphereCells, this.sphereMatrix]) s.render();
    const k = this.k;
    this.readout.textContent =
      this.axis === 'pole' && isPole(this.dragged)
        ? `About the pole the sphere turns like the ring: inside each block, its numbers only ` +
          `rotate in pairs, by m times the angle. The wave that wraps ${k} times sits in block ${k}.`
        : `About any other axis the ${2 * k + 1} numbers of degree ${k} mix with each other and ` +
          `the whole block fills in. Nothing ever leaks into another block.`;
  }

  private drawRing(): void {
    const { ctx, w, h } = this.ring;
    const [cx, cy] = [w / 2, h / 2];
    const R = Math.min(w, h) / 2 - 12;
    const r = R * 0.62;
    const x = this.ringCoords;
    const [a, b] = [x[2 * this.k - 1]!, x[2 * this.k]!];
    const seg = 240;
    for (let j = 0; j < seg; j++) {
      const t0 = (TAU * j) / seg;
      const t1 = (TAU * (j + 1)) / seg + 0.004;
      const t = (t0 + t1) / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, R, -t1, -t0);
      ctx.arc(cx, cy, r, -t0, -t1, true);
      ctx.closePath();
      ctx.fillStyle = colorAt(a * Math.cos(this.k * t) + b * Math.sin(this.k * t));
      ctx.fill();
    }
    ctx.strokeStyle = RULE;
    ctx.lineWidth = 1;
    for (const rr of [R, r]) {
      ctx.beginPath();
      ctx.arc(cx, cy, rr, 0, TAU);
      ctx.stroke();
    }
    // The phase of c_k turns k times as far as the crests.
    const hand = Math.atan2(b, a);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * 0.8 * Math.cos(hand), cy - r * 0.8 * Math.sin(hand));
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2.25;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.font = '11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`k = ${this.k}`, 2, h - 2);
  }

  private drawRingCells(): void {
    drawCells(this.ringCells, this.ringCoords, [1, ...Array<number>(L).fill(2)]);
  }

  private drawSphereCells(): void {
    const x = rotateDofs(this.harmonic, this.rotation);
    drawCells(
      this.sphereCells,
      x,
      Array.from({ length: L + 1 }, (_, l) => 2 * l + 1),
    );
  }
}

function head(title: string, hint: string): HTMLElement {
  return el(
    'div',
    { class: 'rw-panel-head' },
    el('h4', { text: title }),
    el('span', { class: 'rw-hint', text: hint }),
  );
}

const isPole = (m: Mat3): boolean => Math.abs(m[8]! - 1) < 1e-9;

export const mount: Mount = async (root) => new BridgeFigure(root);
