/**
 * Illustrations for the deck: the two usual fixes on the `fixes` slide (augmentation, invariant
 * pooling), the two directions of a complete invariant on the `want` slide (one orbit, one
 * invariant; another orbit, another invariant), and the three uses on the `enables` slide (pooling,
 * invariant features, inversion). Colors and type come from `deck.css`; orange marks the
 * bispectrum, or the digit.
 */

import { f, shift } from './icons';

export type ArtName =
  'augment' | 'collapse' | 'orbit-x' | 'orbit-y' | 'pool' | 'features' | 'invert';

/** A feature map over the 8 poses of a group convolution, relative to its maximum. */
const POSES = [0.55, 0.85, 0.7, 1, 0.4, 0.75, 0.95, 0.6];
/** The pose map of another digit: a different shape, the same maximum. */
const OTHER = [0.3, 0.45, 0.2, 0.35, 0.6, 0.25, 1, 0.4];
/** Tints of the cells of an invariant, from sand to orange. */
const TINTS = [2, 3, 1, 3, 2, 0, 3, 1];
/** The invariants of two digits on the `want` slide: one per orbit, and different. */
const OF_X = [2, 3, 1, 3, 2, 0, 3, 1, 2, 0];
const OF_Y = [0, 1, 3, 0, 2, 3, 1, 0, 3, 2];

const text = (x: number, y: number, s: string, cls: string): string =>
  `<text class="${cls}" x="${x}" y="${y}">${s}</text>`;

/** A pose map as bars standing on `base`, `step` apart. */
const poses = (x: number, base: number, h: number, values = POSES, step = 11): string =>
  values
    .map(
      (v, i) =>
        `<rect class="bar" x="${x + step * i}" y="${f(base - v * h)}" width="${step - 3}" height="${f(v * h)}" rx="1"/>`,
    )
    .join('');

/** A digit turned by `a` degrees on a sand square of side `s`. */
function tile(x: number, y: number, s: number, a: number, cls: string): string {
  const [cx, cy] = [x + s / 2, y + s / 2];
  return (
    `<rect class="cell t0" x="${x}" y="${y}" width="${s}" height="${s}" rx="6"/>` +
    `<text class="${cls}" x="${cx}" y="${cy}" transform="rotate(${a} ${cx} ${cy})">2</text>`
  );
}

/** A digit turned by `a` degrees on a sand globe of radius `r`. */
const globe = (cx: number, cy: number, digit: string, a = 0, r = 28, cls = 'digit'): string =>
  `<circle class="globe" cx="${cx}" cy="${cy}" r="${r}"/>` +
  `<ellipse class="thin" cx="${cx}" cy="${cy}" rx="${r}" ry="${f(r / 3)}"/>` +
  `<text class="${cls}" x="${cx}" y="${cy}" transform="rotate(${a} ${cx} ${cy})">${digit}</text>`;

/** The cells of an invariant, each `(dx, dy)` from the last, of side `s`. */
const cells = (x: number, y: number, dx: number, dy: number, tints = TINTS, s = 10): string =>
  tints
    .map(
      (t, i) =>
        `<rect class="cell t${t}" x="${f(x + dx * i)}" y="${f(y + dy * i)}" width="${s}" height="${s}" rx="${f(s * 0.15)}"/>`,
    )
    .join('');

/** Six poses of a digit round a dashed loop, its orbit, and the one invariant they all map to. */
function orbit(digit: string, tints: number[], label: string): string {
  const poses = Array.from({ length: 6 }, (_, k) => {
    const t = (Math.PI / 3) * k;
    const [x, y] = [Math.round(132 + 106 * Math.cos(t)), Math.round(65 + 40 * Math.sin(t))];
    return globe(x, y, digit, 60 * k, 23, 'digit sm');
  }).join('');
  return (
    `<ellipse class="loop" cx="132" cy="65" rx="106" ry="40"/>` +
    `<g class="poses">${poses}</g>` +
    shift(274, 65, 336, 65, 'act') +
    text(305, 52, 'β', 'lbl mid on') +
    cells(350, 56, 22, 0, tints, 18) +
    text(458, 104, label, 'lbl mid')
  );
}

/** A lopsided closed curve, so that a turn of it shows. */
function blob(cx: number, cy: number, turn: number, cls: string): string {
  const d = Array.from({ length: 145 }, (_, i) => {
    const a = (2 * Math.PI * i) / 144;
    const r = 31 + 7 * Math.cos(3 * a) + 5 * Math.sin(2 * a + 0.5) + 2 * Math.cos(5 * a + 1);
    return `${i ? 'L' : 'M'}${f(cx + r * Math.cos(a + turn))} ${f(cy + r * Math.sin(a + turn))}`;
  }).join(' ');
  return `<path class="${cls}" d="${d} Z"/>`;
}

/** A small MLP: layers of nodes, every node wired to the next layer. */
function mlp(xs: number[], counts: number[], cy: number): string {
  const ys = counts.map((n) => Array.from({ length: n }, (_, i) => cy + 22 * (i - (n - 1) / 2)));
  let wires = '';
  for (let l = 0; l + 1 < xs.length; l++) {
    for (const y0 of ys[l]!) {
      for (const y1 of ys[l + 1]!) {
        wires += `<path class="wire" d="M${xs[l]} ${f(y0)} L${xs[l + 1]} ${f(y1)}"/>`;
      }
    }
  }
  const nodes = xs
    .flatMap((x, l) => ys[l]!.map((y) => `<circle class="node" cx="${x}" cy="${f(y)}" r="6"/>`))
    .join('');
  return wires + nodes;
}

const DRAW: Record<ArtName, () => string> = {
  // Every pose of a 2, one invariant; a 7, in any pose, another.
  'orbit-x': () => orbit('2', OF_X, 'β(g·x) = β(x)'),
  'orbit-y': () => orbit('7', OF_Y, 'β(y) ≠ β(x)'),
  // One digit becomes eight training examples, one per turn.
  augment: () =>
    tile(0, 72, 56, 0, 'digit') +
    text(28, 184, '1 digit', 'lbl mid') +
    shift(66, 100, 98, 100, 'flow') +
    Array.from({ length: 8 }, (_, i) =>
      tile(110 + 52 * (i % 4), 52 + 52 * Math.floor(i / 4), 44, 45 * i, 'digit sm'),
    ).join('') +
    text(210, 184, '8 rotated copies', 'lbl mid') +
    shift(322, 100, 352, 100, 'flow') +
    mlp([366, 400, 434], [3, 3, 2], 100) +
    text(400, 184, 'train', 'lbl mid'),
  // Two digits, two different pose maps, one pooled value: the pose is gone, and so is the digit.
  collapse: () =>
    globe(28, 44, '2') +
    globe(28, 156, '7') +
    shift(62, 44, 84, 44, 'flow') +
    shift(62, 156, 84, 156, 'flow') +
    poses(94, 72, 56, POSES, 14) +
    poses(94, 184, 56, OTHER, 14) +
    shift(212, 44, 266, 90, 'flow') +
    shift(212, 156, 266, 110, 'flow') +
    text(226, 105, 'max', 'lbl mid') +
    `<rect class="bar keep" x="278" y="72" width="14" height="56" rx="1"/>` +
    shift(304, 100, 334, 100, 'flow') +
    text(394, 100, '2 or 7?', 'ask mid'),
  // The same pose map, pooled two ways.
  pool: () =>
    poses(0, 80, 60) +
    shift(98, 56, 146, 56, 'flow') +
    text(122, 44, 'max', 'lbl mid') +
    `<rect class="bar keep" x="160" y="20" width="10" height="60" rx="1"/>` +
    text(165, 102, '1 value', 'lbl mid') +
    text(360, 56, '73.6%', 'num end') +
    poses(0, 190, 60) +
    shift(98, 166, 146, 166, 'act') +
    text(122, 152, 'β', 'lbl mid on') +
    cells(160, 161, 12, 0) +
    text(207, 212, 'complete', 'lbl mid') +
    text(360, 166, '78.5%', 'num end on'),
  // Three poses of a digit, one feature vector, one answer.
  features: () => {
    const globes = (
      [
        [40, 0],
        [110, 115],
        [180, 235],
      ] as const
    )
      .map(([y, a]) => globe(36, y, '2', a))
      .join('');
    const arrows = [40, 110, 180]
      .map((y) => shift(70, y, 126, 110 + (y - 110) * 0.1, 'flow'))
      .join('');
    return (
      globes +
      arrows +
      cells(136, 62, 0, 12) +
      text(141, 194, 'β', 'lbl mid on') +
      shift(156, 110, 186, 110, 'flow') +
      mlp([200, 236, 272], [4, 4, 3], 110) +
      text(236, 194, 'MLP', 'lbl mid') +
      shift(284, 110, 314, 110, 'flow') +
      text(338, 110, '2', 'out')
    );
  },
  // The invariants back to a signal, in a pose of its own; turned back, it is the original.
  invert: () =>
    cells(6, 62, 0, 12) +
    text(11, 194, 'β', 'lbl mid on') +
    shift(28, 110, 90, 110, 'act') +
    text(59, 94, '.invert', 'lbl mid on') +
    blob(144, 110, 1.9, 'rebuilt') +
    text(144, 194, 'rebuilt', 'lbl mid') +
    shift(196, 110, 240, 110, 'flow') +
    text(218, 94, 'align', 'lbl mid') +
    blob(302, 110, 0, 'orig') +
    blob(302, 110, 0, 'rebuilt dash') +
    text(302, 194, 'original', 'lbl mid'),
};

const BOX: Record<ArtName, string> = {
  augment: '0 0 450 200',
  collapse: '0 0 450 200',
  'orbit-x': '0 0 580 130',
  'orbit-y': '0 0 580 130',
  pool: '0 0 360 220',
  features: '0 0 360 220',
  invert: '0 0 360 220',
};

export function art(name: ArtName): string {
  return `<svg class="art" viewBox="${BOX[name]}" aria-hidden="true">${DRAW[name]()}</svg>`;
}

export const isArtName = (s: string): s is ArtName => s in DRAW;
