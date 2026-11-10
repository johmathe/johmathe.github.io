/** Brand tokens (`assets/brand/__init__.py`) and its diverging colormap. */

export const INK = '#1C1C1A';
export const BEIGE = '#F7F3EB';
export const SAND = '#EAE6DB';
export const MUTED = '#5D5852';
export const ASH = '#807B74';
export const STONE = '#A8A298';
export const RULE = '#DDD6CA';
export const ORANGE = '#F57050';
export const ORANGE_TEXT = '#B53E20';
export const GRAPHITE = '#3D3834';
export const NEUTRAL_LIGHT = '#D6CFC3';
export const WHITE = '#FFFFFF';

export const hex = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

export interface Tint {
  hex: string;
  of: string;
  over: string;
  strength: number;
}

/**
 * Lighter variants as opaque mixes, `strength * of + (1 - strength) * over` (`brand.tint`): the
 * brand avoids alpha. `blog/test/brand.test.ts` recomputes every entry and allows no other color.
 */
export const TINTS = {
  /** Panels and figure grounds. */
  panel: { hex: '#FBF9F5', of: BEIGE, over: WHITE, strength: 0.5 },
  /** The second kind of number in the descriptor, next to full orange. */
  orangeLight: { hex: '#F7A792', of: ORANGE, over: '#FBF9F5', strength: 0.6 },
  /** The degree being solved, behind its bars. */
  orangeWash: { hex: '#FAE6DE', of: ORANGE, over: '#FBF9F5', strength: 0.14 },
  /** Every other degree band. */
  sandWash: { hex: '#F2EFE7', of: SAND, over: '#FBF9F5', strength: 0.55 },
} as const satisfies Record<string, Tint>;

/** `brand.METHOD_COLORS`: the accent is the bispectrum's; baselines are warm grays. */
export const METHOD_COLORS = {
  bispectrum: ORANGE_TEXT,
  powerSpectrum: INK,
  cnn: STONE,
} as const;

/** Area fills of the same methods: orange may fill areas, lines and text take `ORANGE_TEXT`. */
export const METHOD_FILLS = {
  bispectrum: ORANGE,
  powerSpectrum: INK,
  cnn: STONE,
} as const;

/** `bispectrum_diverging`: ink (negative), beige (zero), orange, deep orange (positive). */
type Stop = [number, [number, number, number]];

const STOPS: Stop[] = [
  [0, hex(INK)],
  [0.5, hex(BEIGE)],
  [0.8, hex(ORANGE)],
  [1, hex(ORANGE_TEXT)],
];

/** The same map without the accent, for numbers that are not the bispectrum's. */
const GREY_STOPS: Stop[] = [
  [0, hex(INK)],
  [0.5, hex(BEIGE)],
  [1, hex(STONE)],
];

const LUT_SIZE = 512;

function lookup(stops: Stop[]): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1]![0]) k++;
    const [t0, c0] = stops[k]!;
    const [t1, c1] = stops[k + 1]!;
    const f = (t - t0) / (t1 - t0);
    for (let ch = 0; ch < 3; ch++) lut[i * 3 + ch] = Math.round(c0[ch]! + f * (c1[ch]! - c0[ch]!));
  }
  return lut;
}

const LUT = lookup(STOPS);
const GREY_LUT = lookup(GREY_STOPS);

function cssAt(lut: Uint8Array, t: number): string {
  const u = Number.isFinite(t) ? Math.min(1, Math.max(0, 0.5 + 0.5 * t)) : 0.5;
  const k = Math.round(u * (LUT_SIZE - 1)) * 3;
  return `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
}

/** RGBA pixels of `values` on a symmetric scale `[-vmax, vmax]`, written into `out`. */
export function paint(
  values: ArrayLike<number>,
  vmax: number,
  out: Uint8Array | Uint8ClampedArray,
): void {
  const s = vmax > 0 ? 1 / vmax : 0;
  for (let i = 0; i < values.length; i++) {
    const t = Math.min(1, Math.max(0, 0.5 + 0.5 * values[i]! * s));
    const k = Math.round(t * (LUT_SIZE - 1)) * 3;
    out[4 * i] = LUT[k]!;
    out[4 * i + 1] = LUT[k + 1]!;
    out[4 * i + 2] = LUT[k + 2]!;
    out[4 * i + 3] = 255;
  }
}

/** CSS color at `t` in `[-1, 1]` (clamped). */
export const colorAt = (t: number): string => cssAt(LUT, t);

/** Warm grey at `t` in `[-1, 1]`: ink (negative), beige (zero), stone (positive). */
export const greyAt = (t: number): string => cssAt(GREY_LUT, t);

/** Ink of strength `t` in `[0, 1]` on beige, for numbers that cannot be negative. */
export const inkAt = (t: number): string => cssAt(GREY_LUT, -t);

export function cssGradient(): string {
  return `linear-gradient(90deg, ${STOPS.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`).join(', ')})`;
}
