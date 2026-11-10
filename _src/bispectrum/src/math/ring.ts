/**
 * Signals on a ring and their Fourier coefficients, the cyclic case of `CnonCn` (`src/bispectrum/cn_on_cn.py`).
 *
 * A spectrum holds the Fourier coefficients `c_0..c_{m-1}` in the `torch.fft.fft`
 * convention. For the ring figures it is band-limited, a blob of radius
 * `r(θ) = c_0 + 2 Re Σ_{k≥1} c_k e^{ikθ}`, and turning the blob by `α` takes `c_k` to
 * `c_k e^{-ikα}`. The prose writes `c_k` for the conjugates, which turn by `e^{ikα}`. Given a
 * full DFT instead, the chain below is `CnonCn(selective=True)` exactly.
 */

import type { Rng } from './rng';

export interface Spectrum {
  re: Float64Array;
  im: Float64Array;
}

export const spectrum = (m: number): Spectrum => ({
  re: new Float64Array(m),
  im: new Float64Array(m),
});

export const size = (c: Spectrum): number => c.re.length;

export function copy(c: Spectrum): Spectrum {
  return { re: c.re.slice(), im: c.im.slice() };
}

/** Unnormalised DFT, `torch.fft.fft`. */
export function dft(x: ArrayLike<number>): Spectrum {
  const n = x.length;
  const out = spectrum(n);
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * j * k) / n;
      re += x[j]! * Math.cos(a);
      im += x[j]! * Math.sin(a);
    }
    out.re[k] = re;
    out.im[k] = im;
  }
  return out;
}

/** Coefficients `0..K` of the blob whose radius is sampled at `n` evenly spaced angles. */
export function harmonics(samples: ArrayLike<number>, K: number): Spectrum {
  const n = samples.length;
  if (2 * K >= n) throw new Error(`need more than ${2 * K} samples for ${K} harmonics`);
  const full = dft(samples);
  const c = spectrum(K + 1);
  for (let k = 0; k <= K; k++) {
    c.re[k] = full.re[k]! / n;
    c.im[k] = full.im[k]! / n;
  }
  return c;
}

export function radiusAt(c: Spectrum, theta: number): number {
  let r = c.re[0]!;
  for (let k = 1; k < size(c); k++) {
    r += 2 * (c.re[k]! * Math.cos(k * theta) - c.im[k]! * Math.sin(k * theta));
  }
  return r;
}

/** The radius at `n` evenly spaced angles, starting at `θ = 0`. */
export function outline(c: Spectrum, n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, j) => radiusAt(c, (2 * Math.PI * j) / n));
}

/** The blob turned by `alpha`: `c_k e^{-ikα}`. */
export function rotate(c: Spectrum, alpha: number): Spectrum {
  const out = spectrum(size(c));
  for (let k = 0; k < size(c); k++) {
    const [cs, sn] = [Math.cos(k * alpha), -Math.sin(k * alpha)];
    out.re[k] = c.re[k]! * cs - c.im[k]! * sn;
    out.im[k] = c.re[k]! * sn + c.im[k]! * cs;
  }
  return out;
}

/** The same `|c_k|` with every `c_k`, `k ≥ 1`, at a random phase. */
export function scramblePhases(c: Spectrum, rng: Rng): Spectrum {
  const out = copy(c);
  for (let k = 1; k < size(c); k++) {
    const r = Math.hypot(c.re[k]!, c.im[k]!);
    const a = 2 * Math.PI * rng.uniform();
    out.re[k] = r * Math.cos(a);
    out.im[k] = r * Math.sin(a);
  }
  return out;
}

/** `|c_k|^2`, the power spectrum. */
export function power(c: Spectrum): Float64Array {
  return Float64Array.from({ length: size(c) }, (_, k) => c.re[k]! ** 2 + c.im[k]! ** 2);
}

export type Complex = readonly [re: number, im: number];

/** `c_{k1} c_{k2} conj(c_{(k1 + k2) mod m})`, the bispectrum entry `(k1, k2)`. */
export function triple(c: Spectrum, k1: number, k2: number): Complex {
  const m = size(c);
  const k = (k1 + k2) % m;
  const [ar, ai, br, bi] = [c.re[k1]!, c.im[k1]!, c.re[k2]!, c.im[k2]!];
  const [pr, pi] = [ar * br - ai * bi, ar * bi + ai * br];
  const [cr, ci] = [c.re[k]!, -c.im[k]!];
  return [pr * cr - pi * ci, pr * ci + pi * cr];
}

/** Every `(k1, k2)` with `k1 <= k2`, the full bispectrum of `CnonCn(selective=False)`. */
export function fullPairs(m: number): [number, number][] {
  const out: [number, number][] = [];
  for (let k1 = 0; k1 < m; k1++) for (let k2 = k1; k2 < m; k2++) out.push([k1, k2]);
  return out;
}

/** Products of a band-limited blob that need no wrap-around: `k1 <= k2`, `k1 + k2 < m`. */
export function bandPairs(m: number): [number, number][] {
  return fullPairs(m).filter(([k1, k2]) => k1 + k2 < m);
}

/** `(0, 0), (0, 1), (1, 1), (1, 2), ..., (1, m - 2)`: the chain of `CnonCn(selective=True)`. */
export function chainPairs(m: number): [number, number][] {
  if (m < 2) throw new Error('the chain needs at least two coefficients');
  return [
    [0, 0],
    [0, 1],
    ...Array.from({ length: m - 2 }, (_, i) => [1, i + 1] as [number, number]),
  ];
}

export function bispectrum(c: Spectrum, pairs: [number, number][]): Spectrum {
  const out = spectrum(pairs.length);
  pairs.forEach(([k1, k2], i) => {
    [out.re[i], out.im[i]] = triple(c, k1, k2);
  });
  return out;
}

export const chain = (c: Spectrum): Spectrum => bispectrum(c, chainPairs(size(c)));

/** The ring's full bispectrum size, `n (n + 1) / 2`. */
export const fullCount = (n: number): number => (n * (n + 1)) / 2;

const div = (a: Complex, b: Complex): Complex => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};
const mul = (a: Complex, b: Complex): Complex => [
  a[0] * b[0] - a[1] * b[1],
  a[0] * b[1] + a[1] * b[0],
];

const PIVOT = 1e-10;

/**
 * Walks the chain back (`CnonCn.invert`, Algorithm 1 of Mataigne et al. 2024): `c_0` from
 * `β_00`, the length of `c_1` from `β_01`, then `c_{k+1} = conj(β_1k / (c_1 c_k))`. The angle of
 * `c_1` is free, `phi`; changing it turns the result. Throws on a zero pivot `c_k`, `k <= m - 2`.
 */
export function invertChain(beta: Spectrum, phi = 0): Spectrum {
  const m = size(beta);
  const b = (i: number): Complex => [beta.re[i]!, beta.im[i]!];
  const c = spectrum(m);
  const r0 = Math.cbrt(Math.hypot(...b(0)));
  const a0 = Math.atan2(beta.im[0]!, beta.re[0]!);
  [c.re[0], c.im[0]] = [r0 * Math.cos(a0), r0 * Math.sin(a0)];
  if (r0 < PIVOT) throw new Error('cannot invert: c_0 is zero');
  const r1 = Math.sqrt(Math.hypot(...div(b(1), [c.re[0]!, c.im[0]!])));
  if (r1 < PIVOT) throw new Error('cannot invert: c_1 is zero');
  [c.re[1], c.im[1]] = [r1 * Math.cos(phi), r1 * Math.sin(phi)];
  for (let k = 1; k < m - 1; k++) {
    const ck: Complex = [c.re[k]!, c.im[k]!];
    if (Math.hypot(...ck) < PIVOT) throw new Error(`cannot invert: c_${k} is zero`);
    const q = div(b(k + 1), mul([c.re[1]!, c.im[1]!], ck));
    [c.re[k + 1], c.im[k + 1]] = [q[0], -q[1]];
  }
  return c;
}

/**
 * The turn `α` that best lines `c` up with `target` (least squares on the outline): a grid over
 * the circle refined by golden-section search.
 */
export function bestTurn(c: Spectrum, target: Spectrum): number {
  const m = Math.min(size(c), size(target));
  // Overlap of rotate(c, α) with target: Re Σ w_k c_k e^{-ikα} conj(t_k).
  const score = (alpha: number): number => {
    let s = 0;
    for (let k = 0; k < m; k++) {
      const [ca, sa] = [Math.cos(k * alpha), Math.sin(k * alpha)];
      const [xr, xi] = [c.re[k]! * ca + c.im[k]! * sa, c.im[k]! * ca - c.re[k]! * sa];
      s += (k ? 2 : 1) * (xr * target.re[k]! + xi * target.im[k]!);
    }
    return s;
  };
  const steps = 64 * m;
  let best = 0;
  for (let i = 1; i < steps; i++) {
    const a = (2 * Math.PI * i) / steps;
    if (score(a) > score(best)) best = a;
  }
  const g = (Math.sqrt(5) - 1) / 2;
  let [lo, hi] = [best - (2 * Math.PI) / steps, best + (2 * Math.PI) / steps];
  for (let i = 0; i < 40; i++) {
    const [a, b] = [hi - g * (hi - lo), lo + g * (hi - lo)];
    if (score(a) > score(b)) hi = b;
    else lo = a;
  }
  const a = (lo + hi) / 2;
  return ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
}
