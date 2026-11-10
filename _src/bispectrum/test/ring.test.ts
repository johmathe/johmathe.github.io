import { describe, expect, it } from 'vitest';

import {
  bandPairs,
  bestTurn,
  bispectrum,
  chain,
  chainPairs,
  dft,
  fullCount,
  fullPairs,
  harmonics,
  invertChain,
  outline,
  power,
  rotate,
  scramblePhases,
  triple,
  type Spectrum,
} from '../src/math/ring';
import { Rng } from '../src/math/rng';
import { fixture, maxAbsDiff } from './helpers';

interface C {
  re: number[];
  im: number[];
}

interface RingFixture {
  n: number;
  shift: number;
  signals: number[][];
  selective: C[];
  selective_index_map: [number, number][];
  full: C[];
  full_index_map: [number, number][];
  inverted: C[];
  selective_shifted: C[];
}

const ring = fixture<RingFixture>('ring.json');

const flat = (c: Spectrum | C): number[] => [...c.re, ...c.im];

/** A smooth blob: mean 1, hands decaying with k. */
function blob(K: number, rng: Rng): Spectrum {
  const c = { re: new Float64Array(K + 1), im: new Float64Array(K + 1) };
  c.re[0] = 1;
  for (let k = 1; k <= K; k++) {
    c.re[k] = (0.25 * rng.normal()) / k;
    c.im[k] = (0.25 * rng.normal()) / k;
  }
  return c;
}

describe('ring math against CnonCn', () => {
  it('uses the same index maps', () => {
    expect(chainPairs(ring.n)).toEqual(ring.selective_index_map);
    expect(fullPairs(ring.n)).toEqual(ring.full_index_map);
    expect(fullPairs(ring.n).length).toBe(fullCount(ring.n));
  });

  it.each(ring.signals.map((f, i) => [i, f] as const))('signal %i: chain and full', (i, f) => {
    const c = dft(f);
    expect(maxAbsDiff(flat(chain(c)), flat(ring.selective[i]!))).toBeLessThan(1e-9);
    expect(maxAbsDiff(flat(bispectrum(c, fullPairs(ring.n))), flat(ring.full[i]!))).toBeLessThan(
      1e-9,
    );
  });

  it.each(ring.signals.map((_, i) => [i] as const))(
    'signal %i: the chain walks back to CnonCn.invert, which is the signal up to a turn',
    (i) => {
      const beta = {
        re: Float64Array.from(ring.selective[i]!.re),
        im: Float64Array.from(ring.selective[i]!.im),
      };
      const back = invertChain(beta);
      // CnonCn.invert returns the inverse DFT of the coefficients it recovers.
      expect(maxAbsDiff(flat(back), flat(dftComplex(ring.inverted[i]!)))).toBeLessThan(1e-8);
      const c = dft(ring.signals[i]!);
      const alpha = Math.atan2(c.im[1]!, c.re[1]!);
      expect(maxAbsDiff(flat(rotate(c, alpha)), flat(back))).toBeLessThan(1e-8);
    },
  );

  it('a cyclic shift is a turn by 2π shift / n and leaves the chain unchanged', () => {
    const f = ring.signals[0]!;
    const shifted = f.map((_, j) => f[(j - ring.shift + ring.n) % ring.n]!);
    const turned = rotate(dft(f), (2 * Math.PI * ring.shift) / ring.n);
    expect(maxAbsDiff(flat(dft(shifted)), flat(turned))).toBeLessThan(1e-9);
    expect(maxAbsDiff(flat(chain(dft(shifted))), flat(ring.selective_shifted[0]!))).toBeLessThan(
      1e-9,
    );
  });
});

/** Forward DFT of a complex signal given as `{re, im}`. */
function dftComplex(x: C): Spectrum {
  const n = x.re.length;
  const out = { re: new Float64Array(n), im: new Float64Array(n) };
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * j * k) / n;
      out.re[k] += x.re[j]! * Math.cos(a) - x.im[j]! * Math.sin(a);
      out.im[k] += x.re[j]! * Math.sin(a) + x.im[j]! * Math.cos(a);
    }
  }
  return out;
}

describe('a blob and its hands', () => {
  const rng = new Rng(3);
  const K = 7;
  const c = blob(K, rng);

  it('samples and re-reads its hands', () => {
    expect(maxAbsDiff(flat(harmonics(outline(c, 64), K)), flat(c))).toBeLessThan(1e-12);
  });

  it('turning by α turns hand k by k α and the outline by α', () => {
    const alpha = 0.7;
    const t = rotate(c, alpha);
    for (let k = 1; k <= K; k++) {
      const d = Math.atan2(c.im[k]!, c.re[k]!) - Math.atan2(t.im[k]!, t.re[k]!);
      expect(Math.cos(d - k * alpha)).toBeCloseTo(1, 12);
    }
    const n = 360;
    const s = Math.round((alpha / (2 * Math.PI)) * n);
    const turned = outline(rotate(c, (2 * Math.PI * s) / n), n);
    const r = outline(c, n);
    expect(
      maxAbsDiff(
        turned,
        Float64Array.from(r, (_, j) => r[(j - s + n) % n]!),
      ),
    ).toBeLessThan(1e-12);
  });

  it('keeps the power spectrum and every triple product under a turn', () => {
    const t = rotate(c, 2.1);
    expect(maxAbsDiff(power(t), power(c))).toBeLessThan(1e-12);
    for (const [k1, k2] of bandPairs(K + 1)) {
      expect(maxAbsDiff(triple(t, k1, k2), triple(c, k1, k2))).toBeLessThan(1e-12);
    }
  });

  it('a phase scramble keeps the power spectrum and breaks the triple products', () => {
    const s = scramblePhases(c, new Rng(9));
    expect(maxAbsDiff(power(s), power(c))).toBeLessThan(1e-12);
    expect(maxAbsDiff(flat(chain(s)), flat(chain(c)))).toBeGreaterThan(1e-3);
  });

  it('the free angle of the chain is the turn', () => {
    const turned = rotate(c, 1.3);
    const beta = chain(turned);
    for (const phi of [0, 0.4, -2.5]) {
      const back = invertChain(beta, phi);
      const alpha = bestTurn(back, turned);
      expect(maxAbsDiff(flat(rotate(back, alpha)), flat(turned))).toBeLessThan(1e-6);
    }
    expect(bestTurn(turned, c)).toBeCloseTo(2 * Math.PI - 1.3, 6);
  });

  it('counts: n hands, n chain entries, n (n + 1) / 2 products', () => {
    expect(chain(c).re.length).toBe(K + 1);
    expect(fullCount(256 * 256)).toBe(2_147_516_416);
  });
});
