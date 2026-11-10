import { describe, expect, it } from 'vitest';

import { FullBispectrum } from '../src/math/bispectrum';
import { clebschGordan, wigner3j } from '../src/math/cg';
import { Invariant } from '../src/math/invariant';
import { Rng } from '../src/math/rng';
import { randomRotation, rotateDofs } from '../src/math/rotation';
import { loadTables, maxAbsDiff, randn } from './helpers';

const LMAX = 12;
const tables = loadTables();
const key = (t: readonly number[]): string => t.join(',');

describe('Clebsch-Gordan coefficients', () => {
  it('match closed forms', () => {
    expect(wigner3j(1, 1, 0, 0, 0, 0)).toBeCloseTo(-1 / Math.sqrt(3), 14);
    expect(wigner3j(2, 2, 2, 0, 0, 0)).toBeCloseTo(-Math.sqrt(2 / 35), 14);
    expect(clebschGordan(1, 1, 1, -1, 2, 0)).toBeCloseTo(1 / Math.sqrt(6), 14);
    expect(clebschGordan(3, 2, 2, -1, 4, 2)).toBeCloseTo(0, 15);
  });

  it('match the tables exported from the Python library', () => {
    let worst = 0;
    let checked = 0;
    for (let e = 0; e < tables.nBispec; e++) {
      const [l1, l2, l] = tables.triples[e]!;
      for (let k = tables.offsets[e]!; k < tables.offsets[e + 1]!; k++) {
        const m1 = tables.fl1[k]! - l1 * l1 - l1;
        const m2 = tables.fl2[k]! - l2 * l2 - l2;
        const m = tables.fl[k]! - l * l - l;
        worst = Math.max(worst, Math.abs(clebschGordan(l1, m1, l2, m2, l, m) - tables.cg[k]!));
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(10000);
    expect(worst).toBeLessThan(1e-12);
  });

  it('are orthonormal over (m1, m2) at the largest degrees', () => {
    const [l1, l2] = [LMAX, LMAX - 1];
    for (const [la, lb] of [
      [LMAX, LMAX],
      [LMAX, 7],
      [3, 3],
    ] as const) {
      for (let m = -Math.min(la, lb); m <= Math.min(la, lb); m++) {
        let dot = 0;
        for (let m1 = -l1; m1 <= l1; m1++) {
          if (Math.abs(m - m1) > l2) continue;
          dot +=
            clebschGordan(l1, m1, l2, m - m1, la, m) * clebschGordan(l1, m1, l2, m - m1, lb, m);
        }
        expect(dot).toBeCloseTo(la === lb ? 1 : 0, 12);
      }
    }
  });
});

describe('full bispectrum', () => {
  it('has the entry counts of the library', () => {
    expect(new FullBispectrum(10).n).toBe(381);
    expect(new FullBispectrum(12).n).toBe(616);
    expect(new FullBispectrum(15).n).toBe(1124);
  });

  const full = new FullBispectrum(LMAX);
  const index = new Map(full.triples.map((t, i) => [key(t), i]));
  const x = randn(full.lmax ** 2 + 2 * full.lmax + 1, 7);
  const v = full.values(x);
  const scale = Math.max(...v.map(Math.abs));

  it('contains every selective bispectral entry, with the same value', () => {
    const inv = new Invariant(tables, LMAX);
    const sel = inv.values(x);
    let checked = 0;
    inv.triples.forEach((t, i) => {
      if (inv.isPower[i]) return;
      const j = index.get(key(t));
      expect(j, `selective triple ${key(t)}`).toBeDefined();
      expect(Math.abs(v[j!]! - sel[i]!)).toBeLessThan(1e-10 * scale);
      checked++;
    });
    expect(checked).toBe(196);
  });

  it('is rotation invariant', () => {
    const rot = randomRotation(new Rng(3));
    expect(maxAbsDiff(full.values(rotateDofs(x, rot)), v)).toBeLessThan(1e-10 * scale);
  });

  it('vanishes where the library says it does, and only there', () => {
    let zeros = 0;
    full.vanishes.forEach((z, i) => {
      if (z) {
        zeros++;
        expect(Math.abs(v[i]!)).toBeLessThan(1e-11 * scale);
      } else {
        expect(Math.abs(v[i]!)).toBeGreaterThan(1e-8 * scale);
      }
    });
    expect(zeros).toBe(108);
  });

  it('satisfies the permutation identity on real signals', () => {
    let checked = 0;
    full.triples.forEach(([l1, l2, l], i) => {
      const j = index.get(key([l1, l, l2]));
      if (j === undefined || j === i) return;
      const factor = (l1 % 2 ? -1 : 1) * Math.sqrt((2 * l + 1) / (2 * l2 + 1));
      expect(Math.abs(v[i]! - factor * v[j]!)).toBeLessThan(1e-10 * scale);
      checked++;
    });
    expect(checked).toBeGreaterThan(100);
  });
});
