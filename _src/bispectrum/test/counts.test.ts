import { describe, expect, it } from 'vitest';

import { dim } from '../src/math/basis';
import { FullBispectrum } from '../src/math/bispectrum';
import type { InvariantManifest } from '../src/math/invariant';
import { fullCount } from '../src/math/ring';
import { bytes, BYTES_PER_NUMBER, type Counts } from '../src/ui/wall';
import { cgTerms, distinctFull } from './checks';
import { loadDataJson } from './helpers';

const { sphere } = loadDataJson<Counts>('counts.json');
const at = (key: keyof Counts['sphere'], L: number): number => sphere[key][L - 1]!;

describe('counts', () => {
  it('match the paper on the sphere', () => {
    expect([at('selective', 15), at('full', 15)]).toEqual([380, 1124]);
    expect([at('selective', 16), at('full', 16)]).toEqual([426, 1341]);
    expect([at('selective', 64), at('full', 64)]).toEqual([5610, 70257]);
    // App. A.6: 2.67e8 and 1.08e7 terms at L = 64, a 25x reduction.
    expect(at('full_terms', 64) / 1e8).toBeCloseTo(2.67, 2);
    expect(at('selective_terms', 64) / 1e7).toBeCloseTo(1.08, 2);
    expect(Math.round(at('full_terms', 64) / at('selective_terms', 64))).toBe(25);
  });

  it('count each distinct entry of the full bispectrum once on a real signal', () => {
    // The term count behind the distinct entries is the one behind `full_terms`.
    for (const L of [12, 15, 64]) {
      let terms = 0;
      for (let a = 0; a <= L; a++) {
        for (let b = a; b <= L; b++) {
          for (let c = b - a; c <= Math.min(a + b, L); c++) terms += cgTerms(a, b, c);
        }
      }
      expect(terms).toBe(at('full_terms', L));
    }
    expect(distinctFull(12)).toEqual({ entries: 195, terms: 29_315 });
    expect(distinctFull(15)).toEqual({ entries: 352, terms: 79_328 });
    expect(distinctFull(64)).toEqual({ entries: 22_945, terms: 87_204_897 });
  });

  it('match the tables and the full bispectrum that run in the page', () => {
    const inv = loadDataJson<InvariantManifest>('invariant.json');
    for (let L = 1; L <= inv.lmax; L++) {
      expect(at('coefficients', L)).toBe(dim(L));
      expect(at('selective', L)).toBe(inv.row_degree.filter((d) => d <= L).length);
      expect(at('full', L)).toBe(new FullBispectrum(L).n);
    }
  });

  it('match the paper for images under shifts', () => {
    expect(fullCount(256 * 256)).toBe(2_147_516_416);
    expect(fullCount(32 * 32)).toBe(524_800);
    expect(fullCount(128)).toBe(8_256);
    // Figure 5: 32,768x fewer coefficients at |G| = 65,536.
    expect(Math.floor(fullCount(65_536) / 65_536)).toBe(32_768);
    expect(bytes(fullCount(256 * 256) * BYTES_PER_NUMBER)).toBe('17.2 GB');
  });
});
