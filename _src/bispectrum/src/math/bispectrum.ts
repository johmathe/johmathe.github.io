/**
 * The full SO(3) bispectrum of a band-limited real signal, `SO3onS2(selective=False)`: every
 * `beta(l1, l2, l) = sum cg F[l1, m1] F[l2, m2] conj(F[l, m1 + m2])` with `l1 <= l2` and
 * `l2 - l1 <= l <= min(l1 + l2, lmax)`, in the order of `_build_full_index_map` (616 entries at
 * `lmax = 12`, 1,124 at 15). The selective invariant keeps a `Theta(L^2)` subset of these.
 *
 * The Clebsch-Gordan coefficients are computed here instead of being shipped. As in `Invariant`,
 * an entry is real when `l1 + l2 + l` is even and imaginary when odd, so it is one real number.
 */

import { flatMap, toFlat, type FlatMap } from './basis';
import { appendCouplings } from './cg';
import { tripleSum, type InvariantTables } from './invariant';

export class FullBispectrum {
  readonly n: number;
  readonly triples: [number, number, number][] = [];
  /** Largest degree of each triple: the entry is known once that degree is. */
  readonly maxDegree: Int32Array;
  /** 1 when the entry is imaginary on real signals. */
  readonly imagPart: Uint8Array;
  /** 1 when the entry vanishes on every real signal: odd `l1 + l2 + l` with a repeated degree. */
  readonly vanishes: Uint8Array;
  private readonly offsets: Int32Array;
  private readonly table: Pick<InvariantTables, 'cg' | 'fl1' | 'fl2' | 'fl'>;
  private readonly fm: FlatMap;
  private readonly fre: Float64Array;
  private readonly fim: Float64Array;

  constructor(readonly lmax: number) {
    const out = {
      cg: [] as number[],
      fl1: [] as number[],
      fl2: [] as number[],
      fl: [] as number[],
    };
    const offsets = [0];
    for (let l1 = 0; l1 <= lmax; l1++) {
      for (let l2 = l1; l2 <= lmax; l2++) {
        for (let l = l2 - l1; l <= Math.min(l1 + l2, lmax); l++) {
          this.triples.push([l1, l2, l]);
          appendCouplings(l1, l2, l, out);
          offsets.push(out.cg.length);
        }
      }
    }
    this.n = this.triples.length;
    this.offsets = Int32Array.from(offsets);
    this.table = {
      cg: Float64Array.from(out.cg),
      fl1: Uint16Array.from(out.fl1),
      fl2: Uint16Array.from(out.fl2),
      fl: Uint16Array.from(out.fl),
    };
    this.maxDegree = Int32Array.from(this.triples, (t) => Math.max(...t));
    this.imagPart = Uint8Array.from(this.triples, ([a, b, c]) => (a + b + c) % 2);
    this.vanishes = Uint8Array.from(this.triples, ([a, b, c], i) =>
      this.imagPart[i] && (a === b || b === c || a === c) ? 1 : 0,
    );
    this.fm = flatMap(lmax);
    this.fre = new Float64Array((lmax + 1) ** 2);
    this.fim = new Float64Array((lmax + 1) ** 2);
  }

  /** One real number per entry for the signal with real coordinates `x`. */
  values(x: ArrayLike<number>, out: Float64Array = new Float64Array(this.n)): Float64Array {
    toFlat(x, this.fm, this.fre, this.fim);
    for (let e = 0; e < this.n; e++) {
      const v = tripleSum(this.table, this.offsets[e]!, this.offsets[e + 1]!, this.fre, this.fim);
      out[e] = v[this.imagPart[e]!]!;
    }
    return out;
  }
}
