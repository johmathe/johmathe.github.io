/**
 * The augmented selective SO(3) invariant of `bispectrum.SO3onS2(selective=True)` and its exact
 * Jacobian. `blog/scripts/export_tables.py` exports the library's triples; the Clebsch-Gordan
 * tables are computed here from them.
 *
 * Bispectral entry `(l1, l2, l)`: `sum_k cg_k F[fl1_k] F[fl2_k] conj(F[fl_k])`. On real signals it
 * is real when `l1 + l2 + l` is even and imaginary when odd, so the solver keeps one real number
 * per entry. CG power entry `(l1, l2, l)`: `||(F_l1 (x) F_l2)|_l||^2`.
 */

import { dim, flatMap, toFlat, type FlatMap } from './basis';
import { appendCouplings } from './cg';

export type FeatureTag =
  'bootstrap' | 'power' | 'budget_self_coupling' | 'mandatory_even_self' | 'cg_power';

export interface InvariantManifest {
  lmax: number;
  n_entries: number;
  n_bispec: number;
  triples: [number, number, number][];
  tags: FeatureTag[];
  row_degree: number[];
}

export interface InvariantTables {
  lmax: number;
  nBispec: number;
  triples: [number, number, number][];
  tags: FeatureTag[];
  rowDegree: Int32Array;
  offsets: Int32Array;
  cg: Float64Array;
  fl1: Uint16Array;
  fl2: Uint16Array;
  fl: Uint16Array;
}

/**
 * The sparse CG tables of the manifest's triples, CSR by output entry, as
 * `SO3onS2(selective=True)` builds them. `fl1`, `fl2`, `fl` index the flat complex coefficient
 * vector, `l**2 + l + m`.
 */
export function buildTables(manifest: InvariantManifest): InvariantTables {
  const out = { cg: [] as number[], fl1: [] as number[], fl2: [] as number[], fl: [] as number[] };
  const offsets = [0];
  for (const [l1, l2, l] of manifest.triples) {
    appendCouplings(l1, l2, l, out);
    offsets.push(out.cg.length);
  }
  return {
    lmax: manifest.lmax,
    nBispec: manifest.n_bispec,
    triples: manifest.triples,
    tags: manifest.tags,
    rowDegree: Int32Array.from(manifest.row_degree),
    offsets: Int32Array.from(offsets),
    cg: Float64Array.from(out.cg),
    fl1: Uint16Array.from(out.fl1),
    fl2: Uint16Array.from(out.fl2),
    fl: Uint16Array.from(out.fl),
  };
}

/** `sum_k cg_k F[fl1_k] F[fl2_k] conj(F[fl_k])` over the table rows `[k0, k1)`, as `[Re, Im]`. */
export function tripleSum(
  t: Pick<InvariantTables, 'cg' | 'fl1' | 'fl2' | 'fl'>,
  k0: number,
  k1: number,
  fre: Float64Array,
  fim: Float64Array,
): [number, number] {
  let sr = 0;
  let si = 0;
  for (let k = k0; k < k1; k++) {
    const i1 = t.fl1[k]!;
    const i2 = t.fl2[k]!;
    const i3 = t.fl[k]!;
    const g = t.cg[k]!;
    const abr = fre[i1]! * fre[i2]! - fim[i1]! * fim[i2]!;
    const abi = fre[i1]! * fim[i2]! + fim[i1]! * fre[i2]!;
    sr += g * (abr * fre[i3]! + abi * fim[i3]!);
    si += g * (abi * fre[i3]! - abr * fim[i3]!);
  }
  return [sr, si];
}

export class Invariant {
  readonly dim: number;
  /** Number of output entries at this band-limit. */
  readonly n: number;
  readonly triples: [number, number, number][];
  readonly tags: FeatureTag[];
  /** Largest degree in each entry's triple: the march solves for it at that degree. */
  readonly rowDegree: Int32Array;
  readonly isPower: Uint8Array;
  /** 1 when the bispectral entry is imaginary on real signals. */
  readonly imagPart: Uint8Array;
  /** Output index of each entry in the exported (largest band-limit) table. */
  private readonly src: Int32Array;
  private readonly fm: FlatMap;
  private readonly fre: Float64Array;
  private readonly fim: Float64Array;
  private readonly colOf: Int32Array;
  private readonly cRe: Float64Array;
  private readonly cIm: Float64Array;

  constructor(
    private readonly t: InvariantTables,
    readonly lmax: number,
  ) {
    if (lmax > t.lmax) throw new Error(`tables only go to lmax ${t.lmax}, asked for ${lmax}`);
    const src: number[] = [];
    t.rowDegree.forEach((deg, i) => {
      if (deg <= lmax) src.push(i);
    });
    this.src = Int32Array.from(src);
    this.n = src.length;
    this.dim = dim(lmax);
    this.triples = src.map((i) => t.triples[i]!);
    this.tags = src.map((i) => t.tags[i]!);
    this.rowDegree = Int32Array.from(src, (i) => t.rowDegree[i]!);
    this.isPower = Uint8Array.from(src, (i) => (i >= t.nBispec ? 1 : 0));
    this.imagPart = Uint8Array.from(this.triples, ([a, b, c], i) =>
      !this.isPower[i] && (a + b + c) % 2 === 1 ? 1 : 0,
    );
    this.fm = flatMap(lmax);
    this.fre = new Float64Array(this.dim);
    this.fim = new Float64Array(this.dim);
    this.colOf = new Int32Array(this.dim);
    this.cRe = new Float64Array(2 * lmax + 1);
    this.cIm = new Float64Array(2 * lmax + 1);
  }

  /** Entry indices whose row degree satisfies `pred`. */
  rowsWhere(pred: (degree: number) => boolean): Int32Array {
    const out: number[] = [];
    this.rowDegree.forEach((d, i) => {
      if (pred(d)) out.push(i);
    });
    return Int32Array.from(out);
  }

  get allRows(): Int32Array {
    return Int32Array.from({ length: this.n }, (_, i) => i);
  }

  /** Complex value of entry `e` from the current flat coefficients. */
  private entry(e: number): [number, number] {
    const { t, fre, fim } = this;
    const k0 = t.offsets[this.src[e]!]!;
    const k1 = t.offsets[this.src[e]! + 1]!;
    if (this.isPower[e]) {
      const lo = this.triples[e]![2];
      const base = lo * lo;
      const cRe = this.cRe.fill(0, 0, 2 * lo + 1);
      const cIm = this.cIm.fill(0, 0, 2 * lo + 1);
      for (let k = k0; k < k1; k++) {
        const i1 = t.fl1[k]!;
        const i2 = t.fl2[k]!;
        const g = t.cg[k]!;
        const ci = t.fl[k]! - base;
        cRe[ci]! += g * (fre[i1]! * fre[i2]! - fim[i1]! * fim[i2]!);
        cIm[ci]! += g * (fre[i1]! * fim[i2]! + fim[i1]! * fre[i2]!);
      }
      let p = 0;
      for (let m = 0; m <= 2 * lo; m++) p += cRe[m]! ** 2 + cIm[m]! ** 2;
      return [p, 0];
    }
    return tripleSum(t, k0, k1, fre, fim);
  }

  /** The `SO3onS2` output as real `[Re (n), Im (n)]`, the layout of `InvariantModel.features`. */
  features(x: ArrayLike<number>): Float64Array {
    toFlat(x, this.fm, this.fre, this.fim);
    const out = new Float64Array(2 * this.n);
    for (let e = 0; e < this.n; e++) {
      const [re, im] = this.entry(e);
      out[e] = re;
      out[this.n + e] = im;
    }
    return out;
  }

  /** One real number per entry (the nonzero part on real signals), for `rows` or all entries. */
  values(x: ArrayLike<number>, rows: Int32Array = this.allRows, out?: Float64Array): Float64Array {
    toFlat(x, this.fm, this.fre, this.fim);
    const res = out ?? new Float64Array(rows.length);
    for (let r = 0; r < rows.length; r++) {
      const e = rows[r]!;
      const v = this.entry(e);
      res[r] = v[this.imagPart[e]!]!;
    }
    return res;
  }

  /**
   * Values and the exact real Jacobian of `values` at `x`, restricted to `rows` and to the dof
   * columns `dofs`. `jac` is row-major `(rows.length, dofs.length)`.
   */
  jacobian(
    x: ArrayLike<number>,
    rows: Int32Array,
    dofs: Int32Array,
    val: Float64Array = new Float64Array(rows.length),
    jac: Float64Array = new Float64Array(rows.length * dofs.length),
  ): { val: Float64Array; jac: Float64Array } {
    const { t, fre, fim, colOf, fm } = this;
    const fdof = fm.dof;
    const wRe = fm.wRe;
    const wIm = fm.wIm;
    toFlat(x, fm, fre, fim);
    colOf.fill(-1);
    dofs.forEach((d, c) => (colOf[d] = c));
    const nc = dofs.length;
    jac.fill(0);

    for (let r = 0; r < rows.length; r++) {
      const e = rows[r]!;
      const k0 = t.offsets[this.src[e]!]!;
      const k1 = t.offsets[this.src[e]! + 1]!;
      const row = r * nc;

      if (this.isPower[e]) {
        const lo = this.triples[e]![2];
        const base = lo * lo;
        const cRe = this.cRe.fill(0, 0, 2 * lo + 1);
        const cIm = this.cIm.fill(0, 0, 2 * lo + 1);
        for (let k = k0; k < k1; k++) {
          const i1 = t.fl1[k]!;
          const i2 = t.fl2[k]!;
          const g = t.cg[k]!;
          const ci = t.fl[k]! - base;
          cRe[ci]! += g * (fre[i1]! * fre[i2]! - fim[i1]! * fim[i2]!);
          cIm[ci]! += g * (fre[i1]! * fim[i2]! + fim[i1]! * fre[i2]!);
        }
        let p = 0;
        for (let m = 0; m <= 2 * lo; m++) p += cRe[m]! ** 2 + cIm[m]! ** 2;
        val[r] = p;
        // dP/dx = 2 Re sum_k conj(C_m) cg (dF[fl1] F[fl2] + F[fl1] dF[fl2])
        for (let k = k0; k < k1; k++) {
          const ci = t.fl[k]! - base;
          const g2 = 2 * t.cg[k]!;
          const ccr = cRe[ci]!;
          const cci = -cIm[ci]!;
          for (let side = 0; side < 2; side++) {
            const iw = side === 0 ? t.fl1[k]! : t.fl2[k]!;
            const io = side === 0 ? t.fl2[k]! : t.fl1[k]!;
            // q = conj(C) * F[other]
            const qr = ccr * fre[io]! - cci * fim[io]!;
            const qi = ccr * fim[io]! + cci * fre[io]!;
            for (let s = 0; s < 2; s++) {
              const c = colOf[fdof[2 * iw + s]!]!;
              if (c < 0) continue;
              const wr = wRe[2 * iw + s]!;
              const wi = wIm[2 * iw + s]!;
              jac[row + c]! += g2 * (qr * wr - qi * wi);
            }
          }
        }
        continue;
      }

      const im = this.imagPart[e]!;
      let sr = 0;
      let si = 0;
      for (let k = k0; k < k1; k++) {
        const i1 = t.fl1[k]!;
        const i2 = t.fl2[k]!;
        const i3 = t.fl[k]!;
        const g = t.cg[k]!;
        const ar = fre[i1]!;
        const ai = fim[i1]!;
        const br = fre[i2]!;
        const bi = fim[i2]!;
        const cr = fre[i3]!;
        const ci = -fim[i3]!;
        // Products of pairs: b conj(c), a conj(c), a b.
        const bcr = br * cr - bi * ci;
        const bci = br * ci + bi * cr;
        const acr = ar * cr - ai * ci;
        const aci = ar * ci + ai * cr;
        const abr = ar * br - ai * bi;
        const abi = ar * bi + ai * br;
        sr += g * (abr * cr - abi * ci);
        si += g * (abr * ci + abi * cr);
        for (let s = 0; s < 2; s++) {
          let c = colOf[fdof[2 * i1 + s]!]!;
          if (c >= 0) {
            const wr = wRe[2 * i1 + s]!;
            const wi = wIm[2 * i1 + s]!;
            jac[row + c]! += g * (im ? wr * bci + wi * bcr : wr * bcr - wi * bci);
          }
          c = colOf[fdof[2 * i2 + s]!]!;
          if (c >= 0) {
            const wr = wRe[2 * i2 + s]!;
            const wi = wIm[2 * i2 + s]!;
            jac[row + c]! += g * (im ? wr * aci + wi * acr : wr * acr - wi * aci);
          }
          c = colOf[fdof[2 * i3 + s]!]!;
          if (c >= 0) {
            // d conj(F[fl]) = conj(w)
            const wr = wRe[2 * i3 + s]!;
            const wi = -wIm[2 * i3 + s]!;
            jac[row + c]! += g * (im ? wr * abi + wi * abr : wr * abr - wi * abi);
          }
        }
      }
      val[r] = im ? si : sr;
    }
    return { val, jac };
  }
}
