/**
 * Wigner 3j symbols and Clebsch-Gordan coefficients for integer degrees, by Racah's formula, as
 * `bispectrum._cg` computes them: log-factorials below `l1 + l2 + l = 64`, an exact integer sum
 * above, where the alternating series cancels too much for doubles.
 */

/** `bispectrum._cg._EXACT_RACAH_MIN_SUM`. */
const EXACT_RACAH_MIN_SUM = 64;

/** `LOG_FACT[n] = log(n!)`, accumulated as `bispectrum._cg._LOG_FACT` is. */
const LOG_FACT: number[] = [0];
const FACT: bigint[] = [1n];

function logFact(n: number): number {
  while (LOG_FACT.length <= n) LOG_FACT.push(LOG_FACT.at(-1)! + Math.log(LOG_FACT.length));
  return LOG_FACT[n]!;
}

function fact(n: number): bigint {
  while (FACT.length <= n) FACT.push(FACT.at(-1)! * BigInt(FACT.length));
  return FACT[n]!;
}

const sign = (k: number): number => (k & 1 ? -1 : 1);

const comb = (n: number, k: number): bigint => fact(n) / (fact(k) * fact(n - k));

/** `p / q` for positive integers too large for a double, to within an ulp. */
function ratio(p: bigint, q: bigint): number {
  const shift = Math.max(0, 64 - (p.toString(2).length - q.toString(2).length));
  return Number((p << BigInt(shift)) / q) * 2 ** -shift;
}

/** `bispectrum._cg._wigner3j_exact`: Racah's numerator as an exact integer. */
function wigner3jExact(j1: number, j2: number, j3: number, m1: number, m2: number): number {
  const m3 = -m1 - m2;
  const [a, b, c] = [j1 + j2 - j3, j1 - j2 + j3, -j1 + j2 + j3];
  let num = 0n;
  const hi = Math.min(a, j1 - m1, j2 + m2);
  for (let t = Math.max(0, j2 - j3 - m1, j1 - j3 + m2); t <= hi; t++) {
    const term = comb(a, t) * comb(b, j1 - m1 - t) * comb(c, j2 + m2 - t);
    num += t & 1 ? -term : term;
  }
  if (num === 0n) return 0;
  const f = (j: number, m: number): bigint => fact(j + m) * fact(j - m);
  const den = fact(j1 + j2 + j3 + 1) * fact(a) * fact(b) * fact(c);
  const mag = Math.sqrt(ratio(num * num * f(j1, m1) * f(j2, m2) * f(j3, m3), den));
  return sign(j1 - j2 - m3) * (num > 0n ? 1 : -1) * mag;
}

/**
 * The Racah sum of `<l1 m1; l2 m2 | l m> / sqrt(2l + 1)` in log-factorials, term for term as
 * `bispectrum._cg.compute_sparse_cg_entry`: its 3j and CG phases cancel.
 */
function racahLog(l1: number, m1: number, l2: number, m2: number, l: number, m: number): number {
  const logTri =
    logFact(l1 + l2 - l) + logFact(l1 - l2 + l) + logFact(-l1 + l2 + l) - logFact(l1 + l2 + l + 1);
  const logPre =
    0.5 * logTri +
    0.5 *
      (logFact(l1 + m1) +
        logFact(l1 - m1) +
        logFact(l2 + m2) +
        logFact(l2 - m2) +
        logFact(l - m) +
        logFact(l + m));
  let total = 0;
  const hi = Math.min(l1 + l2 - l, l1 - m1, l2 + m2);
  for (let t = Math.max(0, l2 - l - m1, l1 - l + m2); t <= hi; t++) {
    const logDen =
      logFact(t) +
      logFact(l1 + l2 - l - t) +
      logFact(l1 - m1 - t) +
      logFact(l2 + m2 - t) +
      logFact(l - l2 + m1 + t) +
      logFact(l - l1 - m2 + t);
    const v = Math.exp(logPre - logDen);
    total = t & 1 ? total - v : total + v;
  }
  return total;
}

/** `<l1 m1; l2 m2 | l m>`. */
export function clebschGordan(
  l1: number,
  m1: number,
  l2: number,
  m2: number,
  l: number,
  m: number,
): number {
  if (m1 + m2 !== m) return 0;
  if (Math.abs(m1) > l1 || Math.abs(m2) > l2 || Math.abs(m) > l) return 0;
  if (l < Math.abs(l1 - l2) || l > l1 + l2) return 0;
  const scale = Math.sqrt(2 * l + 1);
  if (l1 + l2 + l >= EXACT_RACAH_MIN_SUM) {
    return sign(l1 - l2 + m) * scale * wigner3jExact(l1, l2, l, m1, m2);
  }
  return scale * racahLog(l1, m1, l2, m2, l, m);
}

export function wigner3j(
  j1: number,
  j2: number,
  j3: number,
  m1: number,
  m2: number,
  m3: number,
): number {
  return (sign(j1 - j2 - m3) * clebschGordan(j1, m1, j2, m2, j3, -m3)) / Math.sqrt(2 * j3 + 1);
}

/** Flat index `l**2 + l + m` of the complex coefficient `F[l, m]`. */
const flat = (l: number, m: number): number => l * l + l + m;

/**
 * The nonzero `<l1 m1; l2 m2 | l m>` of one triple, `m` then `m1` ascending, appended to `out`
 * with the flat indices of `F[l1, m1]`, `F[l2, m2]` and `F[l, m]`.
 */
export function appendCouplings(
  l1: number,
  l2: number,
  l: number,
  out: { cg: number[]; fl1: number[]; fl2: number[]; fl: number[] },
): void {
  for (let m = -l; m <= l; m++) {
    for (let m1 = Math.max(-l1, m - l2); m1 <= Math.min(l1, m + l2); m1++) {
      const g = clebschGordan(l1, m1, l2, m - m1, l, m);
      if (g === 0) continue;
      out.cg.push(g);
      out.fl1.push(flat(l1, m1));
      out.fl2.push(flat(l2, m - m1));
      out.fl.push(flat(l, m));
    }
  }
}
