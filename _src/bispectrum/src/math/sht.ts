/**
 * Real spherical harmonic transforms on the torch-harmonics equiangular grid.
 *
 * Matches `torch_harmonics.RealSHT` / `InverseRealSHT` with `grid='equiangular'`, `norm='ortho'`
 * and the Condon-Shortley phase: colatitudes `linspace(0, pi, nlat)` with Clenshaw-Curtis
 * weights, longitudes `2 pi j / nlon`, and coefficients on the half spectrum `m >= 0`.
 */

/** Half-spectrum coefficients `c[l][m]`, `0 <= m <= l <= lmax`, stored at `l * (lmax + 1) + m`. */
export interface HalfSpectrum {
  lmax: number;
  re: Float64Array;
  im: Float64Array;
}

export function halfSpectrum(lmax: number): HalfSpectrum {
  const n = (lmax + 1) * (lmax + 1);
  return { lmax, re: new Float64Array(n), im: new Float64Array(n) };
}

/** Clenshaw-Curtis weights on `[-1, 1]` for nodes `cos(pi j / (n - 1))`, `j = 0..n-1`. */
export function clenshawCurtisWeights(n: number): Float64Array {
  if (n < 2) throw new Error(`need at least 2 nodes, got ${n}`);
  const w = new Float64Array(n);
  if (n === 2) {
    w.fill(1);
    return w;
  }
  const big = n - 1;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 1; k <= Math.floor(big / 2); k++) {
      const b = 2 * k === big ? 1 : 2;
      s += (b / (4 * k * k - 1)) * Math.cos((2 * k * j * Math.PI) / big);
    }
    const c = j === 0 || j === big ? 1 : 2;
    w[j] = (c / big) * (1 - s);
  }
  return w;
}

/**
 * Orthonormal associated Legendre values with Condon-Shortley phase, the `legpoly` recurrence of
 * torch-harmonics. Returns `P[(k * (lmax + 1) + l) * (lmax + 1) + m]` for each `x[k]`.
 */
export function legendreTable(lmax: number, x: ArrayLike<number>): Float64Array {
  const n = lmax + 1;
  const out = new Float64Array(x.length * n * n);
  const p = new Float64Array(n * n);
  for (let k = 0; k < x.length; k++) {
    const xk = x[k]!;
    p.fill(0);
    p[0] = 1 / Math.sqrt(4 * Math.PI);
    for (let l = 1; l < n; l++) {
      p[(l - 1) * n + l] = Math.sqrt(2 * l + 1) * xk * p[(l - 1) * n + l - 1]!;
      p[l * n + l] =
        Math.sqrt(((2 * l + 1) * (1 + xk) * (1 - xk)) / 2 / l) * p[(l - 1) * n + l - 1]!;
    }
    for (let l = 2; l < n; l++) {
      for (let m = 0; m < l - 1; m++) {
        const a = Math.sqrt((((2 * l - 1) / (l - m)) * (2 * l + 1)) / (l + m));
        const b = Math.sqrt(
          (((((l + m - 1) / (l - m)) * (2 * l + 1)) / (2 * l - 3)) * (l - m - 1)) / (l + m),
        );
        p[m * n + l] = a * xk * p[m * n + l - 1]! - b * p[m * n + l - 2]!;
      }
    }
    const base = k * n * n;
    for (let l = 0; l < n; l++) {
      for (let m = 0; m <= l; m++) {
        out[base + l * n + m] = (m % 2 === 1 ? -1 : 1) * p[m * n + l]!;
      }
    }
  }
  return out;
}

/** Values of the band-limited signal `c` at unit vectors `dirs` (`3n`, xyz per point). */
export function evaluateAt(c: HalfSpectrum, dirs: ArrayLike<number>): Float64Array {
  const n = c.lmax + 1;
  const count = dirs.length / 3;
  const z = Float64Array.from({ length: count }, (_, i) =>
    Math.max(-1, Math.min(1, dirs[3 * i + 2]!)),
  );
  const leg = legendreTable(c.lmax, z);
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const phi = Math.atan2(dirs[3 * i + 1]!, dirs[3 * i]!);
    const base = i * n * n;
    let v = 0;
    for (let l = 0; l < n; l++) {
      v += leg[base + l * n]! * c.re[l * n]!;
      for (let m = 1; m <= l; m++) {
        const p = leg[base + l * n + m]!;
        v += 2 * p * (c.re[l * n + m]! * Math.cos(m * phi) - c.im[l * n + m]! * Math.sin(m * phi));
      }
    }
    out[i] = v;
  }
  return out;
}

/** Equiangular grid with precomputed Legendre and Fourier tables for a fixed band-limit. */
export class SphereGrid {
  readonly theta: Float64Array;
  readonly phi: Float64Array;
  /** Clenshaw-Curtis weights per colatitude row (they sum to 2). */
  readonly weights: Float64Array;
  private readonly leg: Float64Array;
  private readonly cosTab: Float64Array;
  private readonly sinTab: Float64Array;
  private readonly rowRe: Float64Array;
  private readonly rowIm: Float64Array;

  /**
   * `nodes = 'centered'` puts samples at cell centers, `((k + 1/2) pi / nlat, (j + 1/2) 2 pi /
   * nlon)`, so they coincide with texel centers of an `(nlat, nlon)` texture; such a grid can only
   * synthesize.
   */
  constructor(
    readonly nlat: number,
    readonly nlon: number,
    readonly lmax: number,
    readonly nodes: 'equiangular' | 'centered' = 'equiangular',
  ) {
    const c = nodes === 'centered';
    this.theta = Float64Array.from({ length: nlat }, (_, k) =>
      c ? (Math.PI * (k + 0.5)) / nlat : (Math.PI * k) / (nlat - 1),
    );
    this.phi = Float64Array.from(
      { length: nlon },
      (_, j) => (2 * Math.PI * (c ? j + 0.5 : j)) / nlon,
    );
    this.weights = c ? new Float64Array(nlat) : clenshawCurtisWeights(nlat);
    this.leg = legendreTable(
      lmax,
      Array.from(this.theta, (t) => Math.cos(t)),
    );
    const n = lmax + 1;
    this.cosTab = new Float64Array(n * nlon);
    this.sinTab = new Float64Array(n * nlon);
    for (let m = 0; m < n; m++) {
      for (let j = 0; j < nlon; j++) {
        this.cosTab[m * nlon + j] = Math.cos(m * this.phi[j]!);
        this.sinTab[m * nlon + j] = Math.sin(m * this.phi[j]!);
      }
    }
    this.rowRe = new Float64Array(n);
    this.rowIm = new Float64Array(n);
  }

  /** Forward real SHT of a row-major `(nlat, nlon)` grid. */
  analyze(f: ArrayLike<number>, out: HalfSpectrum = halfSpectrum(this.lmax)): HalfSpectrum {
    if (this.nodes !== 'equiangular') throw new Error('analysis needs the equiangular grid');
    const { nlat, nlon, lmax } = this;
    const n = lmax + 1;
    const mTop = Math.min(lmax, Math.floor(nlon / 2));
    out.re.fill(0);
    out.im.fill(0);
    for (let k = 0; k < nlat; k++) {
      const row = k * nlon;
      for (let m = 0; m <= mTop; m++) {
        let sr = 0;
        let si = 0;
        const t = m * nlon;
        for (let j = 0; j < nlon; j++) {
          const v = f[row + j]!;
          sr += v * this.cosTab[t + j]!;
          si -= v * this.sinTab[t + j]!;
        }
        this.rowRe[m] = sr / nlon;
        this.rowIm[m] = si / nlon;
      }
      const w = 2 * Math.PI * this.weights[k]!;
      const base = k * n * n;
      for (let l = 0; l < n; l++) {
        for (let m = 0; m <= Math.min(l, mTop); m++) {
          const p = w * this.leg[base + l * n + m]!;
          out.re[l * n + m]! += p * this.rowRe[m]!;
          out.im[l * n + m]! += p * this.rowIm[m]!;
        }
      }
    }
    return out;
  }

  /** Inverse real SHT onto the grid, `f = sum_l c_l0 Y_l0 + 2 Re sum_{m>0} c_lm Y_lm`. */
  synthesize(c: HalfSpectrum, out: Float64Array = new Float64Array(this.nlat * this.nlon)) {
    const { nlat, nlon } = this;
    const lmax = Math.min(this.lmax, c.lmax);
    const n = this.lmax + 1;
    const nc = c.lmax + 1;
    const mTop = Math.min(lmax, Math.floor(nlon / 2));
    for (let k = 0; k < nlat; k++) {
      const base = k * n * n;
      for (let m = 0; m <= mTop; m++) {
        let sr = 0;
        let si = 0;
        for (let l = m; l <= lmax; l++) {
          const p = this.leg[base + l * n + m]!;
          sr += p * c.re[l * nc + m]!;
          si += p * c.im[l * nc + m]!;
        }
        this.rowRe[m] = sr;
        this.rowIm[m] = si;
      }
      const row = k * nlon;
      for (let j = 0; j < nlon; j++) {
        let v = this.rowRe[0]!;
        for (let m = 1; m <= mTop; m++) {
          const t = m * nlon + j;
          const nyquist = 2 * m === nlon;
          const re = this.rowRe[m]! * this.cosTab[t]!;
          const im = nyquist ? 0 : this.rowIm[m]! * this.sinTab[t]!;
          v += (nyquist ? 1 : 2) * (re - im);
        }
        out[row + j] = v;
      }
    }
    return out;
  }
}
