/** Small dense linear algebra on row-major `Float64Array`s. */

/** `A^T A` (`(n, n)`, upper and lower filled) and `A^T r` for row-major `A` of shape `(m, n)`. */
export function normalEquations(
  a: Float64Array,
  r: Float64Array,
  m: number,
  n: number,
  h: Float64Array = new Float64Array(n * n),
  g: Float64Array = new Float64Array(n),
): { h: Float64Array; g: Float64Array } {
  h.fill(0);
  g.fill(0);
  for (let k = 0; k < m; k++) {
    const row = k * n;
    const rk = r[k]!;
    for (let i = 0; i < n; i++) {
      const ai = a[row + i]!;
      if (ai === 0) continue;
      g[i]! += ai * rk;
      const hi = i * n;
      for (let j = i; j < n; j++) h[hi + j]! += ai * a[row + j]!;
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) h[i * n + j] = h[j * n + i]!;
  return { h, g };
}

/** Solve `A x = b` for symmetric positive definite `A` in place; `false` if not SPD. */
export function choleskySolve(a: Float64Array, b: Float64Array, n: number): boolean {
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j]!;
    for (let k = 0; k < j; k++) d -= a[j * n + k]! ** 2;
    if (!(d > 0)) return false;
    const ljj = Math.sqrt(d);
    a[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = a[i * n + j]!;
      for (let k = 0; k < j; k++) s -= a[i * n + k]! * a[j * n + k]!;
      a[i * n + j] = s / ljj;
    }
  }
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= a[i * n + k]! * b[k]!;
    b[i] = s / a[i * n + i]!;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]!;
    for (let k = i + 1; k < n; k++) s -= a[k * n + i]! * b[k]!;
    b[i] = s / a[i * n + i]!;
  }
  return true;
}

/** Solve `A x = b` by LU with partial pivoting in place; `false` if numerically singular. */
export function luSolve(a: Float64Array, b: Float64Array, n: number): boolean {
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let i = c + 1; i < n; i++) if (Math.abs(a[i * n + c]!) > Math.abs(a[p * n + c]!)) p = i;
    const piv = a[p * n + c]!;
    if (!(Math.abs(piv) > 1e-300)) return false;
    if (p !== c) {
      for (let j = 0; j < n; j++) {
        const t = a[c * n + j]!;
        a[c * n + j] = a[p * n + j]!;
        a[p * n + j] = t;
      }
      const t = b[c]!;
      b[c] = b[p]!;
      b[p] = t;
    }
    for (let i = c + 1; i < n; i++) {
      const f = a[i * n + c]! / piv;
      if (f === 0) continue;
      for (let j = c; j < n; j++) a[i * n + j]! -= f * a[c * n + j]!;
      b[i]! -= f * b[c]!;
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]!;
    for (let j = i + 1; j < n; j++) s -= a[i * n + j]! * b[j]!;
    b[i] = s / a[i * n + i]!;
  }
  return true;
}

/** Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi). Ascending eigenvalues; the
 * eigenvectors are the columns of `vectors` (row-major). */
export function symEig3(m: Float64Array): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(m);
  const v = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  for (let sweep = 0; sweep < 60; sweep++) {
    const off = a[1]! ** 2 + a[2]! ** 2 + a[5]! ** 2;
    if (off < 1e-30 * (a[0]! ** 2 + a[4]! ** 2 + a[8]! ** 2 + 1e-300)) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const apq = a[p * 3 + q]!;
      if (apq === 0) continue;
      const theta = (a[q * 3 + q]! - a[p * 3 + p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p]!;
        const akq = a[k * 3 + q]!;
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k]!;
        const aqk = a[q * 3 + k]!;
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p]!;
        const vkq = v[k * 3 + q]!;
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i * 4]! - a[j * 4]!);
  const values = Float64Array.from(order, (i) => a[i * 4]!);
  const vectors = new Float64Array(9);
  order.forEach((src, dst) => {
    for (let k = 0; k < 3; k++) vectors[k * 3 + dst] = v[k * 3 + src]!;
  });
  return { values, vectors };
}
