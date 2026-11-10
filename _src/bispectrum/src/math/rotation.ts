/**
 * Rotations of band-limited signals in the real coordinates of `basis.ts`.
 *
 * `rotateDofs(x, R)` returns the coordinates of `R . f`, `(R . f)(p) = f(R^T p)` (the convention of
 * `bispectrum.rotate_spherical_function`), exactly: the degree-`l` blocks come from the
 * Ivanic-Ruedenberg recursion (J. Phys. Chem. 100, 6342 (1996); errata 102, 9099 (1998)) in the
 * standard real basis without Condon-Shortley phase, where degree 1 is `(y, z, x)`. Our dofs map
 * onto that basis by a signed permutation: `s_{l,m} = (-1)^m x_re(l, m)` and
 * `s_{l,-m} = -(-1)^m x_im(l, m)` for `m > 0`.
 */

import type { Rng } from './rng';

/** Row-major 3x3 matrix. */
export type Mat3 = Float64Array;
/** Unit quaternion `[w, x, y, z]`. */
export type Quat = [number, number, number, number];

export const identity3 = (): Mat3 => Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);

export function mul3(a: Mat3, b: Mat3): Mat3 {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i * 3 + j] = a[i * 3]! * b[j]! + a[i * 3 + 1]! * b[3 + j]! + a[i * 3 + 2]! * b[6 + j]!;
    }
  }
  return out;
}

export function transpose3(a: Mat3): Mat3 {
  return Float64Array.of(a[0]!, a[3]!, a[6]!, a[1]!, a[4]!, a[7]!, a[2]!, a[5]!, a[8]!);
}

export function det3(a: Mat3): number {
  return (
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!)
  );
}

export function quatToMat(q: Quat): Mat3 {
  const n = Math.hypot(...q);
  const [w, x, y, z] = q.map((v) => v / n) as Quat;
  return Float64Array.of(
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  );
}

export function matToQuat(m: Mat3): Quat {
  const t = m[0]! + m[4]! + m[8]!;
  let q: Quat;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [0.25 * s, (m[7]! - m[5]!) / s, (m[2]! - m[6]!) / s, (m[3]! - m[1]!) / s];
  } else if (m[0]! > m[4]! && m[0]! > m[8]!) {
    const s = Math.sqrt(1 + m[0]! - m[4]! - m[8]!) * 2;
    q = [(m[7]! - m[5]!) / s, 0.25 * s, (m[1]! + m[3]!) / s, (m[2]! + m[6]!) / s];
  } else if (m[4]! > m[8]!) {
    const s = Math.sqrt(1 + m[4]! - m[0]! - m[8]!) * 2;
    q = [(m[2]! - m[6]!) / s, (m[1]! + m[3]!) / s, 0.25 * s, (m[5]! + m[7]!) / s];
  } else {
    const s = Math.sqrt(1 + m[8]! - m[0]! - m[4]!) * 2;
    q = [(m[3]! - m[1]!) / s, (m[2]! + m[6]!) / s, (m[5]! + m[7]!) / s, 0.25 * s];
  }
  const n = Math.hypot(...q);
  return q.map((v) => v / n) as Quat;
}

/** The rotation a fraction `t` of the way from `a` to `b`, along the shorter arc. */
export function slerp(a: Mat3, b: Mat3, t: number): Mat3 {
  const qa = matToQuat(a);
  let qb = matToQuat(b);
  let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
  if (dot < 0) {
    qb = qb.map((v) => -v) as Quat;
    dot = -dot;
  }
  const th = Math.acos(Math.min(1, dot));
  if (th < 1e-9) return quatToMat(qb);
  const [wa, wb] = [Math.sin((1 - t) * th) / Math.sin(th), Math.sin(t * th) / Math.sin(th)];
  return quatToMat(qa.map((v, i) => wa * v + wb * qb[i]!) as Quat);
}

/** Rotation by angle `|w|` about `w / |w|` (Rodrigues). */
export function expMap(w: ArrayLike<number>): Mat3 {
  const th = Math.hypot(w[0]!, w[1]!, w[2]!);
  if (th < 1e-300) return identity3();
  const [x, y, z] = [w[0]! / th, w[1]! / th, w[2]! / th];
  const c = Math.cos(th);
  const s = Math.sin(th);
  const v = 1 - c;
  return Float64Array.of(
    c + x * x * v,
    x * y * v - z * s,
    x * z * v + y * s,
    y * x * v + z * s,
    c + y * y * v,
    y * z * v - x * s,
    z * x * v - y * s,
    z * y * v + x * s,
    c + z * z * v,
  );
}

/** Haar-uniform rotation. */
export function randomRotation(rng: Rng): Mat3 {
  return quatToMat([rng.normal(), rng.normal(), rng.normal(), rng.normal()]);
}

/**
 * Ivanic-Ruedenberg blocks `B_l` (row-major `(2l+1)^2`, index `m + l`) acting on standard real
 * SH coefficients: `s'_l = B_l s_l` are the coefficients of `R . f`.
 */
export function realShBlocks(rot: Mat3, lmax: number): Float64Array[] {
  const blocks: Float64Array[] = [Float64Array.of(1)];
  if (lmax === 0) return blocks;
  const perm = [1, 2, 0];
  const r1 = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) r1[i * 3 + j] = rot[perm[i]! * 3 + perm[j]!]!;
  }
  blocks.push(r1);
  const R1 = (i: number, j: number): number => r1[(i + 1) * 3 + (j + 1)]!;

  for (let l = 2; l <= lmax; l++) {
    const prev = blocks[l - 1]!;
    const psz = 2 * l - 1;
    const sz = 2 * l + 1;
    const Rp = (a: number, b: number): number => prev[(a + l - 1) * psz + (b + l - 1)]!;
    const P = (i: number, a: number, b: number): number => {
      if (b === l) return R1(i, 1) * Rp(a, l - 1) - R1(i, -1) * Rp(a, -l + 1);
      if (b === -l) return R1(i, 1) * Rp(a, -l + 1) + R1(i, -1) * Rp(a, l - 1);
      return R1(i, 0) * Rp(a, b);
    };
    const out = new Float64Array(sz * sz);
    for (let m = -l; m <= l; m++) {
      const am = Math.abs(m);
      const d = m === 0 ? 1 : 0;
      for (let n = -l; n <= l; n++) {
        const denom = Math.abs(n) === l ? 2 * l * (2 * l - 1) : (l + n) * (l - n);
        const u = Math.sqrt(((l + m) * (l - m)) / denom);
        const v = 0.5 * Math.sqrt(((1 + d) * (l + am - 1) * (l + am)) / denom) * (1 - 2 * d);
        const w = -0.5 * Math.sqrt(((l - am - 1) * (l - am)) / denom) * (1 - d);
        let val = 0;
        if (u !== 0) val += u * P(0, m, n);
        if (v !== 0) {
          let V: number;
          if (m === 0) V = P(1, 1, n) + P(-1, -1, n);
          else if (m > 0) {
            V = P(1, m - 1, n) * (m === 1 ? Math.SQRT2 : 1);
            if (m !== 1) V -= P(-1, -m + 1, n);
          } else {
            V = P(-1, -m - 1, n) * (m === -1 ? Math.SQRT2 : 1);
            if (m !== -1) V += P(1, m + 1, n);
          }
          val += v * V;
        }
        if (w !== 0) {
          const W = m > 0 ? P(1, m + 1, n) + P(-1, -m - 1, n) : P(1, m - 1, n) - P(-1, -m + 1, n);
          val += w * W;
        }
        out[(m + l) * sz + (n + l)] = val;
      }
    }
    blocks.push(out);
  }
  return blocks;
}

/** Per dof: its standard real-SH slot within the degree block (`m + l`) and the sign. */
function stdSlots(lmax: number): { slot: Int32Array; sign: Float64Array } {
  const n = (lmax + 1) * (lmax + 1);
  const slot = new Int32Array(n);
  const sign = new Float64Array(n);
  for (let l = 0; l <= lmax; l++) {
    const b = l * l;
    slot[b] = l;
    sign[b] = 1;
    for (let m = 1; m <= l; m++) {
      const s = m % 2 === 0 ? 1 : -1;
      slot[b + 2 * m - 1] = l + m;
      sign[b + 2 * m - 1] = s;
      slot[b + 2 * m] = l - m;
      sign[b + 2 * m] = -s;
    }
  }
  return { slot, sign };
}

const slotCache = new Map<number, { slot: Int32Array; sign: Float64Array }>();

/**
 * The rotation as one dense block per degree in dof coordinates: `x'_l = D_l x_l`, with `D_l`
 * row-major `(2l+1)^2` over the dofs `l^2 .. (l+1)^2 - 1`.
 */
export function dofBlocks(rot: Mat3, lmax: number): Float64Array[] {
  let slots = slotCache.get(lmax);
  if (!slots) {
    slots = stdSlots(lmax);
    slotCache.set(lmax, slots);
  }
  const { slot, sign } = slots;
  return realShBlocks(rot, lmax).map((B, l) => {
    const sz = 2 * l + 1;
    const b = l * l;
    const D = new Float64Array(sz * sz);
    for (let i = 0; i < sz; i++) {
      for (let j = 0; j < sz; j++) {
        D[i * sz + j] = sign[b + i]! * sign[b + j]! * B[slot[b + i]! * sz + slot[b + j]!]!;
      }
    }
    return D;
  });
}

export function applyBlocks(
  blocks: Float64Array[],
  x: ArrayLike<number>,
  out: Float64Array = new Float64Array(x.length),
): Float64Array {
  const lmax = Math.round(Math.sqrt(x.length)) - 1;
  for (let l = 0; l <= lmax; l++) {
    const D = blocks[l]!;
    const sz = 2 * l + 1;
    const b = l * l;
    for (let i = 0; i < sz; i++) {
      let s = 0;
      for (let j = 0; j < sz; j++) s += D[i * sz + j]! * x[b + j]!;
      out[b + i] = s;
    }
  }
  return out;
}

/** Coordinates of `R . f` given those of `f`. */
export function rotateDofs(x: ArrayLike<number>, rot: Mat3): Float64Array {
  const lmax = Math.round(Math.sqrt(x.length)) - 1;
  return applyBlocks(dofBlocks(rot, lmax), x);
}

/**
 * Independent random rotation of every degree block: the power spectrum `||F_l||^2` is unchanged,
 * the relative orientation of the degrees (and so the signal) is not.
 */
export function scrambleDegrees(x: ArrayLike<number>, rng: Rng): Float64Array {
  const lmax = Math.round(Math.sqrt(x.length)) - 1;
  const out = new Float64Array(x.length);
  for (let l = 0; l <= lmax; l++) {
    const D = dofBlocks(randomRotation(rng), l)[l]!;
    const sz = 2 * l + 1;
    const b = l * l;
    for (let i = 0; i < sz; i++) {
      let s = 0;
      for (let j = 0; j < sz; j++) s += D[i * sz + j]! * x[b + j]!;
      out[b + i] = s;
    }
  }
  return out;
}
