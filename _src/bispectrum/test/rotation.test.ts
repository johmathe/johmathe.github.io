import { describe, expect, it } from 'vitest';

import { degreePower, dim, toHalfSpectrum } from '../src/math/basis';
import { Invariant } from '../src/math/invariant';
import { Rng } from '../src/math/rng';
import {
  dofBlocks,
  expMap,
  matToQuat,
  mul3,
  quatToMat,
  randomRotation,
  realShBlocks,
  rotateDofs,
  scrambleDegrees,
  slerp,
  transpose3,
} from '../src/math/rotation';
import { evaluateAt } from '../src/math/sht';
import { fixture, fromPythonDofs, loadTables, maxAbsDiff, randn, relErr } from './helpers';

const LMAX = 12;

function randomDirs(n: number, rng: Rng): Float64Array {
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const v = [rng.normal(), rng.normal(), rng.normal()];
    const r = Math.hypot(...v);
    out.set(
      v.map((c) => c / r),
      3 * i,
    );
  }
  return out;
}

describe('rotations', () => {
  const rng = new Rng(7);

  it('interpolates between two rotations along the shorter arc', () => {
    const A = randomRotation(rng);
    const w = [0.3, -1.1, 0.7];
    const B = mul3(expMap(w), A);
    expect(maxAbsDiff(slerp(A, B, 0), A)).toBeLessThan(1e-12);
    expect(maxAbsDiff(slerp(A, B, 1), B)).toBeLessThan(1e-12);
    const quarter = mul3(expMap(w.map((v) => 0.25 * v)), A);
    expect(maxAbsDiff(slerp(A, B, 0.25), quarter)).toBeLessThan(1e-12);
    expect(maxAbsDiff(slerp(A, A, 0.5), A)).toBeLessThan(1e-12);
  });

  it('round-trips quaternions and matrices', () => {
    const R = randomRotation(rng);
    expect(maxAbsDiff(quatToMat(matToQuat(R)), R)).toBeLessThan(1e-14);
    const w = [0.3, -1.1, 0.5];
    const E = expMap(w);
    expect(
      maxAbsDiff(mul3(E, transpose3(E)), Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1)),
    ).toBeLessThan(1e-14);
  });

  it('gives orthogonal blocks and a group homomorphism', () => {
    const A = randomRotation(rng);
    const B = randomRotation(rng);
    const bA = realShBlocks(A, LMAX);
    const bB = realShBlocks(B, LMAX);
    const bAB = realShBlocks(mul3(A, B), LMAX);
    for (let l = 0; l <= LMAX; l++) {
      const sz = 2 * l + 1;
      for (let i = 0; i < sz; i++) {
        for (let j = 0; j < sz; j++) {
          let gram = 0;
          let prod = 0;
          for (let k = 0; k < sz; k++) {
            gram += bA[l]![i * sz + k]! * bA[l]![j * sz + k]!;
            prod += bA[l]![i * sz + k]! * bB[l]![k * sz + j]!;
          }
          expect(Math.abs(gram - (i === j ? 1 : 0))).toBeLessThan(1e-12);
          expect(Math.abs(prod - bAB[l]![i * sz + j]!)).toBeLessThan(1e-12);
        }
      }
    }
  });

  it('rotates the signal pointwise: (R.f)(p) = f(R^T p)', () => {
    const x = randn(dim(LMAX), 11);
    const R = randomRotation(rng);
    const dirs = randomDirs(200, rng);
    const back = new Float64Array(dirs.length);
    const Rt = transpose3(R);
    for (let i = 0; i < 200; i++) {
      for (let a = 0; a < 3; a++) {
        back[3 * i + a] =
          Rt[a * 3]! * dirs[3 * i]! +
          Rt[a * 3 + 1]! * dirs[3 * i + 1]! +
          Rt[a * 3 + 2]! * dirs[3 * i + 2]!;
      }
    }
    const rotated = evaluateAt(toHalfSpectrum(rotateDofs(x, R), LMAX), dirs);
    const original = evaluateAt(toHalfSpectrum(x, LMAX), back);
    expect(relErr(rotated, original)).toBeLessThan(1e-12);
  });

  it('agrees with bispectrum.rotate_spherical_function', () => {
    const fx = fixture<{ lmax: number; x: number[]; rotation: number[]; x_rotated: number[] }>(
      'rotation.json',
    );
    const x = fromPythonDofs(fx.x, fx.lmax);
    const expected = fromPythonDofs(fx.x_rotated, fx.lmax);
    const got = rotateDofs(x, Float64Array.from(fx.rotation));
    // Python interpolates bilinearly on a 256x512 grid; a wrong convention would be O(1) off.
    expect(relErr(got, expected)).toBeLessThan(2e-3);
    expect(
      relErr(rotateDofs(x, transpose3(Float64Array.from(fx.rotation))), expected),
    ).toBeGreaterThan(0.3);
  });

  it.each([4, 8, 12])('leaves the selective invariant unchanged at lmax %i', (lmax) => {
    const inv = new Invariant(loadTables(), lmax);
    const x = randn(dim(lmax), 50 + lmax);
    const before = inv.features(x);
    for (let k = 0; k < 3; k++) {
      const after = inv.features(rotateDofs(x, randomRotation(rng)));
      expect(relErr(after, before)).toBeLessThan(1e-12);
    }
  });

  it('scrambles degrees: same power spectrum, different invariant', () => {
    const lmax = 8;
    const inv = new Invariant(loadTables(), lmax);
    const x = randn(dim(lmax), 99);
    const y = scrambleDegrees(x, rng);
    expect(relErr(degreePower(y, lmax), degreePower(x, lmax))).toBeLessThan(1e-13);
    expect(relErr(inv.features(y), inv.features(x))).toBeGreaterThan(0.1);
    expect(dofBlocks(randomRotation(rng), lmax)).toHaveLength(lmax + 1);
  });
});
