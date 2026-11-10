import { describe, expect, it } from 'vitest';

import { dim, dofsWhere } from '../src/math/basis';
import { Invariant } from '../src/math/invariant';
import {
  fixture,
  fromPythonDofs,
  loadTables,
  maxAbsDiff,
  pythonDofScale,
  randn,
  relErr,
} from './helpers';

interface InvariantFixture {
  lmax: number;
  triples: [number, number, number][];
  x: number[];
  features: number[];
  jacobian?: number[];
}

const tables = loadTables();

describe.each([4, 6, 8, 10, 12])('selective invariant at lmax %i', (lmax) => {
  const fx = fixture<InvariantFixture>(`invariant_L${lmax}.json`);
  const inv = new Invariant(tables, lmax);
  const x = fromPythonDofs(fx.x, lmax);

  it('keeps the Python entry order when filtering the exported table', () => {
    expect(inv.triples).toEqual(fx.triples);
  });

  it('matches InvariantModel.features', () => {
    const f = inv.features(x);
    expect(relErr(f, fx.features)).toBeLessThan(1e-13);
  });

  it('is one real number per entry on real signals', () => {
    const f = inv.features(x);
    const scale = Math.max(...Array.from(f, Math.abs));
    for (let e = 0; e < inv.n; e++) {
      const dropped = inv.isPower[e] ? f[inv.n + e]! : f[inv.imagPart[e] ? e : inv.n + e]!;
      expect(Math.abs(dropped)).toBeLessThan(1e-12 * scale);
    }
    const v = inv.values(x);
    for (let e = 0; e < inv.n; e++) {
      expect(v[e]).toBe(f[inv.imagPart[e] ? inv.n + e : e]);
    }
  });

  it('has an exact Jacobian (central differences)', () => {
    const x = randn(dim(lmax), 100 + lmax);
    const rows = inv.allRows;
    const dofs = Int32Array.from({ length: dim(lmax) }, (_, i) => i);
    const { val, jac } = inv.jacobian(x, rows, dofs);
    expect(maxAbsDiff(val, inv.values(x))).toBe(0);
    const h = 1e-6;
    const fd = new Float64Array(jac.length);
    for (let d = 0; d < dofs.length; d++) {
      const xp = Float64Array.from(x);
      const xm = Float64Array.from(x);
      xp[d]! += h;
      xm[d]! -= h;
      const vp = inv.values(xp);
      const vm = inv.values(xm);
      for (let r = 0; r < rows.length; r++) fd[r * dofs.length + d] = (vp[r]! - vm[r]!) / (2 * h);
    }
    expect(relErr(jac, fd)).toBeLessThan(1e-8);
  });

  it('restricts the Jacobian to row and dof subsets consistently', () => {
    const x = randn(dim(lmax), 200 + lmax);
    const full = inv.jacobian(
      x,
      inv.allRows,
      Int32Array.from({ length: dim(lmax) }, (_, i) => i),
    );
    const rows = inv.rowsWhere((d) => d === lmax - 1);
    const dofs = dofsWhere(lmax, (l) => l === lmax - 1);
    const sub = inv.jacobian(x, rows, dofs);
    for (let r = 0; r < rows.length; r++) {
      for (let c = 0; c < dofs.length; c++) {
        expect(sub.jac[r * dofs.length + c]).toBeCloseTo(
          full.jac[rows[r]! * dim(lmax) + dofs[c]!]!,
          12,
        );
      }
    }
  });

  if (fx.jacobian) {
    it('matches InvariantModel.jacobian', () => {
      const dofs = Int32Array.from({ length: dim(lmax) }, (_, i) => i);
      const { jac } = inv.jacobian(x, inv.allRows, dofs);
      const scale = pythonDofScale(lmax);
      // Python rows are [Re (n), Im (n)]; keep the nonzero part of each entry.
      const pyJac = fx.jacobian!;
      const picked = new Float64Array(jac.length);
      for (let e = 0; e < inv.n; e++) {
        const pyRow = inv.imagPart[e] ? inv.n + e : e;
        for (let d = 0; d < dofs.length; d++) {
          picked[e * dofs.length + d] = pyJac[pyRow * dofs.length + d]! / scale[d]!;
        }
      }
      expect(relErr(jac, picked)).toBeLessThan(1e-13);
    });
  }
});
