import { describe, expect, it } from 'vitest';

import { align, quadrupole } from '../src/math/align';
import { bandlimit, dim, dofsWhere, fromHalfSpectrum, toHalfSpectrum } from '../src/math/basis';
import { Invariant } from '../src/math/invariant';
import { symEig3 } from '../src/math/linalg';
import { lmSolve } from '../src/math/lm';
import { WIDGET_CONFIG } from '../src/math/config';
import { march } from '../src/math/march';
import { Rng } from '../src/math/rng';
import { mul3, randomRotation, rotateDofs, transpose3 } from '../src/math/rotation';
import { SphereGrid } from '../src/math/sht';
import { liftToSphere } from '../src/math/smnist';
import { loadDigits, loadTables, maxAbsDiff, randn, relErr } from './helpers';

const tables = loadTables();

describe('alignment', () => {
  const rng = new Rng(5);

  it('diagonalizes symmetric 3x3 matrices', () => {
    const R = randomRotation(rng);
    const D = Float64Array.of(-1, 0, 0, 0, 0.5, 0, 0, 0, 2);
    const { values, vectors } = symEig3(mul3(mul3(R, D), transpose3(R)));
    expect(maxAbsDiff(values, [-1, 0.5, 2])).toBeLessThan(1e-13);
    const back = mul3(mul3(vectors, D), transpose3(vectors));
    expect(maxAbsDiff(back, mul3(mul3(R, D), transpose3(R)))).toBeLessThan(1e-13);
  });

  it('computes the quadrupole exactly and equivariantly', () => {
    const lmax = 4;
    const x = randn(dim(lmax), 8);
    const grid = new SphereGrid(32, 64, lmax);
    const f = grid.synthesize(toHalfSpectrum(x, lmax));
    const q = new Float64Array(9);
    for (let k = 0; k < grid.nlat; k++) {
      const st = Math.sin(grid.theta[k]!);
      const p0 = [0, 0, Math.cos(grid.theta[k]!)];
      for (let j = 0; j < grid.nlon; j++) {
        const p = [st * Math.cos(grid.phi[j]!), st * Math.sin(grid.phi[j]!), p0[2]!];
        const w = ((grid.weights[k]! * 2 * Math.PI) / grid.nlon) * f[k * grid.nlon + j]!;
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) q[a * 3 + b]! += w * (3 * p[a]! * p[b]! - (a === b ? 1 : 0));
        }
      }
    }
    expect(maxAbsDiff(quadrupole(x), q)).toBeLessThan(1e-13);
    const R = randomRotation(rng);
    const rotated = quadrupole(rotateDofs(x, R));
    expect(maxAbsDiff(rotated, mul3(mul3(R, quadrupole(x)), transpose3(R)))).toBeLessThan(1e-13);
  });

  it('recovers the rotation between a signal and its rotated copy', () => {
    const x = randn(dim(10), 21);
    const R = randomRotation(rng);
    const { rot, rel } = align(x, rotateDofs(x, R), { rng });
    expect(rel).toBeLessThan(1e-10);
    expect(maxAbsDiff(rot, R)).toBeLessThan(1e-9);
  });
});

describe('Levenberg-Marquardt', () => {
  it('converges from a perturbed start', () => {
    const lmax = 6;
    const inv = new Invariant(tables, lmax);
    const x = randn(dim(lmax), 31);
    const target = inv.values(x);
    const noise = randn(dim(lmax), 32);
    const x0 = Float64Array.from(x, (v, i) => v + 1e-2 * noise[i]!);
    const dofs = dofsWhere(lmax, () => true);
    const sol = lmSolve(inv, x0, target, dofs, inv.allRows);
    expect(sol.rel).toBeLessThan(1e-12);
  });
});

describe('degree marching on spherical MNIST', () => {
  // Digit 3 is the hard case: about 1 in 200 seed starts reaches the true orbit, so
  // PAPER_CONFIG's 200-start budget usually runs dry on it.
  it('recovers a digit from its invariant at lmax 8', () => {
    const lmax = 8;
    const digits = loadDigits();
    const grid = new SphereGrid(64, 128, lmax);
    const lifted = liftToSphere(digits.images[3]!, digits.size, grid.theta, grid.phi);
    const x = bandlimit(fromHalfSpectrum(grid.analyze(lifted)), lmax);
    const inv = new Invariant(tables, lmax);
    const target = inv.values(x);

    const t0 = performance.now();
    const degreesSeen: number[] = [];
    const res = march(inv, target, WIDGET_CONFIG, new Rng(1), (e) => {
      if (e.kind === 'degree') degreesSeen.push(e.degree);
    });
    const ms = performance.now() - t0;
    expect(degreesSeen).toEqual([4, 5, 6, 7, 8]);
    expect(res.branches.length).toBeGreaterThan(0);

    const scored = res.branches.map((b) => ({ ...b, al: align(b.x, x, { rng: new Rng(2) }) }));
    scored.sort((a, b) => a.rel - b.rel || a.al.rel - b.al.rel);
    const best = scored[0]!;
    console.log(
      `lmax ${lmax}: ${ms.toFixed(0)} ms, seed starts ${res.seedTried}, ` +
        `branches ${res.branches.length}, feature rel ${best.rel.toExponential(1)}, ` +
        `aligned rel ${best.al.rel.toExponential(1)}`,
    );
    expect(best.rel).toBeLessThan(1e-10);
    expect(best.al.rel).toBeLessThan(1e-6);
    expect(relErr(rotateDofs(best.x, best.al.rot), x)).toBeLessThan(1e-6);
  });
});
