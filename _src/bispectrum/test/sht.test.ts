import { describe, expect, it } from 'vitest';

import { degreePower, dim, fromHalfSpectrum, toHalfSpectrum } from '../src/math/basis';
import { SphereGrid } from '../src/math/sht';
import { liftToSphere } from '../src/math/smnist';
import { fixture, maxAbsDiff, randn, relErr, toPythonDofs } from './helpers';

interface ShtFixture {
  nlat: number;
  nlon: number;
  lmax: number;
  image: number[];
  lifted: number[];
  coeffs_re: number[];
  coeffs_im: number[];
  x: number[];
  inverse: number[];
}

const fx = fixture<ShtFixture>('sht.json');

describe('equiangular real SHT', () => {
  const grid = new SphereGrid(fx.nlat, fx.nlon, fx.lmax);

  it('lifts MNIST exactly like the spherical MNIST cache', () => {
    const image = Float64Array.from(fx.image, (v) => Math.fround(v / 255));
    const lifted = liftToSphere(image, 28, grid.theta, grid.phi);
    expect(maxAbsDiff(lifted, fx.lifted)).toBeLessThan(1e-4);
    expect(relErr(lifted, fx.lifted)).toBeLessThan(1e-5);
  });

  // The cache stores float32; the fixture prints it with 9 digits, which round-trips float32.
  const lifted = Float64Array.from(fx.lifted, Math.fround);

  it('matches torch-harmonics RealSHT on the lifted digit', () => {
    const c = grid.analyze(lifted);
    expect(relErr(c.re, fx.coeffs_re)).toBeLessThan(1e-13);
    expect(maxAbsDiff(c.im, fx.coeffs_im)).toBeLessThan(1e-14);
    expect(relErr(toPythonDofs(fromHalfSpectrum(c), fx.lmax), fx.x)).toBeLessThan(1e-13);
  });

  it('matches torch-harmonics InverseRealSHT', () => {
    const c = {
      lmax: fx.lmax,
      re: Float64Array.from(fx.coeffs_re),
      im: Float64Array.from(fx.coeffs_im),
    };
    expect(relErr(grid.synthesize(c), fx.inverse)).toBeLessThan(1e-13);
  });

  it('round-trips band-limited signals exactly', () => {
    const x = randn(dim(fx.lmax), 3);
    const back = fromHalfSpectrum(grid.analyze(grid.synthesize(toHalfSpectrum(x, fx.lmax))));
    expect(relErr(back, x)).toBeLessThan(1e-12);
  });

  it('uses isometric coordinates (Parseval)', () => {
    const x = randn(dim(fx.lmax), 4);
    const f = grid.synthesize(toHalfSpectrum(x, fx.lmax));
    let integral = 0;
    for (let k = 0; k < grid.nlat; k++) {
      for (let j = 0; j < grid.nlon; j++) {
        integral += ((grid.weights[k]! * 2 * Math.PI) / grid.nlon) * f[k * grid.nlon + j]! ** 2;
      }
    }
    const power = degreePower(x, fx.lmax).reduce((a, b) => a + b, 0);
    expect(Math.abs(integral - power) / power).toBeLessThan(1e-12);
  });
});
