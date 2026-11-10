import { describe, expect, it } from 'vitest';

import { degreePower, dim, fromHalfSpectrum, toHalfSpectrum } from '../src/math/basis';
import {
  BispectrumMlp,
  PixelCnn,
  PowerMlp,
  softmax,
  type ClassifierManifest,
  type ModelName,
} from '../src/math/classifier';
import { Invariant } from '../src/math/invariant';
import { Rng } from '../src/math/rng';
import { randomRotation, rotateDofs, scrambleDegrees } from '../src/math/rotation';
import { SphereGrid } from '../src/math/sht';
import { liftToSphere } from '../src/math/smnist';
import { fixture, loadDataFile, loadDigits, loadTables, maxAbsDiff } from './helpers';

interface ClassifierFixture {
  lmax: number;
  labels: number[];
  features: number[];
  image: number[];
  mlp_logits: number[];
  cnn_logits: number[];
  power_features: number[];
  power_logits: number[];
}

const manifest = JSON.parse(
  Buffer.from(loadDataFile('classifier.json')).toString('utf8'),
) as ClassifierManifest;
const fx = fixture<ClassifierFixture>('classifier.json');
const digits = loadDigits();
const { lmax, nlat, nlon } = manifest;
const grid = new SphereGrid(nlat, nlon, lmax);
const inv = new Invariant(loadTables(), lmax);
const mlp = new BispectrumMlp(manifest.mlp, loadDataFile(manifest.mlp.file));
const cnn = new PixelCnn(manifest.cnn, loadDataFile(manifest.cnn.file), nlat, nlon);
const power = new PowerMlp(manifest.power, loadDataFile(manifest.power.file));

/** The widget's pipeline: mirrored lift, analysis, band-limit. */
function coefficients(i: number): Float64Array {
  const f = liftToSphere(digits.images[i]!, digits.size, grid.theta, grid.phi, { mirror: true });
  return fromHalfSpectrum(grid.analyze(f));
}

const signLog = (v: number): number => Math.sign(v) * Math.log1p(Math.abs(v));
const argmax = (v: ArrayLike<number>): number =>
  Array.from(v).reduce((best, x, i, a) => (x > a[best]! ? i : best), 0);

describe('classifiers', () => {
  it('match the band-limit and size of the widget tables', () => {
    expect(fx.lmax).toBe(lmax);
    expect(inv.n).toBe(manifest.n_features);
    expect(coefficients(0).length).toBe(dim(lmax));
  });

  it('feed the networks the inputs they were trained on', () => {
    const x = coefficients(0);
    expect(maxAbsDiff(Array.from(inv.values(x), signLog), fx.features)).toBeLessThan(1e-5);
    expect(maxAbsDiff(grid.synthesize(toHalfSpectrum(x, lmax)), fx.image)).toBeLessThan(1e-5);
    expect(fx.power_features.length).toBe(lmax + 1);
    expect(
      maxAbsDiff(Array.from(degreePower(x, lmax), Math.log1p), fx.power_features),
    ).toBeLessThan(1e-5);
  });

  it('reproduce the Python logits on the ten widget digits', () => {
    for (let i = 0; i < 10; i++) {
      const x = coefficients(i);
      const a = mlp.logits(inv.values(x));
      const b = cnn.logits(grid.synthesize(toHalfSpectrum(x, lmax)));
      expect(maxAbsDiff(a, fx.mlp_logits.slice(10 * i, 10 * i + 10))).toBeLessThan(1e-3);
      expect(maxAbsDiff(b, fx.cnn_logits.slice(10 * i, 10 * i + 10))).toBeLessThan(1e-3);
      const c = power.logits(degreePower(x, lmax));
      expect(maxAbsDiff(c, fx.power_logits.slice(10 * i, 10 * i + 10))).toBeLessThan(1e-3);
      expect(argmax(b)).toBe(fx.labels[i]);
    }
  });

  it('the bispectrum head ignores rotations, the pixel CNN does not', () => {
    const rng = new Rng(11);
    let moved = 0;
    for (let i = 0; i < 10; i++) {
      const x = coefficients(i);
      const y = rotateDofs(x, randomRotation(rng));
      const p = softmax(mlp.logits(inv.values(x)));
      const q = softmax(mlp.logits(inv.values(y)));
      expect(maxAbsDiff(p, q)).toBeLessThan(1e-9);
      const image = (c: Float64Array): Float64Array => grid.synthesize(toHalfSpectrum(c, lmax));
      moved += maxAbsDiff(softmax(cnn.logits(image(x))), softmax(cnn.logits(image(y))));
    }
    expect(moved / 10).toBeGreaterThan(0.3);
  });

  it('the power-spectrum head ignores rotations, and scrambles too', () => {
    const rng = new Rng(12);
    let moved = 0;
    for (let i = 0; i < 10; i++) {
      const x = coefficients(i);
      const p = softmax(power.logits(degreePower(x, lmax)));
      const turned = rotateDofs(x, randomRotation(rng));
      expect(maxAbsDiff(p, softmax(power.logits(degreePower(turned, lmax))))).toBeLessThan(1e-9);
      // Same power spectrum, another signal: this model cannot tell, the bispectrum one can.
      const other = scrambleDegrees(x, rng);
      expect(maxAbsDiff(p, softmax(power.logits(degreePower(other, lmax))))).toBeLessThan(1e-9);
      moved += maxAbsDiff(
        softmax(mlp.logits(inv.values(x))),
        softmax(mlp.logits(inv.values(other))),
      );
    }
    expect(moved / 10).toBeGreaterThan(0.3);
  });

  it('ships confusion matrices that add up to the accuracies', () => {
    for (const name of ['mlp', 'cnn', 'power'] as ModelName[]) {
      for (const split of ['test_nr', 'test_r'] as const) {
        const c = manifest.confusion[name][split];
        const total = c.flat().reduce((a, b) => a + b, 0);
        const hits = c.reduce((a, row, d) => a + row[d]!, 0);
        expect(total).toBe(10_000);
        expect(hits / total).toBeCloseTo(manifest.accuracy[name][split], 10);
      }
    }
  });
});
