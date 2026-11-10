import { describe, expect, it } from 'vitest';

import { align } from '../src/math/align';
import { bandlimit, toHalfSpectrum } from '../src/math/basis';
import { WIDGET_CONFIG } from '../src/math/config';
import { Invariant } from '../src/math/invariant';
import { march } from '../src/math/march';
import { Rng } from '../src/math/rng';
import { evaluateAt, SphereGrid } from '../src/math/sht';
import {
  DIGIT_PLANE,
  digitSignal,
  EARTH_VIEW,
  earthSignal,
  facing,
  geo,
  type EarthManifest,
} from '../src/signals';
import { loadDataJson, loadDigits, loadTables, maxAbsDiff } from './helpers';

const manifest = loadDataJson<EarthManifest>('earth.json');
/** The widget's band-limit; the hero shows the Earth to `HERO_LMAX`. */
const LMAX = 12;
const HERO_LMAX = 20;

/** Land (about 1) or sea (about 0) at a point of the globe, to the hero's degree. */
const landAt = (lat: number, lon: number): number =>
  evaluateAt(toHalfSpectrum(manifest.x, manifest.lmax), geo(lat, lon))[0]!;

describe('the Earth', () => {
  it('is land and sea where the globe is', () => {
    expect(manifest.lmax).toBe(HERO_LMAX);
    expect(manifest.land_fraction).toBeCloseTo(0.29, 2);
    const land: [string, number, number][] = [
      ['Sahara', 22, 12],
      ['Siberia', 62, 100],
      ['Amazon', -8, -60],
      ['Antarctica', -82, 60],
      ['Australia', -25, 134],
    ];
    const sea: [string, number, number][] = [
      ['central Pacific', 0, -150],
      ['South Pacific', -40, -120],
      ['mid Atlantic', 25, -45],
      ['Indian Ocean', -30, 80],
    ];
    for (const [name, lat, lon] of land) expect(landAt(lat, lon), name).toBeGreaterThan(0.6);
    for (const [name, lat, lon] of sea) expect(landAt(lat, lon), name).toBeLessThan(0.3);
  });

  it('rests with its view centre facing the reader and north up', () => {
    const pose = facing(EARTH_VIEW.lat, EARTH_VIEW.lon);
    const centre = geo(EARTH_VIEW.lat, EARTH_VIEW.lon);
    const apply = (v: number[]): number[] =>
      [0, 1, 2].map(
        (i) => pose[3 * i]! * v[0]! + pose[3 * i + 1]! * v[1]! + pose[3 * i + 2]! * v[2]!,
      );
    expect(maxAbsDiff(apply(centre), [0, 0, 1])).toBeLessThan(1e-12);
    // The north pole lands on the +y side: up on screen.
    expect(apply([0, 0, 1])[1]).toBeGreaterThan(0);
    expect(Math.abs(apply([0, 0, 1])[0]!)).toBeLessThan(1e-12);

    const earth = earthSignal(manifest);
    const posed = toHalfSpectrum(earth.x, HERO_LMAX);
    expect(evaluateAt(posed, [0, 0, 1])[0]).toBeCloseTo(
      landAt(EARTH_VIEW.lat, EARTH_VIEW.lon),
      10,
    );
    // Its world map reads the same land and sea as the globe: row 0 is the north, column 0 180°W.
    const { width, height, dirs } = earth.flat;
    const map = evaluateAt(posed, dirs);
    for (const [row, col] of [
      [10, 20],
      [40, 100],
      [60, 5],
    ] as const) {
      const lat = 90 - (180 * (row + 0.5)) / height;
      const lon = -180 + (360 * (col + 0.5)) / width;
      expect(map[row * width + col]).toBeCloseTo(landAt(lat, lon), 10);
    }
  });

  it.each([LMAX, HERO_LMAX])(
    'is recovered from its descriptor at L = %i, up to a rotation',
    (L) => {
      const x = bandlimit(earthSignal(manifest).x, L);
      const inv = new Invariant(loadTables(), L);
      const res = march(inv, inv.values(x), WIDGET_CONFIG, new Rng(4));
      const scored = res.branches
        .map((b) => ({ rel: b.rel, aligned: align(b.x, x, { rng: new Rng(1) }).rel }))
        .sort((a, b) => a.rel - b.rel || a.aligned - b.aligned);
      expect(scored[0]?.rel).toBeLessThan(WIDGET_CONFIG.okTol);
      expect(scored[0]?.aligned).toBeLessThan(1e-6);
    },
  );
});

describe('the digit plane', () => {
  it('shows a lifted digit the way it was drawn', () => {
    const digits = loadDigits();
    const grid = new SphereGrid(64, 128, LMAX);
    const image = digits.images[2]!;
    const signal = digitSignal(image, digits.size, grid);
    expect(signal.flat).toBe(DIGIT_PLANE);
    const flat = evaluateAt(toHalfSpectrum(signal.x, LMAX), DIGIT_PLANE.dirs);
    // Average the flat picture down to the 28 x 28 pixels of the source and correlate.
    const k = DIGIT_PLANE.width / digits.size;
    const small = Float64Array.from({ length: digits.size ** 2 }, (_, i) => {
      const [r, c] = [Math.floor(i / digits.size), i % digits.size];
      let s = 0;
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) s += flat[(r * k + a) * DIGIT_PLANE.width + c * k + b]!;
      }
      return s / (k * k);
    });
    const mean = (v: ArrayLike<number>): number =>
      Array.from(v).reduce((a, b) => a + b, 0) / v.length;
    const [ma, mb] = [mean(small), mean(image)];
    let [ab, aa, bb] = [0, 0, 0];
    for (let i = 0; i < small.length; i++) {
      ab += (small[i]! - ma) * (image[i]! - mb);
      aa += (small[i]! - ma) ** 2;
      bb += (image[i]! - mb) ** 2;
    }
    expect(ab / Math.sqrt(aa * bb)).toBeGreaterThan(0.8);
  });
});
