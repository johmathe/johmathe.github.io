/**
 * The signals the figures show: MNIST digits and drawings lifted to the sphere, and the Earth.
 * Each one is a coefficient vector at rest plus the flat picture it came from, so a figure can
 * draw "the same thing, unrolled" next to the sphere.
 */

import { dataJson } from './assets';
import { fromHalfSpectrum } from './math/basis';
import { rotateDofs, type Mat3 } from './math/rotation';
import type { SphereGrid } from './math/sht';
import { liftToSphere } from './math/smnist';

/** A flat picture of the sphere: the unit vector each pixel looks at, row-major, 3 per pixel. */
export interface FlatProjection {
  width: number;
  height: number;
  dirs: Float64Array;
}

export interface Signal {
  kind: 'digit' | 'earth';
  /** Real coordinates at rest (`math/basis.ts`), up to the band-limit of the tables. */
  x: Float64Array;
  flat: FlatProjection;
}

/**
 * The image plane of the Spherical MNIST lift, read the way digits show on screen (rows flipped,
 * as `liftToSphere` with `mirror`): pixel `(row, col)` sits at `(x, y) = (2u - 1, 1 - 2v)` and
 * looks at colatitude `2 atan |(x, y)|`, longitude `atan2(y, x)`.
 */
export function digitPlane(size: number): FlatProjection {
  const dirs = new Float64Array(3 * size * size);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const x = (2 * (col + 0.5)) / size - 1;
      const y = 1 - (2 * (row + 0.5)) / size;
      const theta = 2 * Math.atan(Math.hypot(x, y));
      const phi = Math.atan2(y, x);
      dirs.set(
        [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)],
        3 * (row * size + col),
      );
    }
  }
  return { width: size, height: size, dirs };
}

export const DIGIT_PLANE = digitPlane(56);

/** A digit or drawing (`size x size` pixels in `[0, 1]`) lifted to the sphere, upright. */
export function digitSignal(image: ArrayLike<number>, size: number, analysis: SphereGrid): Signal {
  const f = liftToSphere(image, size, analysis.theta, analysis.phi, { mirror: true });
  return { kind: 'digit', x: fromHalfSpectrum(analysis.analyze(f)), flat: DIGIT_PLANE };
}

const rad = (deg: number): number => (deg * Math.PI) / 180;

/** Unit vector of a point of the globe: `z` through the north pole, `x` through 0°N 0°E. */
export function geo(lat: number, lon: number): [number, number, number] {
  const [a, b] = [rad(lat), rad(lon)];
  return [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
}

/**
 * Rotation that brings `(lat, lon)` to the pole facing the reader, with north up: its rows are
 * east, north and the outward normal at that point.
 */
export function facing(lat: number, lon: number): Mat3 {
  const [a, b] = [rad(lat), rad(lon)];
  return Float64Array.of(
    -Math.sin(b),
    Math.cos(b),
    0,
    -Math.sin(a) * Math.cos(b),
    -Math.sin(a) * Math.sin(b),
    Math.cos(a),
    ...geo(lat, lon),
  );
}

/** Equirectangular world map, north up, Greenwich in the middle, for a globe posed by `pose`. */
export function worldMap(width: number, height: number, pose: Mat3): FlatProjection {
  const dirs = new Float64Array(3 * width * height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const d = geo(90 - (180 * (row + 0.5)) / height, -180 + (360 * (col + 0.5)) / width);
      for (let i = 0; i < 3; i++) {
        dirs[3 * (row * width + col) + i] =
          pose[3 * i]! * d[0] + pose[3 * i + 1]! * d[1] + pose[3 * i + 2]! * d[2];
      }
    }
  }
  return { width, height, dirs };
}

/** The view the Earth rests in: Africa and Europe face the reader. */
export const EARTH_VIEW = { lat: 18, lon: 18 };

export interface EarthManifest {
  lmax: number;
  /** Land (1) and sea (0) to degree `lmax`, with `z` through the north pole. */
  x: number[];
  land_fraction: number;
  source: string;
}

export function earthSignal(manifest: EarthManifest): Signal {
  const pose = facing(EARTH_VIEW.lat, EARTH_VIEW.lon);
  return { kind: 'earth', x: rotateDofs(manifest.x, pose), flat: worldMap(128, 64, pose) };
}

export const loadEarth = async (): Promise<Signal> =>
  earthSignal(await dataJson<EarthManifest>('earth.json'));
