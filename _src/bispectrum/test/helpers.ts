import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildTables, type InvariantTables } from '../src/math/invariant';
import { parseDigits, type DigitSet } from '../src/data';

const here = dirname(fileURLToPath(import.meta.url));

export function fixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(here, 'fixtures', name), 'utf8')) as T;
}

const dataFile = (name: string): Buffer => readFileSync(join(here, '..', 'public', 'data', name));

const asArrayBuffer = (b: Buffer): ArrayBuffer =>
  b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

/** A file of `public/data` as an ArrayBuffer. */
export function loadDataFile(name: string): ArrayBuffer {
  return asArrayBuffer(dataFile(name));
}

/** A JSON file of `public/data`. */
export function loadDataJson<T>(name: string): T {
  return JSON.parse(dataFile(name).toString('utf8')) as T;
}

/** A file of this package (`blog/`) as text. */
export function readBlogFile(path: string): string {
  return readFileSync(join(here, '..', path), 'utf8');
}

export const blogRoot = join(here, '..');

export function loadTables(): InvariantTables {
  return buildTables(JSON.parse(dataFile('invariant.json').toString('utf8')));
}

export function loadDigits(): DigitSet {
  return parseDigits(
    JSON.parse(dataFile('digits.json').toString('utf8')),
    asArrayBuffer(dataFile('digits.bin')),
  );
}

/**
 * `RealSHBasis` in `degree_marching.py` builds its `sqrt 2` scalings with `torch.where(..., 1.0,
 * 2.0**0.5)`, a float32 tensor, so Python's real coordinates carry float32-rounded `sqrt 2` and
 * `1 / sqrt 2`. These convert between Python's coordinates and the exact ones used here.
 */
export function fromPythonDofs(x: ArrayLike<number>, lmax: number): Float64Array {
  const s = Math.fround(Math.SQRT1_2) * Math.SQRT2;
  return Float64Array.from({ length: x.length }, (_, d) => (isM0(d, lmax) ? x[d]! : x[d]! * s));
}

export function toPythonDofs(x: ArrayLike<number>, lmax: number): Float64Array {
  const s = Math.fround(Math.SQRT2) * Math.SQRT1_2;
  return Float64Array.from({ length: x.length }, (_, d) => (isM0(d, lmax) ? x[d]! : x[d]! * s));
}

/** Python Jacobian columns are derivatives w.r.t. Python coordinates. */
export function pythonDofScale(lmax: number): Float64Array {
  const s = Math.fround(Math.SQRT1_2) * Math.SQRT2;
  return Float64Array.from({ length: (lmax + 1) ** 2 }, (_, d) => (isM0(d, lmax) ? 1 : s));
}

function isM0(d: number, lmax: number): boolean {
  for (let l = 0; l <= lmax; l++) if (d === l * l) return true;
  return false;
}

export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i]! - b[i]!) ** 2;
    den += b[i]! ** 2;
  }
  return Math.sqrt(num / den);
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

/** Deterministic standard normals for tests. */
export function randn(n: number, seed: number): Float64Array {
  let s = seed >>> 0 || 1;
  const uni = (): number => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) + 0.5) / 4294967296;
  };
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sqrt(-2 * Math.log(uni())) * Math.cos(2 * Math.PI * uni());
  }
  return out;
}
