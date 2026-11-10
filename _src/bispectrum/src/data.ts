import type { ClassifierManifest } from './math/classifier';
import { buildTables, type InvariantManifest, type InvariantTables } from './math/invariant';

export interface DigitManifest {
  n: number;
  size: number;
  labels: number[];
  mnist_test_index: number[];
}

export interface DigitSet {
  size: number;
  labels: number[];
  testIndex: number[];
  /** Pixels in `[0, 1]`, row-major `(size, size)` per digit. */
  images: Float32Array[];
}

export function parseDigits(manifest: DigitManifest, buffer: ArrayBuffer): DigitSet {
  const px = manifest.size * manifest.size;
  const bytes = new Uint8Array(buffer);
  if (bytes.length !== manifest.n * px) {
    throw new Error(`digits.bin has ${bytes.length} bytes, expected ${manifest.n * px}`);
  }
  return {
    size: manifest.size,
    labels: manifest.labels,
    testIndex: manifest.mnist_test_index,
    images: Array.from({ length: manifest.n }, (_, i) =>
      Float32Array.from(bytes.subarray(i * px, (i + 1) * px), (v) => v / 255),
    ),
  };
}

export interface RawAssets {
  invariantManifest: InvariantManifest;
  digitManifest: DigitManifest;
  digitBin: ArrayBuffer;
}

async function get(base: string, name: string): Promise<Response> {
  const res = await fetch(`${base}data/${name}`);
  if (!res.ok) throw new Error(`failed to load ${name}: ${res.status}`);
  return res;
}

export async function fetchJson<T>(base: string, name: string): Promise<T> {
  return (await get(base, name)).json() as Promise<T>;
}

export async function fetchBinary(base: string, name: string): Promise<ArrayBuffer> {
  return (await get(base, name)).arrayBuffer();
}

export async function fetchAssets(base: string): Promise<RawAssets> {
  const [im, dm, db] = await Promise.all([
    get(base, 'invariant.json').then((r) => r.json() as Promise<InvariantManifest>),
    get(base, 'digits.json').then((r) => r.json() as Promise<DigitManifest>),
    get(base, 'digits.bin').then((r) => r.arrayBuffer()),
  ]);
  return { invariantManifest: im, digitManifest: dm, digitBin: db };
}

export function tablesFrom(raw: RawAssets): InvariantTables {
  return buildTables(raw.invariantManifest);
}

export interface ClassifierAssets {
  manifest: ClassifierManifest;
  mlpBin: ArrayBuffer;
  cnnBin: ArrayBuffer;
  powerBin: ArrayBuffer;
}

export async function fetchClassifier(base: string): Promise<ClassifierAssets> {
  const manifest = (await (await get(base, 'classifier.json')).json()) as ClassifierManifest;
  const [mlpBin, cnnBin, powerBin] = await Promise.all(
    [manifest.mlp.file, manifest.cnn.file, manifest.power.file].map(async (f) =>
      (await get(base, f)).arrayBuffer(),
    ),
  );
  return { manifest, mlpBin: mlpBin!, cnnBin: cnnBin!, powerBin: powerBin! };
}
