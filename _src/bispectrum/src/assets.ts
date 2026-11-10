import {
  fetchAssets,
  fetchClassifier,
  fetchJson,
  parseDigits,
  tablesFrom,
  type ClassifierAssets,
  type DigitSet,
  type RawAssets,
} from './data';
import type { InvariantTables } from './math/invariant';

/** What every sphere figure needs: the invariant tables and the bundled digits. */
export interface SharedAssets {
  raw: RawAssets;
  tables: InvariantTables;
  digits: DigitSet;
}

const base = import.meta.env.BASE_URL;
const cache = new Map<string, Promise<unknown>>();

function once<T>(key: string, load: () => Promise<T>): Promise<T> {
  let p = cache.get(key) as Promise<T> | undefined;
  if (!p) {
    p = load();
    cache.set(key, p);
  }
  return p;
}

/** Tables and digits, fetched and parsed once for all figures. */
export const sharedAssets = (): Promise<SharedAssets> =>
  once('shared', async () => {
    const raw = await fetchAssets(base);
    return {
      raw,
      tables: tablesFrom(raw),
      digits: parseDigits(raw.digitManifest, raw.digitBin),
    };
  });

export const classifierAssets = (): Promise<ClassifierAssets> =>
  once('classifier', () => fetchClassifier(base));

/** A JSON file under `public/data`, fetched once. */
export const dataJson = <T>(name: string): Promise<T> =>
  once(`json:${name}`, () => fetchJson<T>(base, name));
