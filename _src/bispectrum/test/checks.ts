/**
 * Numbers the prose takes from page data. In `index.html` and `deck.html` each one is written as
 * `<span data-check="key">value</span>`, and `prose.test.ts` and `deck.test.ts` compare the text
 * with the value computed here from `public/data`, or for the paper's results, from a fixture
 * checked against `paper/`.
 */

import { dim } from '../src/math/basis';
import { FullBispectrum } from '../src/math/bispectrum';
import type { ClassifierManifest } from '../src/math/classifier';
import type { InvariantManifest } from '../src/math/invariant';
import { fullCount } from '../src/math/ring';
import type { EarthManifest } from '../src/signals';
import { ms, type Bench } from '../src/ui/bench';
import { BYTES_PER_NUMBER, type Counts, type SphereCounts } from '../src/ui/wall';
import { fixture, loadDataJson } from './helpers';

export const int = (n: number): string => n.toLocaleString('en-US');
export const pct = (v: number, digits = 1): string => `${(100 * v).toFixed(digits)}%`;

const invariant = (): InvariantManifest => loadDataJson<InvariantManifest>('invariant.json');
const classifier = (): ClassifierManifest => loadDataJson<ClassifierManifest>('classifier.json');
const bench = (): Bench => loadDataJson<Bench>('bench.json');

const sphere = (key: keyof SphereCounts, L: number): number =>
  loadDataJson<Counts>('counts.json').sphere[key][L - 1]!;

/** Clebsch–Gordan terms of the entry at degrees `(a, b, c)`: pairs `(m1, m2)`, `|m1 + m2| <= c`. */
export function cgTerms(a: number, b: number, c: number): number {
  let n = 0;
  for (let m = -a; m <= a; m++) n += Math.max(0, Math.min(b, c - m) - Math.max(-b, -c - m) + 1);
  return n;
}

/**
 * The full spherical bispectrum on a real signal with each entry counted once: orderings of a
 * triple of degrees are proportional, and an entry with an odd sum and a repeated degree is zero.
 */
export function distinctFull(L: number): { entries: number; terms: number } {
  let [entries, terms] = [0, 0];
  for (let a = 0; a <= L; a++) {
    for (let b = a; b <= L; b++) {
      for (let c = b; c <= Math.min(a + b, L); c++) {
        if ((a + b + c) % 2 === 1 && (a === b || b === c)) continue;
        entries++;
        terms += cgTerms(a, b, c);
      }
    }
  }
  return { entries, terms };
}

/** Wrong answers in a confusion matrix `[true digit][predicted digit]`. */
const errors = (c: number[][]): number =>
  c.flat().reduce((a, b) => a + b, 0) - c.reduce((a, row, i) => a + row[i]!, 0);
/** Times digits `a` and `b` were taken for each other. */
const mixed = (c: number[][], a: number, b: number): number => c[a]![b]! + c[b]![a]!;

/** Band-limit of every figure but the opening one. */
const PAGE_L = 12;
/** Band-limit of the opening figure: the Earth's. */
const HERO_L = (): number => loadDataJson<EarthManifest>('earth.json').lmax;

/** Entries of the selective invariant at `L`, all or of one kind. */
const entries = (L: number, kind?: 'triples' | 'energies'): string =>
  int(
    invariant().tags.filter(
      (t, i) =>
        invariant().row_degree[i]! <= L &&
        (kind === undefined || (t === 'cg_power') === (kind === 'energies')),
    ).length,
  );

/** The power-spectrum model's answer on the widget's 2: its digit and its probability. */
function powerOnTwo(): { digit: number; p: number } {
  const fx = fixture<{ labels: number[]; power_logits: number[] }>('classifier.json');
  const i = fx.labels.indexOf(2);
  const z = fx.power_logits.slice(10 * i, 10 * i + 10);
  const e = z.map((v) => Math.exp(v - Math.max(...z)));
  const digit = e.indexOf(Math.max(...e));
  return { digit, p: e[digit]! / e.reduce((a, b) => a + b, 0) };
}

export interface SmnistRow {
  params: string;
  /** Mean accuracy on rotated test digits after training on upright ones (NR/R), in percent. */
  rotated: number;
  ms: string;
}

/** The rows of the paper's Spherical MNIST table, by the start of their label. */
export const SMNIST_ROWS = {
  ours: 'Bispectrum + MLP',
  s2cnn: 'S2CNN + BatchNorm',
  tp: 'Tensor-product net',
  power: 'Power spectrum + MLP',
  cnn: 'CNN (equirectangular)',
} as const;

export type SmnistKey = keyof typeof SMNIST_ROWS;

/**
 * The paper's Spherical MNIST rows and its speed-ups over the spherical CNN and the
 * tensor-product network. The site that vendors this package has no `paper/`, so the numbers are
 * kept in a fixture, which `paper.test.ts` compares with the paper when it is there.
 */
export interface SmnistFixture {
  rows: Record<SmnistKey, SmnistRow>;
  speedups: [string, string];
  /** Unrounded median latencies in ms, from the raw H100 timing records, for the ratios. */
  latency_ms: Record<'ours' | 's2cnn' | 'tp', number>;
}

const smnist = (): SmnistFixture => fixture<SmnistFixture>('smnist.json');

const smnistChecks = Object.fromEntries(
  (Object.keys(SMNIST_ROWS) as SmnistKey[]).flatMap((k) => [
    [`smnist.${k}.params`, () => smnist().rows[k].params],
    [`smnist.${k}.acc`, () => `${smnist().rows[k].rotated.toFixed(1)}%`],
    [`smnist.${k}.ms`, () => smnist().rows[k].ms],
  ]),
) as Record<string, () => string>;

export const CHECKS: Record<string, () => string> = {
  ...smnistChecks,
  'smnist.vsS2': () => (smnist().latency_ms.s2cnn / smnist().latency_ms.ours).toFixed(0),
  'smnist.vsTp': () => (smnist().latency_ms.tp / smnist().latency_ms.ours).toFixed(0),
  'smnist.ours.msRaw': () => smnist().latency_ms.ours.toFixed(3),
  'page.L': () => String(PAGE_L),
  'fp.n': () => entries(PAGE_L),
  'fp.triples': () => entries(PAGE_L, 'triples'),
  'fp.energies': () => entries(PAGE_L, 'energies'),
  'coeffs.n': () => int(dim(PAGE_L)),
  'hero.L': () => String(HERO_L()),
  'hero.coeffs': () => int(dim(HERO_L())),
  'hero.power': () => int(HERO_L() + 1),
  'hero.fp': () => entries(HERO_L()),
  'hero.triples': () => entries(HERO_L(), 'triples'),
  'hero.energies': () => entries(HERO_L(), 'energies'),
  'hero.lower': () => int(dim(HERO_L()) - 3),
  'earth.land': () => pct(loadDataJson<EarthManifest>('earth.json').land_fraction, 0),
  'power.n': () => int(PAGE_L + 1),
  'full.n': () => int(new FullBispectrum(PAGE_L).n),
  'lower.n': () => int(dim(PAGE_L) - 3),
  'wall.pixels': () => int(256 * 256),
  'wall.full': () => int(fullCount(256 * 256)),
  'wall.gb': () => ((fullCount(256 * 256) * BYTES_PER_NUMBER) / 1e9).toFixed(0),
  'torus32.n': () => int(32 * 32),
  'torus32.full': () => int(fullCount(32 * 32)),
  'sphere15.coeffs': () => int(sphere('coefficients', 15)),
  'sphere15.full': () => int(sphere('full', 15)),
  'sphere15.sel': () => int(sphere('selective', 15)),
  'sphere15.cubic': () => entries(15, 'triples'),
  'sphere15.quartic': () => entries(15, 'energies'),
  'sphere15.channels': () => int(2 * sphere('selective', 15)),
  'sphere64.full': () => int(sphere('full', 64)),
  'sphere64.sel': () => int(sphere('selective', 64)),
  'sphere64.fullTerms': () => `${Math.round(sphere('full_terms', 64) / 1e6)} million`,
  'sphere64.ratio': () => (sphere('full_terms', 64) / sphere('selective_terms', 64)).toFixed(2),
  'sphere64.fullTermsExact': () => int(sphere('full_terms', 64)),
  'sphere64.selTermsExact': () => int(sphere('selective_terms', 64)),
  'sphere64.storedRatio': () => (sphere('full', 64) / sphere('selective', 64)).toFixed(2),
  'sphere64.selTerms': () => `${(sphere('selective_terms', 64) / 1e6).toFixed(1)} million`,
  'sphere15.distinct': () => int(distinctFull(15).entries),
  'sphere64.distinct': () => int(distinctFull(64).entries),
  'sphere64.distinctRatio': () => (distinctFull(64).entries / sphere('selective', 64)).toFixed(2),
  'sphere64.distinctTermsRatio': () =>
    (distinctFull(64).terms / sphere('selective_terms', 64)).toFixed(0),
  'bench.version': () => bench().version,
  'bench.trainMin': () => ms(Math.min(...bench().rows.map((r) => r.train_ms))),
  'bench.trainMax': () => ms(Math.max(...bench().rows.map((r) => r.train_ms))),
  'bench.infMin': () => ms(Math.min(...bench().rows.map((r) => r.inference_ms))),
  'bench.infMax': () => ms(Math.max(...bench().rows.map((r) => r.inference_ms))),
  'bench.selMin': () => int(Math.min(...bench().rows.map((r) => r.selective))),
  'bench.selMax': () => int(Math.max(...bench().rows.map((r) => r.selective))),
  'mlp.nr': () => pct(classifier().accuracy.mlp.test_nr),
  'mlp.r': () => pct(classifier().accuracy.mlp.test_r),
  'cnn.nr': () => pct(classifier().accuracy.cnn.test_nr),
  'cnn.r': () => pct(classifier().accuracy.cnn.test_r),
  'power.nr': () => pct(classifier().accuracy.power.test_nr),
  'power.r': () => pct(classifier().accuracy.power.test_r),
  'mlp.errors': () => int(errors(classifier().confusion.mlp.test_nr)),
  'mlp.sixnine': () => int(mixed(classifier().confusion.mlp.test_nr, 6, 9)),
  'mlp.nr2': () => pct(classifier().accuracy.mlp.test_nr, 2),
  'mlp.r2': () => pct(classifier().accuracy.mlp.test_r, 2),
  'mlp.errorsR': () => int(errors(classifier().confusion.mlp.test_r)),
  'mlp.sixnineR': () => int(mixed(classifier().confusion.mlp.test_r, 6, 9)),
  'power.sixnine': () => int(mixed(classifier().confusion.power.test_nr, 6, 9)),
  'two.power': () => String(powerOnTwo().digit),
  'two.powerP': () => pct(powerOnTwo().p, 0),
  'test.n': () =>
    int(
      classifier()
        .confusion.mlp.test_nr.flat()
        .reduce((a, b) => a + b, 0),
    ),
};
