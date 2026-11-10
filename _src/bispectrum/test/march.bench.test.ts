/**
 * Solver benchmark, opt-in: `BENCH=1 npx vitest run test/march.bench.test.ts`.
 * `BENCH_LMAX`, `BENCH_DIGITS` and `BENCH_CONFIG` (a key of `CONFIGS`) select the runs; the Earth
 * is always the last target. `BENCH_SEED` moves the solver's random starts and `BENCH_STRICT=1`
 * fails unless every target is recovered.
 */
import { describe, expect, it } from 'vitest';

import { align } from '../src/math/align';
import { bandlimit, fromHalfSpectrum } from '../src/math/basis';
import { Invariant } from '../src/math/invariant';
import { march, PAPER_CONFIG, type MarchConfig } from '../src/math/march';
import { Rng } from '../src/math/rng';
import { SphereGrid } from '../src/math/sht';
import { liftToSphere } from '../src/math/smnist';
import { THOROUGH_CONFIG, WIDGET_CONFIG } from '../src/math/config';
import { earthSignal, type EarthManifest } from '../src/signals';
import { loadDataJson, loadDigits, loadTables } from './helpers';

const CONFIGS: Record<string, MarchConfig> = {
  paper: PAPER_CONFIG,
  thorough: THOROUGH_CONFIG,
  widget: WIDGET_CONFIG,
  oneSeed: { ...WIDGET_CONFIG, nSeedBranches: 1 },
  seed3: { ...PAPER_CONFIG, seedDegree: 3 },
};

const env = process.env;

describe.runIf(env['BENCH'])('march benchmark', () => {
  const lmaxes = (env['BENCH_LMAX'] ?? '8').split(',').map(Number);
  const nDigits = Number(env['BENCH_DIGITS'] ?? '10');
  const names = (env['BENCH_CONFIG'] ?? 'widget').split(',');
  const seed = Number(env['BENCH_SEED'] ?? '1000');
  const tables = loadTables();
  const digits = loadDigits();
  const grid = new SphereGrid(64, 128, 12);
  const earth = earthSignal(loadDataJson<EarthManifest>('earth.json'));

  for (const name of names) {
    for (const lmax of lmaxes) {
      it(`${name} at lmax ${lmax}`, () => {
        const cfg = CONFIGS[name]!;
        const inv = new Invariant(tables, lmax);
        const rows: string[] = [];
        let ok = 0;
        let total = 0;
        const targets = Array.from({ length: nDigits }, (_, d) => {
          const lifted = liftToSphere(digits.images[d]!, 28, grid.theta, grid.phi);
          return {
            name: `digit ${d} (label ${digits.labels[d]})`,
            x: bandlimit(fromHalfSpectrum(grid.analyze(lifted)), lmax),
          };
        });
        targets.push({ name: 'the Earth', x: bandlimit(earth.x, lmax) });
        for (const [d, { name: label, x }] of targets.entries()) {
          const target = inv.values(x);
          const t0 = performance.now();
          const res = march(inv, target, cfg, new Rng(seed + d));
          const ms = performance.now() - t0;
          total += ms;
          const scored = res.branches.map((b) => ({
            rel: b.rel,
            al: align(b.x, x, { rng: new Rng(d) }).rel,
          }));
          scored.sort((a, b) => a.rel - b.rel || a.al - b.al);
          const best = scored[0];
          const recovered = !!best && best.rel <= cfg.okTol && best.al <= 1e-6;
          ok += recovered ? 1 : 0;
          rows.push(
            `  ${label}: ${recovered ? 'ok  ' : 'FAIL'} ` +
              `${ms.toFixed(0).padStart(6)} ms, seed starts ${String(res.seedTried).padStart(3)}, ` +
              `alive ${res.degrees.map((g) => g.alive).join(',')}, ` +
              `feature ${best?.rel.toExponential(1) ?? 'n/a'}, aligned ${best?.al.toExponential(1) ?? 'n/a'}`,
          );
        }
        console.log(
          `${name} lmax ${lmax}: ${ok}/${targets.length} recovered, mean ${(total / targets.length).toFixed(0)} ms\n` +
            rows.join('\n'),
        );
        expect(ok).toBeGreaterThan(env['BENCH_STRICT'] ? targets.length - 1 : 0);
      });
    }
  }
});
