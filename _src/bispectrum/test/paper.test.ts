import { existsSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { SMNIST_ROWS, type SmnistFixture, type SmnistKey, type SmnistRow } from './checks';
import { blogRoot, fixture, readRepoFile } from './helpers';

const TABLE = 'paper/experiments/smnist_figure/smnist_protocol_table.tex';
const PAPER = 'paper/bispectrum.tex';
const TIMING = 'experiments/tooling/results/smnist_baselines/timing_h100.json';

/** A row of the paper's Spherical MNIST table: model & params & NR/NR & R/R & NR/R & latency. */
function tableRow(tex: string, key: SmnistKey): SmnistRow {
  const line = tex
    .split('\n')
    .find((l) => l.replace(/^\\textbf\{/, '').startsWith(SMNIST_ROWS[key]));
  if (!line) throw new Error(`no Spherical MNIST row for ${key}`);
  const cells = line.split('&').map((c) => c.replace(/\\\\\s*$/, '').trim());
  return { params: cells[1]!, rotated: Number(/\$([\d.]+)/.exec(cells[4]!)![1]), ms: cells[5]! };
}

/** The vendored copy of the blog has no `paper/`. There, the fixture is taken as is. */
const hasPaper = [TABLE, PAPER].every((p) => existsSync(join(blogRoot, '..', p)));

describe.skipIf(!hasPaper)('the Spherical MNIST fixture', () => {
  const fx = fixture<SmnistFixture>('smnist.json');

  it.each(Object.keys(SMNIST_ROWS) as SmnistKey[])('matches the paper table for %s', (key) => {
    expect(fx.rows[key]).toEqual(tableRow(readRepoFile(TABLE), key));
  });

  it('keeps the unrounded latencies of the raw H100 timing records', () => {
    if (!existsSync(join(blogRoot, '..', TIMING))) return;
    const raw = JSON.parse(readRepoFile(TIMING)) as {
      results: { model: string; batch_size: number; median_ms: number; error?: string | null }[];
    };
    const at16 = (model: string): number =>
      raw.results.find((r) => r.model === model && r.batch_size === 16 && !r.error)!.median_ms;
    expect(fx.latency_ms).toEqual({
      ours: at16('bispectrum'),
      s2cnn: at16('s2cnn_bn'),
      tp: at16('tp_c24_l6'),
    });
    const ratio = (k: 's2cnn' | 'tp'): string =>
      (fx.latency_ms[k] / fx.latency_ms.ours).toFixed(0);
    expect([ratio('s2cnn'), ratio('tp')]).toEqual(fx.speedups);
    for (const k of ['ours', 's2cnn', 'tp'] as const) {
      expect(fx.rows[k].ms).toBe(fx.latency_ms[k].toFixed(fx.rows[k].ms.split('.')[1]!.length));
    }
  });

  it('matches the speed-ups in the paper', () => {
    const m = /at\s+\$(\d+)\\times\$ and \$(\d+)\\times\$ lower latency/.exec(readRepoFile(PAPER));
    expect(m, 'no Spherical MNIST speed-ups in the paper').not.toBeNull();
    expect(fx.speedups).toEqual([m![1], m![2]]);
  });
});
