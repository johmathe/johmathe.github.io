import { dataJson } from '../assets';
import type { Mount } from '../figure';
import { el } from './controls';

export interface BenchRow {
  module: string;
  /** What the benchmarked configuration is, in plain words. */
  setting: string;
  /** Numbers in the selective bispectrum, and in the full one where the library has it. */
  selective: number;
  full: number | null;
  /** Median forward time in milliseconds: autograd on, inference, and the full bispectrum. */
  train_ms: number;
  inference_ms: number;
  full_ms: number | null;
}

/** `bench.json`, written by `blog/scripts/export_bench.py` from a run of the paper's benchmark. */
export interface Bench {
  version: string;
  device: string;
  torch: string;
  cuda: string;
  date: string;
  batch: number;
  statistic: string;
  rows: BenchRow[];
}

export const ms = (v: number | null): string => (v === null ? '–' : v.toFixed(2));
const count = (n: number | null): string => (n === null ? '–' : n.toLocaleString('en-US'));

/** The library's timing table, one row per benchmarked module. */
export function benchTable(bench: Bench): HTMLElement {
  const head = (text: string, attrs: Record<string, string> = {}): HTMLElement =>
    el('th', { text, attrs: { scope: 'col', ...attrs } });
  const num = (text: string, cls = ''): HTMLElement =>
    el('td', { class: `num ${cls}`.trim(), text });
  return el(
    'table',
    { class: 'bench' },
    el(
      'thead',
      {},
      el(
        'tr',
        {},
        head('Module', { rowspan: '2' }),
        head('Benchmarked on', { rowspan: '2' }),
        head('Numbers out', { colspan: '2', class: 'group' }),
        head('Forward pass, ms', { colspan: '3', class: 'group' }),
      ),
      el(
        'tr',
        {},
        head('selective', { class: 'num' }),
        head('full', { class: 'num' }),
        head('training', { class: 'num' }),
        head('inference', { class: 'num' }),
        head('full', { class: 'num' }),
      ),
    ),
    el(
      'tbody',
      {},
      ...bench.rows.map((r) =>
        el(
          'tr',
          {},
          el('th', { attrs: { scope: 'row' } }, el('code', { text: r.module })),
          el('td', { text: r.setting }),
          num(count(r.selective), 'ours'),
          num(count(r.full)),
          num(ms(r.train_ms), 'ours'),
          num(ms(r.inference_ms), 'ours'),
          num(ms(r.full_ms)),
        ),
      ),
    ),
  );
}

export const mount: Mount = async (root) => {
  const bench = await dataJson<Bench>('bench.json');
  root.classList.add('bench-wrap');
  root.replaceChildren(benchTable(bench));
  return {};
};
