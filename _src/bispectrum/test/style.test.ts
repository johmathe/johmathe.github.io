import { describe, expect, it } from 'vitest';

import { readBlogFile } from './helpers';
import { hasClass, inside, parse, type Element, type TextRun } from './html';

/** Pages held to the house style of `llm-style-review.md`. */
const PAGES = ['index.html', 'post.html'];

/** Colons inside a sentence, per thousand words. One that ends a block and introduces the next
 * (display math, code, a list) is structural and doesn't count. */
const COLONS_PER_1000 = 6;

const BLOCKS = new Set(['p', 'li', 'figcaption', 'summary', 'h1', 'h2', 'h3', 'h4', 'td', 'th']);
const MATH = /\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g;

const skipped = (e: Element): boolean =>
  e.tag === 'code' ||
  e.tag === 'pre' ||
  e.tag === 'footer' ||
  e.part === 'references' ||
  hasClass(e, 'refs') ||
  hasClass(e, 'bibtex');

describe.each(PAGES)('%s', (file) => {
  const doc = parse(readBlogFile(file));
  const within = (outer: Element, e: Element): boolean =>
    e !== outer && e.start >= outer.start && e.end <= outer.end;
  const blocks = doc.elements.filter((e) => BLOCKS.has(e.tag) && !skipped(e));
  /** Blocks with no block inside them, so that no text is counted twice. */
  const leaves = blocks.filter((b) => !blocks.some((c) => within(b, c) && c.end > c.start));
  const prose = (r: TextRun): boolean => inside(r, (e) => e.tag === 'body') && !inside(r, skipped);
  const texts = leaves
    .map((b) =>
      doc.runs
        .slice(b.start, b.end)
        .filter(prose)
        .map((r) => r.text)
        .join('')
        .replace(MATH, ' ')
        .replace(/\s+/g, ' ')
        .trim(),
    )
    .filter((t) => t.length > 0);

  it('has no semicolons', () => {
    expect(texts.filter((t) => t.includes(';'))).toEqual([]);
  });

  it('has no em dashes', () => {
    expect(texts.filter((t) => t.includes('—'))).toEqual([]);
  });

  it(`keeps colons inside sentences under ${COLONS_PER_1000} per 1,000 words`, () => {
    const words = texts.reduce(
      (n, t) => n + (t.match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu) ?? []).length,
      0,
    );
    const inner = texts.filter((t) => t.slice(0, -1).includes(':'));
    const count = inner.reduce((n, t) => n + (t.slice(0, -1).match(/:/g) ?? []).length, 0);
    expect(
      count * 1000,
      `${count} colons in ${words} words:\n${inner.join('\n')}`,
    ).toBeLessThanOrEqual(COLONS_PER_1000 * words);
  });
});
