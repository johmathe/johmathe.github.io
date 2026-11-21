import { describe, expect, it } from 'vitest';

import { highlightPython } from '../src/highlight';
import { readBlogFile } from './helpers';

const PAGES = ['index.html', 'post.html'];
const PYTHON_BLOCK =
  /<pre[^>]*(?:data-snippet="[^"]+"|data-lang="python")[^>]*><code[^>]*>([\s\S]*?)<\/code><\/pre>/g;

/** What the reader sees of an HTML fragment. */
const text = (html: string): string =>
  html
    .replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');

/** `[class, text]` of every span in `html`. */
const spans = (html: string): [string, string][] =>
  [...html.matchAll(/<span class="(\w)">([^<]*)<\/span>/g)].map((m) => [m[1] ?? '', m[2] ?? '']);

describe.each(PAGES)('%s code blocks', (file) => {
  const blocks = [...readBlogFile(file).matchAll(PYTHON_BLOCK)].map((m) => text(m[1] ?? ''));

  it('has Python to colour', () => {
    expect(blocks.length).toBeGreaterThan(0);
  });

  it('keeps every character of the code', () => {
    for (const src of blocks) expect(text(highlightPython(src))).toBe(src);
  });
});

describe('highlightPython', () => {
  it('marks keywords, called names, argument names, literals and punctuation', () => {
    expect(spans(highlightPython("x = F.interpolate(x, size=(64, 128), mode='bicubic')"))).toEqual(
      [
        ['p', '='],
        ['p', '.'],
        ['f', 'interpolate'],
        ['p', '('],
        ['p', ','],
        ['a', 'size'],
        ['p', '='],
        ['p', '('],
        ['n', '64'],
        ['p', ','],
        ['n', '128'],
        ['p', ')'],
        ['p', ','],
        ['a', 'mode'],
        ['p', '='],
        ['s', "'bicubic'"],
        ['p', ')'],
      ],
    );
    expect(spans(highlightPython('from bispectrum import SO3onS2 as S'))).toEqual([
      ['k', 'from'],
      ['k', 'import'],
      ['k', 'as'],
    ]);
  });

  it('keeps comments whole, quotes included, and never colours inside a name', () => {
    expect(spans(highlightPython("print(err)  # the grid's error"))).toEqual([
      ['f', 'print'],
      ['p', '('],
      ['p', ')'],
      ['c', "# the grid's error"],
    ]);
    expect(spans(highlightPython('SO3onS2 import_x x1'))).toEqual([]);
  });

  it('escapes HTML', () => {
    expect(highlightPython('a < b & c')).toBe(
      'a <span class="p">&lt;</span> b <span class="p">&amp;</span> c',
    );
  });
});
