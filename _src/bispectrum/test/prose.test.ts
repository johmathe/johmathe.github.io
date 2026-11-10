import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { CHECKS } from './checks';
import { blogRoot, readBlogFile } from './helpers';
import { hasClass, inside, parse, type Element, type TextRun } from './html';
import { DEFINED, NAMED_ONCE, PREVIEWED, UNUSED, type Term } from './ledger';

const doc = parse(readBlogFile('index.html'));
const placeholders = doc.elements.filter((e) => hasClass(e, 'placeholder'));
/** The introduction: the masthead, the hook and `#orbits`, up to the first section after it. */
const introEnd = doc.runs.findIndex((r) => r.part === 'ring');
/** Section `id` to the `id` of its chapter, the section of the last `h2` at or before it. */
const chapterOf = new Map<string, string>();
let chapter = '';
for (const e of doc.elements) {
  if (e.tag === 'h2') chapter = e.part;
  if (e.tag === 'h2' || (e.tag === 'section' && e.attrs['id'])) chapterOf.set(e.part, chapter);
}

const isPlaceholder = (e: Element): boolean => hasClass(e, 'placeholder');
const isCode = (e: Element): boolean => e.tag === 'code' || e.tag === 'pre';
const isReference = (e: Element): boolean =>
  e.part === 'references' || e.tag === 'footer' || hasClass(e, 'refs') || hasClass(e, 'bibtex');
const isAside = (e: Element): boolean =>
  e.tag === 'aside' || e.tag === 'details' || 'data-ledger-skip' in e.attrs;
/** Text that counts as running prose for the ledger (in `<body>`). */
const prose = (r: TextRun): boolean =>
  inside(r, (e) => e.tag === 'body') && !inside(r, (e) => isPlaceholder(e) || isCode(e));

const ATTRS = ['title', 'alt', 'aria-label', 'data-loading', 'content'];

function uses(term: Term, ok: (r: TextRun) => boolean): string[] {
  const hits: string[] = [];
  for (const r of doc.runs) if (ok(r) && term.pattern.test(r.text)) hits.push(r.text.trim());
  for (const e of doc.elements) {
    for (const a of ATTRS) {
      const v = e.attrs[a];
      if (v && term.pattern.test(v)) hits.push(`<${e.tag} ${a}="${v}">`);
    }
  }
  return hits;
}

/** String literals in the widget sources, with comments removed. */
function widgetStrings(): { file: string; text: string }[] {
  const out: { file: string; text: string }[] = [];
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (name.endsWith('.ts')) {
        const code = readFileSync(path, 'utf8')
          .replace(/\/\*[\s\S]*?\*\//g, '')
          .replace(/(^|[^:])\/\/.*$/gm, '$1');
        for (const m of code.matchAll(
          /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`/g,
        )) {
          out.push({ file: path.slice(blogRoot.length + 1), text: m[0] });
        }
      }
    }
  };
  walk(join(blogRoot, 'src'));
  return out;
}

describe('numbers in the prose', () => {
  const checked = doc.elements.filter((e) => 'data-check' in e.attrs);

  it('computes every check from the data', () => {
    for (const [key, value] of Object.entries(CHECKS)) expect(value(), key).toMatch(/\S/);
  });

  it.each(checked.map((e) => [e.attrs['data-check']!, doc.text(e)]))(
    '%s reads %s',
    (key, text) => {
      const expected = CHECKS[key];
      expect(expected, `no check named ${key} in test/checks.ts`).toBeDefined();
      expect(text).toBe(expected!());
    },
  );
});

describe('contents', () => {
  const squash = (e: Element): string => doc.text(e).replace(/\s+/g, ' ').trim();
  const within = (outer: Element, e: Element): boolean =>
    e.start >= outer.start && e.end <= outer.end;
  const nav = doc.elements.find((e) => e.tag === 'nav' && hasClass(e, 'toc'));
  const inNav = (tag: string): Element[] =>
    nav ? doc.elements.filter((e) => e.tag === tag && within(nav, e)) : [];
  const links = inNav('a');
  const lists = inNav('ol');
  const items = inNav('li');
  /** 1 for a chapter, 2 for a subsection. */
  const depth = (e: Element): number => lists.filter((ol) => within(ol, e)).length;

  it('lists every heading, in order, chapters above their subsections', () => {
    expect(nav, 'no <nav class="toc">').toBeDefined();
    const headings = doc.elements
      .filter((e) => e.tag === 'h2' || e.tag === 'h3')
      .map((e) => [`#${e.part}`, squash(e), e.tag === 'h2' ? 1 : 2]);
    expect(links.map((a) => [a.attrs['href'], squash(a), depth(a)])).toEqual(headings);
  });

  it('nests each subsection inside the item of its chapter', () => {
    let chapterLink: Element | undefined;
    for (const a of links) {
      if (depth(a) === 1) {
        chapterLink = a;
        continue;
      }
      const item = items.find((li) => depth(li) === 1 && within(li, a));
      expect(item && chapterLink && within(item, chapterLink), a.attrs['href']).toBe(true);
    }
  });
});

describe('term ledger', () => {
  const dfns = doc.elements.filter((e) => e.tag === 'dfn');
  const termOf = (e: Element): string => e.attrs['data-term'] ?? doc.text(e).toLowerCase();

  it('defines no term twice and only ledger terms', () => {
    const terms = dfns.map(termOf);
    expect(new Set(terms).size, `duplicates in ${terms.join(', ')}`).toBe(terms.length);
    const known = new Set(DEFINED.map((t) => t.term));
    expect(terms.filter((t) => !known.has(t))).toEqual([]);
  });

  it.each(DEFINED.map((t) => [t.term, t] as const))(
    '%s is defined in its section before any use',
    (_, term) => {
      const dfn = dfns.find((e) => termOf(e) === term.term);
      // The headings of the chapter that defines a term may name it, and so may their entries
      // in the contents.
      const home = chapterOf.get(term.part);
      const sameChapter = (part: string): boolean =>
        home !== undefined && chapterOf.get(part) === home;
      const heading = (r: TextRun): boolean =>
        (sameChapter(r.part) && inside(r, (e) => e.tag === 'h2' || e.tag === 'h3')) ||
        (inside(r, (e) => e.tag === 'nav') &&
          inside(r, (e) => e.tag === 'a' && sameChapter((e.attrs['href'] ?? '').slice(1))));
      const previewed = (i: number): boolean => PREVIEWED.has(term.term) && i < introEnd;
      // The title is the paper's, ahead of every definition.
      const masthead = (r: TextRun): boolean => inside(r, (e) => e.tag === 'header');
      const first = doc.runs.findIndex(
        (r, i) =>
          prose(r) && !masthead(r) && !heading(r) && !previewed(i) && term.pattern.test(r.text),
      );
      if (!dfn) {
        // A section still written as placeholder beats owes its definitions later.
        if (placeholders.some((e) => e.part === term.part)) return;
        throw new Error(`no <dfn> for "${term.term}"`);
      }
      expect(dfn.part, `the <dfn> of "${term.term}" belongs in #${term.part}`).toBe(term.part);
      expect(
        first,
        `"${term.term}" is used before its <dfn>: "${doc.runs[first]?.text.trim()}"`,
      ).toBeGreaterThanOrEqual(dfn.start);
    },
  );

  it.each(NAMED_ONCE.map((t) => [t.term, t] as const))(
    '%s appears only in fold-outs, asides, code and references',
    (_, term) => {
      const outside = (r: TextRun): boolean =>
        prose(r) && !inside(r, (e) => isAside(e) || isReference(e));
      expect(uses(term, outside)).toEqual([]);
    },
  );

  it.each(UNUSED.map((t) => [t.term, t] as const))('%s is not on the page', (_, term) => {
    expect(uses(term, (r) => prose(r) && !inside(r, isReference))).toEqual([]);
  });

  it('keeps the widget strings free of named-once and unused terms', () => {
    const strings = widgetStrings();
    const hits = [...NAMED_ONCE, ...UNUSED].flatMap((t) =>
      strings.filter((s) => t.pattern.test(s.text)).map((s) => `${s.file}: ${s.text}`),
    );
    expect(hits).toEqual([]);
  });
});
