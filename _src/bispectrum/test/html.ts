/**
 * A small tokenizer for `index.html`: enough to walk our own markup in document order with each
 * text run's ancestors, without a DOM library.
 */

export interface Element {
  tag: string;
  attrs: Record<string, string>;
  /** `id` of the closest `section[id]` at or before this element. */
  part: string;
  /** Text runs `[start, end)` inside the element. */
  start: number;
  end: number;
}

export interface TextRun {
  text: string;
  ancestors: Element[];
  part: string;
}

export interface Document {
  runs: TextRun[];
  elements: Element[];
  /** Text content of `el`, whitespace collapsed. */
  text(el: Element): string;
}

const VOID = new Set(['area', 'br', 'col', 'hr', 'img', 'input', 'link', 'meta', 'source', 'wbr']);
const RAW = new Set(['script', 'style']);
const TOKEN =
  /<!--[\s\S]*?-->|<!doctype[^>]*>|<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|([^<]+)/gi;
const ATTR = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

const decode = (s: string): string =>
  s
    .replace(/&nbsp;/g, '\u00a0')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');

export function parse(html: string): Document {
  const runs: TextRun[] = [];
  const elements: Element[] = [];
  const stack: Element[] = [];
  let part = '';
  let raw: string | null = null;
  for (const m of html.matchAll(TOKEN)) {
    const [, close, tagName, attrText, text] = m;
    if (raw) {
      if (close && tagName?.toLowerCase() === raw) {
        stack.pop()!.end = runs.length;
        raw = null;
      }
      continue;
    }
    if (text !== undefined) {
      runs.push({ text: decode(text), ancestors: [...stack], part });
      continue;
    }
    if (!tagName) continue;
    const tag = tagName.toLowerCase();
    if (close) {
      let i = stack.length - 1;
      while (i >= 0 && stack[i]!.tag !== tag) i--;
      if (i < 0) throw new Error(`stray </${tag}> at offset ${m.index}`);
      for (const e of stack.splice(i)) e.end = runs.length;
      continue;
    }
    const attrs: Record<string, string> = {};
    for (const a of (attrText ?? '').matchAll(ATTR)) {
      attrs[a[1]!.toLowerCase()] = decode(a[2] ?? a[3] ?? a[4] ?? '');
    }
    if (tag === 'section' && attrs['id']) part = attrs['id'];
    const el: Element = { tag, attrs, part, start: runs.length, end: runs.length };
    elements.push(el);
    if (VOID.has(tag) || (attrText ?? '').trimEnd().endsWith('/')) continue;
    stack.push(el);
    if (RAW.has(tag)) raw = tag;
  }
  if (stack.length) throw new Error(`unclosed <${stack.at(-1)!.tag}>`);
  return {
    runs,
    elements,
    text: (el) =>
      runs
        .slice(el.start, el.end)
        .map((r) => r.text)
        .join('')
        .replace(/\s+/g, ' ')
        .trim(),
  };
}

export const hasClass = (el: Element, cls: string): boolean =>
  (el.attrs['class'] ?? '').split(/\s+/).includes(cls);

/** Whether a run sits inside an element matching `pred`. */
export const inside = (run: TextRun, pred: (el: Element) => boolean): boolean =>
  run.ancestors.some(pred);
