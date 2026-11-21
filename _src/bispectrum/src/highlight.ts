/**
 * Python syntax colours for the code blocks, as classes styled in `style.css`: `k` keywords, `s`
 * strings, `n` numbers, `f` called names, `a` keyword-argument names, `p` punctuation and
 * operators, `c` comments. Names that are none of these stay plain.
 */

type Token = 'c' | 's' | 'k' | 'n' | 'f' | 'a' | 'v' | 'p';

const KEYWORDS = [
  'and',
  'as',
  'class',
  'def',
  'elif',
  'else',
  'except',
  'False',
  'finally',
  'for',
  'from',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'None',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'True',
  'try',
  'while',
  'with',
  'yield',
];

// Earlier alternatives win, and every name is consumed whole so that no rule fires mid-word.
const PYTHON = new RegExp(
  [
    String.raw`(?<c>#[^\n]*)`,
    String.raw`(?<s>"""[\s\S]*?"""|'''[\s\S]*?'''|'[^'\n]*'|"[^"\n]*")`,
    String.raw`(?<k>\b(?:${KEYWORDS.join('|')})\b)`,
    String.raw`(?<n>\b\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\b)`,
    String.raw`(?<f>\b[A-Za-z_]\w*(?=\())`,
    String.raw`(?<a>\b[A-Za-z_]\w*(?==[^=]))`,
    String.raw`(?<v>\b[A-Za-z_]\w*)`,
    String.raw`(?<p>[()[\]{},.:;=+\-*/%<>!&|^~@])`,
  ].join('|'),
  'g',
);

const escape = (text: string): string =>
  text.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[ch] ?? ch);

/** `src` as HTML, each token wrapped in a span of its class. */
export function highlightPython(src: string): string {
  let html = '';
  let at = 0;
  for (const m of src.matchAll(PYTHON)) {
    const kind = Object.entries(m.groups ?? {}).find(([, v]) => v !== undefined)?.[0] as Token;
    html += escape(src.slice(at, m.index));
    html += kind === 'v' ? escape(m[0]) : `<span class="${kind}">${escape(m[0])}</span>`;
    at = m.index + m[0].length;
  }
  return html + escape(src.slice(at));
}

/** Recolours every Python block under `root` from its text, replacing any hand-set spans. */
export function highlightCode(root: ParentNode): void {
  for (const code of root.querySelectorAll<HTMLElement>(
    'pre[data-snippet] > code, pre[data-lang="python"] > code',
  )) {
    code.innerHTML = highlightPython(code.textContent ?? '');
  }
}
