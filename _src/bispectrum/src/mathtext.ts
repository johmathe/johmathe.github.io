import katex from 'katex';
import 'katex/dist/katex.min.css';

const DELIMITED = /\\\((.+?)\\\)|\\\[(.+?)\\\]/gs;
/** Punctuation that must not start a line after inline math. */
const TRAILING = /^[.,:;!?)\]]+/;
/** Longer inline math keeps its own break points instead of moving as one block. */
const MAX_GLUED = 100;

/** Typesets `\( inline \)` and `\[ display \]` math in the text under `root`. */
export function renderMath(root: Element, skip: (el: Element) => boolean = () => false): void {
  const nodes: Text[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const t = n as Text;
    if (
      t.parentElement?.closest('script, style, .katex') ||
      (t.parentElement && skip(t.parentElement))
    ) {
      continue;
    }
    if (t.data.includes('\\(') || t.data.includes('\\[')) nodes.push(t);
  }
  for (const t of nodes) {
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of t.data.matchAll(DELIMITED)) {
      frag.append(t.data.slice(last, m.index));
      const span = document.createElement('span');
      const display = m[2] !== undefined;
      const tex = (m[1] ?? m[2])!;
      katex.render(tex, span, { displayMode: display, throwOnError: false });
      last = m.index + m[0].length;
      const tail = !display && tex.length <= MAX_GLUED ? TRAILING.exec(t.data.slice(last)) : null;
      if (tail) {
        // Chrome breaks after KaTeX's inline blocks unless the math itself is nowrap.
        const glued = document.createElement('span');
        glued.className = 'math-glued';
        glued.append(span, tail[0]);
        frag.append(glued);
        last += tail[0].length;
      } else {
        frag.append(span);
      }
    }
    frag.append(t.data.slice(last));
    t.replaceWith(frag);
  }
}
