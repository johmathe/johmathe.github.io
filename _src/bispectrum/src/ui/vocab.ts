/**
 * Widget wording that depends on the page. `post.html` sets `data-vocab="plain"` on `<html>`: it
 * never names the invariant descriptor and quotes only the paper's band-limit, so its widgets say
 * "invariants" and leave out the sizes and scores of the browser's own models.
 */
export const plain = (): boolean =>
  typeof document !== 'undefined' && document.documentElement.dataset['vocab'] === 'plain';

/** `full` on pages that define the descriptor, `plainText` on the others. */
export const say = (full: string, plainText: string): string => (plain() ? plainText : full);
