import type { Figure, Mount } from './figure';
// The brand files live outside this package, so they go through the bundler instead of tags in
// index.html (which the dev server cannot resolve).
import faviconUrl from '../../assets/brand/bispectrum-favicon.svg';
import markUrl from '../../assets/brand/bispectrum-mark.svg';
import wordmarkUrl from '../../assets/brand/bispectrum-wordmark.svg';

const icon = document.createElement('link');
icon.rel = 'icon';
icon.type = 'image/svg+xml';
icon.href = faviconUrl;
document.head.append(icon);

const BRAND: Record<string, string> = { mark: markUrl, wordmark: wordmarkUrl };
// The seven module glyphs, as shipped: `<img data-brand="glyph-CnonCn">`.
const glyphs = import.meta.glob<string>(
  ['../../assets/brand/glyphs/*.svg', '!**/*-reversed.svg'],
  { eager: true, query: '?url', import: 'default' },
);
for (const [path, url] of Object.entries(glyphs)) {
  BRAND[`glyph-${path.slice(path.lastIndexOf('/') + 1, -'.svg'.length)}`] = url;
}
for (const img of document.querySelectorAll<HTMLImageElement>('img[data-brand]')) {
  const url = BRAND[img.dataset.brand ?? ''];
  if (url) img.src = url;
}

void import('./mathtext').then(({ renderMath }) =>
  renderMath(document.body, (el) => el.closest('[data-figure]') !== null),
);

/** One entry per `[data-figure]` element; each figure is its own chunk, loaded near the viewport. */
const FIGURES: Record<string, () => Promise<Mount>> = {
  hero: () => import('./ui/hero').then((m) => m.mount),
  ring: () => import('./ui/ring').then((m) => m.mount),
  bridge: () => import('./ui/bridge').then((m) => m.mount),
  wall: () => import('./ui/wall').then((m) => m.mount),
  recon: () => import('./ui/widget').then((m) => m.mount),
  bench: () => import('./ui/bench').then((m) => m.mount),
  classify: () => import('./ui/classify').then((m) => m.mount),
  playground: () => import('./ui/widget').then((m) => m.mountPlayground),
};

const mounted = new Map<HTMLElement, Promise<Figure | null>>();

async function build(root: HTMLElement): Promise<Figure | null> {
  const load = FIGURES[root.dataset.figure ?? ''];
  if (!load) return null;
  root.classList.add('rw-loading');
  root.textContent = root.dataset.loading ?? 'Loading…';
  try {
    return await (
      await load()
    )(root);
  } catch (err) {
    root.textContent = `The figure failed to load: ${String(err)}`;
    console.error(err);
    return null;
  } finally {
    root.classList.remove('rw-loading');
  }
}

function ensure(root: HTMLElement): Promise<Figure | null> {
  let p = mounted.get(root);
  if (!p) {
    p = build(root);
    mounted.set(root, p);
  }
  return p;
}

const roots = [...document.querySelectorAll<HTMLElement>('[data-figure]')];
const near = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      near.unobserve(e.target);
      void ensure(e.target as HTMLElement);
    }
  },
  { rootMargin: '900px 0px' },
);
roots.forEach((r) => near.observe(r));

// The contents list marks the last section whose top has passed a third of the way down, and
// opens the chapter it belongs to.
const tocLinks = [...document.querySelectorAll<HTMLAnchorElement>('nav.toc a[href^="#"]')];
const tocTargets = tocLinks.map((a) => document.getElementById(a.hash.slice(1)));
const tocChapters = tocLinks.map((a) => a.closest('nav.toc > ol > li'));
const toc = document.querySelector<HTMLElement>('nav.toc');
const masthead = document.querySelector<HTMLElement>('.masthead');
let tocFrame = 0;
function markToc(): void {
  tocFrame = 0;
  if (toc && masthead) {
    const below = masthead.getBoundingClientRect().bottom + 40;
    toc.style.setProperty('--toc-top', `${Math.max(96, below)}px`);
  }
  const atEnd = innerHeight + scrollY >= document.documentElement.scrollHeight - 2;
  let current = -1;
  tocTargets.forEach((t, i) => {
    if (t && t.getBoundingClientRect().top < innerHeight / 3) current = i;
  });
  if (atEnd) current = tocTargets.length - 1;
  tocLinks.forEach((a, i) => {
    if (i === current) a.setAttribute('aria-current', 'location');
    else a.removeAttribute('aria-current');
  });
  const open = tocChapters[current];
  for (const li of new Set(tocChapters)) li?.classList.toggle('toc-open', li === open);
}
addEventListener('scroll', () => (tocFrame ||= requestAnimationFrame(markToc)), { passive: true });
addEventListener('resize', () => (tocFrame ||= requestAnimationFrame(markToc)));
markToc();

// Numbered caption steps put their figure in the state the sentence describes.
document.addEventListener('click', (e) => {
  const btn = (e.target as Element | null)?.closest<HTMLButtonElement>('button[data-step]');
  const root = btn?.closest('figure')?.querySelector<HTMLElement>('[data-figure]');
  if (!btn || !root) return;
  for (const other of btn.closest('ol')?.querySelectorAll('button[data-step]') ?? []) {
    other.classList.toggle('active', other === btn);
  }
  void ensure(root).then((fig) => fig?.apply?.(btn.dataset.step!));
});
