import type { Mount } from './figure';

/** One entry per `[data-figure]` name; each figure is its own chunk. */
export const FIGURES: Record<string, () => Promise<Mount>> = {
  hero: () => import('./ui/hero').then((m) => m.mount),
  ring: () => import('./ui/ring').then((m) => m.mount),
  wave: () => import('./ui/wave').then((m) => m.mount),
  bridge: () => import('./ui/bridge').then((m) => m.mount),
  wall: () => import('./ui/wall').then((m) => m.mount),
  recon: () => import('./ui/widget').then((m) => m.mount),
  bench: () => import('./ui/bench').then((m) => m.mount),
  classify: () => import('./ui/classify').then((m) => m.mount),
  playground: () => import('./ui/widget').then((m) => m.mountPlayground),
};
