import { PAPER_CONFIG, type MarchConfig } from './march';

/**
 * Solver settings used by the widget (see `test/march.bench.test.ts`).
 *
 * Seed starts are cheap here (about 4 ms each), and on some digits only about 1 in 200 random
 * starts reaches the true low-degree orbit, so the seed budget is raised from 200 to 2000. Past
 * the seed, each branch stops at its first consistent completion instead of searching for a
 * second one; that search is what `PAPER_CONFIG` is for (it looks for collisions).
 */
export const WIDGET_CONFIG: MarchConfig = {
  ...PAPER_CONFIG,
  nSeedInits: 2000,
  singleSolutionFromDegree: PAPER_CONFIG.seedDegree + 1,
};

export const THOROUGH_CONFIG: MarchConfig = { ...PAPER_CONFIG, nSeedInits: 2000 };
