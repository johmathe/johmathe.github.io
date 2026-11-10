/**
 * The post's term ledger: which technical words may appear, and where.
 *
 * - `ASSUMED`: the stated reader knows them (an ML researcher at ease with linear algebra, Fourier
 *   transforms, groups and equivariant networks, but not with representation theory).
 * - `DEFINED`: defined at first use, in a `<dfn>` inside the section that owns the term, and
 *   never used before it.
 * - `PREVIEWED`: defined terms the introduction (the masthead through `#orbits`) may name before
 *   the section that develops them; after the introduction they follow the `DEFINED` rule.
 * - `NAMED_ONCE`: named, with a gloss, only in a fold-out, an aside, code or the references.
 * - `UNUSED`: not on the page at all, nor in any widget string.
 */

export interface Term {
  /** As written in `<dfn data-term>` (or as the `<dfn>` text). */
  term: string;
  /** Matches the term and its inflections in running text. */
  pattern: RegExp;
}

export interface DefinedTerm extends Term {
  /** `id` of the section whose `<dfn>` defines it. */
  part: string;
}

const words = (alternation: string): RegExp => new RegExp(`\\b(?:${alternation})\\b`, 'i');

export const ASSUMED = [
  'FFT',
  'Fourier coefficient',
  'frequency',
  'amplitude',
  'phase',
  'complex number',
  'CNN',
  'MLP',
  'pooling',
  'invariant',
  'group',
  'SO(2)',
  'SO(3)',
  'equivariant',
  'group convolution',
  'Jacobian',
  'measure zero',
  'PyTorch',
  'GPU',
  'CUDA graph',
];

export const DEFINED: DefinedTerm[] = [
  {
    term: 'invariant descriptor',
    part: 'orbits',
    pattern: words('(?:invariant )?descriptors?'),
  },
  { term: 'pose', part: 'orbits', pattern: words('poses?') },
  { term: 'orbit', part: 'orbits', pattern: words('orbits?') },
  { term: 'complete', part: 'orbits', pattern: words('complete|completeness') },
  { term: 'power spectrum', part: 'ring', pattern: words('power[ -]spectr(?:um|a)') },
  { term: 'bispectrum', part: 'triple-products', pattern: words('bispectr(?:um|a|al)') },
  {
    term: 'irreducible representation',
    part: 'group-fourier',
    pattern: words('irreducible representations?|irreps?'),
  },
  { term: 'generic', part: 'noncommutative', pattern: words('generic(?:ally)?') },
  {
    term: 'Clebsch-Gordan coefficient',
    part: 'noncommutative',
    pattern: words('Clebsch[-–]Gordan'),
  },
  { term: 'Wigner D-matrix', part: 'sphere', pattern: words('Wigner') },
  { term: 'spherical harmonic', part: 'sphere', pattern: words('spherical harmonics?') },
  { term: 'degree', part: 'sphere', pattern: words('degrees?') },
  { term: 'band-limit', part: 'sphere', pattern: words('band[- ]?limit(?:s|ed)?') },
  { term: 'selective', part: 'ring-finite', pattern: words('selective|selectivity') },
];

export const PREVIEWED = new Set([
  'pose',
  'complete',
  'power spectrum',
  'bispectrum',
  'generic',
  'spherical harmonic',
  'degree',
  'band-limit',
  'selective',
]);

export const NAMED_ONCE: Term[] = [
  { term: 'closure phase', pattern: words('closure phases?') },
  { term: 'speckle masking', pattern: words('speckle masking') },
  { term: 'cryo-EM', pattern: words('cryo-?EM') },
];

export const UNUSED: Term[] = [
  { term: 'fingerprint', pattern: words('fingerprints?') },
  { term: 'hand', pattern: words('hands?') },
  { term: 'Kronecker table', pattern: words('Kronecker') },
  { term: 'Levenberg-Marquardt', pattern: words('Levenberg') },
  { term: 'NR/R', pattern: /\bNR\/R\b/ },
  { term: 'pp', pattern: /\bpp\b/ },
];
