import { defineConfig } from 'vitest/config';

// Link previews (og:image) only work with absolute URLs, and the post does not know where it
// will be served. Build with the public address of the page's folder:
//   SITE_URL=https://example.org/blog/bispectrum/ npm run build
// Without it the card is referenced relative to the page, which browsers resolve and most
// scrapers don't.
const siteUrl = process.env.SITE_URL ? process.env.SITE_URL.replace(/\/?$/, '/') : './';

export default defineConfig({
  base: './',
  plugins: [
    {
      name: 'site-url',
      transformIndexHtml: (html: string): string => html.replaceAll('__SITE_URL__', siteUrl),
    },
  ],
  // The brand fonts live in ../assets/brand.
  server: { fs: { allow: ['..'] } },
  worker: { format: 'es' },
  // three.js is most of the bundle (~150 kB gzipped).
  build: { target: 'es2022', chunkSizeWarningLimit: 800 },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    testTimeout: 120_000,
  },
});
