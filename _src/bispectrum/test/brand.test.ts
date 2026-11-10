import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { hex, TINTS } from '../src/ui/colormap';
import { blogRoot, readBlogFile } from './helpers';

/** Color tokens of `assets/brand/__init__.py`, by name. */
const TOKENS: Record<string, string> = Object.fromEntries(
  [
    ...readFileSync(join(blogRoot, '..', 'assets', 'brand', '__init__.py'), 'utf8').matchAll(
      /^([A-Z_]+): Final = '(#[0-9A-Fa-f]{6})'/gm,
    ),
  ].map((m) => [m[1]!, m[2]!.toUpperCase()]),
);

const ALLOWED = new Set([
  ...Object.values(TOKENS),
  ...Object.values(TINTS).map((t) => t.hex.toUpperCase()),
]);

function sources(): { file: string; text: string }[] {
  const out: { file: string; text: string }[] = [
    { file: 'index.html', text: readBlogFile('index.html') },
  ];
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.(ts|css)$/.test(name)) {
        out.push({ file: path.slice(blogRoot.length + 1), text: readFileSync(path, 'utf8') });
      }
    }
  };
  walk(join(blogRoot, 'src'));
  return out;
}

describe('brand', () => {
  it('reads the tokens from assets/brand', () => {
    expect(TOKENS['INK']).toBe('#1C1C1A');
    expect(TOKENS['ORANGE']).toBe('#F57050');
    expect(TOKENS['ORANGE_TEXT']).toBe('#B53E20');
  });

  it.each(Object.entries(TINTS))('tint %s is the opaque mix it claims to be', (_, t) => {
    const [a, b] = [hex(t.of), hex(t.over)];
    const mix = a.map((c, i) => Math.round(t.strength * c + (1 - t.strength) * b[i]!));
    expect(hex(t.hex)).toEqual(mix);
    expect(ALLOWED.has(t.of.toUpperCase())).toBe(true);
  });

  it('uses no color outside the tokens and declared tints', () => {
    const bad: string[] = [];
    for (const { file, text } of sources()) {
      for (const m of text.matchAll(
        /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![\w-])/g,
      )) {
        if (!ALLOWED.has(m[0].toUpperCase())) bad.push(`${file}: ${m[0]}`);
      }
      for (const m of text.matchAll(/\b(?:rgba|hsla?)\(/g))
        bad.push(`${file}: ${m[0]} (use a tint)`);
    }
    expect(bad).toEqual([]);
  });

  it('sets type in IBM Plex Sans and Mono only', () => {
    const bad: string[] = [];
    for (const { file, text } of sources()) {
      for (const m of text.matchAll(/font-family:\s*([^;]+);/g)) {
        if (!/^(var\(--(sans|mono)\)|'IBM Plex (Sans|Mono)')$/.test(m[1]!.trim())) {
          bad.push(`${file}: font-family: ${m[1]}`);
        }
      }
      for (const m of text.matchAll(/--(sans|mono):\s*([^;]+);/g)) {
        if (!m[2]!.trim().startsWith("'IBM Plex")) bad.push(`${file}: --${m[1]}: ${m[2]}`);
      }
      for (const m of text.matchAll(/\.font = (['`])([^'`]*)\1/g)) {
        if (!/IBM Plex (Sans|Mono)/.test(m[2]!)) bad.push(`${file}: ctx.font = ${m[2]}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it('shows the wordmark from its SVG, never retyped', () => {
    expect(readBlogFile('index.html')).toMatch(/<img[^>]*data-brand="wordmark"/);
    expect(readBlogFile('src/main.ts')).toContain('assets/brand/bispectrum-wordmark.svg');
  });
});
