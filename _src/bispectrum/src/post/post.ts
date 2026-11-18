import '../main';
import { art, isArtName } from '../deck/art';
import { icon, isIconName } from '../deck/icons';
import atmoLogo from '../deck/logos/atmo.svg?raw';
import uclouvainLogo from '../deck/logos/uclouvain.svg?raw';
import ucsbLogo from '../deck/logos/ucsb.svg?raw';

const LOGOS: Record<string, string> = { atmo: atmoLogo, ucsb: ucsbLogo, uclouvain: uclouvainLogo };

for (const slot of document.querySelectorAll<HTMLElement>('[data-logo]')) {
  const svg = LOGOS[slot.dataset.logo ?? ''];
  if (!svg) throw new Error(`no logo named ${slot.dataset.logo}`);
  slot.innerHTML = svg;
}
for (const slot of document.querySelectorAll<HTMLElement>('[data-icon]')) {
  const name = slot.dataset.icon ?? '';
  if (!isIconName(name)) throw new Error(`no icon named ${name}`);
  slot.innerHTML = icon(name);
}
for (const slot of document.querySelectorAll<HTMLElement>('[data-art]')) {
  const name = slot.dataset.art ?? '';
  if (!isArtName(name)) throw new Error(`no art named ${name}`);
  slot.innerHTML = art(name);
}
// Each accuracy cell draws its own bar, as wide as the number it holds.
for (const cell of document.querySelectorAll<HTMLElement>('.rs .acc')) {
  cell.style.setProperty('--v', String(parseFloat(cell.textContent ?? '0')));
}
