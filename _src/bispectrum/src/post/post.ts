import '../main';
import { art, isArtName } from '../deck/art';
import { icon, isIconName } from '../deck/icons';
import { reducedMotion } from '../ui/canvas';
import { animateTiles } from './tiles';

for (const slot of document.querySelectorAll<HTMLElement>('[data-icon]')) {
  const name = slot.dataset.icon ?? '';
  if (!isIconName(name)) throw new Error(`no icon named ${name}`);
  slot.innerHTML = icon(name);
}
for (const list of document.querySelectorAll('.gts')) animateTiles(list);
for (const slot of document.querySelectorAll<HTMLElement>('[data-art]')) {
  const name = slot.dataset.art ?? '';
  if (!isArtName(name)) throw new Error(`no art named ${name}`);
  slot.innerHTML = art(name);
}
// Each accuracy cell draws its own bar, as wide as the number it holds.
for (const cell of document.querySelectorAll<HTMLElement>('.rs .acc')) {
  cell.style.setProperty('--v', String(parseFloat(cell.textContent ?? '0')));
}
// The classifiers spin from the start and have no spin toggle, so with reduced motion they hold
// the 2 still instead. Figures mount on first view, after this runs.
if (reducedMotion()) {
  for (const root of document.querySelectorAll<HTMLElement>('[data-figure="classify"]')) {
    root.dataset.start = 'two';
  }
}
