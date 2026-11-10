import type { DigitSet } from '../data';
import { digitThumb } from './drawpad';

type Child = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: { class?: string; text?: string; title?: string; attrs?: Record<string, string> } = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (props.class) e.className = props.class;
  if (props.text !== undefined) e.textContent = props.text;
  if (props.title) e.title = props.title;
  for (const [k, v] of Object.entries(props.attrs ?? {})) e.setAttribute(k, v);
  for (const c of children) if (c) e.append(c);
  return e;
}

export function button(
  label: string,
  onClick: () => void,
  opts: { kind?: 'primary' | 'quiet'; title?: string } = {},
): HTMLButtonElement {
  const b = el('button', {
    class: `rw-btn ${opts.kind ?? ''}`.trim(),
    text: label,
    ...(opts.title ? { title: opts.title } : {}),
    attrs: { type: 'button' },
  });
  b.addEventListener('click', onClick);
  return b;
}

export interface Slider {
  element: HTMLElement;
  input: HTMLInputElement;
  set(value: number, max?: number): void;
}

export function slider(opts: {
  label: string;
  min: number;
  max: number;
  value: number;
  format?: (v: number) => string;
  onInput: (v: number) => void;
  onChange?: (v: number) => void;
}): Slider {
  const fmt = opts.format ?? String;
  const input = el('input', {
    attrs: {
      type: 'range',
      min: String(opts.min),
      max: String(opts.max),
      step: '1',
      value: String(opts.value),
    },
  });
  const out = el('output', { text: fmt(opts.value) });
  const element = el(
    'label',
    { class: 'rw-slider' },
    el('span', { text: opts.label }),
    input,
    out,
  );
  input.addEventListener('input', () => {
    out.textContent = fmt(input.valueAsNumber);
    opts.onInput(input.valueAsNumber);
  });
  input.addEventListener('change', () => opts.onChange?.(input.valueAsNumber));
  return {
    element,
    input,
    set(value: number, max?: number) {
      if (max !== undefined) input.max = String(max);
      input.value = String(value);
      out.textContent = fmt(value);
    },
  };
}

export interface Toggle {
  element: HTMLElement;
  input: HTMLInputElement;
  set(on: boolean): void;
}

export function toggle(
  label: string,
  checked: boolean,
  onChange: (on: boolean) => void,
  title?: string,
): Toggle {
  const input = el('input', { attrs: { type: 'checkbox' } });
  input.checked = checked;
  const element = el(
    'label',
    { class: 'rw-toggle', ...(title ? { title } : {}) },
    input,
    el('span', { text: label }),
  );
  input.addEventListener('change', () => onChange(input.checked));
  return {
    element,
    input,
    set(on: boolean) {
      input.checked = on;
    },
  };
}

export function select<T extends string>(
  label: string,
  options: { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
): { element: HTMLElement; input: HTMLSelectElement } {
  const input = el('select');
  for (const o of options) {
    const opt = el('option', { text: o.label, attrs: { value: o.value } });
    opt.selected = o.value === value;
    input.append(opt);
  }
  input.addEventListener('change', () => onChange(input.value as T));
  return {
    element: el('label', { class: 'rw-select' }, el('span', { text: label }), input),
    input,
  };
}

export interface Segmented<T extends string> {
  element: HTMLElement;
  /** Selects `value` without calling `onChange`. */
  set(value: T): void;
}

/** Two or more mutually exclusive choices as a pill of buttons. */
export function segmented<T extends string>(
  label: string,
  options: { value: T; label: string; title?: string }[],
  value: T,
  onChange: (v: T) => void,
): Segmented<T> {
  const element = el('div', { class: 'rw-seg', attrs: { role: 'group', 'aria-label': label } });
  const press = (v: T): void =>
    buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i]!.value === v)));
  const buttons = options.map((o) => {
    const b = el('button', {
      text: o.label,
      ...(o.title ? { title: o.title } : {}),
      attrs: { type: 'button', 'aria-pressed': String(o.value === value) },
    });
    b.addEventListener('click', () => {
      if (b.getAttribute('aria-pressed') === 'true') return;
      press(o.value);
      onChange(o.value);
    });
    return b;
  });
  element.append(...buttons);
  return { element, set: press };
}

export type Pick = number | 'draw' | 'earth';

/** Thumbnails of the bundled test digits, an optional Earth tile, and a tile to draw. */
export class DigitPicker {
  readonly element: HTMLElement;
  private readonly tiles = new Map<Pick, HTMLButtonElement>();

  constructor(
    digits: DigitSet,
    shown: number,
    onPick: (p: Pick) => void,
    opts: { earth?: boolean } = {},
  ) {
    this.element = el('div', {
      class: 'rw-picker',
      attrs: { role: 'group', 'aria-label': 'Digit' },
    });
    for (let i = 0; i < Math.min(shown, digits.images.length); i++) {
      const tile = el(
        'button',
        {
          class: 'rw-tile',
          title: `MNIST test image ${digits.testIndex[i]} (label ${digits.labels[i]})`,
          attrs: { type: 'button' },
        },
        digitThumb(digits.images[i]!, digits.size),
      );
      tile.addEventListener('click', () => onPick(i));
      this.tiles.set(i, tile);
      this.element.append(tile);
    }
    if (opts.earth) {
      const earth = el('button', {
        class: 'rw-tile rw-tile-draw',
        text: 'Earth',
        title: 'Land and sea',
        attrs: { type: 'button' },
      });
      earth.addEventListener('click', () => onPick('earth'));
      this.tiles.set('earth', earth);
      this.element.append(earth);
    }
    const draw = el('button', {
      class: 'rw-tile rw-tile-draw',
      text: 'draw',
      attrs: { type: 'button' },
    });
    draw.addEventListener('click', () => onPick('draw'));
    this.tiles.set('draw', draw);
    this.element.append(draw);
  }

  setActive(p: Pick): void {
    for (const [k, t] of this.tiles) t.classList.toggle('active', k === p);
  }
}
