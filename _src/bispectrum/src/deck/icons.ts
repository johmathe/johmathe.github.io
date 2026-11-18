/**
 * Line icons for the seven groups of the `groups` slide, on a 100 × 100 box: the domain in ink, the
 * action of the group on it in orange. Colors come from `deck.css`.
 */

export type IconName = 'ring' | 'circle' | 'torus' | 'polygon' | 'disk' | 'sphere' | 'cube';

const C = 50;
export const f = (n: number): string => n.toFixed(2);
const at = (r: number, a: number, ry = r, cx = C, cy = C): [number, number] => [
  cx + r * Math.cos(a),
  cy + ry * Math.sin(a),
];

/** An arrowhead at `(x, y)`, pointing along angle `t`. */
function head(x: number, y: number, t: number): string {
  return [t + Math.PI - 0.5, t + Math.PI + 0.5]
    .map((b) => `M${f(x + 8 * Math.cos(b))} ${f(y + 8 * Math.sin(b))} L${f(x)} ${f(y)}`)
    .join(' ');
}

/** An elliptical arc from angle `a0` to `a1 > a0` (clockwise on screen), with its arrowhead. */
function turn(rx: number, ry: number, a0: number, a1: number, cx = C, cy = C): string {
  const [x0, y0] = at(rx, a0, ry, cx, cy);
  const [x1, y1] = at(rx, a1, ry, cx, cy);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const t = Math.atan2(ry * Math.cos(a1), -rx * Math.sin(a1));
  return `<path class="act" d="M${f(x0)} ${f(y0)} A${rx} ${ry} 0 ${large} 1 ${f(x1)} ${f(y1)} ${head(x1, y1, t)}"/>`;
}

/** A straight arrow from `(x0, y0)` to `(x1, y1)`. */
export function shift(x0: number, y0: number, x1: number, y1: number, cls = 'act'): string {
  const t = Math.atan2(y1 - y0, x1 - x0);
  return `<path class="${cls}" d="M${f(x0)} ${f(y0)} L${f(x1)} ${f(y1)} ${head(x1, y1, t)}"/>`;
}

const polygon = (n: number, r: number, a0: number): string =>
  Array.from({ length: n }, (_, i) => at(r, a0 + (2 * Math.PI * i) / n))
    .map(([x, y]) => `${f(x)},${f(y)}`)
    .join(' ');

const UP = -Math.PI / 2;

const DRAW: Record<IconName, () => string> = {
  ring: () => {
    const dots = Array.from({ length: 8 }, (_, i) => {
      const [x, y] = at(28, UP + (i * Math.PI) / 4);
      return `<circle class="dot${i === 0 ? ' on' : ''}" cx="${f(x)}" cy="${f(y)}" r="5"/>`;
    }).join('');
    return `<circle class="faint" cx="50" cy="50" r="28"/>${dots}${turn(41, 41, UP + 0.12, UP + Math.PI / 4 - 0.06)}`;
  },
  circle: () => {
    const pts = Array.from({ length: 121 }, (_, i) => {
      const a = (2 * Math.PI * i) / 120;
      return at(27 + 4 * Math.cos(5 * a) + 2.5 * Math.sin(3 * a + 1), a);
    });
    const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x)} ${f(y)}`).join(' ');
    return `<path d="${d} Z"/>${turn(42, 42, UP + 0.1, UP + 1.1)}`;
  },
  torus: () => {
    const shade = ['0100', '1210', '0100', '0001'];
    const cells = shade
      .flatMap((row, i) =>
        [...row].map(
          (v, j) =>
            `<rect class="cell c${v}" x="${18 + 15 * j}" y="${18 + 15 * i}" width="13" height="13" rx="1.5"/>`,
        ),
      )
      .join('');
    return `${cells}${shift(22, 90, 74, 90)}${shift(90, 22, 90, 74)}`;
  },
  polygon: () =>
    `<polygon class="fill" points="${polygon(6, 30, UP)}"/>` +
    `<path class="mirror" d="M50 10 L50 90"/>` +
    turn(42, 42, UP + 0.45, UP + 1.35),
  disk: () =>
    `<circle class="fill" cx="50" cy="50" r="30"/>` +
    `<circle class="blob on" cx="59" cy="40" r="8"/><circle class="blob" cx="41" cy="58" r="10"/>` +
    turn(42, 42, UP + 0.1, UP + 1.1),
  sphere: () =>
    `<circle class="fill" cx="50" cy="48" r="29"/>` +
    `<ellipse class="thin" cx="50" cy="48" rx="29" ry="9"/>` +
    `<ellipse class="thin" cx="50" cy="48" rx="11" ry="29"/>` +
    turn(43, 13, 0.35, 2.75, 50, 52),
  cube: () => {
    const top = '50,22 73,35 50,48 27,35';
    return (
      `<polygon class="fill" points="50,22 73,35 73,61 50,74 27,61 27,35"/>` +
      `<polygon class="face" points="${top}"/>` +
      `<path d="M50 48 L50 74 M50 48 L73 35 M50 48 L27 35"/>` +
      turn(43, 13, 0.35, 2.75, 50, 56)
    );
  },
};

export function icon(name: IconName): string {
  return `<svg class="ic" viewBox="0 0 100 100" aria-hidden="true">${DRAW[name]()}</svg>`;
}

export const isIconName = (s: string): s is IconName => s in DRAW;
