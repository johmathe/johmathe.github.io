/**
 * The group tiles' icons acting out their groups, over and over: steps around the ring, turns of
 * the circle and the disk, shifts of the grid, a turn then a flip of the hexagon, quarter turns of
 * the cube, and the sphere spinning on its axis. Each motion rebuilds what it moves out of the
 * static drawing from `deck/icons.ts`.
 */
import { f, isIconName, TORUS_SHADE, type IconName } from '../deck/icons';
import { reducedMotion } from '../ui/canvas';

const SVG = 'http://www.w3.org/2000/svg';

type Vec3 = [number, number, number];

/** Poses an icon `t` ms after its motion started from rest. */
type Pose = (t: number) => void;

export interface TileMotion {
  /** Puts every icon back at rest and starts the motions over. */
  restart(): void;
}

/** Clip paths need ids that stay unique when a page holds several animated grids. */
let clips = 0;

function add(
  parent: Element,
  tag: string,
  attrs: Record<string, string | number>,
  before: Element | null = parent.querySelector(':scope > .act'),
): SVGElement {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  parent.insertBefore(el, before);
  return el;
}

function find(svg: SVGSVGElement, selector: string, index = 0): Element {
  const el = svg.querySelectorAll(selector)[index];
  if (!el) throw new Error(`the icon has no ${selector} #${index}`);
  return el;
}

/** Step `k` of a motion that moves for `move` ms then rests for `rest` ms, and its eased progress. */
function steps(t: number, move: number, rest: number): [number, number] {
  const k = Math.floor(t / (move + rest));
  const e = Math.min(1, (t - k * (move + rest)) / move);
  return [k, e * e * (3 - 2 * e)];
}

const spin = (t: number): string => `rotate(${f((t * 0.06) % 360)} 50 50)`;

const TILT = Math.atan(Math.SQRT1_2);

/** A point of the cube turned by `phi` about the vertical, then tilted into the isometric view. */
function view([x, y, z]: Vec3, phi: number): Vec3 {
  const [c, s] = [Math.cos(phi), Math.sin(phi)];
  const [x1, z1] = [x * c + z * s, z * c - x * s];
  return [x1, y * Math.cos(TILT) - z1 * Math.sin(TILT), y * Math.sin(TILT) + z1 * Math.cos(TILT)];
}

const MOTIONS: Record<IconName, (svg: SVGSVGElement) => Pose> = {
  ring: (svg) => {
    find(svg, '.dot.on').classList.remove('on');
    const marker = add(svg, 'circle', { class: 'dot on', r: 5.5 });
    return (t) => {
      const [k, e] = steps(t, 260, 260);
      const a = -Math.PI / 2 + ((k + e) * Math.PI) / 4;
      marker.setAttribute('cx', f(50 + 28 * Math.cos(a)));
      marker.setAttribute('cy', f(50 + 28 * Math.sin(a)));
    };
  },
  circle: (svg) => {
    const curve = find(svg, 'path:not(.act)');
    return (t) => curve.setAttribute('transform', spin(t));
  },
  disk: (svg) => {
    const blobs = [...svg.querySelectorAll('.blob')];
    return (t) => {
      for (const blob of blobs) blob.setAttribute('transform', spin(t));
    };
  },
  // The grid slides one cell at a time, right then down, wrapping around: a 5 × 5 patch of the
  // periodic pattern behind a 4 × 4 window.
  torus: (svg) => {
    for (const cell of svg.querySelectorAll('.cell')) cell.remove();
    const id = `tile-torus-${++clips}`;
    add(add(svg, 'clipPath', { id }), 'rect', { x: 17, y: 17, width: 60, height: 60 }, null);
    const grid = add(svg, 'g', { 'clip-path': `url(#${id})` });
    const cells: [number, number, SVGElement][] = [];
    for (let i = -1; i < 4; i++) {
      for (let j = -1; j < 4; j++) {
        const rect = { x: 18 + 15 * j, y: 18 + 15 * i, width: 13, height: 13, rx: 1.5 };
        cells.push([i, j, add(grid, 'rect', rect, null)]);
      }
    }
    const wrap = (n: number): number => ((n % 4) + 4) % 4;
    return (t) => {
      const [k, e] = steps(t, 480, 340);
      const [dx, dy] = [Math.ceil(k / 2), Math.floor(k / 2)];
      const off = f(15 * e);
      grid.setAttribute('transform', k % 2 ? `translate(0 ${off})` : `translate(${off} 0)`);
      for (const [i, j, cell] of cells) {
        cell.setAttribute('class', `cell c${TORUS_SHADE[wrap(i - dy)]?.[wrap(j - dx)]}`);
      }
    };
  },
  polygon: (svg) => {
    const hexagon = find(svg, '.fill');
    return (t) => {
      const [k, e] = steps(t, 600, 450);
      hexagon.setAttribute(
        'transform',
        k % 2
          ? `translate(50 0) scale(${(1 - 2 * e).toFixed(3)} 1) translate(-50 0)`
          : `rotate(${f(60 * e)} 50 50)`,
      );
    };
  },
  sphere: (svg) => {
    const meridians = [
      find(svg, '.thin', 1),
      add(svg, 'ellipse', { class: 'thin', cx: 50, cy: 48, ry: 29 }),
    ] as const;
    return (t) => {
      const m = 0.39 + t * 0.0011;
      meridians[0].setAttribute('rx', f(29 * Math.abs(Math.sin(m))));
      meridians[1].setAttribute('rx', f(29 * Math.abs(Math.cos(m))));
    };
  },
  // The cube in isometric view, turned about its vertical axis: its visible faces, redrawn.
  cube: (svg) => {
    for (const el of svg.querySelectorAll(':scope > :not(.act)')) el.remove();
    const faces = ([0, 1, 2] as const).flatMap((axis) =>
      ([1, -1] as const).map((sign) => {
        const [u, v] = [0, 1, 2].filter((a) => a !== axis) as [number, number];
        const corner = (a: number, b: number): Vec3 => {
          const p: Vec3 = [0, 0, 0];
          [p[axis], p[u], p[v]] = [sign, a, b];
          return p;
        };
        const normal: Vec3 = [0, 0, 0];
        normal[axis] = sign;
        const top = axis === 1 && sign === 1;
        return {
          normal,
          corners: [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)],
          el: add(svg, 'polygon', { class: top ? 'face' : 'fill' }),
        };
      }),
    );
    return (t) => {
      const [k, e] = steps(t, 700, 550);
      const phi = Math.PI / 4 - (Math.PI / 2) * (k + e);
      for (const { normal, corners, el } of faces) {
        const shown = view(normal, phi)[2] > 1e-6;
        // "visible" would show through a hidden ancestor.
        el.setAttribute('visibility', shown ? 'inherit' : 'hidden');
        if (!shown) continue;
        const pts = corners.map((p) => {
          const [x, y] = view(p, phi);
          return `${f(50 + 16.3 * x)},${f(48 - 16.3 * y)}`;
        });
        el.setAttribute('points', pts.join(' '));
      }
    };
  },
};

/**
 * Animates the icons drawn inside `list` while it is on screen, carrying on where they paused when
 * it comes back. Returns null, leaving the drawings still, under reduced motion.
 */
export function animateTiles(list: Element): TileMotion | null {
  if (reducedMotion()) return null;
  const poses = [...list.querySelectorAll<HTMLElement>('[data-icon]')].map((slot) => {
    const name = slot.dataset.icon ?? '';
    const svg = slot.querySelector('svg');
    if (!isIconName(name) || !svg) throw new Error(`no icon drawn for ${name}`);
    return MOTIONS[name](svg);
  });
  let t0 = performance.now();
  let pausedAt: number | null = t0;
  let frame = 0;
  const pose = (now: number): void => {
    for (const p of poses) p(now - t0);
  };
  const tick = (now: number): void => {
    pose(now);
    frame = requestAnimationFrame(tick);
  };
  pose(t0);
  new IntersectionObserver((entries) => {
    const now = performance.now();
    if (entries.at(-1)?.isIntersecting) {
      if (pausedAt === null) return;
      t0 += now - pausedAt;
      pausedAt = null;
      frame = requestAnimationFrame(tick);
    } else if (pausedAt === null) {
      pausedAt = now;
      cancelAnimationFrame(frame);
    }
  }).observe(list);
  return {
    restart: () => {
      t0 = performance.now();
      if (pausedAt !== null) pausedAt = t0;
      pose(t0);
    },
  };
}
