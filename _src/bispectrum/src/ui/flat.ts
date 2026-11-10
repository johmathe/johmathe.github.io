/** Flat views of what a sphere shows: insets painted from the coefficients on display. */

import { toHalfSpectrum } from '../math/basis';
import { evaluateAt, type SphereGrid } from '../math/sht';
import type { FlatProjection } from '../signals';
import { paint } from './colormap';

/** Something a `SphereView` repaints with the coefficients it displays (`null`: nothing). */
export interface Inset {
  paint(x: Float64Array | null, vmax: number): void;
}

const lmaxOf = (x: ArrayLike<number>): number => Math.round(Math.sqrt(x.length)) - 1;

function put(
  canvas: HTMLCanvasElement,
  w: number,
  h: number,
  values: Float64Array | null,
  vmax: number,
) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  if (!values) {
    ctx.clearRect(0, 0, w, h);
    return;
  }
  const img = ctx.createImageData(w, h);
  paint(values, vmax, img.data);
  ctx.putImageData(img, 0, 0);
}

/** The signal through a `FlatProjection`: the un-lifted digit, or a world map. */
export class FlatView implements Inset {
  readonly canvas: HTMLCanvasElement;
  private last: { x: Float64Array | null; vmax: number } = { x: null, vmax: 1 };

  constructor(
    private projection: FlatProjection,
    cls = 'rw-inset',
  ) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = cls;
    this.fit();
  }

  setProjection(projection: FlatProjection): void {
    if (projection === this.projection) return;
    this.projection = projection;
    this.fit();
    this.paint(this.last.x, this.last.vmax);
  }

  paint(x: Float64Array | null, vmax: number): void {
    this.last = { x, vmax };
    const { width, height, dirs } = this.projection;
    const values = x && evaluateAt(toHalfSpectrum(x, lmaxOf(x)), dirs);
    put(this.canvas, width, height, values, vmax);
  }

  private fit(): void {
    this.canvas.style.aspectRatio = String(this.projection.width / this.projection.height);
  }
}

/** The signal on an equiangular grid, as an equirectangular image (what a pixel CNN reads). */
export class GridInset implements Inset {
  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly grid: SphereGrid,
  ) {}

  paint(x: Float64Array | null, vmax: number): void {
    const { nlat, nlon } = this.grid;
    const values = x && this.grid.synthesize(toHalfSpectrum(x, lmaxOf(x)));
    put(this.canvas, nlon, nlat, values, vmax);
  }
}
