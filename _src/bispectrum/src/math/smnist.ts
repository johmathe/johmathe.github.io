/**
 * Spherical MNIST lift: inverse stereographic projection from the north pole, as in
 * `experiments/spherical_mnist/data.py`. Grid point `(theta, phi)` samples the image at
 * `(x, y) = tan(theta / 2) (cos phi, sin phi)` in `[-1, 1]^2`, bilinearly, with zero padding and
 * `align_corners=True` (`F.grid_sample` semantics). The southern hemisphere lands outside the
 * image and is zero.
 */

export interface LiftOptions {
  /**
   * Flip image rows before lifting. The dataset lift reads mirrored when the sphere is viewed
   * from outside (image rows run along +y); flipping makes digits read correctly on screen.
   */
  mirror?: boolean;
}

export function liftToSphere(
  image: ArrayLike<number>,
  size: number,
  theta: ArrayLike<number>,
  phi: ArrayLike<number>,
  opts: LiftOptions = {},
): Float64Array {
  const nlat = theta.length;
  const nlon = phi.length;
  const out = new Float64Array(nlat * nlon);
  const last = size - 1;
  const at = (row: number, col: number): number => {
    if (row < 0 || row > last || col < 0 || col > last) return 0;
    return image[(opts.mirror ? last - row : row) * size + col]!;
  };
  for (let k = 0; k < nlat; k++) {
    const r = Math.tan(theta[k]! / 2);
    for (let j = 0; j < nlon; j++) {
      const ix = ((r * Math.cos(phi[j]!) + 1) / 2) * last;
      const iy = ((r * Math.sin(phi[j]!) + 1) / 2) * last;
      if (!(ix > -1 && ix < size && iy > -1 && iy < size)) continue;
      const x0 = Math.floor(ix);
      const y0 = Math.floor(iy);
      const fx = ix - x0;
      const fy = iy - y0;
      out[k * nlon + j] =
        (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) +
        fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1));
    }
  }
  return out;
}
