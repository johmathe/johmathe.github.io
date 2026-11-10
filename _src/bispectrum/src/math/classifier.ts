/**
 * Inference for the three classifiers exported by `blog/scripts/train_classifiers.py`: the paper's
 * bispectrum MLP at the widget's band-limit and its two controls, a small CNN on the
 * equirectangular image (not invariant) and an MLP on the power spectrum (invariant, incomplete).
 * BatchNorm is already folded into the weights, which ship as float16.
 */

export interface LayerSpec {
  /** `(out, in)` for a dense layer, `(out, in, kh, kw)` for a convolution. */
  shape: number[];
  /** Offset into the weight file, in float16 values: weights, then `out` biases. */
  offset: number;
}

export interface ModelSpec {
  file: string;
  layers: LayerSpec[];
  params: number;
}

export type ModelName = 'mlp' | 'cnn' | 'power';

export interface SplitAccuracy {
  /** Upright test digits. */
  test_nr: number;
  /** Test digits under one random rotation each. */
  test_r: number;
}

export interface ClassifierManifest {
  lmax: number;
  nlat: number;
  nlon: number;
  n_features: number;
  mlp: ModelSpec;
  cnn: ModelSpec;
  power: ModelSpec;
  accuracy: Record<ModelName, SplitAccuracy>;
  /** Answers on the 10,000 test digits: `[true digit][predicted digit]`, per split. */
  confusion: Record<ModelName, Record<keyof SplitAccuracy, number[][]>>;
}

interface Layer {
  shape: number[];
  w: Float64Array;
  b: Float64Array;
}

function float16(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function parseLayers(spec: ModelSpec, buffer: ArrayBuffer): Layer[] {
  const raw = new Uint16Array(buffer);
  return spec.layers.map(({ shape, offset }) => {
    const nw = shape.reduce((a, b) => a * b, 1);
    const nb = shape[0]!;
    if (offset + nw + nb > raw.length) throw new Error(`${spec.file} is shorter than its table`);
    return {
      shape,
      w: Float64Array.from(raw.subarray(offset, offset + nw), float16),
      b: Float64Array.from(raw.subarray(offset + nw, offset + nw + nb), float16),
    };
  });
}

/** `relu?(W x + b)` for a dense layer. */
function dense(layer: Layer, x: Float64Array, relu: boolean): Float64Array {
  const [nOut, nIn] = layer.shape as [number, number];
  if (x.length !== nIn) throw new Error(`dense layer expects ${nIn} inputs, got ${x.length}`);
  const out = new Float64Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = layer.b[o]!;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += layer.w[row + i]! * x[i]!;
    out[o] = relu && s < 0 ? 0 : s;
  }
  return out;
}

export function softmax(logits: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]!);
  const out = Float64Array.from(logits, (v) => Math.exp(v - max));
  const sum = out.reduce((a, b) => a + b, 0);
  return out.map((v) => v / sum);
}

/** The paper's head: `sign(x) log(1 + |x|)` of the selective invariant, then a 3-layer MLP. */
export class BispectrumMlp {
  private readonly layers: Layer[];

  constructor(spec: ModelSpec, buffer: ArrayBuffer) {
    this.layers = parseLayers(spec, buffer);
  }

  /** Logits from `Invariant.values`, one real number per entry. */
  logits(values: ArrayLike<number>): Float64Array {
    let h: Float64Array = Float64Array.from(values, (v) => Math.sign(v) * Math.log1p(Math.abs(v)));
    this.layers.forEach((layer, i) => {
      h = dense(layer, h, i < this.layers.length - 1);
    });
    return h;
  }
}

/** The incomplete control: `log(1 + ||F_l||^2)` per degree, then a 3-layer MLP. */
export class PowerMlp {
  private readonly layers: Layer[];

  constructor(spec: ModelSpec, buffer: ArrayBuffer) {
    this.layers = parseLayers(spec, buffer);
  }

  /** Logits from `degreePower`, one number per degree. */
  logits(power: ArrayLike<number>): Float64Array {
    let h: Float64Array = Float64Array.from(power, Math.log1p);
    this.layers.forEach((layer, i) => {
      h = dense(layer, h, i < this.layers.length - 1);
    });
    return h;
  }
}

const K = 5;
const PAD = 2;

/**
 * `StandardCNN` of `experiments/spherical_mnist/model.py`: conv 5x5 (32) - ReLU - maxpool 2 - conv
 * 5x5 (64) - ReLU - average to 4x4 - dense 128 - ReLU - dense 10, on the `(nlat, nlon)` grid
 * treated as a flat image (zero padding, no wrap in longitude).
 */
export class PixelCnn {
  private readonly layers: Layer[];

  constructor(
    spec: ModelSpec,
    buffer: ArrayBuffer,
    private readonly nlat: number,
    private readonly nlon: number,
  ) {
    this.layers = parseLayers(spec, buffer);
    if (this.layers.length !== 4)
      throw new Error('expected two convolutions and two dense layers');
  }

  logits(image: ArrayLike<number>): Float64Array {
    const { nlat: h, nlon: w } = this;
    if (image.length !== h * w) throw new Error(`expected a ${h} x ${w} image`);
    const [c1, c2, d1, d2] = this.layers as [Layer, Layer, Layer, Layer];
    const a = maxPool2(convRelu(c1, Float64Array.from(image), h, w), c1.shape[0]!, h, w);
    const b = convRelu(c2, a, h / 2, w / 2);
    const pooled = averageTo4(b, c2.shape[0]!, h / 2, w / 2);
    return dense(d2, dense(d1, pooled, true), false);
  }
}

/** 5x5 convolution with zero padding 2, then ReLU. `src` is `(cin, h, w)`, row-major. */
function convRelu(layer: Layer, src: Float64Array, h: number, w: number): Float64Array {
  const [cout, cin] = layer.shape as [number, number];
  const hp = h + 2 * PAD;
  const wp = w + 2 * PAD;
  const pad = new Float64Array(cin * hp * wp);
  // Rows that are entirely zero (common after a ReLU) contribute nothing: skip them.
  const live = new Uint8Array(cin * hp);
  for (let c = 0; c < cin; c++) {
    for (let y = 0; y < h; y++) {
      const row = src.subarray((c * h + y) * w, (c * h + y + 1) * w);
      pad.set(row, (c * hp + y + PAD) * wp + PAD);
      live[c * hp + y + PAD] = row.some((v) => v !== 0) ? 1 : 0;
    }
  }
  const out = new Float64Array(cout * h * w);
  const acc = new Float64Array(w);
  const { w: wt, b: bias } = layer;
  for (let o = 0; o < cout; o++) {
    for (let y = 0; y < h; y++) {
      acc.fill(bias[o]!);
      for (let c = 0; c < cin; c++) {
        for (let ky = 0; ky < K; ky++) {
          if (!live[c * hp + y + ky]) continue;
          const p = (c * hp + y + ky) * wp;
          const k = ((o * cin + c) * K + ky) * K;
          const w0 = wt[k]!;
          const w1 = wt[k + 1]!;
          const w2 = wt[k + 2]!;
          const w3 = wt[k + 3]!;
          const w4 = wt[k + 4]!;
          for (let x = 0; x < w; x++) {
            const q = p + x;
            acc[x] =
              acc[x]! +
              w0 * pad[q]! +
              w1 * pad[q + 1]! +
              w2 * pad[q + 2]! +
              w3 * pad[q + 3]! +
              w4 * pad[q + 4]!;
          }
        }
      }
      const row = (o * h + y) * w;
      for (let x = 0; x < w; x++) out[row + x] = acc[x]! > 0 ? acc[x]! : 0;
    }
  }
  return out;
}

function maxPool2(src: Float64Array, channels: number, h: number, w: number): Float64Array {
  const ho = h / 2;
  const wo = w / 2;
  const out = new Float64Array(channels * ho * wo);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < ho; y++) {
      const r0 = (c * h + 2 * y) * w;
      const r1 = r0 + w;
      for (let x = 0; x < wo; x++) {
        out[(c * ho + y) * wo + x] = Math.max(
          src[r0 + 2 * x]!,
          src[r0 + 2 * x + 1]!,
          src[r1 + 2 * x]!,
          src[r1 + 2 * x + 1]!,
        );
      }
    }
  }
  return out;
}

/** `AdaptiveAvgPool2d((4, 4))` for sizes divisible by 4, flattened as `(channel, row, col)`. */
function averageTo4(src: Float64Array, channels: number, h: number, w: number): Float64Array {
  const bh = h / 4;
  const bw = w / 4;
  const out = new Float64Array(channels * 16);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < h; y++) {
      const row = (c * h + y) * w;
      const cell = c * 16 + Math.floor(y / bh) * 4;
      for (let x = 0; x < w; x++) out[cell + Math.floor(x / bw)]! += src[row + x]!;
    }
  }
  const scale = 1 / (bh * bw);
  for (let i = 0; i < out.length; i++) out[i]! *= scale;
  return out;
}
