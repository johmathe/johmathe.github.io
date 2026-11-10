import { toHalfSpectrum } from '../math/basis';
import { PixelCnn } from '../math/classifier';
import { SphereGrid } from '../math/sht';
import type { ClassifyRequest, ClassifyResult } from './classify.protocol';

interface Scope {
  postMessage(message: ClassifyResult): void;
  addEventListener(type: 'message', fn: (e: MessageEvent<ClassifyRequest>) => void): void;
}

const scope = self as unknown as Scope;
let model: { cnn: PixelCnn; grid: SphereGrid; lmax: number } | null = null;

scope.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    const { nlat, nlon, lmax, cnn } = msg.manifest;
    model = {
      cnn: new PixelCnn(cnn, msg.bin, nlat, nlon),
      grid: new SphereGrid(nlat, nlon, lmax),
      lmax,
    };
    return;
  }
  if (!model) throw new Error('classify before init');
  const t0 = performance.now();
  // The equirectangular image the network was trained on: the signal on the analysis grid.
  const image = model.grid.synthesize(toHalfSpectrum(msg.x, model.lmax));
  const logits = model.cnn.logits(image);
  scope.postMessage({ type: 'result', id: msg.id, logits, ms: performance.now() - t0 });
});
