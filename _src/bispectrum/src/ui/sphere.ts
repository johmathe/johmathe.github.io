import {
  BufferGeometry,
  ClampToEdgeWrapping,
  DataTexture,
  Float32BufferAttribute,
  Group,
  LinearFilter,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  PerspectiveCamera,
  Quaternion,
  RepeatWrapping,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  UnsignedByteType,
  Vector3,
  WebGLRenderer,
} from 'three';
import { toHalfSpectrum } from '../math/basis';
import { identity3, mul3, rotateDofs, transpose3, type Mat3 } from '../math/rotation';
import type { SphereGrid } from '../math/sht';
import { MUTED, paint } from './colormap';
import type { Inset } from './flat';

/** Three.js local frame to ours: `X = x`, `Y = -z`, `Z = y` (rows of `M`). */
const M: Mat3 = Float64Array.from([1, 0, 0, 0, 0, -1, 0, 1, 0]);
const MT = transpose3(M);
/** Base tilt: +90 degrees about x puts our north pole (three's +y) in front of the camera. */
const B: Mat3 = Float64Array.from([1, 0, 0, 0, 0, -1, 0, 1, 0]);
const BT = transpose3(B);

const lmaxOf = (x: ArrayLike<number>): number => Math.round(Math.sqrt(x.length)) - 1;

function toMatrix4(r: Mat3): Matrix4 {
  // prettier-ignore
  return new Matrix4().set(
    r[0]!, r[1]!, r[2]!, 0,
    r[3]!, r[4]!, r[5]!, 0,
    r[6]!, r[7]!, r[8]!, 0,
    0, 0, 0, 1,
  );
}

function fromMatrix4(m: Matrix4): Mat3 {
  const e = m.elements; // column-major
  return Float64Array.from([e[0]!, e[4]!, e[8]!, e[1]!, e[5]!, e[9]!, e[2]!, e[6]!, e[10]!]);
}

/** Mesh orientation that displays `R . f` for a texture holding `f`: `B M^T R M`. */
function meshQuat(r: Mat3): Quaternion {
  return new Quaternion().setFromRotationMatrix(toMatrix4(mul3(B, mul3(MT, mul3(r, M)))));
}

const VERTEX = /* glsl */ `
varying vec3 vLocal;
varying vec3 vNormal;
void main() {
  vLocal = position;
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// The texture is (nlat, nlon) with texel centers on ((k + 1/2) pi / nlat, (j + 1/2) 2 pi / nlon),
// sampled at the exact spherical coordinates of each fragment; without mipmaps the atan seam is
// invisible.
const FRAGMENT = /* glsl */ `
uniform sampler2D map;
uniform float empty;
varying vec3 vLocal;
varying vec3 vNormal;
const float PI = 3.141592653589793;
void main() {
  vec3 p = normalize(vLocal);
  float theta = acos(clamp(p.y, -1.0, 1.0));
  float phi = atan(-p.z, p.x);
  vec3 c = texture2D(map, vec2(fract(phi / (2.0 * PI)), theta / PI)).rgb;
  c = mix(c, vec3(0.918, 0.902, 0.859), empty);
  // Shading is kept faint: the colormap runs from beige down to ink, so any real darkening of
  // the surface would read as negative values.
  vec3 n = normalize(vNormal);
  float diff = max(dot(n, normalize(vec3(-0.35, 0.55, 1.0))), 0.0);
  float rim = pow(1.0 - max(n.z, 0.0), 4.0);
  gl_FragColor = vec4(c * (0.94 + 0.06 * diff) * (1.0 - 0.16 * rim), 1.0);
}
`;

/**
 * One WebGL context for every sphere on the page: browsers cap live contexts (as few as 8 on some
 * phones), and the post shows more spheres than that. Each view renders into the bottom-left
 * corner of the shared canvas and copies the result into its own 2D canvas in the same task, while
 * the drawing buffer is still valid.
 */
class SharedRenderer {
  private renderer: WebGLRenderer | null = null;
  private width = 0;
  private height = 0;

  draw(scene: Scene, camera: PerspectiveCamera, target: HTMLCanvasElement): void {
    const { width: w, height: h } = target;
    if (!w || !h) return;
    const r = this.get();
    if (w > this.width || h > this.height) {
      this.width = Math.max(w, this.width);
      this.height = Math.max(h, this.height);
      r.setSize(this.width, this.height, false);
    }
    r.setViewport(0, 0, w, h);
    r.setScissor(0, 0, w, h);
    r.setScissorTest(true);
    r.render(scene, camera);
    const ctx = target.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(r.domElement, 0, this.height - h, w, h, 0, 0, w, h);
  }

  private get(): WebGLRenderer {
    if (!this.renderer) {
      this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setPixelRatio(1);
      this.renderer.setClearColor(0x000000, 0);
    }
    return this.renderer;
  }
}

const shared = new SharedRenderer();

function graticule(): BufferGeometry {
  const pts: number[] = [];
  const r = 1.003;
  const push = (th: number, ph: number): void => {
    const [X, Y, Z] = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    pts.push(r * X, r * Z, -r * Y);
  };
  const seg = 96;
  for (const th of [30, 60, 90, 120, 150].map((d) => (d * Math.PI) / 180)) {
    for (let i = 0; i < seg; i++) {
      push(th, (2 * Math.PI * i) / seg);
      push(th, (2 * Math.PI * (i + 1)) / seg);
    }
  }
  for (let k = 0; k < 12; k++) {
    const ph = (k * Math.PI) / 6;
    for (let i = 0; i < seg / 2; i++) {
      push((Math.PI * i) / (seg / 2), ph);
      push((Math.PI * (i + 1)) / (seg / 2), ph);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(pts, 3));
  return g;
}

export interface SphereViewOptions {
  /** Display grid (cell-centered); its size is the texture size. */
  grid: SphereGrid;
  /** Flat views repainted with the coefficients on display, whenever those change. */
  insets?: Inset[];
  /** Drag gesture as a rotation in our coordinates; the caller decides what it rotates. */
  onDrag?: (w: Mat3) => void;
  onDragEnd?: () => void;
}

/** Draggable sphere showing `R . f` for coefficients `f` and an orientation `R`. */
export class SphereView {
  readonly canvas: HTMLCanvasElement;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(28, 1, 0.1, 100);
  private readonly mesh: Mesh;
  private readonly material: ShaderMaterial;
  private readonly texture: DataTexture;
  private readonly pixels: Uint8Array;
  private readonly field: Float64Array;
  private readonly observer: ResizeObserver;
  private x: Float64Array | null = null;
  private vmax = 1;
  private rot: Mat3 = identity3();
  private anim: { from: Quaternion; to: Quaternion; t0: number; ms: number } | null = null;
  private frame = 0;
  private insetDirty = false;
  private drag: { id: number; p: Vector3 } | null = null;
  private onScreen = true;
  private missed = false;
  private readonly visibility: IntersectionObserver;

  constructor(
    host: HTMLElement,
    private readonly opts: SphereViewOptions,
  ) {
    const { nlat, nlon } = opts.grid;
    this.canvas = document.createElement('canvas');
    this.canvas.classList.add('sphere-canvas');
    host.appendChild(this.canvas);

    this.pixels = new Uint8Array(nlat * nlon * 4);
    this.field = new Float64Array(nlat * nlon);
    this.texture = new DataTexture(this.pixels, nlon, nlat, RGBAFormat, UnsignedByteType);
    this.texture.wrapS = RepeatWrapping;
    this.texture.wrapT = ClampToEdgeWrapping;
    this.texture.magFilter = LinearFilter;
    this.texture.minFilter = LinearFilter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;

    this.material = new ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: { map: { value: this.texture }, empty: { value: 1 } },
    });
    this.mesh = new Mesh(new SphereGeometry(1, 128, 96), this.material);
    this.mesh.quaternion.copy(meshQuat(this.rot));
    const lines = new LineSegments(
      graticule(),
      new LineBasicMaterial({ color: MUTED, transparent: true, opacity: 0.22 }),
    );
    const fixed = new Group();
    fixed.quaternion.setFromRotationMatrix(toMatrix4(B));
    fixed.add(lines);
    this.scene.add(this.mesh, fixed);
    this.camera.position.set(0, 0, 4.6);

    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.visibility = new IntersectionObserver((entries) => {
      this.onScreen = entries.some((e) => e.isIntersecting);
      if (this.onScreen && this.missed) this.requestRender();
    });
    this.visibility.observe(this.canvas);
    this.resize();
    this.bindPointer();
  }

  /** Whether the canvas is on screen; off-screen frames are skipped and replayed on return. */
  get visible(): boolean {
    return this.onScreen;
  }

  /** Raw coefficients to display (`null` blanks the sphere) and the symmetric color scale. */
  setSignal(x: Float64Array | null, vmax: number): void {
    this.x = x && Float64Array.from(x);
    this.vmax = vmax;
    this.material.uniforms['empty']!.value = x ? 0 : 1;
    if (x) {
      this.opts.grid.synthesize(toHalfSpectrum(x, lmaxOf(x)), this.field);
      paint(this.field, vmax, this.pixels);
      this.texture.needsUpdate = true;
    }
    this.insetDirty = true;
    this.requestRender();
  }

  get orientation(): Mat3 {
    return this.rot;
  }

  setOrientation(r: Mat3, animateMs = 0): void {
    this.rot = Float64Array.from(r);
    const to = meshQuat(r);
    if (animateMs > 0) {
      this.anim = { from: this.mesh.quaternion.clone(), to, t0: performance.now(), ms: animateMs };
    } else {
      this.anim = null;
      this.mesh.quaternion.copy(to);
    }
    this.insetDirty = true;
    this.requestRender();
  }

  dispose(): void {
    this.observer.disconnect();
    this.visibility.disconnect();
    cancelAnimationFrame(this.frame);
    this.texture.dispose();
    this.material.dispose();
    this.canvas.remove();
  }

  private resize(): void {
    const w = this.canvas.parentElement?.clientWidth ?? 300;
    const px = Math.round(w * Math.min(window.devicePixelRatio, 2));
    if (this.canvas.width !== px) this.canvas.width = this.canvas.height = px;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${w}px`;
    this.requestRender();
  }

  private requestRender(): void {
    if (!this.onScreen) {
      this.missed = true;
      return;
    }
    this.missed = false;
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  private render(): void {
    if (this.anim) {
      const t = Math.min(1, (performance.now() - this.anim.t0) / this.anim.ms);
      const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
      this.mesh.quaternion.slerpQuaternions(this.anim.from, this.anim.to, e);
      if (t < 1) this.requestRender();
      else this.anim = null;
    }
    shared.draw(this.scene, this.camera, this.canvas);
    if (this.insetDirty && !this.anim) this.paintInset();
  }

  private paintInset(): void {
    this.insetDirty = false;
    const insets = this.opts.insets;
    if (!insets?.length) return;
    const x = this.x && rotateDofs(this.x, this.rot);
    for (const inset of insets) inset.paint(x, this.vmax);
  }

  /** Pointer position on the virtual arcball, in camera coordinates. */
  private ballPoint(e: PointerEvent): Vector3 {
    const rect = this.canvas.getBoundingClientRect();
    const halfHeight = Math.tan((this.camera.fov * Math.PI) / 360) * this.camera.position.z;
    const x = (((e.clientX - rect.left) / rect.width) * 2 - 1) * halfHeight;
    const y = (1 - ((e.clientY - rect.top) / rect.height) * 2) * halfHeight;
    const d = x * x + y * y;
    return d < 1 ? new Vector3(x, y, Math.sqrt(1 - d)) : new Vector3(x, y, 0).normalize();
  }

  private bindPointer(): void {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => {
      if (!this.opts.onDrag || !this.x) return;
      c.setPointerCapture(e.pointerId);
      this.drag = { id: e.pointerId, p: this.ballPoint(e) };
      this.anim = null;
      c.classList.add('dragging');
    });
    c.addEventListener('pointermove', (e) => {
      if (!this.drag || e.pointerId !== this.drag.id) return;
      const p = this.ballPoint(e);
      const q = new Quaternion().setFromUnitVectors(this.drag.p, p);
      this.drag.p = p;
      const world = fromMatrix4(new Matrix4().makeRotationFromQuaternion(q));
      this.opts.onDrag?.(mul3(M, mul3(BT, mul3(world, mul3(B, MT)))));
    });
    const end = (e: PointerEvent): void => {
      if (!this.drag || e.pointerId !== this.drag.id) return;
      this.drag = null;
      c.classList.remove('dragging');
      this.insetDirty = true;
      this.requestRender();
      this.opts.onDragEnd?.();
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
  }
}
