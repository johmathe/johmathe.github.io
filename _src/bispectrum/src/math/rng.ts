/** Small seeded PRNG (mulberry32) with Box-Muller normals, so solver runs are reproducible. */
export class Rng {
  private s: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  uniform(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  }

  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    const r = Math.sqrt(-2 * Math.log(this.uniform()));
    const a = 2 * Math.PI * this.uniform();
    this.spare = r * Math.sin(a);
    return r * Math.cos(a);
  }

  normals(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.normal();
    return out;
  }
}
