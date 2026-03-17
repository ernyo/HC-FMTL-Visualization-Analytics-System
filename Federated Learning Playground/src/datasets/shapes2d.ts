// dataset.ts
import { Shape, SceneOpts, Scene, SHAPE_UNIVERSE } from './constants';

// -------- rng --------
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function hash32(...nums: number[]) {
  let h = 2166136261 >>> 0;
  for (const n of nums) {
    h ^= (n >>> 0);
    h = Math.imul(h, 16777619) >>> 0;
    h ^= h >>> 13;
    h = Math.imul(h, 0x5bd1e995) >>> 0;
    h ^= h >>> 15;
  }
  return h >>> 0;
}

export function seededShuffle(arr: number[], seed: number) {
  const rnd = mulberry32(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

// -------- SDFs in shape-local coords (centered) --------
function sdfCircle(x: number, y: number, r: number)  { return Math.hypot(x,y) - r; }
function sdfBox(x: number, y: number, hx: number, hy: number) {
  const ax = Math.abs(x) - hx, ay = Math.abs(y) - hy;
  const qx = Math.max(ax,0), qy = Math.max(ay,0);
  return Math.hypot(qx,qy) + Math.min(Math.max(ax,ay),0);
}
// equilateral triangle roughly inscribed in circle of radius r
function sdfTriEq(x: number, y: number, r: number) {
  const k = Math.sqrt(3);
  x /= r;
  y /= r;
  x = Math.abs(x) - 1.0;
  y = y + 1.0 / k;
  if (x + k * y > 0.0) {
    const x2 = (x - k * y) * 0.5;
    const y2 = (-k * x - y) * 0.5;
    x = x2;
    y = y2;
  }
  x = x - Math.max(Math.min(x, 0.0), -2.0);
  const d = -Math.hypot(x, y) * Math.sign(y);
  return d * r;
}
function sdfStar5(x: number, y: number, R: number, r: number) {
  const a = Math.atan2(y,x);
  let ang = a % (2*Math.PI/5); if (ang<0) ang += 2*Math.PI/5;
  const frac = ang / (2*Math.PI/5);
  const edge = frac < 0.5 ? (R + (r-R)*(frac*2)) : (r + (R-r)*((frac-0.5)*2));
  return Math.hypot(x,y) - edge;
}

function rotInv(px:number, py:number, cx:number, cy:number, th:number) {
  const dx = px - cx, dy = py - cy;
  const c = Math.cos(th), s = Math.sin(th);
  return { x:  dx*c + dy*s, y: -dx*s + dy*c }; // inverse rotation
}

// -------- depth synthesis per shape (inside only) --------
type DepthParams = { a: number; b: number; phase: number };
function synthDepth(shape: Shape, x: number, y: number, r: number, p: DepthParams) {
  const Xn = x / r, Yn = y / r;
  if (shape === "circle") {
    const d = 1.0 - Math.min(1, Math.hypot(x, y) / r);
    return d * 8.0;
  } else if (shape === "square") {
    return (p.a * Xn + p.b * Yn) * 8.0;
  } else if (shape === "triangle") {
    return 0.8 * (p.a * Xn - p.b * Yn) * 8.0;
  } else {
    const rr = Math.hypot(x, y);
    return 0.6 * Math.cos(0.2 * rr + p.phase) * 8.0;
  }
}

function classTint(cls: number): [number,number,number] {
  const palette: [number,number,number][] = [
    [30,30,30],      // bg (unused)
    [242,85,85],     // circle - red
    [85,180,245],    // square - blue
    [120,235,120],   // triangle - green
    [245,205,85],    // star - yellow
  ];
  return palette[Math.min(cls, palette.length-1)];
}

// -------- main generator with z-buffer & overlaps --------
export function generateSceneMulti(opts: SceneOpts): Scene {
  const H = opts.H, W = opts.W;
  const probByType = new Map(opts.typeShapes.map(s => [s.type, s.probability] as const));
  const probsRaw = SHAPE_UNIVERSE.map(shape => probByType.get(shape) ?? 0);
  const probs = normalizeProbs(probsRaw);
  const nShapes = opts.nShapes;
  const seed = opts.seed;
  const rnd = mulberry32(seed >>> 0); 

  const shapesThisScene = sampleShapes(rnd, probs, nShapes);

  // outputs
  const rgb = new Uint8ClampedArray(H*W*3);
  const seg = new Uint8ClampedArray(H*W);       // 0 bg
  const edge = new Uint8ClampedArray(H*W);
  const sal = new Uint8ClampedArray(H*W);
  const depthF = new Float32Array(H*W); depthF.fill(-1e9); // z-buffer
  // background
  for (let i=0;i<H*W;i++){ rgb[i*3+0]=30; rgb[i*3+1]=30; rgb[i*3+2]=30; }

  const minS = Math.min(H,W);
  const minScale = opts.minScale ?? 0.1;
  const maxScale = opts.maxScale ?? 0.9;

  // Random light for Lambert shading
  const t = rnd()*2*Math.PI;
  const L = { x: Math.cos(t)*0.5, y: Math.sin(t)*0.5, z: Math.sqrt(1-0.25) };
  const ambient = 0.25;

  // Per shape: place, compute SDF per pixel & compose
  // random draw order (optional)

  for (let q=0;q<nShapes;q++){
    const shape = shapesThisScene[q];

    const scale = minScale + rnd()*(maxScale - minScale);
    const R = scale * 0.5 * minS;

    const cx = W*(0.2 + 0.6*rnd());
    const cy = H*(0.2 + 0.6*rnd());
    const theta = rnd()*2*Math.PI;

    const depthParams: DepthParams = {
      a: (rnd() - 0.5) * 2,
      b: (rnd() - 0.5) * 2,
      phase: rnd() * 2 * Math.PI,
    };

    // per-pixel loop
    for (let y=0;y<H;y++){
      for (let x=0;x<W;x++){
        const idx = y*W + x;

        // transform to shape-local
        const p = rotInv(x+0.5, y+0.5, cx, cy, theta);
        let sdf = 0;
        if (shape === 'circle') sdf = sdfCircle(p.x, p.y, R);
        else if (shape === 'square') sdf = sdfBox(p.x, p.y, R, R);
        else if (shape === 'triangle') sdf = sdfTriEq(p.x, p.y, R);
        else sdf = sdfStar5(p.x, p.y, R, R*0.5);

        if (sdf <= 0) {
          // depth (height) for z-buffer
          const z = synthDepth(shape, p.x, p.y, R, depthParams);

          // keep topmost
          if (z > depthF[idx]) {
            depthF[idx] = z;
            // seg id: 1..K mapped by shape
            const cls = (SHAPE_UNIVERSE.indexOf(shape) + 1) as number;
            seg[idx] = cls;

            // saliency: interior distance (normalize later for display; here just mark)
            // quick 0/255 for now (cheap): you can store 8-bit scaled interior if you want
            sal[idx] = 255;

            // simple Lambert using normal from local slope approx (will refine after normals pass)
            // here just flat shade by (ambient + z contrast): super cheap placeholder
            const shade = Math.max(ambient, ambient + 0.07*z);
            const tint = classTint(cls);
            rgb[idx*3+0] = Math.min(255, (shade * tint[0])|0);
            rgb[idx*3+1] = Math.min(255, (shade * tint[1])|0);
            rgb[idx*3+2] = Math.min(255, (shade * tint[2])|0);
          }

          // edge band (anti-aliased)
          const e = Math.abs(sdf);
          if (e < 1.2) {
            // smooth edge: map 1.2..0 to 0..255
            const v = Math.max(0, Math.min(1, 1 - e/1.2));
            edge[idx] = Math.max(edge[idx], (v*255)|0);
          }
        }
      }
    }
  }

  // Normalize depth and compute normals from final composed depthF
  let minD = +Infinity,
    maxD = -Infinity;
  for (let i = 0; i < H * W; i++) {
    if (depthF[i] > -1e8) {
      if (depthF[i] < minD) minD = depthF[i];
      if (depthF[i] > maxD) maxD = depthF[i];
    }
  }

  const depth = new Uint8ClampedArray(H * W);
  const normal = new Uint8ClampedArray(H * W * 3);

  if (isFinite(minD) && isFinite(maxD) && maxD > minD) {
    const inv = 255 / (maxD - minD);
    for (let i = 0; i < H * W; i++) {
      depth[i] = depthF[i] > -1e8 ? (((depthF[i] - minD) * inv) | 0) : 0;
    }
  }

  // normals via central diffs on depthF
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - 1),
      y1 = Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - 1),
        x1 = Math.min(W - 1, x + 1);

      const z00 = depthF[y0 * W + x0];
      const z10 = depthF[y0 * W + x1];
      const z01 = depthF[y1 * W + x0];
      const z11 = depthF[y1 * W + x1];

      const dzdx = (z10 - z00 + z11 - z01) * 0.5;
      const dzdy = (z01 - z00 + z11 - z10) * 0.5;

      let nx = -dzdx,
        ny = -dzdy,
        nz = 1;
      const invn = 1 / Math.max(1e-6, Math.hypot(nx, ny, nz));
      nx *= invn;
      ny *= invn;
      nz *= invn;

      const i = y * W + x;
      normal[i * 3 + 0] = ((nx + 1) * 127.5) | 0;
      normal[i * 3 + 1] = ((ny + 1) * 127.5) | 0;
      normal[i * 3 + 2] = ((nz + 1) * 127.5) | 0;
    }
  }

  const N = 1; // single scene
  const K = SHAPE_UNIVERSE.length;
  return { H, W, K, N, rgb, seg, edge, sal, depth, normal };
}

function sampleCategorical(rnd: () => number, probs: number[]): number {
  const u = rnd();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return probs.length - 1;
}

function sampleShapes(
  rnd: () => number,
  probs: number[],
  nShapes: number,
): Shape[] {
  const out: Shape[] = [];
  for (let i = 0; i < nShapes; i++) out.push(SHAPE_UNIVERSE[sampleCategorical(rnd, probs)]);
  return out;
}

function normalizeProbs(ps: number[]): number[] {
  const cleaned = ps.map(p => (Number.isFinite(p) && p > 0 ? p : 0));
  const sum = cleaned.reduce((a,b) => a + b, 0);
  if (sum <= 0) {
    const u = 1 / cleaned.length;
    return cleaned.map(() => u);
  }
  return cleaned.map(p => p / sum);
}