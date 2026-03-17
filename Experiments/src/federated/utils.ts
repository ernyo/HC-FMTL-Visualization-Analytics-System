import * as tf from "@tensorflow/tfjs-node-gpu";
import type { TaskName } from "../models/model.js";

export type WeightMap = Record<string, tf.Tensor>;

export const EPS = 1e-8;

/** ---------------- Checkpoint I/O ---------------- */

export function exportCheckpoint(model: tf.LayersModel): WeightMap {
  const names = model.weights.map(w => w.name);
  const values = model.getWeights();

  const ckpt: WeightMap = {};
  for (let i = 0; i < names.length; i++) ckpt[names[i]] = values[i].clone();

  return ckpt;
}

export function loadCheckpoint(
  model: tf.LayersModel,
  ckpt: WeightMap,
) {
  const names = model.weights.map(w => w.name);
  const current = model.getWeights();

  const next: tf.Tensor[] = new Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const k = names[i];
    const ck = canonicalKey(k);
    const t = ckpt[ck];
    next[i] = t ? t : current[i];
  }

  model.setWeights(next);
}


/** ---------------- Key extraction ---------------- */
function canonicalKey(k: string): string {
  // strips TFJS auto-suffix like "_1/" before the slash
  return k.replace(/_\d+$/, "");
}

export function toCanonicalMap(m: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const [key, t] of Object.entries(m)) {
    const ck = canonicalKey(key);
    if (out[ck]) {
      // Optional: fail fast on collisions so you don't silently drop weights
      throw new Error(`toCanonicalMap collision: "${ck}" from "${key}"`);
    }
    out[ck] = t; // keep same tensor reference
  }
  return out;
}

export function getEncoderKeys(allKeys: string[]): string[] {
  // matches encoder_stage{X}_conv{Y}/...
  return allKeys.filter(k => k.startsWith("encoder_stage")).sort();
}

export function getDecoderKeysForTask(allKeys: string[], task: TaskName): string[] {
  // matches `${task}_decoder_stage.../...`
  const decPrefix = `${task}_decoder_`; // works because names are `${task}_decoder_stage...`
  return allKeys.filter(k => k.startsWith(decPrefix)).sort();
}

/** For cross-task decoder alignment we strip the task prefix */
export function toRelativeDecoderKey(fullKey: string, task: TaskName): string {
  const decPrefix = `${task}_decoder_`;
  if (!fullKey.startsWith(decPrefix)) throw new Error(`Not a decoder key: ${fullKey}`);
  return `decoder/${fullKey.slice(decPrefix.length)}`; // e.g. decoder/stage0_conv0/depthwise_kernel
}

export function fromRelativeDecoderKey(relKey: string, task: TaskName): string {
  if (!relKey.startsWith("decoder/")) throw new Error(`Bad rel decoder key: ${relKey}`);
  return `${task}_decoder_${relKey.slice("decoder/".length)}`;
}


/** ---------------- Dict ops ---------------- */

export function meanSoup(dicts: WeightMap[], keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = tf.tidy(() => tf.stack(dicts.map(d => d[k])).mean(0));
  return out;
}

export function weightedMeanSoup(dicts: WeightMap[], keys: string[], weights: number[]): WeightMap {
  if (dicts.length !== weights.length) {
    throw new Error(`weightedMeanSoup: dicts.length=${dicts.length} != weights.length=${weights.length}`);
  }
  const n = dicts.length;
  const out: WeightMap = {};
  if (n === 0 || keys.length === 0) return out;

  // normalize weights
  const w = tf.tensor1d(weights.map(x => (Number.isFinite(x) && x > 0 ? x : 0)), "float32");
  const wsum = tf.maximum(w.sum(), tf.scalar(EPS));

  for (const k of keys) {
    out[k] = tf.tidy(() => {
      const stacked = tf.stack(dicts.map(d => d[k])); // [n, ...]
      // reshape weights to broadcast across stacked tensor
      const wr = w.reshape([n, ...Array(stacked.rank - 1).fill(1)]);
      return stacked.mul(wr).sum(0).div(wsum);
    });
  }

  return out;
}

export function deltaDict(cur: WeightMap, last: WeightMap, keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = tf.tidy(() => cur[k].sub(last[k]));
  return out;
}

export function flatten(dict: WeightMap, keys: string[]): tf.Tensor1D {
  return tf.tidy(() => tf.concat(keys.map(k => dict[k].flatten())) as tf.Tensor1D);
}

export function unflatten(vec: tf.Tensor1D, keys: string[], shapes: number[][]): WeightMap {
  return tf.tidy(() => {
    const out: WeightMap = {};
    let start = 0;

    for (let i = 0; i < keys.length; i++) {
      const size = shapes[i].reduce((a, b) => a * b, 1);
      out[keys[i]] = vec.slice([start], [size]).reshape(shapes[i]);
      start += size;
    }
    return out;
  });
}


export function pick(m: WeightMap, keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = m[k];
  return out;
}

/** ---------------- Conflict-averse delta ---------------- */

function projectToSimplex(v: Float64Array): Float64Array {
  const n = v.length;
  const u = Array.from(v).sort((a, b) => b - a);
  let cssv = 0, rho = -1;
  for (let i = 0; i < n; i++) {
    cssv += u[i];
    const t = (cssv - 1) / (i + 1);
    if (u[i] - t > 0) rho = i;
  }
  const theta = (u.slice(0, rho + 1).reduce((s, x) => s + x, 0) - 1) / (rho + 1);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = Math.max(v[i] - theta, 0);
  return w;
}

function matVec(A: number[][], x: Float64Array): Float64Array {
  const n = A.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}

function dotArr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function quadForm(A: number[][], x: Float64Array): number {
  const Ax = matVec(A, x);
  return dotArr(x, Ax);
}

function objCA(A: number[][], b: Float64Array, c: number, x: Float64Array): number {
  const Ab = matVec(A, b);
  return dotArr(x, Ab) + c * Math.sqrt(quadForm(A, x) + EPS);
}

function gradCA(A: number[][], b: Float64Array, c: number, x: Float64Array): Float64Array {
  const Ab = matVec(A, b);
  const Ax = matVec(A, x);
  const denom = Math.sqrt(quadForm(A, x) + EPS);
  const g = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) g[i] = Ab[i] + (c * Ax[i]) / denom;
  return g;
}

export function solveSimplexProjectedGD(A: number[][], c: number, maxIters = 500, tol = 1e-9): Float64Array {
  const n = A.length;
  let x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = 1 / n;
  const b = x.slice();

  let fx = objCA(A, b, c, x);
  let step = 0.1;

  for (let it = 0; it < maxIters; it++) {
    const g = gradCA(A, b, c, x);
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) z[i] = x[i] - step * g[i];
    const xNew = projectToSimplex(z);
    const fNew = objCA(A, b, c, xNew);

    if (fNew > fx + 1e-12) { 
      step *= 0.5; 
      if (step < 1e-12) break; 
      continue; 
    }

    let diff1 = 0;
    for (let i = 0; i < n; i++) diff1 += Math.abs(xNew[i] - x[i]);

    x.set(xNew); 
    fx = fNew; 
    step = Math.min(step * 1.05, 1.0);
    if (diff1 < tol) break;
  }

  return x;
}


/** -------------------- small tensor helpers -------------------- */
export function flat1d(t: tf.Tensor) {
  return tf.tidy(() => t.flatten().asType("float32") as tf.Tensor1D);
}

export function tensorOrNumberArray(x: tf.Tensor | number[]): number[] {
  if (Array.isArray(x)) return x.slice();
  // tensor: sync OK for small vectors; for large, use async data()
  return Array.from(x.dataSync());
}

export function disposeWeightMap(wm: WeightMap) {
  for (const k of Object.keys(wm)) wm[k].dispose();
}

/** Clone a block dict safely */
export function cloneBlock(b: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const [k, t] of Object.entries(b)) out[k] = t.clone();
  return out;
}

export function cloneWeightMap(wm: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const k of Object.keys(wm)) out[k] = wm[k].clone();
  return out;
}


function crossAttention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const d = q.shape[q.shape.length - 1];
    const scale = 1 / Math.sqrt(d);

    const kT = tf.transpose(k, [
      ...Array.from({ length: k.rank - 2 }, (_, i) => i),
      k.rank - 1,
      k.rank - 2,
    ]);

    const scores = tf.matMul(q, kT).mul(scale);
    const attn = tf.softmax(scores, -1);
    return tf.matMul(attn, v);
  });
}

function l2Normalize(t: tf.Tensor1D): tf.Tensor1D {
  return tf.tidy(() => t.div(t.norm().add(EPS)) as tf.Tensor1D);
}

export function fixedCrossAttnDelta(
  lastList: tf.Tensor1D[],
  deltaList: tf.Tensor1D[],
): tf.Tensor1D[] {
  const L = lastList.length;
  if (L === 0) return [];
  if (deltaList.length !== L) throw new Error("lastList and deltaList must match length");

  const dims = lastList.map(t => t.shape[0]);
  const D = Math.min(...dims);

  // Build Q,K,V and attnV in a tidy, but KEEP the final outputs we want.
  const { attnV } = tf.tidy(() => {
    const Q = tf.stack(lastList.map(t => l2Normalize(t.slice([0], [D])))); // [L,D]
    const K = tf.stack(lastList.map(t => l2Normalize(t.slice([0], [D])))); // [L,D]
    const V = tf.stack(deltaList.map(t => t.slice([0], [D])));             // [L,D]
    const attnV = crossAttention(Q, K, V);                                  // [L,D]
    return { attnV: tf.keep(attnV) }; // keep outside tidy
  });

  const out: tf.Tensor1D[] = [];
  for (let l = 0; l < L; l++) {
    const targetDim = dims[l];
    const t = tf.tidy(() => {
      const head = attnV.gather([l]).reshape([D]) as tf.Tensor1D;
      if (targetDim === D) return head;
      if (targetDim > D) {
        const pad = tf.zeros([targetDim - D], "float32");
        return tf.concat([head, pad], 0) as tf.Tensor1D;
      }
      return head.slice([0], [targetDim]) as tf.Tensor1D;
    });
    out.push(t);
  }

  attnV.dispose();
  return out;
}