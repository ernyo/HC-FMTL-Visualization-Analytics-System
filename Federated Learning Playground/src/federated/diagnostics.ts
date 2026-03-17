import * as tf from "@tensorflow/tfjs";
import type { FederatedClient } from "./client";
import {
  WeightMap,
  getEncoderKeys,
  pick,
  deltaDict,
  flatten,
  EPS,
} from "./utils";

export type InterClientDiag = {
  // divergence
  meanDeltaNorm: number;           // ||mean(Δ)||
  meanClientDeltaNorm: number;     // mean_i ||Δ_i||
  meanDistToMean: number;          // mean_i ||Δ_i - mean(Δ)||

  // conflict
  meanCosine: number;              // mean_{i<j} cos(Δ_i, Δ_j)
  fracNegativeCosine: number;      // fraction of pairs with cos < 0
};

function cosine(a: tf.Tensor1D, b: tf.Tensor1D): tf.Scalar {
  return tf.tidy(() => {
    const dot = tf.sum(a.mul(b));
    const na = tf.norm(a).add(EPS);
    const nb = tf.norm(b).add(EPS);
    return dot.div(na.mul(nb)) as tf.Scalar;
  });
}

/**
 * Cheap diagnostics using encoder deltas (Δ = cur - last).
 * Uses checkpoint differences as a proxy for gradients/updates.
 */
export function computeInterClientDiagnostics(
  clients: FederatedClient[],
  saveCkpt: WeightMap[],
  lastCkpt: WeightMap[],
): InterClientDiag {
  const N = clients.length;
  if (N < 2) {
    return {
      meanDeltaNorm: 0,
      meanClientDeltaNorm: 0,
      meanDistToMean: 0,
      meanCosine: 1,
      fracNegativeCosine: 0,
    };
  }

  const encKeys = getEncoderKeys(Object.keys(saveCkpt[0]));

  // Build Δ_i vectors
  const deltas: tf.Tensor1D[] = [];
  for (let i = 0; i < N; i++) {
    const cur = pick(saveCkpt[i], encKeys);
    const last = pick(lastCkpt[i], encKeys);
    const d = deltaDict(cur, last, encKeys);
    const v = flatten(d, encKeys) as tf.Tensor1D;
    deltas.push(v);
    Object.values(d).forEach(t => t.dispose());
  }

  // mean delta
  const meanDelta = tf.tidy(() => tf.addN(deltas).div(N)) as tf.Tensor1D;
  const meanDeltaNorm = tf.tidy(() => tf.norm(meanDelta).dataSync()[0]);

  let sumClientNorm = 0;
  let sumDistToMean = 0;
  for (let i = 0; i < N; i++) {
    sumClientNorm += tf.tidy(() => tf.norm(deltas[i]).dataSync()[0]);
    sumDistToMean += tf.tidy(() => tf.norm(deltas[i].sub(meanDelta)).dataSync()[0]);
  }
  const meanClientDeltaNorm = sumClientNorm / N;
  const meanDistToMean = sumDistToMean / N;

  // Pairwise cosine stats
  let pairs = 0;
  let sumCos = 0;
  let negCount = 0;

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const c = tf.tidy(() => cosine(deltas[i], deltas[j]).dataSync()[0]);
      sumCos += c;
      if (c < 0) negCount++;
      pairs++;
    }
  }

  const meanCosine = pairs > 0 ? sumCos / pairs : 1;
  const fracNegativeCosine = pairs > 0 ? negCount / pairs : 0;

  meanDelta.dispose();
  deltas.forEach(t => t.dispose());

  return { meanDeltaNorm, meanClientDeltaNorm, meanDistToMean, meanCosine, fracNegativeCosine };
}
