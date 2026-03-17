import * as tf from "@tensorflow/tfjs";
import type { TaskName } from "../models/model.js";
import { FederatedClient } from "./client.js";
import {
  EPS,
  getEncoderKeys,
  getDecoderKeysForTask,
  WeightMap,
  cloneWeightMap,
  pick,
  flatten,
  unflatten,
  deltaDict,
  meanSoup,
  toRelativeDecoderKey,
  fromRelativeDecoderKey,
  solveSimplexProjectedGD,
  loadCheckpoint,
  cloneBlock,
  fixedCrossAttnDelta,
  flat1d,
  weightedMeanSoup,
} from "./utils.js";

/** ---------------- Types ---------------- */
export type EncoderAgg = "none" | "fedavg" | "conflict_averse";
export type DecoderAgg = "none" | "fedavg" | "cross_attention";


/** ---------------- Homo aggregation for encoder deltas ---------------- */
function homoAverageEncoderDeltasInPlace(deltas: tf.Tensor1D[], clients: FederatedClient[]) {
  const N = clients.length;
  let start = 0;

  while (start < N) {
    let end = start + 1;
    while (
      end < N &&
      clients[end].dataname === clients[end - 1].dataname &&
      clients[end].tasks.join("|") === clients[end - 1].tasks.join("|")
    ) end++;

    // average deltas[start:end]
    const avg = tf.tidy(() => tf.addN(deltas.slice(start, end)).div(end - start)) as tf.Tensor1D;

    // dispose old tensors and replace with clones
    for (let i = start; i < end; i++) {
      deltas[i].dispose();
      deltas[i] = (i === start) ? avg : avg.clone();
    }

    // we kept avg as deltas[start], so don't dispose it here
    start = end;
  }
}


export function getCaDelta(flattenDeltaList: tf.Tensor1D[], caC: number): tf.Tensor1D {
  return tf.tidy(() => {
    const N = flattenDeltaList.length;
    const grads = tf.stack(flattenDeltaList).transpose();     // [d, N]
    const GG = grads.transpose().matMul(grads);               // [N, N]
    const g0Norm = tf.sqrt(GG.mean().add(EPS));
    const A = GG.arraySync() as number[][];
    const c = caC * (g0Norm.arraySync() as number) + EPS;

    // solver returns Float64Array
    const ww64 = solveSimplexProjectedGD(A, c) as Float64Array;
    const ww = tf.tensor1d(Array.from(ww64), "float32");      // [N]

    const gw = grads.mul(ww.reshape([1, N])).sum(1); // gw = sum_j w_j * delta_j
    const lambda = tf.scalar(c).div(tf.norm(gw).add(EPS)); // lambda = c / ||gw||
    const g = grads.mean(1).add(gw.mul(lambda)); // globalCA = (mean(delta) + lambda * gw) / (1 + caC^2)

    // rescale=1
    return g.div(tf.scalar(1 + caC * caC)) as tf.Tensor1D;
  });
}

function numTrainSamplesOfClient(c: FederatedClient): number {
  const numSamples = c.state.data.numSamples;
  const percTrain = c.state.data.percTrain;
  if (!Number.isFinite(numSamples)) return 1;
  const nTrain = Math.round((numSamples as number) * (percTrain as number) / 100);
  return Math.max(1, nTrain);
}


/** ---------------- Main aggregate ---------------- */

export async function aggregate(
  clients: FederatedClient[],
  saveCkpt: WeightMap[],
  lastCkpt: WeightMap[],
  encoderAgg: EncoderAgg,
  decoderAgg: DecoderAgg,
  encoderAlpha: number,
  decoderBeta: number,
  caC = 0.4,
) {
  const N = clients.length;
  const update: WeightMap[] = saveCkpt.map(m => cloneWeightMap(m)); // increases memory
  const allKeys0 = Object.keys(saveCkpt[0]);

  // ----- Encoder -----
  if (encoderAgg !== "none") {
    const encKeys = getEncoderKeys(allKeys0);
    const weights  = clients.map(numTrainSamplesOfClient);
    const encShapes = encKeys.map(k => saveCkpt[0][k].shape.slice());

    if (encoderAgg === "fedavg") {
      const encDicts = saveCkpt.map(m => pick(m, encKeys));
      const encAvg = weightedMeanSoup(encDicts, encKeys, weights);
      for (let i = 0; i < N; i++) {
        for (const k of encKeys) {
          const old = update[i][k];
          if (old && old !== encAvg[k]) old.dispose();
          update[i][k] = encAvg[k];
        }
      }
    }

    if (encoderAgg === "conflict_averse") {
      // Compute per-client flattened last encoder and deltas
      const flattenLast: tf.Tensor1D[] = [];
      const flattenDelta: tf.Tensor1D[] = [];

      for (let i = 0; i < N; i++) {
        const current_i = pick(saveCkpt[i], encKeys);
        const last_i = pick(lastCkpt[i], encKeys);

        // Local delta
        const localDelta_i = deltaDict(current_i, last_i, encKeys); // localDelta_i = current_i - last_i
        flattenDelta.push(flatten(localDelta_i, encKeys));
        flattenLast.push(flatten(last_i, encKeys));

        // dispose deltaDict tensors
        Object.values(localDelta_i).forEach(t => t.dispose());
      }

      // Global CA delta (same for all)
      const globalCA = getCaDelta(flattenDelta, caC);
      const globalCAScaled = tf.tidy(() => globalCA.mul(encoderAlpha));
      
      // Homo aggregation on local deltas (matches PyTorch)
      homoAverageEncoderDeltasInPlace(flattenDelta, clients); // this is the homoLocalDelta
    
      // Personalized new encoder
      const flattenNew: tf.Tensor1D[] = flattenLast.map((last_i, i) => 
        tf.tidy(() => last_i.add(flattenDelta[i]).add(globalCAScaled)) as tf.Tensor1D // new_i = last_i + homoLocalDelta_i + fixedEncAlpha_i * globalCA
      ); 
      
      for (let i = 0; i < N; i++) {
        const newEnc = unflatten(flattenNew[i], encKeys, encShapes);
        for (const k of encKeys) {
          const old = update[i][k];
          if (old && old !== newEnc[k]) old.dispose();
          update[i][k] = newEnc[k];
        }
      }

      // local temporaries can go away (cache clones persist)
      flattenNew.forEach(t => t.dispose());
      flattenLast.forEach(t => t.dispose());
      flattenDelta.forEach(t => t.dispose());
      globalCA.dispose();
      globalCAScaled.dispose();
    }
  }

  // ----- Decoder + Head per task block -----
  if (decoderAgg !== "none") {
    const blocks: { clientIdx: number; task: TaskName }[] = [];
    const relKeySet = new Set<string>();

    for (let i = 0; i < N; i++) {
      for (const task of clients[i].tasks as TaskName[]) {
        const full = getDecoderKeysForTask(Object.keys(saveCkpt[i]), task);
        for (const fk of full) relKeySet.add(toRelativeDecoderKey(fk, task));
        blocks.push({ clientIdx: i, task });
      }
    }

    const relKeys = Array.from(relKeySet).sort();

    if (decoderAgg === "fedavg") {
      // Collect tasks present this round
      const taskSet = new Set<TaskName>();
      for (let i = 0; i < N; i++) {
        for (const t of clients[i].tasks as TaskName[]) taskSet.add(t);
      }

      for (const task of Array.from(taskSet)) {
        // Clients that have this task
        const idxs: number[] = [];
        for (let i = 0; i < N; i++) {
          if ((clients[i].tasks as TaskName[]).includes(task)) idxs.push(i);
        }
        if (idxs.length === 0) continue;

        // Define key set from a reference client (assumes same architecture for this task)
        const refIdx = idxs[0];
        const refKeys = Object.keys(saveCkpt[refIdx]);
        const decKeys = getDecoderKeysForTask(refKeys, task);
        if (decKeys.length === 0) continue;

        // Build dicts + FedAvg weights
        const dicts: WeightMap[] = [];
        const weights: number[] = [];

        for (const i of idxs) {
          for (const k of decKeys) {
            if (!saveCkpt[i][k]) {
              throw new Error(`FedAvg missing key "${k}" for clientIdx=${i}, task=${task}`);
            }
          }
          dicts.push(pick(saveCkpt[i], decKeys));
          weights.push(numTrainSamplesOfClient(clients[i]));
        }

        // Weighted average
        const avg = weightedMeanSoup(dicts, decKeys, weights);

        // Write back into each participating client
        for (const i of idxs) {
          const m = update[i];
          for (const k of decKeys) {
            const old = m[k];
            if (old && old !== avg[k]) old.dispose();
            m[k] = avg[k].clone();
          }
        }

        // Dispose the averaged tensors (clones are stored in update[])
        Object.values(avg).forEach(t => t.dispose());
      }
    }

    if (decoderAgg === "cross_attention") {
      // Build block-wise last + delta (cur - last)
      const lastBlocks: WeightMap[] = [];
      const deltaBlocks: WeightMap[] = [];
      const taskOfBlock: TaskName[] = [];
      const clientOfBlock: number[] = [];

      for (const b of blocks) { // increases memory by 72
        const lastM = lastCkpt[b.clientIdx];
        const curM = saveCkpt[b.clientIdx];

        const lb: WeightMap = {};
        const db: WeightMap = {};

        for (const rk of relKeys) {
          const fullKey = fromRelativeDecoderKey(rk, b.task);
          const lastT = lastM[fullKey];
          const curT = curM[fullKey];
          if (!lastT || !curT) continue;

          lb[rk] = lastT.clone();
          db[rk] = tf.tidy(() => curT.sub(lastT)); // delta = cur - last
        }
        
        lastBlocks.push(lb);
        deltaBlocks.push(db);
        taskOfBlock.push(b.task);
        clientOfBlock.push(b.clientIdx);
      }

      const B = lastBlocks.length;
      const newBlocks: WeightMap[] = new Array(B);
      for (let bi = 0; bi < B; bi++) {
        const lastB = lastBlocks[bi];
        const deltaB = deltaBlocks[bi];

        // Flatten per-layer last/delta
        const flatLast: tf.Tensor1D[] = [];
        const flatDelta: tf.Tensor1D[] = [];
        const shapes: Record<string, number[]> = {};

        for (const k of relKeys) {
          const lt = lastB[k];
          const dt = deltaB[k];
          if (!lt || !dt) continue;
          shapes[k] = lt.shape.slice();
          flatLast.push(flat1d(lt));
          flatDelta.push(flat1d(dt));
        }

        // Attention-produced delta (flattened)
        const flatAttnDelta = fixedCrossAttnDelta(flatLast, flatDelta); // compute cross-attention directly

        // Build new params per layer:
        // newDelta = localDelta + beta_l * attnDelta
        // newParam = last + newDelta
        const outBlock: WeightMap = {};
        let layerIdx = 0;
        for (const k of relKeys) {
          const lt = lastB[k];
          const dt = deltaB[k];
          if (!lt || !dt) continue;

          outBlock[k] = tf.tidy(() => {
            const attnD = flatAttnDelta[layerIdx].reshape(dt.shape); 
            const newDelta = dt.add(attnD.mul(decoderBeta)); // newDelta_i,l = localDelta_i,l + fixedDecBeta_i,l * attnDelta_i,l
            return lt.add(newDelta); // newParam_i,l = last_i,l + newDelta_i,l
          });

          layerIdx++;
        }

        // dispose per-block temporaries
        flatLast.forEach(t => t.dispose());
        flatDelta.forEach(t => t.dispose());
        flatAttnDelta.forEach(t => t.dispose());

        newBlocks[bi] = outBlock;
      }

      
      // Write back
      for (let bi = 0; bi < blocks.length; bi++) {
        const b = blocks[bi];
        const m = update[b.clientIdx];
        const nb = newBlocks[bi];

        for (const rk of Object.keys(nb)) {
          const fk = fromRelativeDecoderKey(rk, b.task);
          const old = m[fk];
          if (old) old.dispose();
          m[fk] = nb[rk].clone();
        }
      }

      for (const nb of newBlocks) { // decreases memory by 36
        for (const t of Object.values(nb)) t.dispose();
      }
      
      // dispose temporaries (cache clones persist)
      lastBlocks.forEach(b => Object.values(b).forEach(t => t.dispose()));
      deltaBlocks.forEach(b => Object.values(b).forEach(t => t.dispose())); // decreases memory by 72
    }
  }

  // ----- Load into TFJS models -----
  for (let i = 0; i < N; i++) loadCheckpoint(clients[i].model.model, update[i]);

  // Dispose staging tensors
  const seen = new Set<number>();
  for (const m of update) {
    for (const t of Object.values(m)) {
      const id = (t as any).id; 
      if (id != null && seen.has(id)) continue;
      if (id != null) seen.add(id);
      t.dispose();
    }
  }
}
