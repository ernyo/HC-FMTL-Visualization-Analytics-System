import * as tf from '@tensorflow/tfjs';
import { LossOrMetricFn } from '@tensorflow/tfjs-layers/dist/types';
import { Head } from './model';

export type Metric =
  | 'meanIoU'
  | 'pixelAccuracy'
  | 'meanClassAccuracy'
  | 'precision'
  | 'recall'
  | 'f1'
  | 'odsF'
  | 'absRel'
  | 'rmse'
  | 'rmseLog'
  | 'siLog'
  | 'delta1'
  | 'delta2'
  | 'delta3'
  | 'meanAngularErr'
  | 'pct_11_25'
  | 'pct_22_5'
  | 'pct_30';

export const LOWER_IS_BETTER: Record<string, boolean> = {
  'meanIoU': false,
  'pixelAccuracy': false,
  'meanClassAccuracy': false,
  'precision': false,
  'recall': false,
  'f1': false,
  'odsF': false,
  'absRel': true,
  'rmse': true,
  'rmseLog': true,
  'siLog': true,
  'delta1': false,
  'delta2': false,
  'delta3': false,
  'meanAngularErr': true,
  'pct_11_25': false,
  'pct_22_5': false,
  'pct_30': false,
};

/* --------------------------------- Helpers ----------------------------- */
function _flattenHW(x: tf.Tensor): tf.Tensor2D {
  const [b, h, w, c] = x.shape as [number, number, number, number];
  return x.reshape([b * h * w, c]) as tf.Tensor2D;
}
function _flattenHW1(x: tf.Tensor): tf.Tensor1D {
  const [b, h, w, c] = x.shape as [number, number, number, number];
  return x.reshape([b * h * w * c]) as tf.Tensor1D;
}

/** Binary dilation for tolerance matching (mask must be 0/1 float). */
function dilateBinary(mask01: tf.Tensor4D, r: number): tf.Tensor4D {
  if (r <= 0) return mask01;
  const k = 2 * r + 1;
  return tf.maxPool(mask01, [k, k], [1, 1], "same");
}
type OdsOptions = {
  thresholds?: number[];    // e.g. 0.01..0.99
  tolerancePx?: number;     // 0 = exact, 1/2 = allow small shifts
  eps?: number;
};

// ----------------------- SEMANTIC SEGMENTATION --------------------------
function segPixelAcc(yTrue: tf.Tensor, yLogits: tf.Tensor, numClasses: number) {
  return tf.tidy(() => {
    // yTrue: [B,H,W,C] one-hot; yLogits: [B,H,W,C]
    const yT2 = _flattenHW(yTrue); // [N,C]
    const yP2 = _flattenHW(yLogits); // [N,C]
    const gtIdx = tf.argMax(yT2, 1);        // [N]
    const prIdx = tf.argMax(tf.softmax(yP2), 1); // [N]
    const correct = prIdx.equal(gtIdx);
    const num = tf.sum(tf.cast(correct, 'float32'));
    const den = tf.maximum(tf.sum(tf.cast(yT2, 'float32')), tf.scalar(1e-6));
    return num.div(den);
  });
}

function segMeanClassAcc(yTrue: tf.Tensor, yLogits: tf.Tensor, numClasses: number) {
  return tf.tidy(() => {
    const yT2 = _flattenHW(yTrue); // [N,C]
    const yP2 = _flattenHW(yLogits); // [N,C]
    const gtIdx = tf.argMax(yT2, 1);        // [N]
    const prIdx = tf.argMax(tf.softmax(yP2), 1); // [N]

    // For each class c: acc_c = TP_c / GT_c
    const prOne = tf.oneHot(prIdx, numClasses); // [N,C]
    const gtOne = tf.oneHot(gtIdx, numClasses); // [N,C]
    const TPc = tf.sum(prOne.mul(gtOne), 0); // [C]
    const GTc = tf.sum(gtOne, 0);           // [C]
    const mask = tf.ones([numClasses], 'float32')
    const acc_c = TPc.div(GTc.add(1e-6)).mul(mask);
    const denom = tf.maximum(tf.sum(mask.mul(GTc.greater(0).cast('float32'))), tf.scalar(1.0)); // only classes present
    return tf.sum(acc_c).div(denom);
  });
}

function segMeanIoU(yTrue: tf.Tensor, yLogits: tf.Tensor, numClasses: number) {
  return tf.tidy(() => {
    const yT2 = _flattenHW(yTrue); // [N,C]
    const yP2 = _flattenHW(yLogits); // [N,C]
    const gtIdx = tf.argMax(yT2, 1);
    const prIdx = tf.argMax(tf.softmax(yP2), 1);
    const prOne = tf.oneHot(prIdx, numClasses); // [N,C]
    const gtOne = tf.oneHot(gtIdx, numClasses); // [N,C]
    const inter = tf.sum(prOne.mul(gtOne), 0); // [C]
    const prSum = tf.sum(prOne, 0);            // [C]
    const gtSum = tf.sum(gtOne, 0);            // [C]
    const union = prSum.add(gtSum).sub(inter);        // [C]
    const iou = inter.div(union.add(1e-6));
    const present = gtSum.greater(0).cast('float32');
    const denom = tf.maximum(tf.sum(present), tf.scalar(1.0));
    return tf.sum(iou).div(denom);
  });
}

// ----------------------- EDGES & SALIENCY (binary) ----------------------
function precision(yTrue: tf.Tensor, yLogits: tf.Tensor){
  const thresh = 0.5;
  return tf.tidy(() => {
    const yT = _flattenHW1(yTrue.cast('float32'));               // labels in [0,1] as float
    const yP = _flattenHW1(yLogits.cast('float32').sigmoid());   // logits → probs
    const t = yT.greater(0.5);
    const p = yP.greater(thresh);
    const tp = tf.sum(tf.logicalAnd(t, p).cast('float32'));
    const predPos = tf.maximum(tf.sum(p.cast('float32')), tf.scalar(1e-6));
    return tp.div(predPos);
  })
}

function recall(yTrue: tf.Tensor, yLogits: tf.Tensor) {
  const thresh = 0.5;
  return tf.tidy(() => {
    const yT = _flattenHW1(yTrue.cast('float32'));
    const yP = _flattenHW1(yLogits.cast('float32').sigmoid());
    const t = yT.greater(0.5);
    const p = yP.greater(thresh);
    const tp = tf.sum(tf.logicalAnd(t, p).cast('float32'));
    const truePos = tf.maximum(tf.sum(t.cast('float32')), tf.scalar(1e-6));
    return tp.div(truePos);
  })
}

function f1(yTrue: tf.Tensor, yLogits: tf.Tensor) {
  return tf.tidy(() => {
    const prec = precision(yTrue, yLogits);
    const rec = recall(yTrue, yLogits);
    return prec.mul(rec).mul(2).div(prec.add(rec).add(1e-6));
  })
};

function edgeOdsF(yT: tf.Tensor, yLogits: tf.Tensor, opts: OdsOptions = {}): tf.Scalar {
  const thresholds =
    opts.thresholds ??
    Array.from({ length: 99 }, (_, i) => (i + 1) / 100); // 0.01..0.99

  const tol = opts.tolerancePx ?? 1;
  const eps = opts.eps ?? 1e-8;

  return tf.tidy(() => {
    // Ensure rank-4 [B,H,W,1]
    if (yT.rank === 3) yT = yT.expandDims(-1);
    if (yLogits.rank === 3) yLogits = yLogits.expandDims(-1);

    // Convert yTrue to {0,1} float
    // (If your GT is already 0/1 float, this is safe.)
    const yTrue01 = yT.greater(tf.scalar(0.5)).toFloat() as tf.Tensor4D;

    // Convert predictions to probabilities
    const yProb = tf.sigmoid(yLogits) as tf.Tensor4D;

    // Precompute dilated GT once (used for TP/FP)
    const gtDil = dilateBinary(yTrue01, tol);

    let bestF: tf.Scalar = tf.scalar(0);

    for (const t of thresholds) {
      // binarize predictions at threshold t
      const thr = tf.scalar(t);
      const pred01 = yProb.greaterEqual(thr).toFloat() as tf.Tensor4D;

      // Dilate prediction for FN (coverage check)
      const predDil = dilateBinary(pred01, tol);

      // TP: pred hits dilated GT
      const tp = pred01.mul(gtDil).sum(); // scalar
      // FP: pred outside dilated GT
      const fp = pred01.mul(tf.scalar(1).sub(gtDil)).sum();
      // FN: GT not covered by dilated pred
      const fn = yTrue01.mul(tf.scalar(1).sub(predDil)).sum();

      const TP = tp as tf.Scalar;
      const FP = fp as tf.Scalar;
      const FN = fn as tf.Scalar;

      const precision = TP.div(TP.add(FP).add(eps));
      const recall = TP.div(TP.add(FN).add(eps));
      const F = precision.mul(recall).mul(2).div(precision.add(recall).add(eps)) as tf.Scalar;

      // update bestF (dispose old best to avoid leaks inside tidy)
      const newBest = tf.maximum(bestF, F) as tf.Scalar;
      bestF.dispose();
      bestF = newBest;

      // Clean up loop scalars
      thr.dispose();
      pred01.dispose();
      predDil.dispose();
      tp.dispose();
      fp.dispose();
      fn.dispose();
      precision.dispose();
      recall.dispose();
      F.dispose();
    }

    return bestF;
  });
}

// ----------------------------- DEPTH ------------------------------------
function depthAbsRel(yT: tf.Tensor, yP: tf.Tensor) {
  return tf.tidy(() => {
    const eps = tf.scalar(1e-6);
    const t = _flattenHW1(yT.cast('float32'));
    const p = _flattenHW1(yP.cast('float32'));
    const valid = t.greater(0);
    const tv = t.where(valid, eps);
    const pv = p.where(valid, eps);
    const rel = pv.sub(tv).abs().div(tv);
    const den = tf.maximum(tf.sum(valid.cast('float32')), tf.scalar(1e-6));
    return tf.sum(rel).div(den);
  });
}
function depthRMSE(yT: tf.Tensor, yP: tf.Tensor) {
  return tf.tidy(() => {
    const t = _flattenHW1(yT.cast('float32'));
    const p = _flattenHW1(yP.cast('float32'));
    const valid = t.greater(0);
    const err2 = p.sub(t).square().where(valid, tf.scalar(0));
    const den = tf.maximum(tf.sum(valid.cast('float32')), tf.scalar(1e-6));
    return tf.sqrt(tf.sum(err2).div(den));
  });
}
function depthRMSELog(yT: tf.Tensor, yP: tf.Tensor) {
  return tf.tidy(() => {
    const eps = tf.scalar(1e-6);
    const t = _flattenHW1(yT.cast('float32'));
    const p = _flattenHW1(yP.cast('float32'));
    const valid = t.greater(0);
    const le = tf.log(p.add(eps)).sub(tf.log(t.add(eps)));
    const err2 = le.square().where(valid, tf.scalar(0));
    const den = tf.maximum(tf.sum(valid.cast('float32')), tf.scalar(1e-6));
    return tf.sqrt(tf.sum(err2).div(den));
  });
}
function depthSILog(yT: tf.Tensor, yP: tf.Tensor) {
  return tf.tidy(() => {
    const eps = tf.scalar(1e-6);
    const t = _flattenHW1(yT.cast('float32'));
    const p = _flattenHW1(yP.cast('float32'));
    const valid = t.greater(0);
    const d = tf.log(p.add(eps)).sub(tf.log(t.add(eps))).where(valid, tf.scalar(0));
    const n = tf.maximum(tf.sum(valid.cast('float32')), tf.scalar(1e-6));
    const mu = tf.sum(d).div(n);
    const varTerm = tf.sum(d.square()).div(n).sub(mu.square());
    return tf.sqrt(tf.maximum(varTerm, tf.scalar(0)));
  });
}
function depthDelta(yT: tf.Tensor, yP: tf.Tensor, threshold: number) {
  return tf.tidy(() => {
    const eps = tf.scalar(1e-6);
    const t = _flattenHW1(yT.cast('float32'));
    const p = _flattenHW1(yP.cast('float32'));
    const valid = t.greater(0);
    const tv = t.where(valid, tf.scalar(1)); // avoid 0 division for invalid
    const pv = p.where(valid, tf.scalar(1));
    const ratio = tf.maximum(pv.div(tv), tv.div(pv));
    const ok = ratio.less(threshold).logicalAnd(valid);
    const num = tf.sum(ok.cast('float32'));
    const den = tf.maximum(tf.sum(valid.cast('float32')), tf.scalar(1e-6));
    return num.div(den);
  });
}

// ----------------------------- NORMALS ----------------------------------
function normalsMeanAngularErr(yT: tf.Tensor, yP: tf.Tensor) {
  return tf.tidy(() => {
    // yT,yP: [B,H,W,3]
    const eps = tf.scalar(1e-6);
    const t = _flattenHW(yT.cast('float32')).div(_flattenHW(yT.cast('float32')).norm('euclidean', 1, true).maximum(eps));
    const p = _flattenHW(yP.cast('float32')).div(_flattenHW(yP.cast('float32')).norm('euclidean', 1, true).maximum(eps));
    const dots = tf.sum(t.mul(p), 1).clipByValue(-1, 1); // [N]
    const ang = tf.acos(dots).mul(180/Math.PI); // degrees
    return tf.mean(ang);
  });
}
function normalsWithin(yT: tf.Tensor, yP: tf.Tensor, thresholdDeg: number) {
  return tf.tidy(() => {
    const eps = tf.scalar(1e-6);
    const t = _flattenHW(yT.cast('float32')).div(_flattenHW(yT.cast('float32')).norm('euclidean', 1, true).maximum(eps));
    const p = _flattenHW(yP.cast('float32')).div(_flattenHW(yP.cast('float32')).norm('euclidean', 1, true).maximum(eps));
    const dots = tf.sum(t.mul(p), 1).clipByValue(-1, 1);
    const ang = tf.acos(dots).mul(180/Math.PI);
    const ok = ang.less(thresholdDeg);
    return tf.mean(ok.cast('float32'));
  });
}

// ----------------------------- SUITE ------------------------------------
function createMetricForHead(head: Head): LossOrMetricFn {
  const config = head.taskConfig;
  const metric: Metric = config.metric;
  switch (metric) {
    /* --------- segmentation metrics (seg head) ---------- */
    case 'meanIoU':
      return (yT, yP) => segMeanIoU(yT, yP, config.filters);

    case 'pixelAccuracy':
      return (yT, yP) => segPixelAcc(yT, yP, config.filters);

    case 'meanClassAccuracy':
      return (yT, yP) => segMeanClassAcc(yT, yP, config.filters);

    /* --------- binary (edge/saliency) ---------- */
    case 'precision':
      return (yT, yP) => precision(yT, yP);

    case 'recall':
      return (yT, yP) => recall(yT, yP);

    case 'f1':
      return (yT, yP) => f1(yT, yP);

    case 'odsF':
      return (yT, yP) => edgeOdsF(yT, yP, { tolerancePx: 1 });

    /* --------- depth ---------- */
    case 'absRel':
      return (yT, yP) => depthAbsRel(yT, yP);

    case 'rmse':
      return (yT, yP) => depthRMSE(yT, yP);

    case 'rmseLog':
      return (yT, yP) => depthRMSELog(yT, yP);

    case 'siLog':
      return (yT, yP) => depthSILog(yT, yP);

    case 'delta1':
      return (yT, yP) => depthDelta(yT, yP, 1.25);

    case 'delta2':
      return (yT, yP) => depthDelta(yT, yP, 1.25 ** 2);

    case 'delta3':
      return (yT, yP) => depthDelta(yT, yP, 1.25 ** 3);

    /* --------- normals ---------- */
    case 'meanAngularErr':
      return (yT, yP) => normalsMeanAngularErr(yT, yP);

    case 'pct_11_25':
      return (yT, yP) => normalsWithin(yT, yP, 11.25);

    case 'pct_22_5':
      return (yT, yP) => normalsWithin(yT, yP, 22.5);

    case 'pct_30':
      return (yT, yP) => normalsWithin(yT, yP, 30.0);

    default:
      throw new Error(`Unknown metric name: ${metric}`);
  }
}


/** Object suitable for tf.Model.compile({ metrics }) */
export function getMetricsObject(heads: Head[]): { [outputName: string]: string | LossOrMetricFn } {
  const metrics = {};

  for (const head of heads) {
    const metric = createMetricForHead(head)
    metrics[head.name] = metric;
  }

  return metrics;
}