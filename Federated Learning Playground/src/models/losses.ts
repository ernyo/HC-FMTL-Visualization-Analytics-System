import * as tf from '@tensorflow/tfjs';
import { LossOrMetricFn } from '@tensorflow/tfjs-layers/dist/types';
import { Head } from './model';

export type Loss = 
  | 'categoricalCrossEntropy'
  | 'softmaxCrossEntropy'
  | 'binaryCrossEntropy'
  | 'sigmoidCrossEntropy'
  | 'meanSquaredError'
  | 'mse'
  | 'absoluteDifference'
  | 'mae'
  | 'meanAbsoluteError'
  | 'huber'
  | 'huberLoss'
  | 'hinge'
  | 'hingeLoss'
  | 'cosineDistance'
  | 'computeWeightedLoss';

/* ----------------------- LossSuite: weighted losses ----------------------- */
function weighted(fn: LossOrMetricFn, w: number) {
  return (yT: tf.Tensor, yP: tf.Tensor) => tf.tidy(() => fn(yT, yP).mul(w));
}

function createLossForHead(head: Head): LossOrMetricFn {
  const loss = head.taskConfig.loss;
  switch (loss) {
    /* -------- cross-entropy style losses ---------- */

    // multi-class, one-hot labels + logits
    case 'categoricalCrossEntropy':
    case 'softmaxCrossEntropy':
      return (yT, yP) =>
        tf.losses.softmaxCrossEntropy(
          yT,          // one-hot labels
          yP,          // logits
          undefined,   // weights
          undefined,   // labelSmoothing
          1           // reduction axis (last)
        );

    // binary labels + logits
    case 'binaryCrossEntropy':
    case 'sigmoidCrossEntropy':
      return (yT, yP) =>
        tf.losses.sigmoidCrossEntropy(
          yT,          // labels
          yP           // logits
          // weights?, labelSmoothing?, reduction? left default
        );

    /* -------- L2 / L1-ish losses ---------- */

    case 'meanSquaredError':
    case 'mse':
      return (yT, yP) =>
        tf.losses.meanSquaredError(
          yT,
          yP
          // weights?, reduction? left default
        );

    case 'absoluteDifference':
    case 'mae':
    case 'meanAbsoluteError':
      return (yT, yP) =>
        tf.losses.absoluteDifference(
          yT,
          yP
          // weights?, reduction? left default
        );

    case 'huber':
    case 'huberLoss':
      return (yT, yP) =>
        tf.losses.huberLoss(
          yT,
          yP
          // weights?, delta?, reduction? left default
        );

    /* -------- margin / hinge ---------- */

    case 'hinge':
    case 'hingeLoss':
      return (yT, yP) =>
        tf.losses.hingeLoss(
          yT,
          yP
          // weights?, reduction? left default
        );

    /* -------- cosine distance ---------- */

    case 'cosineDistance':
      return (yT, yP) => {
        const EPS = tf.scalar(1e-6);
        const yTn = yT.div(yT.norm('euclidean', -1, true).maximum(EPS));
        const yPn = yP.div(yP.norm('euclidean', -1, true).maximum(EPS));
        // 1 - cos similarity
        const cosSim = yTn.mul(yPn).sum(-1);
        return tf.sub(1, cosSim).mean();
      };

    /* -------- generic weighted loss wrapper ---------- */

    // If you already computed some per-example loss externally and just
    // want tf.losses.computeWeightedLoss, you can expose it like this:
    case 'computeWeightedLoss':
      // Assumes yP already contains per-example loss values.
      return (perExampleLoss, weights) =>
        tf.losses.computeWeightedLoss(
          perExampleLoss,
          weights
          // reduction? left default
        );

    default:
      throw new Error(`Unknown loss name: ${name}`);
  }
}

/** Order: [seg_logits, edge_logits, sal_logits, depth, normal] */
export function getLossArray(heads: Head[]): { [outputName: string]: LossOrMetricFn }{
  const losses = {};
  for (const head of heads) {
    const name = head.name // e.g. 'semseg', 'edge', ...
    const lossFn = createLossForHead(head);
    losses[name] = weighted(lossFn, head.taskConfig.lossWeight);
  }

  return losses;
}