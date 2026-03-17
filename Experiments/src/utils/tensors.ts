import * as tf from "@tensorflow/tfjs-node-gpu";
import { Scene } from '../datasets/constants.js';
import { TaskName } from '../models/model.js';

export function batchToTensors(batch: Scene, tasks: TaskName[]) {
  return tf.tidy(() => {
    const { H, W, N, rgb, seg, edge, sal, depth, normal } = batch;

    // Inputs [N, H, W, 3] → float32 [0, 1]
    const xs = tf.tensor4d(Float32Array.from(rgb), [N, H, W, 3]).div(255) as tf.Tensor4D;
    const ys: tf.Tensor[] = [];
    for (const task of tasks) {
      switch(task) {
        case 'semseg':
        // seg to one-hot [N, H, W, K+1]
          const segY = tf.tidy(() => tf.oneHot(tf.tensor(Int32Array.from(seg), [N, H, W, 1], 'int32'), batch.K+1).squeeze([-2]));
          ys.push(segY);
          break;
        case 'edge':
          const edgeY = tf.tensor4d(Float32Array.from(edge), [N, H, W, 1]).div(255) as tf.Tensor4D;
          ys.push(edgeY);
          break;
        case 'saliency':
          const salY = tf.tensor4d(Float32Array.from(sal), [N, H, W, 1]).div(255) as tf.Tensor4D;
          ys.push(salY);
          break;
        case 'depth':
          const depthY = tf.tensor4d(Float32Array.from(depth), [N, H, W, 1]).div(255) as tf.Tensor4D;
          ys.push(depthY);
          break;
        case 'normal':
          const normY = tf.tensor4d(Float32Array.from(normal), [N, H, W, 3]).div(127.5).sub(1) as tf.Tensor4D;
          ys.push(normY);
          break;
      }
    }
    return { xs, ys };
  });
}
