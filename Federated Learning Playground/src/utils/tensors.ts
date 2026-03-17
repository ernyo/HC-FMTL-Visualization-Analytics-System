import * as tf from '@tensorflow/tfjs';
import { Scene } from '../datasets/constants';
import { TaskName } from 'src/models/model';

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

/**
 * Convert a 3D tensor [H, W, C] to ImageData.
 *
 * mode:
 *  - 'semseg': treat channels as class scores, argmax → color label map.
 *  - 'normal': treat channels as surface normals in [-1,1] and map to RGB.
 *  - 'auto'  : 1-chan → grayscale (min–max), 3-chan → RGB-ish (min–max).
 */
function tensorToImage(
  t: tf.Tensor3D,
  mode: 'auto' | 'normal' | 'semseg' = 'auto'
): ImageData {
  const [H, W, C] = t.shape;

  // --- SEMSEG BRANCH ------------------------------------------------------
  if (mode === 'semseg') {
    // t: [H, W, numClasses] (logits or probs or one-hot).
    // Reduce to [H, W] class IDs.
    const classIds = tf.tidy(() => {
      const classTensor = t.argMax(2);          // along channel axis
      return classTensor.dataSync() as Int32Array;
    });

    // Simple fixed palette (extend as needed)
    const palette: Array<[number, number, number]> = [
      [0, 0, 0],        // class 0: background
      [255, 0, 0],      // 1
      [0, 255, 0],      // 2
      [0, 0, 255],      // 3
      [255, 255, 0],    // 4
    ];

    const imgData = new ImageData(W, H);
    const out = imgData.data;

    for (let i = 0; i < H * W; i++) {
      const cls = classIds[i] ?? 0;
      const [r, g, b] = palette[cls % palette.length];
      const o = i * 4;
      out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
    }
    return imgData;
  }

  // --- NORMAL BRANCH ------------------------------------------------------
  if (mode === 'normal') {
    // Expect normals in [-1, 1] per channel, at least 3 channels.
    const data = t.dataSync();
    const imgData = new ImageData(W, H);
    const out = imgData.data;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const base = (y * W + x) * C;
        const n0 = data[base]     ?? 0; // x
        const n1 = data[base + 1] ?? 0; // y
        const n2 = data[base + 2] ?? 0; // z

        // Map [-1,1] → [0,1] → [0,255]
        const r = Math.round((n0 + 1) * 0.5 * 255);
        const g = Math.round((n1 + 1) * 0.5 * 255);
        const b = Math.round((n2 + 1) * 0.5 * 255);

        const o = (y * W + x) * 4;
        out[o]     = r;
        out[o + 1] = g;
        out[o + 2] = b;
        out[o + 3] = 255;
      }
    }

    return imgData;
  }

  // --- AUTO / GENERIC BRANCH ---------------------------------------------
  // Handle 1-channel (edge/depth/saliency) and generic 3-channel tensors.
  const data = t.dataSync();

  // Per-image min–max so even tiny ranges show contrast.
  let min = +Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  let range = max - min;
  if (range === 0) range = 1;

  const imgData = new ImageData(W, H);
  const out = imgData.data;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const base = (y * W + x) * C;

      let r: number, g: number, b: number;

      if (C === 1) {
        // Grayscale
        const v = (data[base] - min) / range;
        const v255 = Math.round(v * 255);
        r = g = b = v255;
      } else {
        // Take first 3 channels as RGB-ish, min–max normalized.
        const vr = (data[base]     - min) / range;
        const vg = (data[base + 1] - min) / range;
        const vb = (data[base + 2] - min) / range;
        r = Math.round(vr * 255);
        g = Math.round(vg * 255);
        b = Math.round(vb * 255);
      }

      const o = (y * W + x) * 4;
      out[o]     = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }

  return imgData;
}


export function taskTensorToImage(
  t: tf.Tensor3D,
  task: TaskName,
): ImageData {
  switch (task) {
    case 'normal':
      return tensorToImage(t, 'normal');
    case 'semseg':
      // if one-hot/logits: argmax → id map → colormap → ImageData
      return tensorToImage(t, 'semseg');
    case 'edge':
    case 'saliency':
      return tensorToImage(t.sigmoid() as tf.Tensor3D, 'auto');
    case 'depth':
      return tensorToImage(t, 'auto');  // grayscale min–max
  }
}