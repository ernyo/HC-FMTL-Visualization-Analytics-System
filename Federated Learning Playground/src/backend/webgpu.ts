// webgpu.ts
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';  // registers 'webgpu'
import '@tensorflow/tfjs-backend-webgl';   // registers 'webgl'
import '@tensorflow/tfjs-backend-wasm';    // registers 'wasm'

export async function initBackend() {
  // Prefer WebGPU if available
  const canWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  
  if (canWebGPU) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      console.log('Using TFJS WebGPU backend');
      return;
    } catch (e) {
      console.warn('WebGPU backend failed, falling back to WebGL:', e);
    }
  }

  // Fallbacks
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('Using TFJS WebGL backend');
  } catch {
    await tf.setBackend('wasm');
    await tf.ready();
    console.log('Using TFJS WASM backend');
  }

  if (tf.getBackend() === 'webgl') {
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
    tf.env().set('WEBGL_PACK', true);
    tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
  }
}

