export type Shape = 'circle' | 'square' | 'triangle' | 'star';

export const SHAPE_UNIVERSE = ['circle', 'square', 'triangle', 'star'] as const;

export interface ShapeConfig {
  type: Shape;
  probability: number; 
}

export interface SceneOpts {
  H: number; 
  W: number;
  nShapes?: number;                 // 1..4
  typeShapes?: ShapeConfig[];          // default all
  seed?: number;                    // deterministic RNG
  minScale?: number; 
  maxScale?: number; // fraction of min(H,W)
}

export interface Scene {
  H: number; 
  W: number; 
  K: number;
  N: number; // batch size
  rgb: Uint8ClampedArray;      // [H*W*3] 0..255
  seg: Uint8ClampedArray;      // [H*W] class ids: 0 bg, 1..K
  edge: Uint8ClampedArray;     // [H*W] 0..255
  sal: Uint8ClampedArray;      // [H*W] 0..255
  depth: Uint8ClampedArray;    // [H*W] 0..255 (normalized)
  normal: Uint8ClampedArray;   // [H*W*3] normals encoded to 0..255
}