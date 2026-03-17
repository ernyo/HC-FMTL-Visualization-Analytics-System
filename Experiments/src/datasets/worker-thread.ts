import { parentPort } from "node:worker_threads";
import { generateSceneMulti } from "./shapes2d.js";
import type { Scene, ShapeConfig } from "./constants.js";

export type Req = {
  H: number;
  W: number;
  nShapes: number;
  typeShapes: ShapeConfig[];
  batch: number;
  baseSeed: number;
  indices: number[];
};

function isReq(data: any): data is Req {
  return (
    data &&
    typeof data.H === "number" &&
    typeof data.W === "number" &&
    typeof data.nShapes === "number" &&
    typeof data.batch === "number" &&
    typeof data.baseSeed === "number" &&
    Array.isArray(data.indices)
  );
}

if (!parentPort) {
  throw new Error("worker-thread: parentPort is null");
}

parentPort.on("message", (msg: any) => {
  const { id, req } = msg ?? {};
  if (typeof id !== "number" || !req || !isReq(req)) return;

  const { H, W, nShapes, typeShapes, batch, baseSeed, indices } = req;

  const scenes: Scene[] = new Array(batch);
  for (let i = 0; i < batch; i++) {
    const sampleIdx = indices[i] ?? 0;
    scenes[i] = generateSceneMulti({
      H,
      W,
      nShapes,
      typeShapes,
      seed: baseSeed + sampleIdx,
    });
  }

  // Flatten batch into big typed arrays
  const N = batch;
  const C = 3;
  const K = scenes[0].K;
  const HxW = H * W;

  const rgb = new Uint8ClampedArray(N * HxW * C);
  const seg = new Uint8ClampedArray(N * HxW);
  const edge = new Uint8ClampedArray(N * HxW);
  const sal = new Uint8ClampedArray(N * HxW);
  const depth = new Uint8ClampedArray(N * HxW);
  const normal = new Uint8ClampedArray(N * HxW * C);

  for (let n = 0; n < N; n++) {
    const s = scenes[n];
    rgb.set(s.rgb, n * HxW * C);
    seg.set(s.seg, n * HxW);
    edge.set(s.edge, n * HxW);
    sal.set(s.sal, n * HxW);
    depth.set(s.depth, n * HxW);
    normal.set(s.normal, n * HxW * C);
  }

  // Transfer buffers (zero-copy)
  parentPort!.postMessage(
    { id, scene: { H, W, K, N, rgb, seg, edge, sal, depth, normal } },
    [rgb.buffer, seg.buffer, edge.buffer, sal.buffer, depth.buffer, normal.buffer]
  );
});