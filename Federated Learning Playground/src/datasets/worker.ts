// worker.ts
import { generateSceneMulti } from './shapes2d';
import { Scene, ShapeConfig } from './constants';

export type Req = {
  H: number;
  W: number;
  nShapes: number;
  typeShapes: ShapeConfig[];
  batch: number;
  baseSeed: number;
  indices: number[]; // length == batch
};

/*------------------ Main Thread ---------------------*/
let _reqId = 0;

export function requestBatchFromWorker(worker: Worker, req: Req): Promise<Scene> {
  const id = _reqId++;

  return new Promise(resolve => {
    const handler = (ev: MessageEvent<any>) => {
      if (!ev.data || ev.data.id !== id) return;
      worker.removeEventListener('message', handler);
      resolve(ev.data.scene);
    };

    worker.addEventListener('message', handler);
    worker.postMessage({ id, req });
  });
}

/*------------------ Worker Thread ---------------------*/
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

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (ev: MessageEvent<{ id: number; req: Req }>) => {
  const { id, req } = ev.data ?? {};
  if (!req || !isReq(req)) return;

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

  // Flatten batch into big typed arrays for zero-copy transfer
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

  (self as any).postMessage(
    { id, scene: { H, W, K, N, rgb, seg, edge, sal, depth, normal } },
    [rgb.buffer, seg.buffer, edge.buffer, sal.buffer, depth.buffer, normal.buffer]
  );
};