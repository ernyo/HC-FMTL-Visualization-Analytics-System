import { Worker } from "node:worker_threads";
import type { Scene, ShapeConfig } from "./constants.js";

export type Req = {
  H: number;
  W: number;
  nShapes: number;
  typeShapes: ShapeConfig[];
  batch: number;
  baseSeed: number;
  indices: number[]; // length == batch
};

let _reqId = 0;

export function requestBatchFromWorker(worker: Worker, req: Req): Promise<Scene> {
  const id = _reqId++;

  return new Promise((resolve, reject) => {
    const onMessage = (msg: any) => {
      if (!msg || msg.id !== id) return;
      cleanup();
      resolve(msg.scene as Scene);
    };

    const onError = (err: any) => {
      cleanup();
      reject(err);
    };

    const cleanup = () => {
      worker.off("message", onMessage);
      worker.off("error", onError);
      worker.off("exit", onExit);
    };

    const onExit = (code: number) => {
      cleanup();
      reject(new Error(`Worker exited before responding (code=${code})`));
    };

    worker.on("message", onMessage);
    worker.on("error", onError);
    worker.on("exit", onExit);

    worker.postMessage({ id, req });
  });
}