import * as tf from "@tensorflow/tfjs-node-gpu";
import { batchToTensors } from '../utils/tensors.js';
import { hash32, seededShuffle } from './shapes2d.js';
import { TaskName } from '../models/model.js';
import { ShapeConfig } from './constants.js';
import { Worker } from "node:worker_threads";
import { requestBatchFromWorker, Req } from "./worker-client.js";

export interface DataConfig {
  seed: number;
  numSamples: number;
  numShapes: number;
  typeShapes: ShapeConfig[];
  resolution: number;
  batchSize: number;
  percTrain: number;
}

export class Dataloader {
  private config: DataConfig;
  private tasks: TaskName[];

  trainSet: tf.data.Dataset<{ xs: tf.Tensor4D; ys: tf.Tensor[] }>;
  testXs: tf.Tensor4D;
  testYs: tf.Tensor[];

  testGlobalXs: tf.Tensor4D;
  testGlobalYs: tf.Tensor[];

  ready: Promise<void>;
  private worker: Worker;
  private trainGenFactory: () => AsyncGenerator<{ xs: tf.Tensor4D; ys: tf.Tensor[] }, void, unknown>;

  constructor(config: DataConfig, tasks: TaskName[]) {
    this.worker = new Worker(new URL("./worker-thread.js", import.meta.url), {
    type: "module",
  } as any);

    this.ready = this.reset(tasks, config);
  }

  private generateBatch(
    worker: Worker,
    baseReq: Omit<Req, "baseSeed" | "indices">,
    baseSeed: number,
    tasks: TaskName[]
  ) {
    const numTrain = Math.max(1, Math.floor(this.config.percTrain/100 * this.config.numSamples));
    const batchSize = baseReq.batch;

    // Pre-allocate index list [0..numTrain-1]
    const baseIndices = new Array<number>(numTrain);
    for (let i = 0; i < numTrain; i++) baseIndices[i] = i;

    return async function* () {
      let step = 0;
      let currentEpoch = -1;
      let perm: number[] = [];

      while (true) {
        const sampleOffset = (step * batchSize) % numTrain;
        const epoch = Math.floor((step * batchSize) / numTrain);

        if (epoch !== currentEpoch) {
          currentEpoch = epoch;
          perm = baseIndices.slice();
          const epochSeed = hash32(baseSeed, epoch, 0xE0C0);
          seededShuffle(perm, epochSeed);
        }

        const indices: number[] = [];
        for (let k = 0; k < batchSize; k++) {
          indices.push(perm[(sampleOffset + k) % numTrain]);
        }

        // Request this batch from the worker using explicit indices
        const req: Req = {
          ...baseReq,
          batch: batchSize,
          baseSeed,
          indices,
        };

        const batch = await requestBatchFromWorker(worker, req);
        const { xs, ys } = batchToTensors(batch, tasks);

        yield { xs, ys };
        step++;
      }
    };
  }

  private async generateTestData() {
    if (this.testXs) this.testXs.dispose();
    if (this.testYs) this.testYs.forEach(t => t.dispose());

    const numTrain = Math.max(1, Math.floor(this.config.percTrain/100 * this.config.numSamples));
    const numTest = Math.max(1, this.config.numSamples - numTrain);

    const testIndices = new Array<number>(numTest);
    for (let i = 0; i < numTest; i++) testIndices[i] = numTrain + i;

    const req: Req = {
      H: this.config.resolution,
      W: this.config.resolution,
      nShapes: this.config.numShapes,
      typeShapes: this.config.typeShapes,
      batch: numTest,
      baseSeed: this.config.seed,
      indices: testIndices,
    };

    const batch = await requestBatchFromWorker(this.worker, req);
    const { xs, ys } = batchToTensors(batch, this.tasks);

    this.testXs = xs;
    this.testYs = ys;
  }

  private async generateGlobalTestData() {
    // dispose old
    if (this.testGlobalXs) this.testGlobalXs.dispose();
    if (this.testGlobalYs) this.testGlobalYs.forEach(t => t.dispose());

    const numTest = 1000;
    const batchSize = this.config.batchSize;

    // fixed deterministic indices (don’t overlap with train)
    const testIndices = Array.from({ length: numTest }, (_, i) => 12345 + i);

    const baseReq: Omit<Req, "indices" | "baseSeed" | "batch"> = {
      H: 32,
      W: 32,
      nShapes: 1,
      typeShapes: [
        { type: "circle", probability: 0.25 },
        { type: "square", probability: 0.25 },
        { type: "triangle", probability: 0.25 },
        { type: "star", probability: 0.25 },
      ],
    };

    const xsChunks: tf.Tensor4D[] = [];
    const ysChunks: tf.Tensor[][] = []; // [chunkIdx][headIdx]

    for (let off = 0; off < numTest; off += batchSize) {
      const inds = testIndices.slice(off, off + batchSize);

      const req: Req = {
        ...baseReq,
        batch: inds.length,        // IMPORTANT: last chunk may be smaller
        baseSeed: 123456789,
        indices: inds,
      };

      const sceneBatch = await requestBatchFromWorker(this.worker, req);
      const { xs, ys } = batchToTensors(sceneBatch, this.tasks);

      xsChunks.push(xs);
      ysChunks.push(ys);
    }

    // concat xs along batch dimension
    this.testGlobalXs = tf.concat(xsChunks, 0) as tf.Tensor4D;

    // concat each head’s y along batch dimension
    const nHeads = ysChunks[0].length;
    this.testGlobalYs = Array.from({ length: nHeads }, (_, h) =>
      tf.concat(ysChunks.map(ch => ch[h]), 0)
    );

    // cleanup chunk tensors (we already concatenated)
    xsChunks.forEach(t => t.dispose());
    ysChunks.forEach(arr => arr.forEach(t => t.dispose()));
  }

  async reset(tasks: TaskName[], config: DataConfig) {
    this.tasks = tasks;
    this.config = config;
    this.trainGenFactory = this.generateBatch(
      this.worker,
      {
        H: this.config.resolution,
        W: this.config.resolution,
        nShapes: this.config.numShapes,
        typeShapes: this.config.typeShapes,
        batch: this.config.batchSize,
      },
      this.config.seed,
      this.tasks
    );

    this.trainSet = tf.data.generator(this.trainGenFactory).prefetch(1);
    await this.generateTestData();
    await this.generateGlobalTestData();
  }
}