import * as tf from '@tensorflow/tfjs';
import { Req, requestBatchFromWorker } from './worker';
import { batchToTensors } from '../utils/tensors';
import { hash32, seededShuffle } from './shapes2d';
import { TaskName } from '../models/model';
import { ShapeConfig } from './constants';

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
  gtImages: ImageData[]; // ground truth images for test set
  private worker: Worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  private trainGenFactory: () => AsyncGenerator<{ xs: tf.Tensor4D; ys: tf.Tensor[] }, void, unknown>;
  ready: Promise<void>;

  constructor(config: DataConfig, tasks: TaskName[]) {
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
  }
}