// client.ts
import { MTLModel, TaskName } from "../models/model.js";
import { State } from "../models/state.js";
import { Dataloader } from "../datasets/dataloader.js";

/**
 * Minimal metrics payload for UI.
 */
export interface ClientMetrics {
  id: string;
  epoch: number;
  lossTrain: number;
  lossTest: number;
  lossTestGlobal: number;
  perHead: Array<{
    task: TaskName;
    trainLoss: number;
    trainMetric: number;
    testLoss: number;
    testMetric: number;
    testLossGlobal: number;
    testMetricGlobal: number;
  }>;
}

/**
 * UI-driven client:
 * - owns its own State/Dataloader/MTLModel
 * - supports cooperative cancellation (stopTraining)
 * - exposes lastMetrics for UI charts
 */
export class FederatedClient {
  readonly state: State;
  readonly dataloader: Dataloader;
  readonly model: MTLModel;

  /** Optional stable id/name for UI lists. */
  readonly id: string = crypto.randomUUID();

  /** Enabled tasks (drives per-task aggregation blocks). */
  get tasks(): TaskName[] {
    // MTLModel builds enabled heads/decoders from config.tasks
    return this.state.model.taskConfig.filter(t => t.enabled).map(t => t.name);
  }

  /** Used by the server for grouping / homo aggregation (like your python dataname). */
  get dataname(): string {
    // adapt to your State shape if needed
    return (this.state as any).datasetName ?? (this.state as any).scene ?? "default";
  }

  get snapshotConfig() {
    const clientConfig = {
      id: this.id,
      data: this.state.data,
      model: {
        activation: this.state.model.activation,
        regularization: this.state.model.regularization,
        regularizationRate: this.state.model.regularizationRate,
        learningRate: this.state.model.learningRate,
        taskConfig: this.state.model.taskConfig
      },
    }
    return clientConfig
  }

  /** Prevent concurrent training from UI. */
  isTraining = false;

  /** Cooperative cancellation flag. */
  private cancelRequested = false;

  /** Latest metrics for UI / server hooks. */
  lastMetrics: ClientMetrics;

  constructor(state?: State) {
    this.state = state || new State();
    this.dataloader = new Dataloader(this.state.data, this.tasks);
    this.model = new MTLModel(this.state.model, [this.state.data.resolution, this.state.data.resolution, 3]);
  }

  /** UI can request cancellation; takes effect between epochs. */
  stopTraining() {
    this.cancelRequested = true;
  }

  /** Convenience: rebuild model when UI changes config/state. */
  async reset() {
    this.state.epochs = 0;
    await this.dataloader.reset(this.tasks, this.state.data);
    this.model.rebuild( 
      [this.state.data.resolution, this.state.data.resolution, 3],
      this.state.model.activation,
      this.state.model.regularization,
      this.state.model.regularizationRate,
      this.state.model.learningRate,
      this.state.model.taskConfig
    );
    await this.initialEvaluation();
  }

  /**
   * Train for `epochs` epochs (or until stopped).
   * Returns last epoch metrics.
   */
  async update(epochs: number) {
    if (this.isTraining) return;
    this.isTraining = true;
    this.cancelRequested = false;

    try {
      for (let i = 0; i < epochs; i++) {
        if (this.cancelRequested) break;

        const batchesPerEpoch = Math.ceil(this.state.data.numSamples / this.state.data.batchSize);
        
        // Train one epoch (your MTLModel.train already returns total loss for that epoch)
        const { totalTrainLoss, trainLosses, trainMetrics } = await this.model.train(
          this.dataloader.trainSet,
          this.state.epochs,
          batchesPerEpoch
        );
        // increment global epoch counter (matches your python usage)
        this.state.epochs++;
        
        // Evaluate (your evaluate returns total loss and updates per-head stats internally)
        const { totalTestLoss, testLosses, testMetrics } = this.model.evaluate(
          this.dataloader.testXs,
          this.dataloader.testYs,
          this.state.data.batchSize
        );

        const { totalTestLoss: totalTestGlobalLoss, testLosses: testGlobalLosses, testMetrics: testGlobalMetrics } = this.model.evaluate(
          this.dataloader.testGlobalXs,
          this.dataloader.testGlobalYs,
          this.state.data.batchSize
        );
        
        this.model.updateStats(trainLosses, trainMetrics, testLosses, testMetrics, testGlobalLosses, testGlobalMetrics);

        this.updateLastMetrics(totalTrainLoss, totalTestLoss, totalTestGlobalLoss);
      }
    } finally {
      this.isTraining = false;
    }
  }

  async initialEvaluation() {
    await this.dataloader.ready;
    const { totalTestLoss, testLosses, testMetrics} = this.model.evaluate(
      this.dataloader.testXs,
      this.dataloader.testYs,
      this.state.data.batchSize
    );
    const { totalTestLoss: totalTestGlobalLoss, testLosses: testGlobalLosses, testMetrics: testGlobalMetrics } = this.model.evaluate(
      this.dataloader.testGlobalXs,
      this.dataloader.testGlobalYs,
      this.state.data.batchSize
    );
    this.model.updateStats(undefined, undefined, testLosses, testMetrics, testGlobalLosses, testGlobalMetrics);
    this.updateLastMetrics(undefined, totalTestLoss, totalTestGlobalLoss);
  }

  private updateLastMetrics(totalTrainLoss: number | undefined, totalTestLoss: number, totalTestGlobalLoss: number) {
    const perHead = this.model.heads.map((h) => ({
      task: h.taskConfig.name,
      trainLoss: h.stats.loss[h.stats.loss.length - 1][0],
      trainMetric: h.stats.metric[h.stats.metric.length - 1][0],
      testLoss: h.stats.loss[h.stats.loss.length - 1][1],
      testMetric: h.stats.metric[h.stats.metric.length - 1][1],
      testLossGlobal: h.stats.globalLoss[h.stats.globalLoss.length - 1],
      testMetricGlobal: h.stats.globalMetric[h.stats.globalMetric.length - 1],
    }));

    this.lastMetrics = {
      id: this.id,
      epoch: this.state.epochs,
      lossTestGlobal: totalTestGlobalLoss,
      lossTrain: totalTrainLoss,
      lossTest: totalTestLoss,
      perHead,
    };
  }

  /**
   * Full cleanup (e.g., when removing client from UI).
   */
  dispose() {
    this.stopTraining();

    // model graph & weights
    this.model.dispose();
  }
}