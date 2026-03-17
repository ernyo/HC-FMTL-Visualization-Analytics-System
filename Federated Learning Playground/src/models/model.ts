// model.ts
import * as tf from '@tensorflow/tfjs';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';
import { getLossArray, Loss } from './losses';
import { getMetricsObject, Metric } from './metrics';
import { taskTensorToImage } from '../utils/tensors';
import { NUM_TO_SHOW } from '../main';

export type Activations =
  | 'relu' | 'elu' | 'selu' | 'tanh' | 'sigmoid'
  | 'softplus' | 'softsign' | 'linear' | 'hardSigmoid' | 'softmax';

export type Regularizations =
  | 'l1' | 'l2' | 'l1l2'
  | 'none';

function makeRegularizer(
  reg: Regularizations,
  rate: number
): Regularizer {
  if (reg === 'l1l2') return tf.regularizers.l1l2({ l1: rate, l2: rate });
  if (reg === 'l1') return tf.regularizers.l1({ l1: rate });
  if (reg === 'l2') return tf.regularizers.l2({ l2: rate });
  return undefined;
}


/* --------------------------- Config definitions ------------------------ */

export type FlexShape = [number | null, number | null, 3]; // dynamic H,W

export interface EncoderConfig {
  readonly name: string;
  /** Number of encoder stages, e.g. 4 (x1..x4). */
  layers: number;
  /** Filters at stage 0; later stages scale by growthFactor^i. */
  baseFilters: number;
  /** Per-stage growth factor, e.g. 2: 32→64→128→256. */
  growthFactor: number;
  /** Number of Conv blocks per stage, e.g. 2 for Conv(Conv(...)). */
  convsPerStage: number;

  convBlock: {
    type: 'separableConv2d' | 'conv2d';
    kernelSize: number;
    strides: number;
    padding: 'same' | 'valid';
    useBias: boolean;
    depthwiseInitializer: string;
    pointwiseInitializer: string;
  };

  poolBlock: {
    type: 'max' | 'avg';
    poolSize: number | [number, number];
  };
}

export interface DecoderConfig {
  /** Number of decoder stages; typically encoder.layers - 1. */
  layers: number;
  /**
   * Scale of the top encoder filters at the first decoder stage.
   * Example: top encoder = 256, startScaleFromTop = 0.5 → 128.
   */
  startScaleFromTop: number;
  /**
   * Per-stage growth factor in decoder.
   * Example: 0.5 for 128→64→32 taper.
   */
  growthFactor: number;
  /** Number of Conv blocks per decoder stage. */
  convsPerStage: number;

  convBlock: {
    type: 'separableConv2d' | 'conv2d';
    kernelSize: number;
    strides: number;
    padding: 'same' | 'valid';
    useBias: boolean;
    depthwiseInitializer: string;
    pointwiseInitializer: string;
  };  
}

export type TaskName = 'semseg' | 'edge' | 'saliency' | 'depth' | 'normal';

export interface TaskConfig {
  enabled: boolean;
  name: TaskName;
  filters: number;
  loss: Loss;
  lossWeight: number;
  metric: Metric;
}

export interface HeadConfig {
  type: string;
  kernelSize: number | [number, number];
  padding: 'same' | 'valid';
  useBias: boolean;
  depthwiseInitializer: string;
  pointwiseInitializer: string;
}

export interface ModelConfig {
  /** Default backbone activation, e.g. "relu". */
  name: string;
  activation: Activations;
  learningRate: number;
  regularization: Regularizations;
  regularizationRate: number;

  /** UNet backbone config (tapering pattern controlled here). */
  encoder: EncoderConfig;
  decoder: DecoderConfig;

  /** Multi-task prediction heads. */
  head: HeadConfig;
  taskConfig: TaskConfig[];
}

/* ----------------------------- tiny utils ------------------------------ */

function ConvBlock(
  name: string, 
  x: tf.SymbolicTensor,
  filters: number,
  block: EncoderConfig['convBlock'] | DecoderConfig['convBlock'],
  activation: Activations,
  regularizer: Regularizer,
): tf.SymbolicTensor {

  const common = {
    name: name,
    filters: filters,
    kernelSize: block.kernelSize,
    strides: block.strides,
    padding: block.padding,
    useBias: block.useBias,
  };

  let convOut: tf.SymbolicTensor;
  if (block.type === 'separableConv2d') {
    convOut = tf.layers.separableConv2d({
      ...common,
      depthwiseInitializer: block.depthwiseInitializer,
      pointwiseInitializer: block.pointwiseInitializer,
      depthwiseRegularizer: regularizer,
      pointwiseRegularizer: regularizer,
    }).apply(x) as tf.SymbolicTensor;
  } else if (block.type === 'conv2d') {
    convOut = tf.layers.conv2d({
      ...common,
      kernelInitializer: block.pointwiseInitializer,
      kernelRegularizer: regularizer,
    }).apply(x) as tf.SymbolicTensor;
  } else {
    throw new Error(`Unsupported conv block type: ${block.type}. Choose 'separableConv2d' or 'conv2d'.`);
  }

  return tf.layers.activation({ activation }).apply(convOut) as tf.SymbolicTensor;
}

function PoolBlock(
  name: string, 
  x: tf.SymbolicTensor,
  block: EncoderConfig['poolBlock'],
) {
  if (block.type === 'avg') {
    return tf.layers.averagePooling2d({ name: name, poolSize: block.poolSize, strides: block.poolSize }).apply(x) as tf.SymbolicTensor;
  } else if (block.type === 'max') {
    return tf.layers.maxPooling2d({ name: name, poolSize: block.poolSize, strides: block.poolSize }).apply(x) as tf.SymbolicTensor;
  } else {
    throw new Error(`Unsupported pool block type: ${block.type}. Choose 'avg' or 'max'.`);
  }
}

function up2(name: string, x: tf.SymbolicTensor) {
  return tf.layers.upSampling2d({ name: name, size: [2, 2] }).apply(x) as tf.SymbolicTensor;
}

/* -------------------------------- Encoder ------------------------------ */

export class Encoder {
  readonly name: string;
  private activation: Activations;
  private regularizer: Regularizer;

  constructor(
    private readonly config: EncoderConfig,
    activation: Activations,
    regularizer: Regularizer
  ) {
    this.name = config.name;
    this.activation = activation;
    this.regularizer = regularizer;
  }

  setActivation(a: Activations) {
    this.activation = a;
  }

  setRegularizer(r: Regularizer) {
    this.regularizer = r;
  }

  /**
   * Returns hierarchical features for UNet-style decoding.
   * Output: array of feature maps [stage0, stage1, ..., stageN].
   */
  build(input: tf.SymbolicTensor): tf.SymbolicTensor[] {
    const { layers, baseFilters, growthFactor, convsPerStage } = this.config;
    const features: tf.SymbolicTensor[] = [];
    let x: tf.SymbolicTensor = input;

    for (let i = 0; i < layers; i++) {
      const filters = baseFilters * Math.pow(growthFactor, i);

      for (let c = 0; c < convsPerStage; c++) {
        const name = `encoder_stage${i}_conv${c}`;
        x = ConvBlock(name, x, filters, this.config.convBlock, this.activation, this.regularizer);
      }

      features.push(x);

      // Don't pool after the last stage.
      if (i < layers - 1) {
        x = PoolBlock(`encoder_stage${i}_pool`, x, this.config.poolBlock);
      }
    }

    return features;
  }
}

/* -------------------------------- Decoder ------------------------------ */

export class Decoder {
  readonly name: string;
  private readonly task: TaskName;
  readonly link: Link;
  private activation: Activations;
  private regularizer: Regularizer;

  constructor(
    task: TaskName,
    private readonly decoderConfig: DecoderConfig,
    private readonly encoderConfig: EncoderConfig,
    activation: Activations,
    regularizer: Regularizer,
  ) {
    this.name = `${task}_decoder`;
    this.activation = activation;
    this.regularizer = regularizer;
    this.link = new Link('encoder', this.name);
    this.task = task;
  }

  setActivation(a: Activations) {
    this.activation = a;
  }

  setRegularizer(r: Regularizer) {
    this.regularizer = r;
  }

  /**
   * UNet skip-concat decoder returning a shared full-res feature map.
   * Accepts encoder features from lowest to highest resolution:
   * [x1, x2, ..., xN], where xN is bottleneck.
   */
  build(encFeatures: tf.SymbolicTensor[]): tf.SymbolicTensor {
    if (encFeatures.length < 2) {
      throw new Error('Decoder expects at least 2 encoder feature maps.');
    }

    const { layers, startScaleFromTop, growthFactor, convsPerStage } = this.decoderConfig;
    const { baseFilters, growthFactor: encGrowth, layers: encStages } = this.encoderConfig;

    const topIdx = encFeatures.length - 1;
    const topEncoderFilters = baseFilters * Math.pow(encGrowth, encStages - 1);

    // Start at bottleneck upsampled
    let y = up2(`${this.task}_decoder_up0`, encFeatures[topIdx]);

    // Initial decoder filters derived from top encoder features
    let filters = topEncoderFilters * startScaleFromTop;

    for (let d = 0; d < layers; d++) {
      const skipIdx = topIdx - 1 - d;
      if (skipIdx < 0) {
        throw new Error(
          `Decoder has more stages (${layers}) than available encoder skip connections (${encFeatures.length - 1}).`
        );
      }

      // Skip connection
      y = tf.layers.concatenate({ name: `${this.task}_decoder_concat${d}` }).apply([y, encFeatures[skipIdx]]) as tf.SymbolicTensor;

      // Conv blocks at this decoder stage
      for (let c = 0; c < convsPerStage; c++) {
        const name = `${this.task}_decoder_stage${d}_conv${c}`;
        y = ConvBlock(name, y, filters, this.decoderConfig.convBlock, this.activation, this.regularizer);
      }

      // Upsample for next stage (except after last decoder stage)
      if (d < layers - 1) {
        y = up2(`${this.task}_decoder_up${d + 1}`, y);
        filters *= growthFactor;
      }
    }

    return y;
  }
}

/* --------------------------------- Heads ------------------------------- */
export class Link {
  constructor(
    readonly source: string,
    readonly dest: string,
    public weight: number = 0, // initial weight, will be refreshed later
    public bias: number = 0,
  ) {}
}

export class Stats {
  constructor(
    public loss: number[][] = [],
    public metric: number[][] = [],
    public predictions: ImageData[] = [],
    public gts: ImageData[] = [],
  ) {}
}

// One logical task head (semseg, edge, saliency, depth, normal)
export class Head {
  readonly name: string;
  readonly link: Link;
  readonly stats: Stats = new Stats();
  readonly headConfig: HeadConfig;
  readonly taskConfig: TaskConfig;
  private regularizer: Regularizer;

  constructor(
    headConfig: HeadConfig,
    taskConfig: TaskConfig,
    regularizer: Regularizer,
  ) {
    this.name = `${taskConfig.name}_head`;
    this.headConfig = headConfig;
    this.taskConfig = taskConfig;
    this.regularizer = regularizer;
    this.link = new Link(`${taskConfig.name}_decoder`, this.name);
  }

  setRegularizer(r: Regularizer) {
    this.regularizer = r;
  }

  /** Build this head's output tensor from the shared decoder features. */
  build(shared: tf.SymbolicTensor): tf.SymbolicTensor {
    return tf.layers.separableConv2d({
      name: this.name,
      filters: this.taskConfig.filters,
      kernelSize: this.headConfig.kernelSize,
      padding: this.headConfig.padding,
      useBias: this.headConfig.useBias,
      depthwiseInitializer: this.headConfig.depthwiseInitializer,
      pointwiseInitializer: this.headConfig.pointwiseInitializer,
      depthwiseRegularizer: this.regularizer,
      pointwiseRegularizer: this.regularizer,
    }).apply(shared) as tf.SymbolicTensor;
  }
}

/* ------------------------------- Top-level ----------------------------- */

export class MTLModel {
  readonly name: string;
  private activation: Activations;
  private learningRate: number;
  private regularizer: Regularizer;
  private optimizer?: tf.Optimizer;

  readonly encoder: Encoder;
  public decoders: Decoder[] = [];
  public heads: Head[] = [];

  readonly config: ModelConfig;

  model!: tf.LayersModel;
  private inputShape: FlexShape;

  /**
   * @param config     ModelConfig loaded from JSON/YAML or constructed in code.
   */
  constructor(
    config: ModelConfig,
    inputShape: FlexShape
  ) {
    this.config = config;
    this.name = config.name;
    this.inputShape = inputShape;

    this.activation = config.activation;
    this.regularizer = makeRegularizer(config.regularization, config.regularizationRate);
    this.learningRate = config.learningRate;

    // Architecture
    this.encoder = new Encoder(this.config.encoder, this.activation, this.regularizer);
    this.buildDecodersHeads();
    this.build();
    this.compile();
  }

  /* ---------------------------- SETTERS ------------------------------ */
  setActivation(activation: Activations) {
    if (activation === this.activation) return;
    this.activation = activation;
    this.encoder.setActivation(activation);
    this.decoders.forEach(decoder => decoder.setActivation(activation));
  }

  setRegularizer(regularization: Regularizations, rate: number) {
    const reg = makeRegularizer(regularization, rate);
    if (reg === this.regularizer) return;
    this.regularizer = reg;
    // propagate to modules
    this.encoder.setRegularizer(reg);
    this.decoders.forEach(decoder => decoder.setRegularizer(reg));
    this.heads.forEach(h => h.setRegularizer(reg));
  }

  setLearningRate(lr: number) {
    if (lr === this.learningRate) return;
    this.learningRate = lr;
    this.compile();
  }

  setTaskConfig(newConfig: TaskConfig[]) {
    this.config.taskConfig = newConfig;
  }


  /* ------------------------ METADATA ------------------------ */
  private refreshLinks() {
    this.heads.forEach(head => {
      const layer = this.model.getLayer(head.name);
      const weights = layer.getWeights();

      if (weights.length > 0) {
        const kernel = weights[0];                    // depthwise (or first kernel)
        const bias   = weights[weights.length - 1];   // last weight tensor → bias
        head.link.weight = kernel.dataSync()[0];
        head.link.bias = bias.dataSync()[0];
      }
    });
  }

  updateStats(
    trainLosses?: Array<number>,
    trainMetrics?: Array<number>,
    testLosses?: Array<number>,
    testMetrics?: Array<number>,
    predictions?: Array<tf.Tensor>,
    gts?: Array<tf.Tensor>,
  ) {
    this.heads.forEach((head, i) => {
      // initial evaluation
      if (trainLosses === undefined && testLosses !== undefined && trainMetrics === undefined && testMetrics !== undefined) {
        head.stats.loss.push([undefined, testLosses[i]]);
        head.stats.metric.push([undefined, testMetrics[i]]);
      } 

      // losses
      if (trainLosses && trainLosses[i] !== undefined && testLosses && testLosses[i] !== undefined) {
        if (head.stats.loss[0][0] === undefined) head.stats.loss = []; // reset if first entry was undefined
        head.stats.loss.push([trainLosses[i], testLosses[i]]);
      }

      // metrics
      if (trainMetrics && trainMetrics[i] !== undefined && testMetrics && testMetrics[i] !== undefined) {
        if (head.stats.metric[0][0] === undefined) head.stats.metric = []; // reset if first entry was undefined
        head.stats.metric.push([trainMetrics[i], testMetrics[i]]);
      }

      // Predictions (tensors)
      if (predictions && predictions[i] !== undefined) {
        let predImages = [];
        for (let j = 0; j < NUM_TO_SHOW; j++) {
          tf.engine().startScope();
          try {
            const pred = predictions[i]
              .slice([j, 0, 0, 0], [1, -1, -1, -1])
              .squeeze([0]) as tf.Tensor3D;

            predImages.push(taskTensorToImage(pred, head.taskConfig.name));
          } finally {
            tf.engine().endScope();
          }
        }
        head.stats.predictions = predImages;
        predictions[i].dispose(); // Clean up!!
      }
      if (gts && gts[i] !== undefined) {
        let gtImages = [];
        for (let j = 0; j < NUM_TO_SHOW; j++) {
          tf.engine().startScope();
          try {
            const gt = gts[i]
              .slice([j, 0, 0, 0], [1, -1, -1, -1])
              .squeeze([0]) as tf.Tensor3D;

            gtImages.push(taskTensorToImage(gt, head.taskConfig.name));
          } finally {
            tf.engine().endScope();
          }
        }
        head.stats.gts = gtImages;
      }
    });
  }


  /* ------------------------ TRAINING / PREDICTION ----------------------- */
  buildDecodersHeads() {
    this.decoders = [];
    this.heads = [];
    this.config.taskConfig.filter(taskConfig => taskConfig.enabled)     // ignore disabled ones
      .forEach(taskConfig => {
        const decoder = new Decoder(
          taskConfig.name,
          this.config.decoder,
          this.config.encoder,
          this.activation,
          this.regularizer,
        );
        this.decoders.push(decoder);

        const head = new Head(
          this.config.head,
          taskConfig,
          this.regularizer,
        );
        this.heads.push(head);
      });
  }

  rebuild(
    inputShape: FlexShape,
    activation: Activations, 
    regularization: Regularizations, 
    regularizationRate: number, 
    learningRate: number, 
    taskConfig: TaskConfig[],
  ) {
    // clean up
    this.inputShape = inputShape;
    // build heads first
    this.setTaskConfig(taskConfig);
    this.buildDecodersHeads();

    // activation and regularizer update heads
    this.setActivation(activation);
    this.setRegularizer(regularization, regularizationRate);
    this.setLearningRate(learningRate);

    // recompile model
    this.dispose();
    this.build();
    this.compile();
  }

  private build() {
    const inp = tf.input({ shape: this.inputShape });
    const encFeatures = this.encoder.build(inp);
    const shared = this.decoders.map(decoder => decoder.build(encFeatures));
    const outs = this.heads.map((head, idx) => head.build(shared[idx]));

    this.model = tf.model({
      inputs: inp,
      outputs: outs,
      name: this.name,
    });
  }

  dispose() {
    this.model?.dispose();
    this.optimizer?.dispose?.();
  }

  compile() {
    this.optimizer?.dispose?.();
    this.optimizer = tf.train.adam(this.learningRate);
    this.model.compile({
      optimizer: this.optimizer,
      loss: getLossArray(this.heads),
      metrics: getMetricsObject(this.heads),
    });
  }

  async train(dataset: tf.data.Dataset<{ xs: tf.Tensor4D; ys: tf.Tensor[] }>, currentEpoch: number, batchesPerEpoch: number) {
    tf.engine().startScope();
    const history = await this.model.fitDataset(dataset, {
      epochs: currentEpoch + 1,
      initialEpoch: currentEpoch,
      batchesPerEpoch,
      verbose: 0,
    });
    tf.engine().endScope();
    await tf.nextFrame(); // ensure UI updates before next training step

    this.refreshLinks();

    // ---- parse history.history ----
    const h = history.history;
    const epochIdx = h.loss.length - 1; // last epoch in this call

    // total loss:
    const totalTrainLoss = h.loss[epochIdx] as number;

    // per-head loss & metric:
    let trainLosses = [];
    let trainMetrics = [];
    if (this.heads.length === 1) {
      const trainLoss = h["loss"][epochIdx] as number;
      const trainMetric = h[""][epochIdx] as number;
      trainLosses.push(trainLoss); 
      trainMetrics.push(trainMetric);
    } else {
      this.heads.forEach(head => {
        const lossKey = `${head.name}_loss`;
        const metricEntry = Object.entries(h).find(([k]) => k.startsWith(`${head.name}_`) && !k.endsWith('_loss') && k !== 'loss');
        const trainLoss = h[lossKey][epochIdx];
        const trainMetric = metricEntry[1][epochIdx];

        trainLosses.push(trainLoss); 
        trainMetrics.push(trainMetric);
      });
    }

    return { totalTrainLoss, trainLosses, trainMetrics };
  }

  /** Evaluate; returns one tensor per head in `this.order`. */
  evaluate(xs: tf.Tensor4D, y: tf.Tensor[], batchSize: number) {
    // 1. Evaluate losses/metrics
    const rawOutputs = this.model.evaluate(xs, y, { batchSize });
    let outputs = Array.isArray(rawOutputs)
      ? (rawOutputs as tf.Scalar[])
      : [rawOutputs as tf.Scalar];
    const index = this.heads.length - 1;
    for (let i = 0; i < index; i++) outputs[i].dispose(); // prevent leaks
    outputs = outputs.slice(index, outputs.length); // outputs are in order of model.metricsNames, but tfjs duplicates (this.heads.length - 1) values at start for some reason

    let totalLossTensor: tf.Scalar;
    let testLossTensors: tf.Scalar[];
    let testMetricTensors: tf.Scalar[];

    if (this.heads.length === 1) {
      // outputs[0] = total loss (and only head's loss)
      // outputs[1] = only head's metric (if you compiled with one metric)
      totalLossTensor = outputs[0];
      testLossTensors = [outputs[0]];
      testMetricTensors = [outputs[1]];
    } else {
      // multi-head: assume layout:
      // 0: total loss
      // 1..H         : per-head losses
      // H+1..H+H     : per-head metrics
      totalLossTensor   = outputs[0];
      testLossTensors   = outputs.slice(1, 1 + this.heads.length);
      testMetricTensors = outputs.slice(1 + this.heads.length, 1 + 2 * this.heads.length);
    }

    const totalTestLoss = totalLossTensor.dataSync()[0];
    const testLosses    = testLossTensors.map(t => t.dataSync()[0]);
    const testMetrics   = testMetricTensors.map(t => t.dataSync()[0]);

    totalLossTensor.dispose();
    testLossTensors.forEach(t => t.dispose());
    testMetricTensors.forEach(t => t.dispose());

    // 2. Predict per head
    const xsSmall = xs.slice([0,0,0,0], [NUM_TO_SHOW, -1, -1, -1]);
    const rawPredictions = this.model.predict(xsSmall);
    const predictions = Array.isArray(rawPredictions) ? (rawPredictions as tf.Tensor[]) : [rawPredictions as tf.Tensor];
    xsSmall.dispose();

    return { totalTestLoss, testLosses, testMetrics, predictions, y };
  }
}