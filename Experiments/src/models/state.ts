import { TaskName, ModelConfig } from './model.js';
import { Loss } from './losses.js';
import { Metric } from './metrics.js';
import { DataConfig } from '../datasets/dataloader.js';

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";

/** A map between names and loss functions. */
export let losses: { [K in TaskName]: { [label: string]: Loss } } = {
  "semseg": {
    "Categorical cross entropy": "categoricalCrossEntropy",
  },
  "edge": {
    "Binary cross entropy": "binaryCrossEntropy",
  },
  "saliency": {
    "Binary cross entropy": "binaryCrossEntropy",
  },
  "depth": {
    "Mean squared error": "meanSquaredError",
    "Absolute difference": "absoluteDifference",
    "Huber": "huberLoss",
  },
  "normal": {
    "Mean squared error": "meanSquaredError",
    "Absolute difference": "absoluteDifference",
    "Huber": "huberLoss",
  },
};

export let metrics: { [K in TaskName]: { [label: string]: Metric } } = {
  "semseg": {
    "Mean intersection over union": "meanIoU",
    "Pixel accuracy": "pixelAccuracy",
    "Mean class accuracy": "meanClassAccuracy",
  },
  "edge": {
    "Recall": "recall",
    "Precision": "precision",
    "F1 Score": "f1",
    "ODS F1 Score": "odsF",
  },
  "saliency": {
    "Recall": "recall",
    "Precision": "precision",
    "F1 Score": "f1",
  },
  "depth": {
    "Root mean squared error": "rmse",
    "Absolute relative difference": "absRel",
    "Root mean squared log error": "rmseLog",
    "Scale invariant log error": "siLog",
    "Delta 1": "delta1",
    "Delta 2": "delta2",
    "Delta 3": "delta3",
  },
  "normal": {
    "Mean angular error": "meanAngularErr",
    "Percentage of pixels with angular error < 11.25°": "pct_11_25",
    "Percentage of pixels with angular error < 22.5°": "pct_22_5",
    "Percentage of pixels with angular error < 30°": "pct_30",
  },
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substring(s.length - suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// Add the GUI state.
export class State {
  epochs = 0;

  // configurable 
  data: DataConfig = {
    seed: 0,
    numSamples: 100,
    numShapes: 1,
    typeShapes: [
      { type: 'circle', probability: 0.25 },
      { type: 'square', probability: 0.25 },
      { type: 'triangle', probability: 0.25 },
      { type: 'star', probability: 0.25 },
    ],
    resolution: 32,
    batchSize: 32,
    percTrain: 80,
  };

  model: ModelConfig = {
    name: "mtl_unet_configurable",
    activation: "relu", // configurable
    learningRate: 0.001, // configurable
    regularization: "l2", // configurable
    regularizationRate: 0.0001, // configurable
    encoder: {
      name: "encoder",
      layers: 4,
      baseFilters: 32,
      growthFactor: 2.0,
      convsPerStage: 2,
      convBlock: {
        type: "separableConv2d",
        kernelSize: 3,
        strides: 1,
        padding: "same",
        depthwiseInitializer: "heNormal",
        pointwiseInitializer: "heNormal",
        useBias: false
      },
      poolBlock: {
        type: "max",
        poolSize: [2, 2],
      }
    },
    decoder: {
      layers: 3,   
      startScaleFromTop: 0.5,
      growthFactor: 0.5,
      convsPerStage: 2,
      convBlock: {
        type: "separableConv2d",
        kernelSize: 3,
        strides: 1,
        padding: "same",
        depthwiseInitializer: "heNormal",
        pointwiseInitializer: "heNormal",
        useBias: false
      },
    },
    head: {
      type: "separableConv2d",
      kernelSize: 1,
      padding: "same",
      useBias: true,
      depthwiseInitializer: "heNormal",
      pointwiseInitializer: "heNormal",
    },

    // configurable
    taskConfig: [
      { name: 'semseg', enabled: true, loss: 'categoricalCrossEntropy', metric: 'meanIoU', lossWeight: 1, filters: 5 }, 
      { name: 'edge', enabled: true, loss: 'binaryCrossEntropy', metric: 'odsF', lossWeight: 1, filters: 1 }, 
      { name: 'saliency', enabled: true, loss: 'binaryCrossEntropy', metric: 'f1', lossWeight: 1, filters: 1 }, 
      { name: 'depth', enabled: true, loss: 'meanSquaredError', metric: 'rmse', lossWeight: 1, filters: 1 }, 
      { name: 'normal', enabled: true, loss: 'meanSquaredError', metric: 'meanAngularErr', lossWeight: 1, filters: 3 }
    ]
  }
}