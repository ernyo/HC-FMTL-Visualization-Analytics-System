import { TaskName, ModelConfig } from './model';
import { Loss } from './losses';
import { Metric } from './metrics';
import { DataConfig } from '../datasets/dataloader';

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

  // /**
  //  * Deserializes the state from the url hash.
  //  */
  // static deserializeState(): State {
  //   let map: {[key: string]: string} = {};
  //   for (let keyvalue of window.location.hash.slice(1).split("&")) {
  //     let [name, value] = keyvalue.split("=");
  //     map[name] = value;
  //   }
  //   let state = new State();

  //   function hasKey(name: string): boolean {
  //     return name in map && map[name] != null && map[name].trim() !== "";
  //   }

  //   function parseArray(value: string): string[] {
  //     return value.trim() === "" ? [] : value.split(",");
  //   }

  //   // Deserialize regular properties.
  //   State.PROPS.forEach(({name, type, keyMap}) => {
  //     switch (type) {
  //       case Type.OBJECT:
  //         if (keyMap == null) {
  //           throw Error("A key-value map must be provided for state " +
  //               "variables of type Object");
  //         }
  //         if (hasKey(name) && map[name] in keyMap) {
  //           state[name] = keyMap[map[name]];
  //         }
  //         break;
  //       case Type.NUMBER:
  //         if (hasKey(name)) {
  //           // The + operator is for converting a string to a number.
  //           state[name] = +map[name];
  //         }
  //         break;
  //       case Type.STRING:
  //         if (hasKey(name)) {
  //           state[name] = map[name];
  //         }
  //         break;
  //       case Type.BOOLEAN:
  //         if (hasKey(name)) {
  //           state[name] = (map[name] === "false" ? false : true);
  //         }
  //         break;
  //       case Type.ARRAY_NUMBER:
  //         if (name in map) {
  //           state[name] = parseArray(map[name]).map(Number);
  //         }
  //         break;
  //       case Type.ARRAY_STRING:
  //         if (name in map) {
  //           state[name] = parseArray(map[name]);
  //         }
  //         break;
  //       default:
  //         throw Error("Encountered an unknown type for a state variable");
  //     }
  //   });

  //   if (hasKey("taskConfig")) {
  //     try {
  //       state.taskConfig = JSON.parse(decodeURIComponent(map["taskConfig"]));
  //     } catch (e) {
  //       console.warn("Failed to parse taskConfig from URL, using defaults", e);
  //     }
  //   }

  //   // Deserialize state properties that correspond to hiding UI controls.
  //   getHideProps(map).forEach(prop => {
  //     state[prop] = (map[prop] === "true") ? true : false;
  //   });
  //   if (state.seed == null) {
  //     state.seed = Math.random();
  //   }
  //   Math.seed(state.seed);
  //   return state;
  // }

  // /**
  //  * Serializes the state into the url hash.
  //  */
  // serialize() {
  //   // Serialize regular properties.
  //   let props: string[] = [];
  //   State.PROPS.forEach(({name, type, keyMap}) => {
  //     let value = this[name];
  //     // Don't serialize missing values.
  //     if (value == null) {
  //       return;
  //     }
  //     if (name === "taskConfig") {
  //       value = encodeURIComponent(JSON.stringify(this.taskConfig));
  //     } else if (type === Type.OBJECT) {
  //       value = getKeyFromValue(keyMap, value);
  //     } else if (type === Type.ARRAY_NUMBER ||
  //         type === Type.ARRAY_STRING) {
  //       value = value.join(",");
  //     }
  //     props.push(`${name}=${value}`);
  //   });
  //   // Serialize properties that correspond to hiding UI controls.
  //   getHideProps(this).forEach(prop => {
  //     props.push(`${prop}=${this[prop]}`);
  //   });
  //   window.location.hash = props.join("&");
  // }

  // /** Returns all the hidden properties. */
  // getHiddenProps(): string[] {
  //   let result: string[] = [];
  //   for (let prop in this) {
  //     if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
  //       result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
  //     }
  //   }
  //   return result;
  // }

  // setHideProperty(name: string, hidden: boolean) {
  //   this[name + HIDE_STATE_SUFFIX] = hidden;
  // }
}