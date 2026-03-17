// main.ts
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';
import './utils/rng';
import { initBackend } from './backend/webgpu';
import { Activations, Regularizations } from './models/model';
import { State } from './models/state';
import { Player, PlayerDriver } from './drivers/player';
import { ExperimentDriver, ExperimentRun } from './drivers/experiment';
import { FederatedClient } from './federated/client';
import { FederatedServer } from './federated/server';
import { EncoderAgg, DecoderAgg } from "./federated/aggregate";
import { ClientView } from './views/client-view';
import { ServerView } from "./views/server-view";
import { ConfigView } from "./views/config-view";
import { ExperimentView } from './views/experiment-view';

await initBackend();
let isTraining = false;
let userHasInteracted = false;

await tf.ready();

// ---------- Globals / app state ----------
const clients: FederatedClient[] = [];
const clientViews = new Map<string, ClientView>();
export const NUM_TO_SHOW = 3; // number of heatmaps
let experimentMode = false;

// global config
let activation: Activations = "relu";
let lr: number = 0.001;
let regularization: Regularizations = "l2";//"none"; //
let regularizationRate: number = 0.0001;//0; //

// ---------- Clients ----------
const CLIENT_LIMIT = 10;
let numClients = 0;
for (let i = 0; i < 3; i++) {
  await addClient();
}
let activeClient: FederatedClient = clients[0];
let activeView: ClientView = clientViews.get(activeClient.id);

// ---------- Server ----------
const server = new FederatedServer();

async function addClient() {
  if (numClients >= CLIENT_LIMIT) {
    return;
  }
  numClients++;
  // global config
  let state = new State();
  state.model.activation = activation;
  state.model.learningRate = lr;
  state.model.regularization = regularization;
  state.model.regularizationRate = regularizationRate;
  const c = new FederatedClient(state);

  await c.initialEvaluation();
  clients.push(c);

  const v = new ClientView(c, reset);
  clientViews.set(c.id, v);
}

function removeClient(client?: FederatedClient) {
  if (numClients <= 1) {
    return;
  }
  numClients--;

  if (!client) client = clients[clients.length - 1];
  const idx = clients.findIndex(c => c.id === client!.id);
  if (idx < 0) return;
  clients.splice(idx, 1);

  clientViews.get(client.id).hide();
  clientViews.delete(client.id);

  // if active removed, switch to another
  if (activeClient.id === client.id) {
    showClient(clients[0]);
  }

  updateUI();
}

// ---------- Switching clientViews ----------
function showClient(client: FederatedClient) {
  if (activeView) activeView.hide();
  activeClient = client;
  activeView = clientViews.get(client.id);
  activeView.show();
  syncClientControls(client);
  bindActiveClientGUI();
}

function syncClientControls(client: FederatedClient) {
  activeView.select("#resolution").property("value", Math.round(Math.log2(client.state.data.resolution / 32)));
  activeView.select(`label[for='resolution'] .value`).text(String(client.state.data.resolution));

  activeView.select("#nShapes").property("value", client.state.data.numShapes);
  activeView.select(`label[for='nShapes'] .value`).text(String(client.state.data.numShapes));

  activeView.select("#percTrainData").property("value", client.state.data.percTrain);
  activeView.select(`label[for='percTrainData'] .value`).text(String(client.state.data.percTrain));
  activeView.select("#batchSize").property("value", Math.round(Math.log2(client.state.data.batchSize / 8)));
  activeView.select(`label[for='batchSize'] .value`).text(String(client.state.data.batchSize));

  activeView.select("#activations").property("value", client.state.model.activation);
  activeView.select("#learningRate").property("value", client.state.model.learningRate);
  activeView.select("#regularizations").property("value", client.state.model.regularization);
  activeView.select("#regularRate").property("value", client.state.model.regularizationRate);
}

// ---------- Server view ----------
const configView = new ConfigView(
  (c) => {showClient(c); configView.render(clients);},
  async (c) => {await removeClient(c); configView.render(clients); await reset();},
  () => activeClient,
  (c) => c.lastMetrics, 
);
const serverView = new ServerView();

// ---------- Player ----------
const driver = new PlayerDriver({oneStep})
let player = new Player(driver);

// ---------- Experiment mode ----------
const experimentRun = new ExperimentRun();
const experimentDriver = new ExperimentDriver({
  experimentRun,
  server,
  clients,
  oneStep,
  reset,
  updateCards: () => {experimentView.render();}
});
const experimentView = new ExperimentView(experimentRun);

function updateExperimentUI() {
  let suffix = experimentRun.numExperiments !== 1 ? "s" : "";
  d3.select("#experiment-label").text("Experiment" + suffix);
  d3.select("#num-experiments").text(experimentRun.numExperiments);
  experimentView.render();
}

// ---------- UI controls ----------
function makeGUI() {
  d3.select("#experiment-button").on("click", async () => {
    if (experimentMode) {
      experimentMode = false;
      player.setDriver(driver);
      experimentView.hide();
    } else {
      experimentMode = true;
      player.setDriver(experimentDriver);
      experimentView.show();
    }
    if (userHasInteracted) {
      await reset();
    }
    d3.select("#experiment-button").classed("active", experimentMode);
  });

  d3.select("#add-experiment").on("click", () => {
    if (!experimentMode) {
      return;
    }
    experimentRun.addExperiment();
    updateExperimentUI();
  });

  d3.select("#remove-experiment").on("click", () => {
    if (!experimentMode || experimentRun.numExperiments <= 1) {
      return;
    }
    const lastExp = experimentRun.experiments[experimentRun.experiments.length - 1];
    experimentRun.removeExperiment(lastExp.id);
    updateExperimentUI();
  });

  // SERVER CONTROLS
  d3.select("#reset-button").on("click", async () => {
    if (experimentMode) {
      experimentRun.reset();
      experimentDriver.reset();
    }
    await reset();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", async () => {
    player.pause();
    console.log('before step', tf.memory());
    await oneStep();
    console.log('after step', tf.memory());
  });

  // server config
  let epochsInput = d3.select("#epochs-per-client").on("change", async function() {
    const input = this as HTMLInputElement;
    if (!experimentMode) {
      server.epochsPerClient = +input.value;
    } else {
      experimentRun.refresh();
    }
    if (userHasInteracted) await reset();
  });
  epochsInput.property("value", server.epochsPerClient);

  d3.select("#rounds").on("change", async function() {
    if (experimentMode) {
      experimentRun.refresh();
    }
    if (userHasInteracted) await reset();
  });

  let encoderAgg = d3.select("#encoder-agg").on("change", async function() {
    const input = this as HTMLInputElement;
    server.encoderAgg = input.value as EncoderAgg;
    if (userHasInteracted) await reset();
  });
  encoderAgg.property("value", server.encoderAgg);

  let decoderAgg = d3.select("#decoder-agg").on("change", async function() {
    const input = this as HTMLInputElement;
    server.decoderAgg = input.value as DecoderAgg;
    if (userHasInteracted) await reset();
  });
  decoderAgg.property("value", server.decoderAgg);

  let cac = d3.select("#cac").on("input", async function() {
    const input = this as HTMLInputElement;
    server.caC = +input.value;
    if (userHasInteracted) await reset();
  });
  cac.property("value", server.caC);

  // global config
  let activationDropdown = d3.select("#activations").on("change", async function() {
    const input = this as HTMLInputElement;
    for (const c of clients) {
      c.state.model.activation = input.value as Activations;
    }
    activation = input.value as Activations;
    // activeClient.state.model.activation = input.value as Activations;
    userHasInteracted = true;
    await reset();
  });
  activationDropdown.property("value", activeClient.state.model.activation);

  let learningRate = d3.select("#learningRate").on("change", async function() {
    const input = this as HTMLInputElement;
    for (const c of clients) {
      c.state.model.learningRate = +input.value;
    }
    lr = +input.value;
    // activeClient.state.model.learningRate = +input.value;
    userHasInteracted = true;
    await reset();
  });
  learningRate.property("value", activeClient.state.model.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change", async function() {
    const input = this as HTMLInputElement;
    for (const c of clients) {
      c.state.model.regularization = input.value as Regularizations;
    }
    regularization = input.value as Regularizations;
    // activeClient.state.model.regularization = input.value as Regularizations;
    userHasInteracted = true;
    await reset();
  });
  regularDropdown.property("value", activeClient.state.model.regularization);

  let regularRate = d3.select("#regularRate").on("change",  async function() {
    const input = this as HTMLInputElement;
    for (const c of clients) {
      c.state.model.regularizationRate = +input.value;
    }
    regularizationRate = +input.value;
    // activeClient.state.model.regularizationRate = +input.value;
    userHasInteracted = true;
    await reset();
  });
  regularRate.property("value", activeClient.state.model.regularizationRate);

  // CLIENT CONTROLS
  d3.select("#add-clients").on("click", async () => {
    if (numClients >= CLIENT_LIMIT) {
      return;
    }
    await addClient();
    updateUI();
    await reset();
  });

  d3.select("#remove-clients").on("click", async () => {
    if (numClients <= 1) {
      return;
    }
    removeClient();
    await reset();
  });

  // Resize: redraw only visible view (or all if you want)
  window.addEventListener("resize", () => {
    // cached clientViews: easiest is resize all, but you can do only active
    for (const v of clientViews.values()) v.resize();
  });
}

async function oneStep() {
  userHasInteracted = true;
  if (isTraining) {
    return;
  }
  isTraining = true;
  await server.step(clients);
  updateUI();
  isTraining = false;
}

function updateUI() {
  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  // Update number of clients.
  let suffix = numClients !== 1 ? "s" : "";
  d3.select("#clients-label").text("Client" + suffix);
  d3.select("#num-clients").text(numClients);

  // Update  iteration number.
  d3.select("#iter-number").text(addCommas(zeroPad(server.round)));
  // update all clients series
  for (const v of clientViews.values()) v.tick();

  // Update the links visually.
  // Update the bias values visually.
  // Update charts
  // Update heatmaps
  // Update task-specific line charts
  configView.render(clients);
  serverView.update(server.lastDiagnostics);
  activeView.update();
}


function bindActiveClientGUI() {
  // let activationDropdown = activeView.select("#activations").on("change", async function() {
  //   const input = this as HTMLInputElement;
  //   // for (const c of clients) {
  //   //   c.state.model.activation = input.value as Activations;
  //   // }
  //   activeClient.state.model.activation = input.value as Activations;
  //   userHasInteracted = true;
  //   await reset();
  // });
  // activationDropdown.property("value", activeClient.state.model.activation);

  // let learningRate = activeView.select("#learningRate").on("change", async function() {
  //   const input = this as HTMLInputElement;
  //   // for (const c of clients) {
  //   //   c.state.model.learningRate = +input.value;
  //   // }
  //   activeClient.state.model.learningRate = +input.value;
  //   userHasInteracted = true;
  //   await reset();
  // });
  // learningRate.property("value", activeClient.state.model.learningRate);

  // let regularDropdown = activeView.select("#regularizations").on("change", async function() {
  //   const input = this as HTMLInputElement;
  //   // for (const c of clients) {
  //   //   c.state.model.regularization = input.value as Regularizations;
  //   // }
  //   activeClient.state.model.regularization = input.value as Regularizations;
  //   userHasInteracted = true;
  //   await reset();
  // });
  // regularDropdown.property("value", activeClient.state.model.regularization);

  // let regularRate = activeView.select("#regularRate").on("change",  async function() {
  //   const input = this as HTMLInputElement;
  //   // for (const c of clients) {
  //   //   c.state.model.regularizationRate = +input.value;
  //   // }
  //   activeClient.state.model.regularizationRate = +input.value;
  //   userHasInteracted = true;
  //   await reset();
  // });
  // regularRate.property("value", activeClient.state.model.regularizationRate);

  // per client config
  activeView.select("#add-tasks").on("click", async () => {
    const enabledCount = activeClient.state.model.taskConfig.filter(task => task.enabled).length;
    if (enabledCount >= 5) {
      return;
    }
    const nextDisabled = activeClient.state.model.taskConfig.find(task => !task.enabled);
    if (!nextDisabled) {
      return; // nothing to enable
    }

    nextDisabled.enabled = true;
    await reset();
  });

  activeView.select("#remove-tasks").on("click", async () => {
    const enabledIndices = activeClient.state.model.taskConfig
      .map((task, i) => ({ task, i }))
      .filter(({ task }) => task.enabled);

    if (enabledIndices.length <= 1) {
      return; // need at least 1 head
    }

    const lastEnabledIndex = enabledIndices[enabledIndices.length - 1].i;
    activeClient.state.model.taskConfig[lastEnabledIndex].enabled = false;

    await reset();
  });

  
  // let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  // dataThumbnails.on("click", async function() {
  //   const shape = (this as HTMLCanvasElement).getAttribute("data-dataset");
  //   if (activeClient.state.data.typeShapes.find(s => s.type === shape && s.enabled)) {
  //     // remove shape
  //     if (activeClient.state.data.typeShapes.filter(s => s.enabled).length <= 1) {
  //       return; // need at least 1 shape
  //     }
  //     activeClient.state.data.typeShapes = activeClient.state.data.typeShapes.map(s => {
  //       if (s.type === shape) {
  //         return { ...s, enabled: false };
  //       }
  //       return s;
  //     });
  //     activeView.select(`canvas[data-dataset='${shape}']`).classed("selected", false);
  //   } else {
  //     // add shape
  //     activeClient.state.data.typeShapes = activeClient.state.data.typeShapes.map(s => {
  //       if (s.type === shape) {
  //         return { ...s, enabled: true };
  //       }
  //       return s;
  //     });
  //     activeView.select(`canvas[data-dataset='${shape}']`).classed("selected", true);
  //   }
  //   const enabledShapes = activeClient.state.data.typeShapes.filter(s => s.enabled).map(s => s.type);
  //   activeClient.state.model.taskConfig.find(t => t.name === 'semseg').filters = enabledShapes.length + 1;
  //   await reset();
  // });

  let shapeProbInputs = d3.selectAll("input[shape-prob]");
  shapeProbInputs.on("input", async function() {
    const input = this as HTMLInputElement;
    const shape = input.getAttribute("shape-prob");
    activeClient.state.data.typeShapes.find(s => s.type === shape).probability = +input.value;
    await reset();
  });

  let randomSeed = activeView.select("#random-seed")
    .on("click", async () => {
      activeClient.state.data.seed = Math.floor(Math.random() * 1e8);
      activeView.select('#seed').property("value", activeClient.state.data.seed);
      await reset();
    })
  randomSeed.property("value", activeClient.state.data.seed);
  let seed = activeView.select("#seed")
    .on("input", async function() {
      const input = this as HTMLInputElement;
      activeClient.state.data.seed = Math.floor(Number(input.value)) || 0;
    })
    .on("wheel", (event) => {
      event.preventDefault();
    }, { passive: false } as any)
    .on("change", async function () {
      await reset();
    });
  seed.property("value", activeClient.state.data.seed);

  let resolution = activeView.select("#resolution")
    .on("input", async function() {
      const input = this as HTMLInputElement;
      const raw = Number(input.value);
      const snapped = 32 * Math.pow(2, raw);
      activeClient.state.data.resolution = snapped;
      activeView.select("label[for='resolution'] .value").text(String(snapped));
    })
    .on("change", async function () {
      await reset();
    });
  resolution.property("value", Math.round(Math.log2(activeClient.state.data.resolution / 32)));
  activeView.select("label[for='resolution'] .value").text(activeClient.state.data.resolution);

  let nShapes = activeView.select("#nShapes")
    .on("input", async function() {
      const input = this as HTMLInputElement;
      activeClient.state.data.numShapes = Number(input.value);
      activeView.select("label[for='nShapes'] .value").text(input.value);
    })
    .on("change", async function () {
      await reset();
    });
  nShapes.property("value", activeClient.state.data.numShapes);
  activeView.select("label[for='nShapes'] .value").text(activeClient.state.data.numShapes);

  let percTrain = activeView.select("#percTrainData")
  .on("input", async function() {
    const input = this as HTMLInputElement;
    activeClient.state.data.percTrain = Number(input.value);
    activeView.select("label[for='percTrainData'] .value").text(input.value);
  })
  .on("change", async function () {
    await reset();
  });
  percTrain.property("value", activeClient.state.data.percTrain);
  activeView.select("label[for='percTrainData'] .value").text(activeClient.state.data.percTrain);

  let batchSize = activeView.select("#batchSize")
  .on("input", async function() {
    const input = this as HTMLInputElement;
    const raw = Number(input.value);
    const snapped = 8 * Math.pow(2, raw);
    activeClient.state.data.batchSize = snapped;
    activeView.select("label[for='batchSize'] .value").text(String(snapped));
  })
  .on("change", async function () {
    await reset();
  });
  batchSize.property("value", Math.round(Math.log2(activeClient.state.data.batchSize / 8)));
  activeView.select("label[for='batchSize'] .value").text(activeClient.state.data.batchSize);

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener("resize", () => {
    activeView.resize();
  });
}

async function reset(pausePlayer = true) {
  if (pausePlayer) {
    player.pause();
  }
  while (isTraining) {
    await new Promise(resolve => setTimeout(resolve, 0)); // yield to event loop
  }
  
  if (!userHasInteracted) {
    // Reset only active client/view
    await activeClient.reset();
    activeView.reset();
  } else {
    // Reset all clients
    for (const c of clients) {
      await c.reset();
    }
    // Reset all clientViews
    for (const v of clientViews.values()) {
      v.reset();
    }
    userHasInteracted = false;
  }
  await server.reset();
  serverView.reset();
  updateUI();
  updateExperimentUI();
  console.log("reset", tf.memory());
};

// ---------- Boot ----------
makeGUI();
await reset();
configView.render(clients);
showClient(clients[0]);