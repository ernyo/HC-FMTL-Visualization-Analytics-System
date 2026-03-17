// server.ts
import * as tf from "@tensorflow/tfjs";
import { FederatedClient } from "./client";
import { aggregate, EncoderAgg, DecoderAgg } from "./aggregate";
import { exportCheckpoint, WeightMap, disposeWeightMap, toCanonicalMap } from "./utils";
import { computeInterClientDiagnostics } from "./diagnostics";


export class FederatedServer {
  round = 0;
  lastDiagnostics: ReturnType<typeof computeInterClientDiagnostics> | null = null;

  /** Configurable parameters; can be updated live. */
  epochsPerClient: number = 1;
  encoderAgg: EncoderAgg = "none";
  decoderAgg: DecoderAgg = "none";
  encoderAlpha: number = 0.1;
  decoderBeta: number = 0.1;
  caC: number = 0.0;
  evalEvery: number = 1;
  checkpointEvery: number = 1;

  /** Internal state. */
  private lastCkpt: WeightMap[] | null = null;

  get snapshotConfig() {
    return {
      epochsPerClient: this.epochsPerClient,
      encoderAgg: this.encoderAgg,
      decoderAgg: this.decoderAgg,
      caC: this.caC,
    }
  }

  async reset() {
    // clear server bookkeeping
    this.round = 0;
    this.lastDiagnostics = null;

    // dispose old checkpoints
    if (this.lastCkpt) {
      this.lastCkpt.forEach(disposeWeightMap);
      this.lastCkpt = null;
    }
  }

  /** One federated round (train -> (optional) hyperweight update -> aggregate -> (optional) eval). */
  async step(clients: FederatedClient[]) {
    console.log("before", tf.memory());
    
    // Initialize lastCkpt on first round
    if (!this.lastCkpt) {
      this.lastCkpt = clients.map(c => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors
    }

    // --- 1) Local training (sequential; you can parallelize if your environment allows)
    for (const client of clients) {
      await client.update(this.epochsPerClient);
    }

    // --- 2) Collect checkpoints
    const saveCkpt = clients.map((c) => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors
    this.lastDiagnostics = computeInterClientDiagnostics(clients, saveCkpt, this.lastCkpt!);

    // --- 3) Aggregate 
    await aggregate( // increases memory by 319 tensors in first round, then 228 tensors
      clients,
      saveCkpt,
      this.lastCkpt,
      this.encoderAgg,
      this.decoderAgg,
      this.encoderAlpha,
      this.decoderBeta,
      this.caC,
    );

    // --- 4) Update lastCkpt for next round
    saveCkpt.forEach(disposeWeightMap); // decreases memory by 93 tensors
    this.lastCkpt.forEach(disposeWeightMap); // decreases memory by 93 tensors
    this.lastCkpt = clients.map(c => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors

    this.round++;
    console.log("after", tf.memory());
  }
}