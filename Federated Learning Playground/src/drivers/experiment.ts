import * as d3 from "d3";
import { DecoderAgg, EncoderAgg } from "../federated/aggregate";
import { FederatedClient, ClientMetrics } from "../federated/client";
import { FederatedServer } from "../federated/server";
import { Driver } from "./player";
import { mean, isFiniteNumber, deltaM } from "../utils/helpers";
import { LOWER_IS_BETTER } from "../models/metrics";
import { TaskName } from "../models/model";
import { Metric } from "../models/metrics";

export type ExperimentPlan = {
  encoderAgg: EncoderAgg;   // e.g. "none" | "fedavg" | "conflict_averse"
  decoderAgg: DecoderAgg;   // e.g. "none" | "fedavg" | "cross_attention"
  cac: number;          // e.g. "0" | "0.1" ...
  epochsPerClient: number;  // e.g. 1 | 5 | 10
  rounds: number;       // number of rounds to run this setup for
};

export type ExperimentLog = {
  header: {
    timestamp: string;
    clients: any[];
    server: any;
  };
  rounds: Array<{
    round: number; 
    clientMetrics: Array<ClientMetrics>;
    serverMetrics?: Record<string, number | null>;
    deltaM?: number | null; 
  }>;
  footer: {
    timestamp: string;
    totalTime: number;
    deltaM?: number | null;
  };
}

type ExperimentSummary = {
  done: boolean;
  finalRound?: number;
  avgLossTest?: number;
  avgLossTrain?: number;
  totalTime?: number;
  deltaM?: number | null;
};

export type ExperimentStatus = "Not started" | "In progress" | "Completed" | "Failed";

export class Experiment {
  id: string = crypto.randomUUID();
  plan: ExperimentPlan;
  log: ExperimentLog = {
    header: { timestamp: new Date().toISOString(), clients: [], server: {} },
    rounds: [],
    footer: { timestamp: new Date().toISOString(), totalTime: 0 },
  };
  startTime: number = undefined;
  status: ExperimentStatus = "Not started";

  constructor(
    encoderAgg: EncoderAgg = 'none',
    decoderAgg: DecoderAgg = 'none',
    cac: number = 0,
    epochsPerClient: number = 1,
    rounds: number = 1,
  ) {
    this.plan = { encoderAgg, decoderAgg, cac, epochsPerClient, rounds };
  }

  reset() {
    this.log = {
      header: { timestamp: new Date().toISOString(), clients: [], server: {} },
      rounds: [],
      footer: { timestamp: new Date().toISOString(), totalTime: 0 },
    };
    this.startTime = undefined;
    this.status = "Not started";
  }

  updatePlan(updatedFields: Partial<ExperimentPlan>) {
    this.plan = { ...this.plan, ...updatedFields };
  }

  downloadLog() {
    const blob = new Blob([JSON.stringify(this.log, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `results_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  summarizeExperiment(baseTask?: Record<string, number> | null): ExperimentSummary {
    const rounds = this.log.rounds;
    const footer = this.log.footer;

    if (!rounds.length) return { done: false };

    const last = rounds[rounds.length - 1];
    const metricsArr = (last?.clientMetrics ?? []) as any[];

    const taskMetricMap: Partial<Record<TaskName, Metric>> = {};
    this.log.header.clients[0].model.taskConfig.forEach((tc: any) => {
      if (tc.name && tc.metric) {
        taskMetricMap[tc.name] = tc.metric;
      }
    });

    const lossTestVals = metricsArr.map(m => m?.lossTest).filter(isFiniteNumber);
    const lossTrainVals = metricsArr.map(m => m?.lossTrain).filter(isFiniteNumber);

     // compute Δm if baseline provided and this experiment has task metrics
    let dm: number | null = null;
    if (baseTask) {
      const fedTask = this.finalTaskMetricsFromLog();
      if (fedTask) dm = deltaM(fedTask, baseTask, taskMetricMap, LOWER_IS_BETTER);
    }

    return {
      done: this.status === "Completed",
      finalRound: last?.round,
      avgLossTest: mean(lossTestVals),
      avgLossTrain: mean(lossTrainVals),
      totalTime: isFiniteNumber(footer?.totalTime) ? footer.totalTime : undefined,
      deltaM: dm,
    };
  }

  taskMetricMapFromHeader(): Partial<Record<TaskName, Metric>> {
    const out: Partial<Record<TaskName, Metric>> = {};
    const firstClient = this.log.header.clients?.[0];
    const cfg = firstClient?.model?.taskConfig ?? [];
    cfg.forEach((tc: any) => {
      if (tc?.name && tc?.metric) out[tc.name] = tc.metric;
    });
    return out;
  }

  taskMetricsFromClientMetrics(metrics: ClientMetrics[]) {
    const sums: Record<string, number> = {};
    const counts: Record<string, number> = {};

    for (const cm of metrics ?? []) {
      const perHead = (cm as any)?.perHead ?? [];
      for (const h of perHead) {
        const task = h?.task;
        const v = Number(h?.testMetric);
        if (!task || !Number.isFinite(v)) continue;

        sums[task] = (sums[task] ?? 0) + v;
        counts[task] = (counts[task] ?? 0) + 1;
      }
    }

    const out: Record<string, number> = {};
    for (const task of Object.keys(sums)) {
      out[task] = sums[task] / counts[task];
    }
    return out;
  }

  finalTaskMetricsFromLog() {
    const rounds = this.log.rounds;
    if (!rounds.length) return null;

    const last = rounds[rounds.length - 1];
    const clientMetrics = last.clientMetrics;

    const sums: Record<string, number> = {};
    const counts: Record<string, number> = {};

    for (const cm of clientMetrics) {
      const perHead = cm.perHead;
      for (const h of perHead) {
        const task = h.task;
        const v = Number(h.testMetric);
        if (!task || !Number.isFinite(v)) continue;

        sums[task] = (sums[task] ?? 0) + v;
        counts[task] = (counts[task] ?? 0) + 1;
      }
    }

    const out: Record<string, number> = {};
    for (const task of Object.keys(sums)) {
      out[task] = sums[task] / counts[task];
    }
    return out;
  }
}

export class ExperimentRun {
  experiments: Experiment[] = [];
  constructor() {
    this.addExperiment(); // start with one default experiment
  }

  private get epochsPerClient() {
    return Number(d3.select("#epochs-per-client").property("value"));
  }
  private get rounds() {
    return Number(d3.select("#rounds").property("value"));
  }

  get numExperiments() {
    return this.experiments.length;
  }

  addExperiment(encoderAgg: EncoderAgg = 'none', decoderAgg: DecoderAgg = 'none', cac: number = 0) {
    const exp = new Experiment(encoderAgg, decoderAgg, cac, this.epochsPerClient, this.rounds);
    this.experiments.push(exp);
  }

  updateExperiment(id: string, updatedFields: Partial<ExperimentPlan>) {
    const experiment = this.experiments.find(e => e.id === id);
    if (!experiment) return;
    experiment.updatePlan(updatedFields);
  }

  removeExperiment(id: string) {
    this.experiments = this.experiments.filter(e => e.id !== id);
  }

  refresh() {
    for (const e of this.experiments) {
      this.updateExperiment(e.id, {
        epochsPerClient: this.epochsPerClient,
        rounds: this.rounds,
      });
    }
  }

  reset() {
    for (const exp of this.experiments) {
      exp.reset();
    }
  }
}

export class ExperimentDriver implements Driver {
  private expIdx = 0;
  private active: Experiment | null = null;
  private roundsLeft = 0;
  private finished = false;

  constructor(
    private deps: {
      experimentRun: ExperimentRun;
      server: FederatedServer;
      clients: FederatedClient[];
      oneStep: (epochsPerClient: number) => Promise<void>;
      reset: (pausePlayer?: boolean) => Promise<void>;
      updateCards: () => void; // e.g. to trigger re-render or update button states
    }
  ) {}

  async step(): Promise<boolean> {
    const run = this.deps.experimentRun;
    if (!run?.experiments?.length) return false;

    // stop when done
    if (this.finished) return false;

    // start next experiment if needed
    if (!this.active || this.roundsLeft <= 0) {
      // if we just finished one, finalize it
      if (this.active && this.roundsLeft <= 0) {
        this.finishExperiment(this.active);
        this.deps.updateCards();
        this.active = null;
      }

      // no more experiments
      if (this.expIdx >= run.experiments.length) {
        this.finished = true;
        // optional: reset everything back to baseline state here (WITHOUT pausing player)
        // await this.deps.reset(false);
        return false;
      }

      await this.startExperiment(run.experiments[this.expIdx]);
      this.expIdx += 1;
    }

    // run exactly ONE round
    const exp = this.active!;
    try {
      const before = this.deps.server.round;
      await this.deps.oneStep(exp.plan.epochsPerClient);
      const after = this.deps.server.round;

      // Only log if round advanced (safety)
      if (after !== before) {
        const roundJustRan = before;
        const clientMetrics = this.deps.clients.map(c => c.lastMetrics);

        // 1) snapshot server metrics
        const serverMetrics = this.deps.server.lastDiagnostics ?? null;

        // 2) per-round deltaM vs baseline (same round)
        const baseline = run.experiments[0];
        let dmThisRound: number | null = null;

        if (baseline && baseline !== exp) {
          const baseRound = baseline.log.rounds.find(r => r.round === roundJustRan);
          if (baseRound) {
            const taskMetricMap = exp.taskMetricMapFromHeader(); // or baseline's
            const fedTask = exp.taskMetricsFromClientMetrics(clientMetrics);

            // IMPORTANT: baseRound must have the same property name you log with
            const baseTask = baseline.taskMetricsFromClientMetrics(baseRound.clientMetrics);

            dmThisRound = deltaM(fedTask, baseTask, taskMetricMap, LOWER_IS_BETTER);
          }
        }

        exp.log.rounds.push({
          round: roundJustRan,
          clientMetrics,
          serverMetrics,
          deltaM: dmThisRound,
        });

        this.roundsLeft -= 1;
      }

      this.deps.updateCards();

      // finalize immediately if that was the last round
      if (this.roundsLeft <= 0) {
        this.finishExperiment(exp);
        this.deps.updateCards();
        exp.downloadLog(); // if you still want auto-download
        this.active = null;
      }

      return true;
    } catch (err) {
      console.error("Error during experiment step:", err);
      exp.status = "Failed";
      this.deps.updateCards();
      this.finished = true;
      return false;
    }
  }

  /** Call this before pressing play to restart the whole run */
  reset() {
    this.expIdx = 0;
    this.active = null;
    this.roundsLeft = 0;
    this.finished = false;
  }

  private async startExperiment(exp: Experiment) {
    // IMPORTANT: apply plan BEFORE reset if reset uses server config
    this.deps.server.encoderAgg = exp.plan.encoderAgg;
    this.deps.server.decoderAgg = exp.plan.decoderAgg;
    this.deps.server.caC = exp.plan.cac;
    this.deps.server.epochsPerClient = exp.plan.epochsPerClient;

    exp.status = "In progress";
    this.deps.updateCards();

    // reset models without pausing the player
    await this.deps.reset(false);

    // init log AFTER reset so snapshotConfig reflects the actual reset state
    exp.log.header.timestamp = new Date().toISOString();
    exp.log.header.clients = this.deps.clients.map(c => c.snapshotConfig);
    exp.log.header.server = { ...this.deps.server.snapshotConfig, rounds: exp.plan.rounds };

    exp.log.rounds = [];
    exp.startTime = performance.now();

    this.active = exp;
    this.roundsLeft = exp.plan.rounds;
  }

  private finishExperiment(exp: Experiment) {
    exp.status = "Completed";
    const end = performance.now();

    const baseline = this.deps.experimentRun.experiments[0];
    const baseTask =
      baseline && baseline !== exp && baseline.status === "Completed"
        ? baseline.finalTaskMetricsFromLog()
        : null;

    const dm =
      baseTask ? exp.summarizeExperiment(baseTask).deltaM : null;

    exp.log.footer = {
      timestamp: new Date().toISOString(),
      totalTime: (end - (exp.startTime ?? end)) / 1000,
      deltaM: dm,
    };
  }

}
