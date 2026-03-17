import { performance } from "node:perf_hooks";

function quantile(sorted: number[], q: number) {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1];
  return next === undefined ? sorted[base] : sorted[base] + rest * (next - sorted[base]);
}

function summarizeMs(xs: number[]) {
  const s = xs.slice().sort((a,b)=>a-b);
  return {
    n: s.length,
    median: quantile(s, 0.5),
    p95: quantile(s, 0.95),
    max: s.length ? s[s.length - 1] : NaN,
  };
}

// EXPERIMENT 1: LABEL SKEW
console.log("label_skew.ts: loaded");
import fs from "node:fs";
import path from "node:path";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";

// IMPORTANT: must be first TF import for CUDA
import * as tf from "@tensorflow/tfjs-node-gpu";
import { execSync } from "node:child_process";


import { FederatedClient } from "../federated/client.js";
import { FederatedServer } from "../federated/server.js";

import { deltaM, isFiniteNumber } from "../utils/helpers.js";
import { LOWER_IS_BETTER } from "../models/metrics.js";
import type { Metric } from "../models/metrics.js";
import type { TaskName } from "../models/model.js";
import type { ClientMetrics } from "../federated/client.js";

type ExperimentPlan = {
  encoderAgg: string;   // or EncoderAgg if you want
  decoderAgg: string;   // or DecoderAgg
  cac: number;
  epochsPerClient: number;
  rounds: number;
};

type ExperimentLog = {
  header: {
    timestamp: string;
    clients: any[];
    server: any;
  };
  rounds: Array<{
    round: number;
    clientMetrics: Array<ClientMetrics>;
    serverMetrics?: Record<string, any> | null;
    deltaM?: number | null;
    timings?: { stepMs: number; totalMs: number };
  }>;
  footer: {
    timestamp: string;
    totalTime: number;
    deltaM?: number | null;
  };
};

function taskMetricMapFromClientSnapshot(clientSnap: any): Partial<Record<TaskName, Metric>> {
  const out: Partial<Record<TaskName, Metric>> = {};
  const cfg = clientSnap?.model?.taskConfig ?? [];
  for (const tc of cfg) {
    if (tc?.name && tc?.metric) out[tc.name] = tc.metric;
  }
  return out;
}

function taskMetricsFromClientMetrics(metrics: ClientMetrics[]) {
  const sums: Record<string, number> = {};
  const counts: Record<string, number> = {};

  for (const cm of metrics ?? []) {
    for (const h of cm?.perHead ?? []) {
      const task = h?.task;
      const v = Number(h?.testMetric);
      if (!task || !Number.isFinite(v)) continue;
      sums[task] = (sums[task] ?? 0) + v;
      counts[task] = (counts[task] ?? 0) + 1;
    }
  }

  const out: Record<string, number> = {};
  for (const k of Object.keys(sums)) out[k] = sums[k] / counts[k];
  return out;
}

function finalTaskMetricsFromLog(log: ExperimentLog) {
  const last = log.rounds[log.rounds.length - 1];
  if (!last) return null;
  return taskMetricsFromClientMetrics(last.clientMetrics);
}

function atomicWriteJson(filePath: string, obj: any) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });

  const tmp = `${filePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), "utf8");
  fs.renameSync(tmp, filePath); // atomic on same filesystem
}

// Deterministic 32-bit hash from strings + numbers (FNV-1a)
function hash32(...parts: Array<string | number>): number {
  let h = 2166136261 >>> 0;
  for (const p of parts) {
    const s = String(p);
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    // separator
    h ^= 0xff;
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function clientSeed(args: {
  runSeed: number;
  ablation: string;
  variant: string;
  agg: string;
  clientIndex: number;   // 0,1,2
}) {
  // keep within JS safe int and tfjs seed expectations
  return hash32(args.runSeed, args.ablation, args.variant, args.agg, args.clientIndex);
}

type AggRun = {
  name: "none" | "fedavg" | "fedhca2" | "encoder_fedavg_decoder_none" | "encoder_conflict_averse_decoder_none" | "encoder_none_decoder_fedavg" | "encoder_none_decoder_cross_attention";
  serverPatch: Partial<Pick<FederatedServer, "encoderAgg" | "decoderAgg" | "caC">>;
};

const AGG_RUNS: AggRun[] = [
  {
    name: "none",
    serverPatch: { encoderAgg: "none", decoderAgg: "none", caC: 0.0 },
  },
  {
    name: "fedavg",
    serverPatch: { encoderAgg: "fedavg", decoderAgg: "fedavg", caC: 0.0 },
  },
  {
    name: "fedhca2",
    serverPatch: { encoderAgg: "conflict_averse", decoderAgg: "cross_attention", caC: 0.5 },
  },
  {
    name: "encoder_fedavg_decoder_none",
    serverPatch: { encoderAgg: "fedavg", decoderAgg: "none", caC: 0.0 },
  },
  {
    name: "encoder_conflict_averse_decoder_none",
    serverPatch: { encoderAgg: "conflict_averse", decoderAgg: "none", caC: 0.5 },
  },
  {
    name: "encoder_none_decoder_fedavg",
    serverPatch: { encoderAgg: "none", decoderAgg: "fedavg", caC: 0.0 },
  },
  {
    name: "encoder_none_decoder_cross_attention",
    serverPatch: { encoderAgg: "none", decoderAgg: "cross_attention", caC: 0.5 },
  },
];

// -----------------------------
// Baseline constants (uniform shapes)
// -----------------------------
type ShapeSpec = { type: string; probability: number };

// Matches the baseline paragraph you gave
const BASELINE = {
  clients: 3 as const,

  // training schedule
  rounds: 150,
  epochsPerClient: 1,

  // server
  caC: 0.5,
  evalEvery: 1,

  // client data
  data: {
    numSamples: 100,
    numShapes: 1,
    resolution: 32,
    percTrain: 80,
    batchSize: 16,
    typeShapes: [
    { type: "circle", probability: 0.25 },
    { type: "square", probability: 0.25 },
    { type: "triangle", probability: 0.25 },
    { type: "star", probability: 0.25 },
    ],
  },

  // client model
  model: {
    activation: "relu" as const,
    learningRate: 0.01,
    regularization: "l2" as const,
    regularizationRate: 0.0001,
    lossWeight: 1.0,
  },
};

type RunConfig = {
  runId: string;
  seed: number;
  logDir: string;
  // optionally override baseline rounds from CLI
  rounds?: number;
};

// -----------------------------
// Ablation types
// -----------------------------
type ClientPatch = {
  data?: Partial<{
    seed: number;
    numSamples: number;
    numShapes: number;
    typeShapes: ShapeSpec[];
    resolution: number;
    batchSize: number;
    percTrain: number;
  }>;
  model?: Partial<{
    activation: "relu" | "tanh" | "sigmoid";
    learningRate: number;
    regularization: "none" | "l1" | "l2";
    regularizationRate: number;
    lossWeight: number; // apply to all tasks
  }>;
};

type Variant = {
  name: string;
  clientPatches: [ClientPatch, ClientPatch, ClientPatch];
  serverPatch?: Partial<Pick<FederatedServer, "epochsPerClient" | "evalEvery" | "caC">>;
};

type Ablation = { name: string; variants: Variant[] };

// -----------------------------
// Presets: LABEL SKEW
// -----------------------------
type LabelSkewPreset = ShapeSpec[][];
const LABEL_SKEW_LOW: LabelSkewPreset = [
  [
    { type: "circle", probability: 0.26 },
    { type: "square", probability: 0.24 },
    { type: "triangle", probability: 0.25 },
    { type: "star", probability: 0.25 },
  ],
  [
    { type: "circle", probability: 0.23 },
    { type: "square", probability: 0.27 },
    { type: "triangle", probability: 0.24 },
    { type: "star", probability: 0.26 },
  ],
  [
    { type: "circle", probability: 0.25 },
    { type: "square", probability: 0.25 },
    { type: "triangle", probability: 0.27 },
    { type: "star", probability: 0.23 },
  ],
];

const LABEL_SKEW_MED: LabelSkewPreset = [
  [
    { type: "circle", probability: 0.55 },
    { type: "square", probability: 0.25 },
    { type: "triangle", probability: 0.15 },
    { type: "star", probability: 0.05 },
  ],
  [
    { type: "circle", probability: 0.10 },
    { type: "square", probability: 0.60 },
    { type: "triangle", probability: 0.20 },
    { type: "star", probability: 0.10 },
  ],
  [
    { type: "circle", probability: 0.15 },
    { type: "square", probability: 0.10 },
    { type: "triangle", probability: 0.60 },
    { type: "star", probability: 0.15 },
  ],
];

const LABEL_SKEW_HIGH: LabelSkewPreset = [
  [
    { type: "circle", probability: 0.85 },
    { type: "square", probability: 0.10 },
    { type: "triangle", probability: 0.03 },
    { type: "star", probability: 0.02 },
  ],
  [
    { type: "circle", probability: 0.04 },
    { type: "square", probability: 0.90 },
    { type: "triangle", probability: 0.03 },
    { type: "star", probability: 0.03 },
  ],
  [
    { type: "circle", probability: 0.03 },
    { type: "square", probability: 0.04 },
    { type: "triangle", probability: 0.90 },
    { type: "star", probability: 0.03 },
  ],
];

// -----------------------------
// Logging
// -----------------------------
class JsonlLogger {
  private stream: fs.WriteStream;
  constructor(public filePath: string) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    this.stream = fs.createWriteStream(filePath, { flags: "a" });
  }
  log(event: Record<string, any>) {
    const row = { ts: new Date().toISOString(), ...event };
    this.stream.write(JSON.stringify(row) + "\n");
    if (event.level !== "debug") {
      // eslint-disable-next-line no-console
      console.log(`[${row.ts}] ${event.event ?? "log"} ${event.msg ?? ""}`);
    }
  }
  close() {
    this.stream.end();
  }
}

function validateShapes(typeShapes: ShapeSpec[]) {
  const sum = typeShapes.reduce((s, x) => s + x.probability, 0);
  if (Math.abs(sum - 1) > 1e-6) throw new Error(`typeShapes must sum to 1. Got ${sum}`);
}

// -----------------------------
// CUDA init
// -----------------------------
async function initTf(logger: JsonlLogger, requireGpu: boolean = true) {
  const ok = await tf.setBackend("tensorflow");
  await tf.ready();

  const backend = tf.getBackend();

  // Force TF runtime initialization (and GPU init if available)
  const a = tf.randomNormal([256, 256]);
  const b = tf.randomNormal([256, 256]);
  const c = a.matMul(b);
  await c.data();
  a.dispose(); b.dispose(); c.dispose();

  // Strong, external signal: does this node have NVIDIA GPUs?
  let nvidiaGpuCount: number | null = null;
  let nvidiaNames: string[] | null = null;
  try {
    const out = execSync(
      "nvidia-smi --query-gpu=name --format=csv,noheader",
      { encoding: "utf8" }
    ).trim();
    nvidiaNames = out ? out.split("\n").map(s => s.trim()).filter(Boolean) : [];
    nvidiaGpuCount = nvidiaNames.length;
  } catch {
    nvidiaGpuCount = null;  // nvidia-smi not available
    nvidiaNames = null;
  }

  logger.log({
    event: "tf_init",
    msg: "TF ready",
    meta: { ok, backend, nvidiaGpuCount, nvidiaNames },
  });

  if (requireGpu) {
    if (backend !== "tensorflow") {
      throw new Error(`requireGpu=true but backend is '${backend}' (expected 'tensorflow').`);
    }
    // Only fail on strong negative evidence:
    if (nvidiaGpuCount === 0) {
      throw new Error("requireGpu=true but nvidia-smi reports 0 GPUs on this node.");
    }
    // If nvidia-smi isn't available (null), don't fail here—Slurm may still provide GPU access.
  }
}

// -----------------------------
// Baseline application
// -----------------------------
function applyBaseline(client: FederatedClient, seed: number) {
  // baseline data
  Object.assign(client.state.data as any, BASELINE.data, { seed });

  // baseline model
  client.state.model.activation = BASELINE.model.activation;
  client.state.model.learningRate = BASELINE.model.learningRate;
  client.state.model.regularization = BASELINE.model.regularization;
  client.state.model.regularizationRate = BASELINE.model.regularizationRate;

  // loss weights baseline
  client.state.model.taskConfig = client.state.model.taskConfig.map((t: any) => ({
    ...t,
    lossWeight: BASELINE.model.lossWeight,
  }));

  validateShapes((client.state.data as any).typeShapes);
}

function applyPatch(client: FederatedClient, patch: ClientPatch) {
  if (patch.data) {
    Object.assign(client.state.data as any, patch.data);
    if (patch.data.typeShapes) validateShapes(patch.data.typeShapes);
  }
  if (patch.model) {
    if (patch.model.activation) client.state.model.activation = patch.model.activation;
    if (typeof patch.model.learningRate === "number") client.state.model.learningRate = patch.model.learningRate;
    if (patch.model.regularization) client.state.model.regularization = patch.model.regularization;
    if (typeof patch.model.regularizationRate === "number") client.state.model.regularizationRate = patch.model.regularizationRate;

    if (typeof patch.model.lossWeight === "number") {
      client.state.model.taskConfig = client.state.model.taskConfig.map((t: any) => ({
        ...t,
        lossWeight: patch.model.lossWeight!,
      }));
    }
  }
}

async function buildSystem(cfg: RunConfig, variant: Variant, agg: AggRun) {
  const clients: [FederatedClient, FederatedClient, FederatedClient] = [
    new FederatedClient(),
    new FederatedClient(),
    new FederatedClient(),
  ];

  clients.forEach((c, i) => {
    const s = clientSeed({
      runSeed: cfg.seed,
      ablation: "label_skew",
      variant: variant.name,
      agg: agg.name,
      clientIndex: i,
    });

    applyBaseline(c, s);       // baseline uses seed you pass in
    applyPatch(c, variant.clientPatches[i]);
  });

  for (const c of clients) await c.reset();

  const server = new FederatedServer();
  await server.reset();

  // baseline schedule
  server.epochsPerClient = BASELINE.epochsPerClient;
  server.evalEvery = BASELINE.evalEvery;
  server.caC = BASELINE.caC;

  // apply aggregation run patch (always)
  Object.assign(server as any, agg.serverPatch);

  // (optional) allow variant-specific server overrides too
  if (variant.serverPatch) Object.assign(server as any, variant.serverPatch);

  return { clients, server };
}

// -----------------------------
// Run loops
// -----------------------------
async function runVariant(
  cfg: RunConfig,
  ablationName: string,
  variant: Variant,
  agg: AggRun,
  baselineLog: ExperimentLog | null,
  outPath: string
): Promise<ExperimentLog> {
  const { clients, server } = await buildSystem(cfg, variant, agg);
  const rounds = cfg.rounds ?? BASELINE.rounds;

  const plan: ExperimentPlan = {
    encoderAgg: (server as any).encoderAgg ?? "none",
    decoderAgg: (server as any).decoderAgg ?? "none",
    cac: (server as any).caC ?? 0,
    epochsPerClient: server.epochsPerClient,
    rounds,
  };

  const start = Date.now();

  // Header snapshot like UI
  const log: ExperimentLog = {
    header: {
      timestamp: new Date().toISOString(),
      clients: clients.map(c => c.snapshotConfig),
      server: { ...(server as any).snapshotConfig, ...plan },
    },
    rounds: [],
    footer: {
      timestamp: new Date().toISOString(),
      totalTime: 0,
      deltaM: null,
    },
  };

  // metric map for deltaM directionality
  const taskMetricMap = taskMetricMapFromClientSnapshot(log.header.clients[0]);

  // For per-round deltaM, we need baseline round-by-round
  const baselineRoundsByRound = new Map<number, ClientMetrics[]>();
  if (baselineLog) {
    for (const r of baselineLog.rounds) baselineRoundsByRound.set(r.round, r.clientMetrics);
  }

  try {
    for (let r = 1; r <= rounds; r++) {
      const tRound0 = performance.now(); // time

      await server.step(clients);

      const tAfterStep = performance.now(); // time

      const roundJustRan = server.round;

      const clientMetrics = clients.map(c => c.lastMetrics);
      const serverMetrics = server.lastDiagnostics ?? null;

      let dmThisRound: number | null = null;
      if (baselineLog) {
        const baseMetrics = baselineRoundsByRound.get(roundJustRan);
        if (baseMetrics) {
          const fedTask = taskMetricsFromClientMetrics(clientMetrics);
          const baseTask = taskMetricsFromClientMetrics(baseMetrics);
          dmThisRound = deltaM(fedTask, baseTask, taskMetricMap, LOWER_IS_BETTER);
        }
      }

      const roundRecord = {
        round: roundJustRan,
        clientMetrics,
        serverMetrics,
        deltaM: dmThisRound,
        timings: {
          stepMs: tAfterStep - tRound0,
          totalMs: 0,
        },
      };

      log.rounds.push(roundRecord);

      // checkpoint EVERY round
      atomicWriteJson(outPath, log);

      const tAfterWrite = performance.now(); // time
      roundRecord.timings!.totalMs = tAfterWrite - tRound0;
    }

    // footer deltaM vs baseline FINAL
    let finalDm: number | null = null;
    if (baselineLog) {
      const baseTask = finalTaskMetricsFromLog(baselineLog);
      const fedTask = finalTaskMetricsFromLog(log);
      if (baseTask && fedTask) {
        finalDm = deltaM(fedTask, baseTask, taskMetricMap, LOWER_IS_BETTER);
      }
    }

    const totals = log.rounds.map(r => r.timings?.totalMs).filter((x): x is number => Number.isFinite(x));
    log.footer = {
      timestamp: new Date().toISOString(),
      totalTime: (Date.now() - start) / 1000,
      deltaM: finalDm,
      perf: {
        totalMs: summarizeMs(totals),
        stepMs: summarizeMs(log.rounds.map(r => r.timings?.stepMs).filter((x): x is number => Number.isFinite(x))),
      },
    } as any;
    atomicWriteJson(outPath, log);

    return log;
  } finally {
    clients.forEach(c => c.dispose());
    // tf.disposeVariables();
    await tf.nextFrame();
  }
}

async function runAblation(cfg: RunConfig, ablation: Ablation) {
  fs.mkdirSync(cfg.logDir, { recursive: true });

  for (const variant of ablation.variants) {
    // 1) Run "none" first for this SAME skew -> comparator baseline
    const noneAgg = AGG_RUNS.find(a => a.name === "none")!;
    const outNone = path.join(cfg.logDir, `${cfg.runId}_${ablation.name}_${variant.name}_none.json`);
    const baselineNoneLog = await runVariant(cfg, ablation.name, variant, noneAgg, null, outNone);

    // 2) Run the other aggs, comparing to the none baseline log
    for (const agg of AGG_RUNS) {
      if (agg.name === "none") continue;

      const outPath = path.join(cfg.logDir, `${cfg.runId}_${ablation.name}_${variant.name}_${agg.name}.json`);
      await runVariant(cfg, ablation.name, variant, agg, baselineNoneLog, outPath);
    }
  }
}

// -----------------------------
// Variant builders
// -----------------------------
function labelSkewVariant(name: string, preset: LabelSkewPreset): Variant {
  return {
    name,
    clientPatches: [
      { data: { typeShapes: preset[0] } },
      { data: { typeShapes: preset[1] } },
      { data: { typeShapes: preset[2] } },
    ],
  };
}

function buildAblations(): Ablation {
  return {
    name: "label_skew",
    variants: [
      // labelSkewVariant("low", LABEL_SKEW_LOW),
      labelSkewVariant("med", LABEL_SKEW_MED),
      // labelSkewVariant("high", LABEL_SKEW_HIGH),
    ],
  };
}

// -----------------------------
// Main
// -----------------------------
async function main() {
  console.log("label_skew.ts: main() starting");
  const argv = await yargs(hideBin(process.argv))
    .option("seed", { type: "number", default: 0 })
    .option("logDir", { type: "string", default: "results" })
    .option("runId", { type: "string", default: "" })
    .option("rounds", { type: "number", default: BASELINE.rounds })
    .strict()
    .parse();

  const runId = argv.runId || `abl_${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const cfg: RunConfig = { runId, seed: argv.seed, logDir: argv.logDir, rounds: argv.rounds };

  const logPath = path.join(cfg.logDir, `${cfg.runId}.jsonl`);
  const logger = new JsonlLogger(logPath);

  logger.log({ event: "run_start", msg: "Starting run", meta: { cfg, baseline: BASELINE } });
  await initTf(logger);

  const ablation = buildAblations();

  try {
    await runAblation(cfg, ablation);
    logger.log({ event: "run_end", msg: "All done", meta: { logPath } });
  } catch (err: any) {
    logger.log({ event: "run_error", msg: "Failed", meta: { message: err?.message, stack: err?.stack } });
    process.exitCode = 1;
  } finally {
    logger.close();
  }
}

void main();