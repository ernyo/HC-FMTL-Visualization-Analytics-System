import { TaskName } from "../models/model.js";
import { Metric } from "../models/metrics.js";

export function isFiniteNumber(x: any): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

export function mean(nums: number[]) {
  if (!nums.length) return undefined;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

export function formatNum(x?: number) {
  return isFiniteNumber(x) ? x.toFixed(4) : "—";
}

export function formatDelta(delta?: number) {
  if (!isFiniteNumber(delta)) return "—";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(4)}`;
}

export function humanReadable(n: number): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return n.toFixed(2);
}

// Computes Δm using the provided per-task metrics.
export function deltaM(
  fed: Record<TaskName, number>,
  local: Record<TaskName, number>,
  taskMetricMap: Partial<Record<TaskName, Metric>>,
  lowerIsBetter: Record<Metric, boolean>
): number | null {
  const tasks = Object.keys(local).filter(t => Number.isFinite(local[t]) && Number.isFinite(fed[t]));
  if (tasks.length === 0) return null;

  let sum = 0;
  let N = 0;

  for (const t of tasks) {
    const MLocal = local[t];
    const MFed = fed[t];

    // guard: avoid divide-by-zero / tiny baseline
    if (!Number.isFinite(MLocal) || !Number.isFinite(MFed) || Math.abs(MLocal) < 1e-12) continue;

    const sign = lowerIsBetter[taskMetricMap[t]] ? -1 : 1;
    sum += sign * ((MFed - MLocal) / MLocal);
    N += 1;
  }

  return N > 0 ? sum / N : null;
}