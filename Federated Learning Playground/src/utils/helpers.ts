import { TaskName } from "../models/model";
import { Shape } from "../datasets/constants";
import { Metric } from "../models/metrics";

const COLORS = {
  circle: [255, 0, 0],
  square: [0, 255, 0],
  triangle: [0, 0, 255],
  star: [255, 255, 0],
}

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

export function drawThumbnail(canvas: HTMLCanvasElement, shape: Shape) {
  const size = 36; // pick whatever looks good
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // background
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size, size);

  // shape fill

  const [r, g, b] = (COLORS as any)[shape] ?? [200, 200, 200];
  ctx.fillStyle = `rgb(${r},${g},${b})`;

  const cx = size / 2;
  const cy = size / 2;

  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.28, 0, Math.PI * 2);
    ctx.fill();
  }

  if (shape === "square") {
    const s = size * 0.56;
    ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
  }

  if (shape === "triangle") {
    const s = size * 0.62;
    ctx.beginPath();
    ctx.moveTo(cx, cy - s * 0.55);
    ctx.lineTo(cx - s * 0.5, cy + s * 0.35);
    ctx.lineTo(cx + s * 0.5, cy + s * 0.35);
    ctx.closePath();
    ctx.fill();
  }

  if (shape === "star") {
    const spikes = 5;
    const outerR = size * 0.30;
    const innerR = size * 0.14;
    let rot = -Math.PI / 2;
    const step = Math.PI / spikes;

    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
    for (let i = 0; i < spikes; i++) {
      rot += step;
      ctx.lineTo(cx + Math.cos(rot) * innerR, cy + Math.sin(rot) * innerR);
      rot += step;
      ctx.lineTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
    }
    ctx.closePath();
    ctx.fill();
  }
}
