import * as d3 from "d3";
import { humanReadable } from "../utils/helpers";
import { LineChart } from "../visualizations/linechart";

export interface ServerMetrics {
  meanDeltaNorm: number;
  meanClientDeltaNorm: number;
  meanDistToMean: number;
  meanCosine: number;
  fracNegativeCosine: number;
}

type MetricKey = keyof ServerMetrics;

const METRICS: readonly MetricKey[] = [
  "meanDeltaNorm",
  "meanClientDeltaNorm",
  "meanDistToMean",
  "meanCosine",
  "fracNegativeCosine",
] as const;

export class ServerView {
  private container: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;

  // quick access to DOM nodes per metric
  private valueContainer = new Map<MetricKey, d3.Selection<HTMLElement, unknown, HTMLElement, any>>();
  private chartContainer = new Map<MetricKey, d3.Selection<HTMLDivElement, unknown, HTMLElement, any>>();

  // charts + history per metric
  private charts = new Map<MetricKey, LineChart>();

  constructor() {
    this.container = d3.select<HTMLDivElement, unknown>("#server-metrics");

    this.bindElements();
    this.createCharts();
    this.reset();
  }

  private bindElements() {
    // Each metric lives inside: .output-stats > ... > .value and .linechart[data-metric]
    const rows = this.container.selectAll<HTMLDivElement, unknown>(".server-chart");

    rows.each((_, i, nodes) => {
      const row = d3.select(nodes[i]);

      const chartDiv = row.select<HTMLDivElement>(".linechart");
      const key = chartDiv.attr("data-metric") as MetricKey | null;

      if (!key) return;

      // store value span + chart div for this metric
      this.valueContainer.set(key, row.select<HTMLElement>(".value"));
      this.chartContainer.set(key, chartDiv);
    });
  }

  private createCharts() {
    for (const k of METRICS) {
      const container = this.chartContainer.get(k);
      if (!container) continue;

      // clear any prior svg if hot-reloading
      container.selectAll("*").remove();
      // Your LineChart takes a container selection; adjust if it expects a node instead
      const chart = new LineChart(container, ["#777"]);
      this.charts.set(k, chart);
    }
  }

  reset() {
    // reset text + history + charts
    for (const k of METRICS) {
      this.valueContainer.get(k)?.text("—");
      this.charts.get(k)?.reset();
    }
  }

  update(m: ServerMetrics | null | undefined) {
    // Update text values
    for (const k of METRICS) {
      const v = m?.[k];
      this.valueContainer.get(k)?.text(humanReadable(v) ?? "—");
    }

    // update charts
    for (const k of METRICS) {
      if (m?.[k] !== undefined) {
        this.charts.get(k)?.addDataPoint([m[k]]);
      }
    }
  }
}
