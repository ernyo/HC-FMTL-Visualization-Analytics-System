import * as d3 from "d3";
import type { FederatedClient, ClientMetrics } from "../federated/client";
import { humanReadable } from "../utils/helpers";
import { DataConfig } from "../datasets/dataloader";
import { Shape } from "../datasets/constants";

const MAX_TASK_ROWS = 5;
const CONFIG_ROWS: Array<{ label: string; key: ConfigKey }> = [
  { label: "Seed", key: "seed" },
  { label: "Number of samples", key: "numSamples" },
  { label: "Number of shapes", key: "numShapes" },
  { label: "Shape types", key: "typeShapes" },
  { label: "Resolution", key: "resolution" },
  { label: "Batch size", key: "batchSize" },
  { label: "Percentage train data", key: "percTrain" },
];
const SHAPE_ORDER = ["circle", "square", "triangle", "star"] as const;
const SHAPE_SYMBOLS: Record<string, string> = {
  circle: "●",
  square: "■",
  triangle: "▲",
  star: "★",
};

type ConfigKey = keyof DataConfig;
type ConfigKV = { key: ConfigKey; label: string; value: string[] | number };


export class ConfigView {
  private container: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;

  constructor(
    private onSelect: (client: FederatedClient) => void,
    private onRemove: (client: FederatedClient) => Promise<void>,
    private getActiveClient: () => FederatedClient | null,
    private getMetrics?: (client: FederatedClient) => ClientMetrics | null | undefined
  ) {
    this.container = d3.select<HTMLDivElement, unknown>("#clients");
  }

  render(clients: FederatedClient[]) {
    const mismatchedKeys = this.findMismatchedKeys(clients);
    const cards = this.container
      .selectAll<HTMLDivElement, FederatedClient>("div.client-card")
      .data(clients, d => d.id);

    cards.exit().remove();

    const enter = cards.enter()
      .append("div")
      .attr("class", "client-card")
      .on("click", (_, client) => {
        if (this.getActiveClient()?.id === client.id) return;
        this.onSelect(client);
      });

    enter.append("button")
      .attr("class", "close")
      .attr("type", "button")
      .text("×")
      .on("click", async (event, client) => {
        event.stopPropagation();
        await this.onRemove(client);
      });


    // data config
    const configTable = enter.append("div").attr("class", "config-table");
    configTable.selectAll("div.row.config-row")
      .data(CONFIG_ROWS, (d: ConfigKV) => d.key)
      .enter()
      .append("div")
      .attr("class", "row config-row")
      .each(function (d) {
        const row = d3.select(this);
        row.append("div").attr("class", "cell").append("label").text(d.label);
        row.append("div").attr("class", "cell val");
      });

    // stats table container
    const statsTable = enter.append("div").attr("class", "table");

    // header row (matches CSS: .row.header)
    const header = statsTable.append("div").attr("class", "row header");
    header.append("div").attr("class", "cell task").append("label").text("Task");
    header.append("div").attr("class", "cell loss").append("label").text("Loss");
    header.append("div").attr("class", "cell metric").append("label").text("Metric");

    // overall row (matches CSS: .row.overall)
    const overall = statsTable.append("div").attr("class", "row overall");
    overall.append("div").attr("class", "cell task").append("label").text("Overall");
    overall.append("div").attr("class", "cell val loss");
    overall.append("div").attr("class", "cell val metric");


    const merged = enter.merge(cards as any);
    
    merged.select("button.close").style("display", clients.length > 1 ? null : "none");

    merged.classed("active", d => this.getActiveClient()?.id === d.id);

    merged.each((client, idx, nodes) => {
      const root = d3.select(nodes[idx]);

      // update configurable data
      const configRows = this.getConfigRows(client.state.data);

      const configSel = root
        .select(".config-table")
        .selectAll<HTMLDivElement, ConfigKV>("div.row.config-row")
        .data(configRows, d => d.key);
      configSel.exit().remove();

      const configEnter = configSel
        .enter()
        .append("div")
        .attr("class", "row config-row");

      configEnter.append("div").attr("class", "cell label");
      configEnter.append("div").attr("class", "cell val");

      const configMerged = configEnter.merge(configSel as any);

      // Add a warning class if this key differs across clients
      configMerged.classed("mismatch", d => mismatchedKeys.has(d.key));

      configMerged
        .select(".cell:not(.val)")
        .html("") // clear
        .append("label")
        .text(d => d.label);

      configMerged.select<HTMLDivElement>(".cell.val")
        .each(function (d: any) {
          const cell = d3.select(this);

          // Clear previous contents (important when switching between text and spans)
          cell.selectAll("*").remove();

          if (d.key === "typeShapes") {
            const raw = Array.isArray(d.value) ? (d.value as Shape[]) : [];

            // Normalize order: only keep shapes that are present, in fixed order
            const present = new Set(raw);
            const shapes = SHAPE_ORDER.filter(s => present.has(s));

            cell.selectAll("span.shape")
              .data(shapes, (s: Shape) => s)
              .join("span")
              .attr("class", s => `shape ${s}`)
              .text(s => SHAPE_SYMBOLS[s]);
          } else {
            cell.text(String(d.value));
          }
        });
 
      // update training metrics
      const m = this.getMetrics?.(client) ?? null;

      const rowsData = this.rowModel(m);

      const rowsSel = root.select(".table")
        .selectAll<HTMLDivElement, any>("div.row.task-row")
        .data(rowsData, (_d, i) => String(i)); // fixed keys 0..4

      rowsSel.exit().remove();

      const rowEnter = rowsSel.enter()
        .append("div")
        .attr("class", "row task-row");

      rowEnter.append("div").attr("class", "cell task");
      rowEnter.append("div").attr("class", "cell val loss");
      rowEnter.append("div").attr("class", "cell val metric");

      const rowMerged = rowEnter.merge(rowsSel as any);

      rowMerged.select(".cell.task")
        .text(d => d ? String(d.task).toUpperCase() : "—");

      rowMerged.select(".cell.val.loss")
        .text(d => d ? `${humanReadable(d.trainLoss)}/${humanReadable(d.testLoss)}` : "—");

      rowMerged.select(".cell.val.metric")
        .text(d => d ? `${humanReadable(d.trainMetric)}/${humanReadable(d.testMetric)}` : "—");

      // overall row stays separate (always exists)
      root.select(".row.overall .cell.val.loss")
        .text(m ? `${humanReadable(m.lossTrain)}/${humanReadable(m.lossTest)}` : "—");
      root.select(".row.overall .cell.val.metric")
        .text("—");
    });
  }

  private canonicalConfigValue(key: ConfigKey, value: unknown): string {
    // Make typeShapes order stable for comparison
    if (key === "typeShapes" && Array.isArray(value)) {
      const present = new Set(value as Shape[]);
      return SHAPE_ORDER.filter(s => present.has(s)).join("|"); // stable string
    }

    if (Array.isArray(value)) return value.join("|");
    return String(value);
  }

  private findMismatchedKeys(clients: FederatedClient[]): Set<ConfigKey> {
    const mismatched = new Set<ConfigKey>();
    if (clients.length <= 1) return mismatched;

    for (const { key } of CONFIG_ROWS) {
      const first = this.canonicalConfigValue(key, clients[0].state.data[key]);

      for (let i = 1; i < clients.length; i++) {
        const curr = this.canonicalConfigValue(key, clients[i].state.data[key]);
        if (curr !== first) {
          mismatched.add(key);
          break;
        }
      }
    }
    return mismatched;
  }

  private rowModel(m?: ClientMetrics | null) {
    const perHead = m?.perHead ?? [];
    // pad to 5 rows so layout stays fixed
    return Array.from({ length: MAX_TASK_ROWS }, (_, i) => perHead[i] ?? null);
  }

  private getConfigRows(dataConfig: DataConfig): ConfigKV[] {
    return CONFIG_ROWS.map(r => ({
      ...r,
      value: dataConfig[r.key],
    }));
  }
}