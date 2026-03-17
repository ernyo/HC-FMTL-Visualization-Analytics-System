import * as d3 from "d3";
import type { ExperimentRun, Experiment } from "../drivers/experiment";
import type { EncoderAgg, DecoderAgg } from "../federated/aggregate";
import { isFiniteNumber } from "../utils/helpers";

export class ExperimentView {
  private view = d3.select<HTMLDivElement, unknown>("#experiment-view");
  private cards = d3.select<HTMLDivElement, unknown>("#experiments");
  private globalControls = d3.selectAll<HTMLDivElement, unknown>(".global");

  constructor(
    private experimentRun: ExperimentRun,
  ) {}

  show() { 
    this.view.style("display", null); 
    this.globalControls.classed("experiment-mode", true);
  }
  hide() { 
    this.view.style("display", "none"); 
    this.globalControls.classed("experiment-mode", false);
  }

  render() {
    const data = this.experimentRun.experiments;

    const cardSel = this.cards
      .selectAll<HTMLDivElement, Experiment>("div.experiment-card")
      .data(data, d => d.id);

    cardSel.exit().remove();

    const enter = cardSel.enter()
      .append("div")
      .attr("class", "experiment-card");

    // top row: title
    const top = enter.append("div").attr("class", "top");
    
    // left label
    top.append("div").attr("class", "label");

    // right: status + download
    const statusbar = top.append("div").attr("class", "statusbar");

    const pill = statusbar.append("div").attr("class", "status-pill not-started");
    pill.append("span").attr("class", "text").text("Not started");

    statusbar.append("button")
      .attr("class", "btn-download")
      .attr("type", "button")
      .text("Download")
      .style("display", "none"); // only show on completed

    // controls grid
    const grid = enter.append("div").attr("class", "grid");

    // Encoder agg
    const enc = grid.append("div").attr("class", "experiment-control");
    enc.append("div").attr("class", "label").text("Encoder aggregation");
    enc.append("div").attr("class", "select").append("select").attr("class", "encoder-agg")

    // CA C
    const cac = grid.append("div").attr("class", "experiment-control cac");
    cac.append("div").attr("class", "label").text("CA C");
    cac.append("div").attr("class", "select").append("select").attr("class", "cac");

    // Decoder agg
    const dec = grid.append("div").attr("class", "experiment-control");
    dec.append("div").attr("class", "label").text("Decoder aggregation");
    dec.append("div").attr("class", "select").append("select").attr("class", "decoder-agg");

    // results
    const results = enter.append("div").attr("class", "results");
    results.append("div").attr("class", "row delta");
    results.append("div").attr("class", "row time");

    // merge
    const merged = enter.merge(cardSel as any);
    merged.classed("is-baseline", (_d, i) => i === 0);

    // label "Experiment N"
    merged.select(".top .label").html(""); // clear
    merged.select(".top .label")
      .append("span")
      .attr("class", "badge")
      .classed("badge-baseline", (_d, i) => i === 0)
      .classed("badge-variant", (_d, i) => i !== 0)
      .text((_d, i) => (i === 0 ? "Baseline" : `Experiment ${i}`));


    // Compute baseline once (for Δm)
    const baseline = data[0];
    const baseTask = baseline.finalTaskMetricsFromLog();
    const baselineDone = baseline.status === "Completed";

    // Populate selects + wire updates with correct experiment id
    merged.each((exp, i, nodes) => {
      const root = d3.select(nodes[i]);

      // status pill
      const pill = root.select<HTMLDivElement>(".status-pill");
      pill.classed("not-started", exp.status === "Not started")
        .classed("in-progress", exp.status === "In progress")
        .classed("completed", exp.status === "Completed")
        .classed("failed", exp.status === "Failed");

      pill.select(".text").text(() => {
        if (exp.status === "In progress") return "In progress";
        if (exp.status === "Completed") return "Completed";
        if (exp.status === "Failed") return "Failed";
        return "Not started";
      });

      // download button
      const dl = root.select<HTMLButtonElement>("button.btn-download");
      dl.style("display", exp.status === "Completed" ? null : "none");
      dl.on("click", (event) => {
        event.stopPropagation();
        exp.downloadLog(); // or exp.downloadLog() if you prefer full log
      });

      // Encoder options
      const encSel = root.select<HTMLSelectElement>("select.encoder-agg");
      encSel.selectAll("option").remove();
      encSel.append("option").attr("value", "none").text("None");
      encSel.append("option").attr("value", "fedavg").text("FedAvg");
      encSel.append("option").attr("value", "conflict_averse").text("Conflict Averse");

      encSel.property("value", exp.plan.encoderAgg);
      encSel.on("change", (event) => {
        const v = (event.currentTarget as HTMLSelectElement).value as EncoderAgg;
        this.experimentRun.updateExperiment(exp.id, { encoderAgg: v });
        const enabled = v === "conflict_averse";
        this.setCacEnabled(root as any, enabled);
      });

      // Decoder options
      const decSel = root.select<HTMLSelectElement>("select.decoder-agg");
      decSel.selectAll("option").remove();
      decSel.append("option").attr("value", "none").text("None");
      decSel.append("option").attr("value", "fedavg").text("FedAvg");
      decSel.append("option").attr("value", "cross_attention").text("Cross Attention");

      decSel.property("value", exp.plan.decoderAgg);
      decSel.on("change", (event) => {
        const v = (event.currentTarget as HTMLSelectElement).value as DecoderAgg;
        this.experimentRun.updateExperiment(exp.id, { decoderAgg: v });
      });

      // CA C options (matching your existing dropdown values)
      const cacEnabled = exp.plan.encoderAgg === "conflict_averse";
      this.setCacEnabled(root as any, cacEnabled);

      const cacSel = root.select<HTMLSelectElement>("select.cac");
      const cacVals = ["0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"];
      cacSel.selectAll("option").remove();
      cacSel.selectAll("option")
        .data(cacVals)
        .enter()
        .append("option")
        .attr("value", d => d)
        .text(d => d);

      cacSel.property("value", String(exp.plan.cac));
      cacSel.on("change", (event) => {
        const v = Number((event.currentTarget as HTMLSelectElement).value);
        this.experimentRun.updateExperiment(exp.id, { cac: v });
      });

      // ----- Results -----
      const s = exp.summarizeExperiment(i > 0 && baselineDone ? baseTask : null);
      const deltaText = i === 0 ? "Reference" : s.deltaM === null ? "Δm —" : `Δm ${(s.deltaM * 100).toFixed(2)}%`;
      const elapsedSec = exp.status === "In progress" && isFiniteNumber(exp.startTime)
        ? (performance.now() - exp.startTime) / 1000
        : (s.done && isFiniteNumber(s.totalTime) ? s.totalTime : null);

      root.select(".results .row.delta")
        .html("")
        .call(sel => {
          const row = sel as any;
          row.append("div").attr("class", "k").text("Delta vs. baseline");
          row.append("div").attr("class", "v").text(deltaText);
        });

      root.select(".results .row.time")
        .html("")
        .call(sel => {
          const row = sel as any;
          row.append("div").attr("class", "k").text("Time elapsed");
          row.append("div").attr("class", "v")
            .text(elapsedSec == null ? "—" : `${elapsedSec.toFixed(2)}s`);
        });
    });
  }

  private setCacEnabled(root: d3.Selection<HTMLElement, unknown, null, undefined>, enabled: boolean) {
    root.select(".experiment-control.cac").style("visibility", enabled ? null : "hidden").style("pointer-events", enabled ? null : "none");
  }

}
