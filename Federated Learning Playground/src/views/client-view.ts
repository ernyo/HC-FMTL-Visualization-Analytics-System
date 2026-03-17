import * as d3 from "d3";
import { FederatedClient } from "../federated/client";
import { LineChart } from "../visualizations/linechart";
import { HeatMap } from "../visualizations/heatmap";
import { Head, Link, Encoder, Decoder, MTLModel, TaskName } from "../models/model";
import { losses, metrics } from "../models/state";
import { NUM_TO_SHOW } from "../main";
import { humanReadable, drawThumbnail } from "../utils/helpers";
import { Shape, SHAPE_UNIVERSE } from "../datasets/constants";

const MARGIN = 8;
const RECT_SIZE = 50;
const BIAS_SIZE = 5;
enum HoverType {
  BIAS, WEIGHT
}

let linkWidthScale = d3.scaleLinear()
                        .domain([0, 5])
                        .range([1, 10])
                        .clamp(true);
let colorScale = d3.scaleLinear<string, number>()
                    .domain([-1, 0, 1])
                    .range(["#f59322", "#e8eaeb", "#0877bd"])
                    .clamp(true);

export class ClientView {
  private root = d3.select<HTMLDivElement, unknown>("#main-container").append("div").style("display", "none");
  private visible = false;

  // per-client UI objects (cached)
  private totalLossChart!: LineChart;
  private taskLossCharts: LineChart[] = [];
  private taskMetricCharts: LineChart[] = [];

  private totalLossSeries: Array<[number, number]> = [];       // [train,test]

  constructor(private client: FederatedClient, private globalReset: () => Promise<void>) {
    this.buildSkeleton();
    this.reset();
    this.update();
  }

  select<T extends d3.BaseType = any>(selector: string) {
    return this.root.select<T>(selector);
  }

  show() {
    if (this.visible) return;
    this.visible = true;
    this.root.style("display", null);
    this.drawNetwork();
    this.createLineCharts();
    this.renderAll();
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.root.style("display", "none");
  }

  /** Full rebuild (e.g., taskConfig changed order/count). */
  reset() {
    this.totalLossSeries = [];
    this.createTaskPanel();
    if (this.visible) {
      this.createLineCharts();
      this.drawNetwork();
      this.renderAll();
    }
  }

  /** Lightweight refresh after training step. */
  update() {
    this.updateWeightsUI();
    this.updateBiasesUI();
    this.renderAll();
  }

  /** For window resize: recompute sizes and redraw network; charts often resize too. */
  resize() {
    // simplest: redraw network and update
    this.drawNetwork();
    this.update();
  }

  // ------------------------
  // Initialization
  // ------------------------
  private buildSkeleton() {
    // Main Part
    const main = this.root
      .attr("id", "main-part")
      .attr("class", `l--page client-view ${this.client.id}`);

    // -------------------------
    // Training Config
    // -------------------------
    
    // <div class="container l--page">
    //     <div class="control ui-learningRate">
    //       <label for="learningRate">Learning rate</label>
    //       <div class="select">
    //         <select id="learningRate">
    //           <option value="0.00001">0.00001</option>
    //           <option value="0.0001">0.0001</option>
    //           <option value="0.001">0.001</option>
    //           <option value="0.003">0.003</option>
    //           <option value="0.01">0.01</option>
    //           <option value="0.03">0.03</option>
    //           <option value="0.1">0.1</option>
    //           <option value="0.3">0.3</option>
    //           <option value="1">1</option>
    //           <option value="3">3</option>
    //           <option value="10">10</option>
    //         </select>
    //       </div>
    //     </div>

    //     <div class="control ui-activation">
    //       <label for="activations">Activation</label>
    //       <div class="select">
    //         <select id="activations">
    //           <option value="relu">ReLU</option>
    //           <option value="tanh">Tanh</option>
    //           <option value="sigmoid">Sigmoid</option>
    //           <option value="linear">Linear</option>
    //         </select>
    //       </div>
    //     </div>

    //     <div class="control ui-regularization">
    //       <label for="regularizations">Regularization</label>
    //       <div class="select">
    //         <select id="regularizations">
    //           <option value="none">None</option>
    //           <option value="L1">L1</option>
    //           <option value="L2">L2</option>
    //         </select>
    //       </div>
    //     </div>

    //     <div class="control ui-regularizationRate">
    //       <label for="regularRate">Regularization rate</label>
    //       <div class="select">
    //         <select id="regularRate">
    //           <option value="0">0</option>
    //           <option value="0.001">0.001</option>
    //           <option value="0.003">0.003</option>
    //           <option value="0.01">0.01</option>
    //           <option value="0.03">0.03</option>
    //           <option value="0.1">0.1</option>
    //           <option value="0.3">0.3</option>
    //           <option value="1">1</option>
    //           <option value="3">3</option>
    //           <option value="10">10</option>
    //         </select>
    //       </div>
    //     </div>
    //   </div>
    // </div>

    // -------------------------
    // Data Column
    // -------------------------
    const dataCol = main.append("div").attr("class", "column data");
    dataCol.append("h4").append("span").text("Data");

    // seed
    const seed = dataCol.append("div").attr("class", "ui-seed");
    seed.append("label").text("Seed number:");
    seed.append("input")
      .attr("type", "number")
      .attr("id", "seed")
    seed.append("button")
      .attr("id", "random-seed")
      .attr("class", "basic-button")
      .text("Randomize");


    // typeShapes
    const typeShapes = dataCol.append("div").attr("class", "ui-dataset");
    typeShapes.append("p")
      .text("Which shapes do you want to use?");
    const container = typeShapes.append("div").attr("class", "dataset-list");
    SHAPE_UNIVERSE.forEach(shape => {
      const canvas = container
        .append("div")
        .attr("class", "dataset")
        .append("canvas")
        .attr("class", "data-thumbnail")
        .attr("data-dataset", shape.toLowerCase())
        .classed("selected", true)
        .node() as HTMLCanvasElement;
      drawThumbnail(canvas, shape);
      container.append("input")
        .attr("type", "number")
        .attr("shape-prob", shape.toLowerCase())
        .attr("min", "0")
        .attr("max", "1")
        .attr("step", "0.01")
        .style("width", "72px")
    });

    // nShapes
    const nShapes = dataCol.append("div").attr("class", "ui-shapesCount");
    nShapes.append("label")
      .attr("for", "nShapes")
      .html(`Number of shapes:&nbsp;&nbsp;<span class="value">1</span>`);
    nShapes.append("p").attr("class", "slider")
      .append("input")
      .attr("class", "mdl-slider mdl-js-slider")
      .attr("id", "nShapes")
      .attr("type", "range")
      .attr("min", 1)
      .attr("max", 10)
      .attr("step", 1)
      .attr("value", 1);
  
    // resolution
    const resolution = dataCol.append("div").attr("class", "ui-resolution");
    resolution.append("label")
      .attr("for", "resolution")
      .html(`Pixel resolution:&nbsp;&nbsp;<span class="value">32</span>`);
    resolution.append("p").attr("class", "slider")
      .append("input")
      .attr("class", "mdl-slider mdl-js-slider")
      .attr("id", "resolution")
      .attr("type", "range")
      .attr("min", 0)
      .attr("max", 2)
      .attr("step", 1)
      .attr("value", 0);

    // percTrain
    const percTrain = dataCol.append("div").attr("class", "ui-percTrainData");
    percTrain.append("label")
      .attr("for", "percTrainData")
      .html(`Ratio of training to test data:&nbsp;&nbsp;<span class="value">XX</span>%`);
    percTrain.append("p").attr("class", "slider")
      .append("input")
      .attr("class", "mdl-slider mdl-js-slider")
      .attr("type", "range")
      .attr("id", "percTrainData")
      .attr("min", 10)
      .attr("max", 90)
      .attr("step", 10);

    // batchSize
    const batchSize = dataCol.append("div").attr("class", "ui-batchSize");
    batchSize.append("label")
      .attr("for", "batchSize")
      .html(`Batch size:&nbsp;&nbsp;<span class="value">8</span>`);
    batchSize.append("p").attr("class", "slider")
      .append("input")
      .attr("class", "mdl-slider mdl-js-slider")
      .attr("id", "batchSize")
      .attr("type", "range")
      .attr("min", 0)
      .attr("max", 3)
      .attr("step", 1)
      .attr("value", 0);

    // -------------------------
    // Task Panels (Config Column)
    // -------------------------
    const configCol = main.append("div").attr("class", "column config");

    const configH4 = configCol.append("h4");
    
    const numTasks = configH4.append("div").attr("class", "ui-numTasks");
    numTasks.append("button")
    .attr("id", "add-tasks")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .append("i")
    .attr("class", "material-icons")
    .text("add");
    
    numTasks.append("button")
    .attr("id", "remove-tasks")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .append("i")
    .attr("class", "material-icons")
    .text("remove");
    
    configH4.append("span").attr("id", "num-tasks");
    configH4.append("span").attr("id", "tasks-label").attr("class", "heading");

    configCol.append("div")
      .attr("class", "control-container")
      .attr("style", "margin-top: -8px;");

    // -------------------------
    // Architecture Column
    // -------------------------
    const archCol = main.append("div").attr("class", "column architecture");

    archCol.append("h4")
      .append("span")
      .text("Architecture");

    const featuresCol = archCol.append("div").attr("class", "column features");

    const network = featuresCol.append("div").attr("class", "network");
    const svg = network.append("svg").attr("class", "svg");

    const defs = svg.append("defs");
    const marker = defs.append("marker")
      .attr("id", `markerArrow${this.client.id}`)
      .attr("markerWidth", 7)
      .attr("markerHeight", 13)
      .attr("refX", 1)
      .attr("refY", 6)
      .attr("orient", "auto")
      .attr("markerUnits", "userSpaceOnUse");

    marker.append("path")
      .attr("d", "M2,11 L7,6 L2,2");

    // Hover card
    const hover = network.append("div").attr("id", "hovercard")

    hover.append("div")
      .style("font-size", "10px")
      .text("Click anywhere to edit.");

    const hoverLine = hover.append("div");
    hoverLine.append("span").attr("class", "type").text("Weight/Bias");
    hoverLine.append("span").text(" is ");
    hoverLine.append("span").attr("class", "value").text("0.2");
    hoverLine.append("span").append("input").attr("type", "number").style("display", "none");

    // // Callouts
    // const calloutThumb = network.append("div")
    //   .attr("class", `callout thumbnail`)

    // const thumbSvg = calloutThumb.append("svg").attr("viewBox", "0 0 30 30");
    // const thumbDefs = thumbSvg.append("defs");
    // thumbDefs.append("marker")
    //   .attr("id", `arrow`)
    //   .attr("markerWidth", 5)
    //   .attr("markerHeight", 5)
    //   .attr("refx", 5)
    //   .attr("refy", 2.5)
    //   .attr("orient", "auto")
    //   .attr("markerUnits", "userSpaceOnUse")
    //   .append("path").attr("d", "M0,0 L5,2.5 L0,5 z");

    // thumbSvg.append("path")
    //   .attr("d", "M12,30C5,20 2,15 12,0")
    //   .attr("marker-end", `url(#arrow)`);

    // calloutThumb.append("div").attr("class", "label")
    //   .html(`This is the output from one <b>head</b>. Hover to see it larger.`);

    // const calloutWeights = network.append("div")
    //   .attr("class", `callout weights`)

    // const wSvg = calloutWeights.append("svg").attr("viewBox", "0 0 30 30");
    // const wDefs = wSvg.append("defs");
    // wDefs.append("marker")
    //   .attr("id", `arrow`)
    //   .attr("markerWidth", 5)
    //   .attr("markerHeight", 5)
    //   .attr("refx", 5)
    //   .attr("refy", 2.5)
    //   .attr("orient", "auto")
    //   .attr("markerUnits", "userSpaceOnUse")
    //   .append("path").attr("d", "M0,0 L5,2.5 L0,5 z");

    // wSvg.append("path")
    //   .attr("d", "M12,30C5,20 2,15 12,0")
    //   .attr("marker-end", `url(#arrow)`);

    // calloutWeights.append("div").attr("class", "label")
    //   .html(`The outputs are mixed with varying <b>weights</b>, shown by the thickness of the lines.`);

    // -------------------------
    // Output Column
    // -------------------------
    const outCol = main.append("div").attr("class", "column output");
    outCol.append("h4").text("Output");
    outCol.append("div").attr("class", "heatmaps");

    // -------------------------
    // Stats Column
    // -------------------------
    const statsCol = main.append("div").attr("class", "column stats");
    statsCol.append("h4").text("Statistics");
    statsCol.append("div").attr("class", "charts");
  }

  private createLineCharts() {
    // reset per-client charts
    this.taskLossCharts = [];
    this.taskMetricCharts = [];

    const charts = this.root.select(".charts");
    charts.selectAll("*").remove();

    // overall chart area
    const statsContainer = charts.append("div").attr("class", "task-charts");
    const overall = charts.append("div").attr("class", "metrics total-loss");

    const trainStats = overall.append("div").attr("class", "output-stats train");
    trainStats.append("span").text("Total train loss");
    trainStats.append("span").text("\u00A0");
    trainStats.append("div").attr("class", "value").attr("id", "loss-train");

    const testStats = overall.append("div").attr("class", "output-stats ui-percTrainData");
    testStats.append("span").text("Total test loss");
    testStats.append("span").text("\u00A0");
    testStats.append("div").attr("class", "value loss-test").attr("id", "loss-test");
    
    const linechart = overall.append("div").attr("class", "linechart");
    this.totalLossChart = new LineChart(linechart, ["#777", "black"]);

    this.client.model.heads.forEach(head => {
      const container = statsContainer.append("div")
        .attr("class", "task-chart")
        .attr("data-head", head.name)
        .style("display", "grid")
        .style("grid-template-columns", "1fr 1fr")
        .style("grid-gap", `20px`)
        .style("margin-bottom", `${MARGIN}px`)
        .style("height", `${2 * RECT_SIZE}px`);

      // Loss panel
      const lossContainer = container.append("div").attr("class", "metrics loss");
      const lossTrain = lossContainer.append("div").attr("class", "output-stats train");
      lossTrain.append("span").text("Train loss");
      lossTrain.append("span").text("\u00A0");
      lossTrain.append("div").attr("class", "value loss-train");

      const lossTest = lossContainer.append("div").attr("class", "output-stats test");
      lossTest.append("span").text("Test loss");
      lossTest.append("span").text("\u00A0");
      lossTest.append("div").attr("class", "value loss-test");

      this.taskLossCharts.push(new LineChart(lossContainer, ["#777", "black"]));

      // Metric panel
      const metricContainer = container.append("div").attr("class", "metrics metric");
      const metTrain = metricContainer.append("div").attr("class", "output-stats train");
      metTrain.append("span").text("Train metric");
      metTrain.append("span").text("\u00A0");
      metTrain.append("div").attr("class", "value metric-train");

      const metTest = metricContainer.append("div").attr("class", "output-stats test");
      metTest.append("span").text("Test metric");
      metTest.append("span").text("\u00A0");
      metTest.append("div").attr("class", "value metric-test");

      this.taskMetricCharts.push(new LineChart(metricContainer, ["#777", "black"]));
    });
  }

  private drawEncoderDecoder(
    cx: number,
    cy: number,
    node: Encoder | Decoder,
    container: d3.Selection<SVGGElement, unknown, HTMLElement, any>
  ): void {
    const width  = RECT_SIZE;
    const height = RECT_SIZE * 0.9;
    const cornerR = RECT_SIZE * 0.12;   // corner radius
    const topWidth  = width * 0.7;      // shorter side
    const bottomWidth = width;          // longer side = RECT_SIZE

    const isEncoder = node instanceof Encoder;
    const fillColor = isEncoder ? "#2196f3" : "#ff9800"; // blue / orange

    // Local coordinates before rotation
    let topLeftX: number, topRightX: number;
    let bottomLeftX: number, bottomRightX: number;

    if (isEncoder) {
      // encoder: narrow at top, wide at bottom
      topLeftX    = (width - topWidth) / 2;
      topRightX   = topLeftX + topWidth;
      bottomLeftX = 0;
      bottomRightX= bottomWidth;
    } else {
      // decoder: flipped → narrow at bottom, wide at top
      topLeftX    = 0;
      topRightX   = bottomWidth;
      bottomLeftX = (width - topWidth) / 2;
      bottomRightX= bottomLeftX + topWidth;
    }

    const topY = 0;
    const bottomY = height;

    // Rounded trapezoid path (unrotated, in 0..RECT_SIZE coords)
    const tlx = topLeftX,  tly = topY;
    const trx = topRightX, try_ = topY;
    const blx = bottomLeftX, bly = bottomY;
    const brx = bottomRightX, bry = bottomY;
    const r = cornerR;

    const d = [
      `M ${blx + r},${bly}`,
      `L ${brx - r},${bry}`,
      `Q ${brx},${bry} ${brx},${bry - r}`,
      `L ${trx},${try_ + r}`,
      `Q ${trx},${try_} ${trx - r},${try_}`,
      `L ${tlx + r},${tly}`,
      `Q ${tlx},${tly} ${tlx},${tly + r}`,
      `L ${blx},${bly - r}`,
      `Q ${blx},${bly} ${blx + r},${bly}`,
      "Z"
    ].join(" ");

    // Position group so its local (0,0) is top-left of the shape
    const x = cx - RECT_SIZE / 2;
    const y = cy - RECT_SIZE / 2;

    const nodeGroup = container.append("g")
      .attr("class", node.name)
      .attr("id", node.name)
      .attr("transform", `translate(${x},${y})`);

    // Rotate the trapezoid 90° clockwise around its center
    nodeGroup.append("path")
      .attr("d", d)
      .attr("fill", fillColor)
      .attr("transform", `rotate(90 ${RECT_SIZE / 2} ${RECT_SIZE / 2})`);

    // nodeGroup.append("text")
    //   .attr("x", RECT_SIZE / 2)
    //   .attr("y", -6)
    //   .attr("text-anchor", "middle")
    //   .text(node.name.toUpperCase())
    //   .style("font-size", "10px");
  }

  private drawHead(network: MTLModel, cx: number, cy: number, head: Head, container: d3.Selection<SVGGElement, unknown, HTMLElement, any>) {
    let x = cx - RECT_SIZE / 2;
    let y = cy - RECT_SIZE / 2;

    let headGroup = container.append("g")
      .attr("class", "head")
      .attr("id", `head-${this.client.id}-${head.name}`)
      .attr("transform", `translate(${x},${y})`);

    // Draw the node's bias.
    headGroup.append("rect")
      .attr("id", `bias-${this.client.id}-${head.name}`)
      .attr("x", -BIAS_SIZE - 2)
      .attr("y", RECT_SIZE - BIAS_SIZE + 3)
      .attr("width", BIAS_SIZE)
      .attr("height", BIAS_SIZE)
      .on("mouseenter", (event) => {
        const networkDiv = this.root.select<HTMLDivElement>(".network").node()!;
        const coords = d3.pointer(event, networkDiv);
        this.updateHoverCard(network, HoverType.BIAS, head.link, coords);
      })
      .on("mouseleave", () => {
        this.updateHoverCard(network, null);
      });

    // headGroup.append("text")
    //   .attr("x", RECT_SIZE / 2)
    //   .attr("y", -6)
    //   .attr("text-anchor", "middle")
    //   .text(head.name.toUpperCase())
    //   .style("font-size", "10px");

    // Draw the node's canvas.
    // Heatmap container replacing the node box
    this.root.select(".heatmaps")
      .append("div")
      .attr("id", `canvas-${this.client.id}-${head.name}`)
      .style("display", "grid")
      .style("grid-template-rows", "auto auto") // first row: preds, second row: GT
      .style("margin-bottom", `${MARGIN}px`)
  }

  private drawLink(
    input: Link, 
    node2coord: {[id: string]: {cx: number, cy: number}},
    network: MTLModel, 
    container,
  ) {
    let line = container.insert("path", ":first-child");
    let source = node2coord[input.source];
    let dest = node2coord[input.dest];
    let datum = {
      source: {
        x: source.cx + RECT_SIZE / 2,
        y: source.cy
      },
      target: {
        x: dest.cx - RECT_SIZE / 2 + 6,
        y: dest.cy
      }
    };
    const diagonal = d3.linkHorizontal<any, any>()
      .x(d => d.x)
      .y(d => d.y);
    line
      .attr("marker-start", `url(#markerArrow${this.client.id})`)
      .attr("class", "link")
      .attr("id", `link-${this.client.id}-${input.source}-${input.dest}`)
      .attr("d", diagonal(datum));

    // Add an invisible thick link that will be used for
    // showing the weight value on hover.
    container.append("path")
      .attr("d", diagonal(datum))
      .attr("class", "link-hover")
      .on("mouseenter", (event) => {
        const networkDiv = this.root.select<HTMLDivElement>(".network").node()!;
        const coords = d3.pointer(event, networkDiv);
        this.updateHoverCard(network, HoverType.WEIGHT, input, coords);
      })
      .on("mouseleave", () => {
        this.updateHoverCard(network, null);
      });

    return line;
  }

  private drawNetwork() {
    const svg = this.root.select("svg.svg");
    // Remove all svg elements.
    svg.select("g.core").remove();
    // Remove all div elements.
    this.root.select(".network").selectAll("div.canvas").remove();
    this.root.select(".heatmaps").selectAll("*").remove();

    // compute width using *scoped* columns
    const co = this.root.select(".column.output").node() as HTMLDivElement;
    const ch = this.root.select(".column.architecture").node() as HTMLDivElement;
    const width = co.offsetLeft - ch.offsetLeft;
    svg.attr("width", width);

    const network = this.client.model;

    // Map of all node coordinates.
    let node2coord: {[id: string]: {cx: number, cy: number}} = {};
    let container = svg.append("g").classed("core", true);

    // X-Coordinates
    const encoderX = RECT_SIZE / 2;
    const headX   = width;
    const decoderX = (encoderX + headX) / 2;

    // Y-Coordinates
    let tasks = this.client.state.model.taskConfig.filter(task => task.enabled);
    let numTasks = tasks.length;
    const nodeIndexScale = (idx: number) => {
      let totalHeight = RECT_SIZE / 2;
      for (let i = idx; i >= 1; i--) {
        const h = this.root.select(".control-container").select(`.control[data-head="${network.heads[i].name}"]`).node() as HTMLDivElement;
        totalHeight += h.offsetHeight + MARGIN;
      }
      return totalHeight;
    }

    // Center encoder/decoder vertically in the middle of the heads
    const encoder = network.encoder;
    let encoderY = 0;
    for (let i = 0; i < numTasks; i++) {
      encoderY += nodeIndexScale(i);
    }
    encoderY /= numTasks;
    node2coord[encoder.name] = { cx: encoderX, cy: encoderY };
    this.drawEncoderDecoder(encoderX, encoderY, encoder, container);

    // Draw the encoder and decoders
    const decoders = network.decoders;
    decoders.forEach((decoder, i) => {
      let decoderY = nodeIndexScale(i);
      node2coord[decoder.name] = { cx: decoderX, cy: decoderY};
      this.drawEncoderDecoder(decoderX, decoderY, decoder, container);
      this.drawLink(decoder.link, node2coord, network, container);
    });

    // Draw the network layer by layer.
    let calloutThumb = this.root.select(".callout.thumbnail").style("display", "none");
    let calloutWeights = this.root.select(".callout.weights").style("display", "none");
    let idWithCallout = null;
    let targetIdWithCallout = null;

    // Draw the heads
    network.heads.forEach((head, i) => {
      let headY = node2coord[decoders[i].name].cy;
      node2coord[head.name] = {cx: headX, cy: headY};
      this.drawHead(network, headX, headY, head, container);
      
      // Show callout to thumbnails.
      if (idWithCallout == null && i === numTasks - 1) {
        calloutThumb
          .style("display", null)
          .style("top", `${50 + headY}px`)
          .style("left", `${headX}px`);
        idWithCallout = head.name;
      }

      // Draw links.
      let path: SVGPathElement = this.drawLink(head.link, node2coord, network, container).node() as any;
      // Show callout to weights.
      if (targetIdWithCallout == null && i === numTasks - 1) {
        let midPoint = path.getPointAtLength(path.getTotalLength() * 0.6);
        calloutWeights
          .style("display", null)
          .style("top", `${midPoint.y + 20}px`)
          .style("left", `${midPoint.x - 35}px`);
        targetIdWithCallout = head.name;
      }
    });

    svg.attr("height", nodeIndexScale(numTasks - 1) + RECT_SIZE + MARGIN);
  }

  private createTaskPanel() {
    const container = this.root.select(".column.config").select(".control-container");
    container.selectAll("*").remove();

    const network = this.client.model;
    const numTasks = this.client.tasks.length;

    let suffix = numTasks !== 1 ? "s" : "";
    this.root.select("#tasks-label").text("Task" + suffix);
    this.root.select("#num-tasks").text(numTasks);

    network.heads.forEach(head => {
      const panel = container.append("div")
        .attr("class", "control")
        .attr("data-head", head.name)
        .style("height", `${2 * RECT_SIZE}px`)
        .style("margin-bottom", `${MARGIN}px`);
      const currentTaskConfig = this.client.state.model.taskConfig.find(task => head.name.includes(task.name))!;

      // TASK (which logical task this head is assigned to)
      const taskRow = panel.append("div").attr("class", "control-row");
      taskRow.append("label").text("Task");
      const taskWrapper = taskRow.append("div").attr("class", "select");

      const taskSelect = taskWrapper.append("select")
        .on("change", async (event) => {
          const newTask = (event.currentTarget as HTMLSelectElement).value as TaskName;
          const newConfig = this.client.state.model.taskConfig.find(task => task.name === newTask);
          newConfig.enabled = true;
          currentTaskConfig.enabled = false; 

          // switch order of tasks in array
          const currentIndex = this.client.state.model.taskConfig.indexOf(currentTaskConfig);
          const newIndex = this.client.state.model.taskConfig.indexOf(newConfig);
          this.client.state.model.taskConfig[currentIndex] = newConfig;
          this.client.state.model.taskConfig[newIndex] = currentTaskConfig;

          await this.globalReset();
        });

      const tasks = this.client.state.model.taskConfig.filter(task => !task.enabled || task.name === currentTaskConfig.name).map(task => task.name);
      tasks.forEach(task => {
        taskSelect.append("option")
          .attr("value", task)
          .property("selected", task === currentTaskConfig.name)
          .text(task.toUpperCase());
      });

      /* ---------- LOSS ROW ---------- */
      const lossRow = panel.append("div").attr("class", "control-row");
      lossRow.append("label").text("Loss");
      const lossWrapper = lossRow.append("div").attr("class", "select");

      const lossSelect = lossWrapper.append("select")
        .on("change", async (event) => {
          const key = (event.currentTarget as HTMLSelectElement).value;
          const value = losses[currentTaskConfig.name][key];
          currentTaskConfig.loss = value;
          await this.globalReset();
        });

      Object.entries(losses[currentTaskConfig.name]).forEach(([label, loss]) => {
        lossSelect.append("option")
          .attr("value", label)
          .property("selected", loss === currentTaskConfig.loss)
          .text(label);
      });

      /* ---------- WEIGHT ROW ---------- */
      const weightRow = panel.append("div").attr("class", "control-row");
      weightRow.append("label").text("Weight");

      // wrapper for slider + value
      const weightTrack = weightRow.append("div").attr("class", "weight-track");

      const weightValue = weightTrack.append("span")
        .attr("class", "value")
        .text(String(currentTaskConfig.lossWeight));

      weightTrack.append("input")
        .attr("type", "range")
        .attr("class", "mdl-slider")
        .attr("step", "1")
        .attr("min", "1")
        .attr("max", "5")
        .property("value", currentTaskConfig.lossWeight)
        .on("input", function () {
          const value = parseFloat((this as HTMLInputElement).value);
          if (!isNaN(value)) {
            weightValue.text(String(value));   // live update
          }
        })
        .on("change", async (event) => {
          const value = parseFloat((event.currentTarget as HTMLInputElement).value);
          if (!isNaN(value)) {
            currentTaskConfig.lossWeight = value;
            await this.globalReset();
          }
        });

      /* ---------- METRIC ROW ---------- */
      const metricRow = panel.append("div").attr("class", "control-row");
      metricRow.append("label").text("Metric");
      const metricWrapper = metricRow.append("div").attr("class", "select");

      const metricSelect = metricWrapper.append("select")
        .on("change", async (event) => {
          const key = (event.currentTarget as HTMLSelectElement).value;
          const value = metrics[currentTaskConfig.name][key];
          currentTaskConfig.metric = value;
          await this.globalReset();
        });

      Object.entries(metrics[currentTaskConfig.name]).forEach(([label, metric]) => {
        metricSelect.append("option")
          .attr("value", label)
          .property("selected", metric === currentTaskConfig.metric)
          .text(label);
      });
    });
  }

  // ------------------------
  //  For active view
  // ------------------------
  private updateBiasesUI() {
    this.client.model.heads.forEach(head => {
      this.root.select(`rect#bias-${this.client.id}-${head.name}`).style("fill", colorScale(head.link.bias));
    });
  }

  private updateWeightsUI() {
    const core = this.root.select("g.core"); // your svg group
    this.client.model.heads.forEach(head => {
      core.select(`#link-${this.client.id}-${head.link.source}-${head.link.dest}`)
        .style("stroke-dashoffset", -this.client.state.epochs / 3)
        .style("stroke-width", linkWidthScale(Math.abs(head.link.weight)))
        .style("stroke", colorScale(head.link.weight))
        .datum(head.link);
    });
  }

  private updateHoverCard(network: MTLModel, type: HoverType | null, link?: Link, coordinates?: [number, number]) {
    const hovercard = this.root.select<HTMLDivElement>(".hovercard");
    const svg = this.root.select<SVGSVGElement>("svg.svg");

    if (type == null) {
      hovercard.style("display", "none");
      svg.on("click", null);
      return;
    }

    const value = (type === HoverType.WEIGHT) ? link.weight : link.bias;
    const name  = (type === HoverType.WEIGHT) ? "Weight" : "Bias";

    hovercard
      .style("left", `${coordinates[0] + 20}px`)
      .style("top", `${coordinates[1]}px`)
      .style("display", "block");

    hovercard.select(".type").text(name);
    hovercard.select(".value")
      .style("display", null)
      .text(value.toPrecision(2));

    hovercard.select("input")
      .property("value", value.toPrecision(2))
      .style("display", "none");

    // Click-to-edit (scoped to this svg only)
    svg.on("click", () => {
      hovercard.select(".value").style("display", "none");
      const inputSel = hovercard.select<HTMLInputElement>("input");
      inputSel.style("display", null);

      inputSel.on("input", (event) => {
        const inputEl = event.currentTarget as HTMLInputElement;
        if (inputEl.value !== "") {
          const v = +inputEl.value;
          if (type === HoverType.WEIGHT) link.weight = v;
          else link.bias = v;
          this.update(); 
        }
      });

      inputSel.on("keydown", (event) => {
        if ((event as KeyboardEvent).key === "Enter") {
          this.updateHoverCard(network, type, link, coordinates);
        }
      });

      (inputSel.node() as HTMLInputElement).focus();
    });
  }

  // ------------------------
  //  For switching views
  // ------------------------
  private renderAll() {
    if (!this.visible) return;
    this.renderLineCharts();
    this.renderHeatMaps();
  }

  private renderLineCharts() {
    this.client.model.heads.forEach((head, i) => {
      const taskChart = this.root.select(`.task-chart[data-head="${head.name}"]`);
      taskChart.select(".loss .loss-train").text(humanReadable(this.client.lastMetrics.perHead[i].trainLoss));
      taskChart.select(".loss .loss-test").text(humanReadable(this.client.lastMetrics.perHead[i].testLoss));
      taskChart.select(".metric .metric-train").text(humanReadable(this.client.lastMetrics.perHead[i].trainMetric));
      taskChart.select(".metric .metric-test").text(humanReadable(this.client.lastMetrics.perHead[i].testMetric));
    });
    
    this.root.select("#loss-train").text(humanReadable(this.client.lastMetrics.lossTrain));
    this.root.select("#loss-test").text(humanReadable(this.client.lastMetrics.lossTest));
    
    this.client.model.heads.forEach((head, i) => {
      const lossSeries = head.stats.loss;       // Array<[train?, test?]>
      const metricSeries = head.stats.metric;   // Array<[train?, test?]>
      if (!lossSeries?.length || !metricSeries?.length) return;

      const lastLoss = lossSeries[lossSeries.length - 1];
      const lastMet  = metricSeries[metricSeries.length - 1];
      const hasLoss = (lastLoss?.[0] !== undefined) && (lastLoss?.[1] !== undefined);
      const hasMet  = (lastMet?.[0] !== undefined)  && (lastMet?.[1] !== undefined);
      if (!hasLoss || !hasMet) return;
      this.taskLossCharts[i].setData(head.stats.loss);
      this.taskMetricCharts[i].setData(head.stats.metric);
    });

    const hasLoss = (this.totalLossSeries?.[0]?.[0] !== undefined) && (this.totalLossSeries?.[0]?.[1] !== undefined);
    if (this.totalLossSeries?.length && hasLoss) {
      this.totalLossChart.setData(this.totalLossSeries);
    }
  }

  private renderHeatMaps() {
    const tasks = this.client.tasks;

    for (const taskName of tasks) {
      const head = this.client.model.heads.find(h => h.name.includes(taskName));
      if (!head) continue;

      const container = this.root.select(`#canvas-${this.client.id}-${head.name}`);
      container.selectAll("*").remove();

      const predRow = container.append("div")
        .style("display", "grid")
        .style("grid-template-columns", `repeat(${NUM_TO_SHOW}, ${RECT_SIZE}px)`);

      const gtRow = container.append("div")
        .style("display", "grid")
        .style("grid-template-columns", `repeat(${NUM_TO_SHOW}, ${RECT_SIZE}px)`);

      for (let i = 0; i < NUM_TO_SHOW; i++) {
        const predCell = predRow.append("div");
        new HeatMap(50, this.client.state.data.resolution, predCell)
          .updateImage(head.stats.predictions[i]);

        const gtCell = gtRow.append("div");
        new HeatMap(50, this.client.state.data.resolution, gtCell)
          .updateImage(head.stats.gts[i]);
      }
    }
  }

  /** Called after each training step. Safe when hidden: updates series only. */
  tick() {
    if (!this.client.lastMetrics) return;

    // store series (data only)
    if (this.totalLossSeries.length > 0 && this.totalLossSeries[0][0] === undefined) {
      this.totalLossSeries = [];
    }
    this.totalLossSeries.push([this.client.lastMetrics.lossTrain, this.client.lastMetrics.lossTest]); 
  }
}