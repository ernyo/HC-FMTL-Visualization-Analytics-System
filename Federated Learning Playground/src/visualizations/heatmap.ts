import * as d3 from 'd3';

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

export class HeatMap {
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false,
  };
  private xScale;f
  private yScale;
  private canvas;
  private svg;

  constructor(display: number, resolution: number, container: d3.Selection<d3.BaseType, unknown, any, any>, userSettings?: HeatMapSettings) {
    let height = display;
    let width = display;
    let padding = this.settings.showAxes ? 20 : 0;

    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3.scaleLinear()
      .domain([0, resolution])
      .range([0, width - 2 * padding]);

    this.yScale = d3.scaleLinear()
      .domain([0, resolution])
      .range([height - 2 * padding, 0]);

    container = container.append("div")
      .style("width", `${width}px`)
      .style("height", `${height}px`)
      .style("position", "relative")
      .style("top", `-${padding}px`)
      .style("left", `-${padding}px`);
    this.canvas = container.append("canvas")
      .attr("width", resolution)
      .attr("height", resolution)
      .style("width", (width - 2 * padding) + "px")
      .style("height", (height - 2 * padding) + "px")
      .style("position", "absolute")
      .style("top", `${padding}px`)
      .style("left", `${padding}px`);

    if (!this.settings.noSvg) {
      this.svg = container.append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("position", "absolute")
        .style("left", "0")
        .style("top", "0")
        .append("g")
        .attr("transform", `translate(${padding},${padding})`);

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    if (this.settings.showAxes) {
      let xAxis = d3.axisBottom(this.xScale)
      let yAxis = d3.axisRight(this.yScale);

      this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(0,${height - 2 * padding})`)
        .call(xAxis);

      this.svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (width - 2 * padding) + ",0)")
        .call(yAxis);
    }
  }

   // ---------- image-based overlay for semseg / depth / edge / saliency / normal ----------

  /**
   * Draws the predicted image over the actual image on the internal canvas.
   * Both images must be the same size and be ImageData (e.g. from a <canvas>).
   *
   * You can use this for:
   * - semseg: pass colorized GT & prediction
   * - depth: pass colormapped depth GT & prediction
   * - edge: pass grayscale edge maps as RGB
   * - saliency: pass grayscale saliency as RGB
   * - normal: pass normal maps as RGB images
   */
  updateImage(image: ImageData): void {
    const canvasEl = this.canvas.node() as HTMLCanvasElement;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) {
      throw new Error("Could not get 2D context from canvas");
    }

    const width = image.width;
    const height = image.height;

    // Make the internal canvas match the image resolution so we don't stretch.
    canvasEl.width = width;
    canvasEl.height = height;

    const composite = ctx.createImageData(width, height);
    const cData = composite.data;

    for (let i = 0; i < image.data.length; i += 4) {
      // Simple linear blend per channel.
      cData[i]     = image.data[i];     // R
      cData[i + 1] = image.data[i + 1]; // G
      cData[i + 2] = image.data[i + 2]; // B
      cData[i + 3] = 255; // Fully opaque
    }
    
    ctx.putImageData(composite, 0, 0);
  }
}