// chart.ts
import * as d3 from 'd3';

type DataPoint = {
  x: number;
  y: number[]; 
};

export class LineChart {
  private svg: d3.Selection<SVGGElement, unknown, null, undefined>;

  private xScale!: d3.ScaleLinear<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;

  private data: DataPoint[] = [];
  private paths: d3.Selection<SVGPathElement, unknown, null, undefined>[] = [];

  private numLines: number;
  private lineColors: string[];

  private minY = Number.MAX_VALUE;
  private maxY = Number.MIN_VALUE;

  constructor(
    container: d3.Selection<d3.BaseType, unknown, any, any>,
    lineColors: string[],
  ) {
    this.lineColors = lineColors;
    this.numLines = lineColors.length;
    let node = container.node() as HTMLElement;
    let totalWidth = node.offsetWidth;
    let totalHeight = node.offsetHeight;
    let margin = {top: 2, right: 0, bottom: 2, left: 2};
    let width = totalWidth - margin.left - margin.right;
    let height = totalHeight - margin.top - margin.bottom;

    this.xScale = d3.scaleLinear()
      .domain([0, 0])
      .range([0, width]);

    this.yScale = d3.scaleLinear()
      .domain([0, 0])
      .range([height, 0]);

    this.svg = container.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    this.paths = new Array(this.numLines);
    for (let i = 0; i < this.numLines; i++) {
      this.paths[i] = this.svg.append("path")
        .attr("class", "line")
        .attr('fill', 'none')
        .attr('stroke', this.lineColors[i])
        .attr('stroke-width', 1.5);
    }

    this.reset();
    this.redraw();
  }

  /* ---------- public API ---------- */

  /** Clear all data */
  reset() {
    this.data = [];
    this.redraw();
    this.minY = Number.MAX_VALUE;
    this.maxY = Number.MIN_VALUE;
  }

  setData(series: number[][]) {
    this.data = [];
    this.minY = Number.POSITIVE_INFINITY;
    this.maxY = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < series.length; i++) {
      const point = series[i];
      if (point.length !== this.numLines) {
        throw new Error(`setData: expected ${this.numLines} values, got ${point.length}`);
      }
      for (const y of point) {
        this.minY = Math.min(this.minY, y);
        this.maxY = Math.max(this.maxY, y);
      }
      this.data.push({ x: i + 1, y: point });
    }
    this.redraw();
  }

  /** Append one multi-series point (length must equal number of lines). */
  addDataPoint(point: number[]) {
    if (point.length !== this.numLines) {
      throw new Error(`addDataPoint: expected ${this.numLines} values, got ${point.length}`);
    }
    point.forEach(y => {
      this.minY = Math.min(this.minY, y);
      this.maxY = Math.max(this.maxY, y);
    });
    this.data.push({x: this.data.length + 1, y: point});
    this.redraw();
  }

  /* ---------- internals ---------- */
  private redraw() {
    // update domains
    this.xScale.domain([1, this.data.length]);
    this.yScale.domain([this.minY, this.maxY]);

    let getPathMap = (lineIndex: number) => {
      return d3.line<DataPoint>()
      .x(d => this.xScale(d.x))
      .y(d => this.yScale(d.y[lineIndex]));
    };

    // draw each series path from the windowed data
    // build per-line arrays on the fly for the generator
    for (let line = 0; line < this.numLines; line++) {
      this.paths[line].datum(this.data).attr("d", getPathMap(line));
    }
  }
}
