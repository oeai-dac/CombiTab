/**
 * MatrixRenderer — WebGL2-Renderer der Kombinationstabelle.
 *
 * Vom ProjectV2 getrieben:
 *  - Zellfarbe = Farbe der Materialgruppe der Spalte (columnMetadata.color),
 *    Sättigung nach Wert.
 *  - Anzeige-Reihenfolge über zwei kleine Order-Lookup-Texturen (Zeilen/Spalten):
 *    Umsortieren ist damit O(1) (nur eine 1D-Textur neu hochladen), nicht O(Zellen).
 *  - Interaktion: Pan, Zoom (um den Cursor), Hover, Klick-Selektion und
 *    Drag-Reorder von Zeilen/Spalten an ihren Beschriftungen.
 *  - Framework-frei; React mountet nur die beiden Canvasse und liest Callbacks.
 */
import type { ProjectV2 } from "../core/model.js";
import { annotationKey } from "../core/model.js";
import { applySeriation, moveFree, toggleFixed } from "./orderModel.js";
import { certaintyColor } from "../annotations/annotations.js";
import { mixCell } from "./cellColor.js";

export interface CellRef { kind: "cell" | "row" | "col"; row: number; col: number; } // row/col = ANZEIGE-Index
export interface RendererCallbacks {
  onHover?: (ref: CellRef | null) => void;
  onSelect?: (ref: CellRef | null) => void;
  onReorderStart?: () => void;
  onReorder?: () => void;
  onChange?: () => void;
  onAreaSelect?: (cells: Array<[number, number]>) => void;
}
export interface OrderSnapshot { rows: number[]; cols: number[]; rowFixed: number[]; colFixed: number[]; }

const MARGIN = { x: 108, y: 116 };

/** Theme-abhängige Farben des Renderers. Werden aus den CSS-Tokens
 *  (`theme.css`) gelesen und via `setTheme()` gesetzt; `bg` als 0..1-RGB für WebGL. */
export type MatrixTheme = {
  bg: [number, number, number]; // Hintergrund/Leerzellen-Mischfarbe (WebGL, 0..1)
  label: string;                // normale Beschriftung
  labelDim: string;             // Achsentitel / Fußnote
  labelActive: string;          // fixiert / aktiv (Akzent)
};

const LIGHT_THEME: MatrixTheme = { bg: [0.965, 0.957, 0.949], label: "#5d584f", labelDim: "#8b857c", labelActive: "#a81d26" };

export class MatrixRenderer {
  private gl: WebGL2RenderingContext | null;
  private ctx2d: CanvasRenderingContext2D | null = null; // 2D-Fallback-Zellfläche
  private octx: CanvasRenderingContext2D;
  private prog!: WebGLProgram;
  private uni: Record<string, WebGLUniformLocation | null> = {};
  private vao!: WebGLVertexArrayObject;
  private dataTex!: WebGLTexture; private colorTex!: WebGLTexture;
  private rowOrderTex!: WebGLTexture; private colOrderTex!: WebGLTexture;
  private dpr = Math.min(window.devicePixelRatio || 1, 2);

  private p!: ProjectV2;
  private NR = 0; private NC = 0; private vmax = 1;
  private disp!: Uint8Array;              // NR*NC, canonical row-major
  private colRGB = new Uint8Array(0);     // NC*3, Spaltenfarben 0..255 (2D-Fallback)
  private rowOrder!: Uint32Array; private colOrder!: Uint32Array;
  private rowFixed = new Set<number>(); private colFixed = new Set<number>();

  private view = { cell: 18, panX: 0, panY: 0 };
  private hover: CellRef | null = null;
  private sel: CellRef | null = null;
  private linkedCtx = -1; private linkedType = -1;
  private drag: { axis: "row" | "col"; from: number; canon: number } | null = null;
  private interactive = true;
  private mode: "navigate" | "select" = "navigate";
  private area: { r0: number; c0: number; r1: number; c1: number } | null = null;
  private areaDrag: { r0: number; c0: number } | null = null;
  private raf = 0; private needsDraw = true;
  private theme: MatrixTheme = LIGHT_THEME;
  private axisContext = "Kontext ▾"; private axisType = "Typ ▸"; // i18n
  // Performance-Instrumentierung
  private profiling = false; private forceDraw = false; private lastTs = 0;
  private frameBuf: number[] = []; private drawBuf: number[] = []; private readonly perfCap = 180;

  constructor(private glCanvas: HTMLCanvasElement, private overlay: HTMLCanvasElement, private cb: RendererCallbacks = {}) {
    this.gl = glCanvas.getContext("webgl2", { antialias: false, alpha: false });
    this.octx = overlay.getContext("2d")!;
    if (this.gl) this.initGL();
    else this.ctx2d = glCanvas.getContext("2d"); // WebGL2 nicht verfügbar → Canvas-2D-Fallback
    this.bindEvents();
    const loop = (ts: number) => {
      if (this.profiling) { if (this.lastTs) this.push(this.frameBuf, ts - this.lastTs); this.lastTs = ts; }
      if (this.needsDraw || this.forceDraw) {
        if (this.profiling) { const t0 = performance.now(); this.draw(); this.push(this.drawBuf, performance.now() - t0); }
        else this.draw();
        this.needsDraw = false;
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  /* ── Performance-Instrumentierung ── */
  private push(buf: number[], v: number): void { buf.push(v); if (buf.length > this.perfCap) buf.shift(); }
  private static pct(a: number[], p: number): number { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
  private static avg(a: number[]): number { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }

  /** Frame-/Draw-Messung ein-/ausschalten (Perf-HUD). */
  setProfiling(on: boolean): void { this.profiling = on; if (on) { this.frameBuf = []; this.drawBuf = []; this.lastTs = 0; } }

  /** Momentaufnahme der Messwerte für das HUD. */
  get perfStats(): { fps: number; frameAvg: number; frameP95: number; drawAvg: number; drawP95: number; backend: string; cells: number; visibleCells: number; samples: number } {
    const { visCols, visRows } = this.visible();
    const frameAvg = MatrixRenderer.avg(this.frameBuf);
    return {
      fps: frameAvg > 0 ? 1000 / frameAvg : 0,
      frameAvg, frameP95: MatrixRenderer.pct(this.frameBuf, 0.95),
      drawAvg: MatrixRenderer.avg(this.drawBuf), drawP95: MatrixRenderer.pct(this.drawBuf, 0.95),
      backend: this.backend, cells: this.NR * this.NC, visibleCells: visCols * visRows, samples: this.frameBuf.length,
    };
  }

  /**
   * In-Browser-Benchmark: zeichnet für `durationMs` in jedem Frame (auch ohne
   * Interaktion) und liefert die erreichte Bildrate und Frame-/Draw-Zeiten.
   * Deckt das GPU-Budget ab (1.000² @ 60 fps, Draw < 16 ms), das headless nicht
   * messbar ist. Erwartet, dass zuvor ein passendes Projekt/Zoom gesetzt wurde.
   */
  benchmark(durationMs = 2000): Promise<{ fps: number; frameAvg: number; frameP95: number; drawAvg: number; drawP95: number; frames: number; backend: string; cells: number; visibleCells: number; durationMs: number }> {
    return new Promise((resolve) => {
      const wasProfiling = this.profiling;
      this.setProfiling(true); this.forceDraw = true; this.needsDraw = true;
      const t0 = performance.now();
      const done = () => {
        this.forceDraw = false;
        const s = this.perfStats;
        const elapsed = performance.now() - t0;
        if (!wasProfiling) this.setProfiling(false);
        resolve({ fps: s.fps, frameAvg: s.frameAvg, frameP95: s.frameP95, drawAvg: s.drawAvg, drawP95: s.drawP95, frames: this.frameBuf.length, backend: s.backend, cells: s.cells, visibleCells: s.visibleCells, durationMs: Math.round(elapsed) });
      };
      setTimeout(done, durationMs);
    });
  }

  get supported(): boolean { return !!this.gl || !!this.ctx2d; }
  /** Aktiver Zeichenpfad: WebGL2 (schnell), Canvas-2D (Fallback) oder keiner. */
  get backend(): "webgl2" | "canvas2d" | "none" { return this.gl ? "webgl2" : this.ctx2d ? "canvas2d" : "none"; }
  get order() { return { rows: Array.from(this.rowOrder), cols: Array.from(this.colOrder) }; }
  get linkedState() { return { ctx: this.linkedCtx, type: this.linkedType }; }

  destroy(): void {
    cancelAnimationFrame(this.raf);
    const [move, up] = this.usesPointer ? ["pointermove", "pointerup"] : ["mousemove", "mouseup"];
    window.removeEventListener(move, this.hMove);
    window.removeEventListener(up, this.hUp);
    if (this.usesPointer) window.removeEventListener("pointercancel", this.hUp);
  }

  setProject(p: ProjectV2): void {
    this.p = p; this.NR = p.contexts.length; this.NC = p.types.length;
    // Anzeige-Ordnung: p.order (auf kanonische Indizes abgebildet)
    const rIdx = new Map(p.contexts.map((c, i) => [c, i] as const));
    const cIdx = new Map(p.types.map((t, j) => [t, j] as const));
    this.rowOrder = Uint32Array.from(p.order.rows.map((r) => rIdx.get(r) ?? 0));
    this.colOrder = Uint32Array.from(p.order.cols.map((c) => cIdx.get(c) ?? 0));
    if (this.rowOrder.length !== this.NR) this.rowOrder = Uint32Array.from(p.contexts.keys());
    if (this.colOrder.length !== this.NC) this.colOrder = Uint32Array.from(p.types.keys());

    // Fixierungen aus den Metadaten übernehmen (kanonische Indizes)
    this.rowFixed = new Set(p.contexts.map((c, i) => (p.rowMetadata[c]?.isFixed ? i : -1)).filter((i) => i >= 0));
    this.colFixed = new Set(p.types.map((t, j) => (p.columnMetadata[t]?.isFixed ? j : -1)).filter((j) => j >= 0));

    // Anzeige-Puffer (kanonisch) + vmax
    this.vmax = 1;
    for (let i = 0; i < this.NR; i++) for (let j = 0; j < this.NC; j++) { const v = p.matrix[i][j]; if (v > this.vmax) this.vmax = v; }
    this.disp = new Uint8Array(this.NR * this.NC);
    for (let i = 0; i < this.NR; i++) for (let j = 0; j < this.NC; j++) { const v = p.matrix[i][j]; this.disp[i * this.NC + j] = v ? Math.max(28, Math.round((v / this.vmax) * 255)) : 0; }

    // Spaltenfarben-Cache (0..255) — vom Canvas-2D-Fallback genutzt, immer aktuell halten.
    this.colRGB = new Uint8Array(this.NC * 3);
    for (let j = 0; j < this.NC; j++) { const [r, g, b] = hexToRgb(p.columnMetadata[p.types[j]]?.color ?? "#808080"); this.colRGB[j * 3] = r; this.colRGB[j * 3 + 1] = g; this.colRGB[j * 3 + 2] = b; }

    if (this.gl) { this.uploadData(); this.uploadColors(); this.uploadOrder(); }
    this.fit(); this.needsDraw = true;
  }

  /** Neue Anzeige-Ordnung setzen (z. B. nach Seriation). Indizes sind kanonisch. */
  setOrder(rows: number[], cols: number[]): void {
    this.rowOrder = Uint32Array.from(rows); this.colOrder = Uint32Array.from(cols);
    if (this.gl) this.uploadOrder();
    this.syncProjectOrder(); this.needsDraw = true;
  }

  /** Seriation anwenden, fixierte Elemente behalten ihre Position (orderModel). */
  applySeriationOrder(sortedRows: number[], sortedCols: number[]): void {
    const rows = applySeriation(Array.from(this.rowOrder), this.rowFixed, sortedRows);
    const cols = applySeriation(Array.from(this.colOrder), this.colFixed, sortedCols);
    this.setOrder(rows, cols);
  }

  getSnapshot(): OrderSnapshot {
    return { rows: Array.from(this.rowOrder), cols: Array.from(this.colOrder), rowFixed: [...this.rowFixed], colFixed: [...this.colFixed] };
  }
  restore(s: OrderSnapshot): void {
    this.rowFixed = new Set(s.rowFixed); this.colFixed = new Set(s.colFixed);
    this.rowOrder = Uint32Array.from(s.rows); this.colOrder = Uint32Array.from(s.cols);
    if (this.gl) this.uploadOrder();
    this.syncFixedToMeta(); this.syncProjectOrder(); this.needsDraw = true;
  }

  /** Fixierung an einer Anzeige-Position umschalten (Achse). */
  toggleFix(axis: "row" | "col", displayPos: number): void {
    if (axis === "row") this.rowFixed = toggleFixed(Array.from(this.rowOrder), this.rowFixed, displayPos);
    else this.colFixed = toggleFixed(Array.from(this.colOrder), this.colFixed, displayPos);
    this.syncFixedToMeta(); this.needsDraw = true; this.cb.onChange?.();
  }
  isFixedAt(axis: "row" | "col", displayPos: number): boolean {
    return axis === "row" ? this.rowFixed.has(this.rowOrder[displayPos]) : this.colFixed.has(this.colOrder[displayPos]);
  }

  /** Von anderen Ansichten verlinkte Hervorhebung (Brushing & Linking). */
  setLinked(ctxName: string | null, typeName: string | null): void {
    this.linkedCtx = ctxName ? this.p.contexts.indexOf(ctxName) : -1;
    this.linkedType = typeName ? this.p.types.indexOf(typeName) : -1;
    this.needsDraw = true;
  }

  setMode(m: "navigate" | "select"): void { this.mode = m; if (m === "navigate") this.area = null; this.needsDraw = true; }
  /** Read-only schalten (z. B. bei aktiver gefilterter Ansicht): Drag/Area deaktiviert. */
  setInteractive(on: boolean): void { this.interactive = on; if (!on) { this.drag = null; this.areaDrag = null; } }

  /** Setzt die theme-abhängigen Farben (Hell/Dunkel) und zeichnet neu. */
  setTheme(t: MatrixTheme): void { this.theme = t; this.needsDraw = true; }

  /** Übersetzte Achsentitel setzen (DE/EN) und neu zeichnen. */
  setAxisLabels(context: string, type: string): void { this.axisContext = context; this.axisType = type; this.needsDraw = true; }
  clearArea(): void { this.area = null; this.needsDraw = true; }
  refresh(): void { this.needsDraw = true; }
  private areaCells(): Array<[number, number]> {
    if (!this.area) return [];
    const { r0, c0, r1, c1 } = this.area; const cells: Array<[number, number]> = [];
    for (let dr = Math.min(r0, r1); dr <= Math.max(r0, r1); dr++)
      for (let dc = Math.min(c0, c1); dc <= Math.max(c0, c1); dc++)
        cells.push([this.rowOrder[dr], this.colOrder[dc]]);
    return cells;
  }

  fit(): void {
    const w = this.glCanvas.clientWidth, h = this.glCanvas.clientHeight;
    this.view.cell = Math.max(2, Math.min((w - MARGIN.x - 12) / this.NC, (h - MARGIN.y - 12) / this.NR, 34));
    this.view.panX = 0; this.view.panY = 0; this.needsDraw = true;
  }
  setCell(px: number): void { this.view.cell = Math.max(2, Math.min(60, px)); this.needsDraw = true; }
  /** Aktuelles Sichtfenster in Display-Positionen (für die Minimap). */
  getViewport(): { c0: number; r0: number; cols: number; rows: number; NC: number; NR: number } {
    const { c0, r0, visCols, visRows } = this.visible();
    return { c0, r0, cols: visCols, rows: visRows, NC: this.NC, NR: this.NR };
  }
  /** Verschiebt das Sichtfenster so, dass die Display-Zelle (rowPos,colPos) zentriert ist. */
  centerOnCell(rowPos: number, colPos: number): void {
    const w = this.glCanvas.width / this.dpr, h = this.glCanvas.height / this.dpr, cs = this.view.cell;
    this.view.panX = w / 2 - MARGIN.x - (colPos + 0.5) * cs;
    this.view.panY = h / 2 - MARGIN.y - (rowPos + 0.5) * cs;
    this.needsDraw = true;
  }
  resize(): void {
    const w = this.glCanvas.clientWidth, h = this.glCanvas.clientHeight;
    for (const c of [this.glCanvas, this.overlay]) { c.width = w * this.dpr; c.height = h * this.dpr; }
    if (this.gl) this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    this.needsDraw = true;
  }

  /* ── WebGL-Setup ── */
  private initGL(): void {
    const gl = this.gl!;
    const vs = `#version 300 es
    precision highp float; precision highp int;
    uniform vec2 uRes,uPan,uMargin; uniform float uCell;
    uniform int uC0,uR0,uVisCols,uCols,uRows;
    uniform vec3 uBg;
    uniform sampler2D uData; uniform sampler2D uColor;
    uniform highp usampler2D uRowOrder; uniform highp usampler2D uColOrder;
    out vec3 vColor;
    void main(){
      int gid=gl_InstanceID; int lc=gid%uVisCols; int lr=gid/uVisCols;
      int scol=uC0+lc, srow=uR0+lr; vColor=vec3(0.0);
      if(scol>=uCols||srow>=uRows){gl_Position=vec4(2.,2.,2.,1.);return;}
      int dcol=int(texelFetch(uColOrder,ivec2(scol,0),0).r);
      int drow=int(texelFetch(uRowOrder,ivec2(srow,0),0).r);
      float v=texelFetch(uData,ivec2(dcol,drow),0).r;
      if(v<=0.003){gl_Position=vec4(2.,2.,2.,1.);return;}
      int vid=gl_VertexID; vec2 cr;
      if(vid==0)cr=vec2(0.,0.);else if(vid==1)cr=vec2(1.,0.);else if(vid==2)cr=vec2(0.,1.);
      else if(vid==3)cr=vec2(0.,1.);else if(vid==4)cr=vec2(1.,0.);else cr=vec2(1.,1.);
      float gap=uCell>3.0?0.10*uCell:0.0;
      vec2 origin=uMargin+uPan+vec2(float(scol),float(srow))*uCell;
      vec2 px=origin+cr*(uCell-gap);
      gl_Position=vec4(px.x/uRes.x*2.0-1.0, 1.0-px.y/uRes.y*2.0, 0.,1.);
      vec3 base=texelFetch(uColor,ivec2(dcol,0),0).rgb;
      vColor=mix(uBg,base,0.25+0.75*clamp(v,0.,1.));
    }`;
    const fs = `#version 300 es
    precision highp float; in vec3 vColor; out vec4 o; void main(){o=vec4(vColor,1.0);}`;
    const sh = (t: number, src: string) => { const s = gl.createShader(t)!; gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || "shader"); return s; };
    this.prog = gl.createProgram()!; gl.attachShader(this.prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(this.prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(this.prog);
    if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.prog) || "link");
    gl.useProgram(this.prog);
    for (const n of ["uRes", "uPan", "uMargin", "uCell", "uC0", "uR0", "uVisCols", "uCols", "uRows", "uBg", "uData", "uColor", "uRowOrder", "uColOrder"]) this.uni[n] = gl.getUniformLocation(this.prog, n);
    gl.uniform1i(this.uni.uData, 0); gl.uniform1i(this.uni.uColor, 1); gl.uniform1i(this.uni.uRowOrder, 2); gl.uniform1i(this.uni.uColOrder, 3);
    this.vao = gl.createVertexArray()!;
    this.dataTex = gl.createTexture()!; this.colorTex = gl.createTexture()!; this.rowOrderTex = gl.createTexture()!; this.colOrderTex = gl.createTexture()!;
  }
  private uploadData(): void {
    const gl = this.gl!; gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.NC, this.NR, 0, gl.RED, gl.UNSIGNED_BYTE, this.disp);
    this.nearest();
  }
  private uploadColors(): void {
    const gl = this.gl!; const rgba = new Uint8Array(this.NC * 4);
    for (let j = 0; j < this.NC; j++) { const hex = this.p.columnMetadata[this.p.types[j]]?.color ?? "#808080"; const [r, g, b] = hexToRgb(hex); rgba[j * 4] = r; rgba[j * 4 + 1] = g; rgba[j * 4 + 2] = b; rgba[j * 4 + 3] = 255; }
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.NC, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba); this.nearest();
  }
  private uploadOrder(): void {
    const gl = this.gl!;
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.rowOrderTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, this.NR, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, this.rowOrder); this.nearest();
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.colOrderTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, this.NC, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, this.colOrder); this.nearest();
  }
  private nearest(): void { const gl = this.gl!; for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST); for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE); }

  /* ── Zeichnen ── */
  private visible() {
    const w = this.glCanvas.width / this.dpr, h = this.glCanvas.height / this.dpr, cs = this.view.cell;
    let c0 = Math.floor((-MARGIN.x - this.view.panX) / cs), c1 = Math.ceil((w - MARGIN.x - this.view.panX) / cs);
    let r0 = Math.floor((-MARGIN.y - this.view.panY) / cs), r1 = Math.ceil((h - MARGIN.y - this.view.panY) / cs);
    c0 = Math.max(0, c0); r0 = Math.max(0, r0); c1 = Math.min(this.NC, c1); r1 = Math.min(this.NR, r1);
    return { c0, r0, visCols: Math.max(0, c1 - c0), visRows: Math.max(0, r1 - r0) };
  }
  private draw(): void {
    if (this.gl) this.drawGL();
    else if (this.ctx2d) this.drawCanvas2D();
    this.drawOverlay();
  }
  private drawGL(): void {
    const gl = this.gl!; gl.clearColor(this.theme.bg[0], this.theme.bg[1], this.theme.bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const { c0, r0, visCols, visRows } = this.visible(); const inst = visCols * visRows;
    if (inst > 0) {
      gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.rowOrderTex);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.colOrderTex);
      gl.uniform2f(this.uni.uRes, this.glCanvas.width, this.glCanvas.height);
      gl.uniform2f(this.uni.uPan, this.view.panX * this.dpr, this.view.panY * this.dpr);
      gl.uniform2f(this.uni.uMargin, MARGIN.x * this.dpr, MARGIN.y * this.dpr);
      gl.uniform1f(this.uni.uCell, this.view.cell * this.dpr);
      gl.uniform1i(this.uni.uC0, c0); gl.uniform1i(this.uni.uR0, r0); gl.uniform1i(this.uni.uVisCols, visCols);
      gl.uniform1i(this.uni.uCols, this.NC); gl.uniform1i(this.uni.uRows, this.NR);
      gl.uniform3f(this.uni.uBg, this.theme.bg[0], this.theme.bg[1], this.theme.bg[2]);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, inst);
    }
  }
  /**
   * Canvas-2D-Fallback. Zeichnet dieselben sichtbaren Zellen wie der
   * WebGL-Pfad mit identischem Farbmodell (Basis-Materialfarbe, nach Häufigkeit
   * gegen die Hintergrundfarbe ausgemischt) und identischer Kulling-/Gap-Geometrie.
   * Arbeitet im selben CSS-Pixel-Koordinatenraum wie das Overlay, damit Zellen und
   * Beschriftungen deckungsgleich sind.
   */
  private drawCanvas2D(): void {
    const g = this.ctx2d!; g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const Wc = this.glCanvas.width / this.dpr, Hc = this.glCanvas.height / this.dpr;
    const bg = this.theme.bg, b0 = bg[0] * 255, b1 = bg[1] * 255, b2 = bg[2] * 255;
    g.fillStyle = `rgb(${Math.round(b0)},${Math.round(b1)},${Math.round(b2)})`;
    g.fillRect(0, 0, Wc, Hc);
    const cs = this.view.cell, gap = cs > 3 ? 0.10 * cs : 0, size = cs - gap;
    const { c0, r0, visCols, visRows } = this.visible();
    const ox = MARGIN.x + this.view.panX, oy = MARGIN.y + this.view.panY;
    for (let lr = 0; lr < visRows; lr++) {
      const srow = r0 + lr, drow = this.rowOrder[srow], y = oy + srow * cs;
      for (let lc = 0; lc < visCols; lc++) {
        const scol = c0 + lc, dcol = this.colOrder[scol];
        const v = this.disp[drow * this.NC + dcol] / 255;
        if (v <= 0.003) continue;
        const k = dcol * 3;
        const [R, G, B] = mixCell(bg, [this.colRGB[k], this.colRGB[k + 1], this.colRGB[k + 2]], v);
        g.fillStyle = `rgb(${R},${G},${B})`;
        g.fillRect(ox + scol * cs, y, size, size);
      }
    }
  }
  private drawOverlay(): void {
    const g = this.octx, cs = this.view.cell; g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.overlay.width, this.overlay.height);
    const { c0, r0, visCols, visRows } = this.visible();
    // Selektions-Kreuz
    if (this.sel) {
      g.fillStyle = "rgba(210,38,48,.07)";
      const y = MARGIN.y + this.view.panY + this.sel.row * cs, x = MARGIN.x + this.view.panX + this.sel.col * cs;
      if (this.sel.kind !== "col") g.fillRect(MARGIN.x, y, this.NC * cs, cs);
      if (this.sel.kind !== "row") g.fillRect(x, MARGIN.y, cs, this.NR * cs);
    }
    // Verlinkte Hervorhebung (aus anderen Ansichten)
    if (this.linkedCtx >= 0) { const dp = this.rowOrder.indexOf(this.linkedCtx); if (dp >= 0) { const y = MARGIN.y + this.view.panY + dp * cs; g.fillStyle = "rgba(210,38,48,.11)"; g.fillRect(MARGIN.x, y, this.NC * cs, cs); g.fillStyle = "#d22630"; g.fillRect(MARGIN.x - 3, y, 3, cs); } }
    if (this.linkedType >= 0) { const dp = this.colOrder.indexOf(this.linkedType); if (dp >= 0) { const x = MARGIN.x + this.view.panX + dp * cs; g.fillStyle = "rgba(210,38,48,.11)"; g.fillRect(x, MARGIN.y, cs, this.NR * cs); g.fillStyle = "#d22630"; g.fillRect(x, MARGIN.y - 3, cs, 3); } }
    // Zeilenbeschriftung
    g.textBaseline = "middle"; g.font = '500 11px "JetBrains Mono",monospace';
    for (let lr = 0; lr < visRows; lr++) {
      const srow = r0 + lr, y = MARGIN.y + this.view.panY + (srow + 0.5) * cs; if (y < MARGIN.y - 2) continue;
      const name = this.p.contexts[this.rowOrder[srow]];
      const fixed = this.rowFixed.has(this.rowOrder[srow]);
      const active = this.hover?.row === srow || this.sel?.row === srow || (this.drag?.axis === "row" && this.drag.from === srow);
      g.fillStyle = fixed || active ? this.theme.labelActive : this.theme.label;
      if (fixed) { g.beginPath(); g.arc(4, y, 2.4, 0, 7); g.fill(); }
      g.fillText(clip(name, 13), fixed ? 12 : 8, y);
    }
    // Spaltenbeschriftung (vertikal) nur wenn breit genug
    if (cs >= 7) {
      g.font = '500 9px "JetBrains Mono",monospace';
      for (let lc = 0; lc < visCols; lc++) {
        const scol = c0 + lc, x = MARGIN.x + this.view.panX + (scol + 0.5) * cs; if (x < MARGIN.x - 2) continue;
        const name = this.p.types[this.colOrder[scol]];
        const fixed = this.colFixed.has(this.colOrder[scol]);
        const active = this.hover?.col === scol || this.sel?.col === scol || (this.drag?.axis === "col" && this.drag.from === scol);
        g.save(); g.translate(x + 3, MARGIN.y - 6); g.rotate(-Math.PI / 2);
        g.fillStyle = fixed || active ? this.theme.labelActive : this.theme.label;
        if (fixed) { g.beginPath(); g.arc(-4, 0, 2.4, 0, 7); g.fill(); }
        g.fillText(clip(name, 15), fixed ? 4 : 0, 0); g.restore();
      }
    }
    // Hover-Rahmen
    if (this.hover && this.hover.kind === "cell") {
      g.strokeStyle = "#d22630"; g.lineWidth = 1.5;
      g.strokeRect(MARGIN.x + this.view.panX + this.hover.col * cs + 0.75, MARGIN.y + this.view.panY + this.hover.row * cs + 0.75, cs - 1.5, cs - 1.5);
    }
    // Annotations-Marker (Ecke, Ampelfarbe nach Sicherheit)
    if (cs >= 7 && this.p && Object.keys(this.p.cellAnnotations).length) {
      const vr = this.visible();
      for (let lr = 0; lr < vr.visRows; lr++) for (let lc = 0; lc < vr.visCols; lc++) {
        const srow = vr.r0 + lr, scol = vr.c0 + lc;
        const a = this.p.cellAnnotations[annotationKey(this.rowOrder[srow], this.colOrder[scol])];
        if (!a) continue;
        const x = MARGIN.x + this.view.panX + scol * cs, y = MARGIN.y + this.view.panY + srow * cs;
        const m = Math.min(5, cs * 0.42);
        g.fillStyle = certaintyColor(a.certainty); g.beginPath();
        g.moveTo(x + cs - 0.5, y + 0.5); g.lineTo(x + cs - 0.5, y + m); g.lineTo(x + cs - m, y + 0.5); g.closePath(); g.fill();
      }
    }
    // „Nicht erfasst"-Marker (§9.6): diagonales Kreuz — klar unterscheidbar von leer (Absenz)
    if (cs >= 4 && this.p && this.p.missingCells && Object.keys(this.p.missingCells).length) {
      const vr = this.visible();
      g.strokeStyle = this.theme.labelDim; g.lineWidth = 1;
      for (let lr = 0; lr < vr.visRows; lr++) for (let lc = 0; lc < vr.visCols; lc++) {
        const srow = vr.r0 + lr, scol = vr.c0 + lc;
        if (!this.p.missingCells[annotationKey(this.rowOrder[srow], this.colOrder[scol])]) continue;
        const x = MARGIN.x + this.view.panX + scol * cs, y = MARGIN.y + this.view.panY + srow * cs;
        const pad = Math.max(1, cs * 0.2);
        g.beginPath();
        g.moveTo(x + pad, y + pad); g.lineTo(x + cs - pad, y + cs - pad);
        g.moveTo(x + cs - pad, y + pad); g.lineTo(x + pad, y + cs - pad);
        g.stroke();
      }
    }
    // Bereichsselektion (Annotieren-Modus)
    if (this.area) {
      const ar0 = Math.min(this.area.r0, this.area.r1), ar1 = Math.max(this.area.r0, this.area.r1);
      const ac0 = Math.min(this.area.c0, this.area.c1), ac1 = Math.max(this.area.c0, this.area.c1);
      const x = MARGIN.x + this.view.panX + ac0 * cs, y = MARGIN.y + this.view.panY + ar0 * cs, w = (ac1 - ac0 + 1) * cs, h = (ar1 - ar0 + 1) * cs;
      g.fillStyle = "rgba(210,38,48,.12)"; g.fillRect(x, y, w, h);
      g.strokeStyle = "#d22630"; g.lineWidth = 1.5; g.strokeRect(x + 0.75, y + 0.75, w - 1.5, h - 1.5);
    }
    // Ecke: Achsentitel
    g.fillStyle = this.theme.labelDim; g.font = '600 10px "Outfit",sans-serif';
    g.fillText(this.axisContext, 8, 18); g.save(); g.translate(18, MARGIN.y - 8); g.rotate(-Math.PI / 2); g.fillText(this.axisType, 0, 0); g.restore();
  }

  /* ── Interaktion ── */
  // Pointer- statt Maus-Events, damit die Matrix auch per Touch bedienbar
  // ist (Pan, Drag-Reorder, Bereichsauswahl auf Tablets — die App ist eine PWA).
  // PointerEvent erbt von MouseEvent; die Handler bleiben unverändert. Nur der
  // Primär-Zeiger wird verarbeitet (kein Mehrfinger-Chaos); Zoom bleibt über
  // Mausrad bzw. den Zellgrößen-Regler erreichbar. `touch-action: none` (CSS)
  // verhindert, dass der Browser die Geste fürs Seiten-Scrollen abfängt.
  private usesPointer = typeof window !== "undefined" && "PointerEvent" in window;
  private onlyPrimary = (h: (ev: MouseEvent) => void) => (ev: Event) => {
    const pe = ev as PointerEvent;
    if (this.usesPointer && pe.isPrimary === false) return;
    h(pe);
  };
  private hDown = this.onlyPrimary((ev) => this.onDown(ev));
  private hMove = this.onlyPrimary((ev) => this.onMove(ev));
  private hUp = this.onlyPrimary(() => this.onUp());
  private bindEvents(): void {
    const [down, move, up] = this.usesPointer ? ["pointerdown", "pointermove", "pointerup"] : ["mousedown", "mousemove", "mouseup"];
    this.glCanvas.addEventListener(down, this.hDown);
    this.glCanvas.addEventListener("wheel", this.onWheel, { passive: false });
    window.addEventListener(move, this.hMove);
    window.addEventListener(up, this.hUp);
    if (this.usesPointer) window.addEventListener("pointercancel", this.hUp);
    this.glCanvas.style.touchAction = "none";
  }
  private local(ev: MouseEvent) { const r = this.glCanvas.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; }
  /** Bildschirmposition → Anzeige-Zelle/Zeile/Spalte (oder null). */
  hitTest(x: number, y: number): CellRef | null {
    const cs = this.view.cell;
    const col = Math.floor((x - MARGIN.x - this.view.panX) / cs), row = Math.floor((y - MARGIN.y - this.view.panY) / cs);
    const inCols = col >= 0 && col < this.NC, inRows = row >= 0 && row < this.NR;
    if (inCols && inRows) return { kind: "cell", row, col };
    if (x < MARGIN.x && inRows) return { kind: "row", row, col: -1 };
    if (y < MARGIN.y && inCols) return { kind: "col", row: -1, col };
    return null;
  }
  private panStart: { x: number; y: number; px: number; py: number } | null = null;
  private onDown = (ev: MouseEvent) => {
    const { x, y } = this.local(ev); const hit = this.hitTest(x, y);
    if (this.mode === "select") {
      if (this.interactive && hit?.kind === "cell") { this.areaDrag = { r0: hit.row, c0: hit.col }; this.area = { r0: hit.row, c0: hit.col, r1: hit.row, c1: hit.col }; this.needsDraw = true; }
      return;
    }
    if (this.interactive && hit?.kind === "row") {
      if (this.rowFixed.has(this.rowOrder[hit.row])) return;      // fixiert → nicht ziehbar
      this.drag = { axis: "row", from: hit.row, canon: this.rowOrder[hit.row] }; this.cb.onReorderStart?.();
    } else if (this.interactive && hit?.kind === "col") {
      if (this.colFixed.has(this.colOrder[hit.col])) return;
      this.drag = { axis: "col", from: hit.col, canon: this.colOrder[hit.col] }; this.cb.onReorderStart?.();
    } else { this.panStart = { x: ev.clientX, y: ev.clientY, px: this.view.panX, py: this.view.panY }; }
  };
  private onMove = (ev: MouseEvent) => {
    const { x, y } = this.local(ev);
    if (this.areaDrag) {
      const cs = this.view.cell;
      let r1 = Math.floor((y - MARGIN.y - this.view.panY) / cs); r1 = Math.max(0, Math.min(this.NR - 1, r1));
      let c1 = Math.floor((x - MARGIN.x - this.view.panX) / cs); c1 = Math.max(0, Math.min(this.NC - 1, c1));
      this.area = { r0: this.areaDrag.r0, c0: this.areaDrag.c0, r1, c1 }; this.needsDraw = true; return;
    }
    if (this.drag) {
      const cs = this.view.cell;
      if (this.drag.axis === "row") {
        let to = Math.floor((y - MARGIN.y - this.view.panY) / cs); to = Math.max(0, Math.min(this.NR - 1, to));
        const next = moveFree(Array.from(this.rowOrder), this.rowFixed, this.drag.from, to);
        const nf = next.indexOf(this.drag.canon);
        if (nf !== this.drag.from) { this.rowOrder = Uint32Array.from(next); this.drag.from = nf; if (this.gl) this.uploadOrder(); this.syncProjectOrder(); this.needsDraw = true; }
      } else {
        let to = Math.floor((x - MARGIN.x - this.view.panX) / cs); to = Math.max(0, Math.min(this.NC - 1, to));
        const next = moveFree(Array.from(this.colOrder), this.colFixed, this.drag.from, to);
        const nf = next.indexOf(this.drag.canon);
        if (nf !== this.drag.from) { this.colOrder = Uint32Array.from(next); this.drag.from = nf; if (this.gl) this.uploadOrder(); this.syncProjectOrder(); this.needsDraw = true; }
      }
      return;
    }
    if (this.panStart) { this.view.panX = this.panStart.px + (ev.clientX - this.panStart.x); this.view.panY = this.panStart.py + (ev.clientY - this.panStart.y); this.needsDraw = true; return; }
    const hit = this.hitTest(x, y); const changed = JSON.stringify(hit) !== JSON.stringify(this.hover);
    this.hover = hit; if (changed) { this.needsDraw = true; this.cb.onHover?.(hit); }
  };
  private onUp = () => {
    if (this.areaDrag) { this.areaDrag = null; this.cb.onAreaSelect?.(this.areaCells()); this.needsDraw = true; return; }
    if (this.drag) { this.drag = null; this.cb.onReorder?.(); this.needsDraw = true; return; }
    if (this.panStart) { const moved = false; this.panStart = null; if (moved) return; }
    this.panStart = null;
  };
  // Klick-Selektion getrennt behandeln (kein Pan)
  private onWheel = (ev: WheelEvent) => {
    ev.preventDefault(); const { x, y } = this.local(ev);
    const f = ev.deltaY < 0 ? 1.12 : 1 / 1.12; const nc = Math.max(2, Math.min(60, this.view.cell * f)); const k = nc / this.view.cell;
    this.view.panX = x - MARGIN.x - (x - MARGIN.x - this.view.panX) * k;
    this.view.panY = y - MARGIN.y - (y - MARGIN.y - this.view.panY) * k;
    this.view.cell = nc; this.needsDraw = true;
  };
  /** Klick (Selektion) — wird von der View über click-Event aufgerufen. */
  click(ev: MouseEvent): void {
    const { x, y } = this.local(ev); const hit = this.hitTest(x, y);
    this.sel = hit; this.needsDraw = true; this.cb.onSelect?.(hit);
  }
  /** Datenzugriff für den Inspektor (Anzeige-Index → Werte). */
  info(ref: CellRef) {
    const dcol = ref.col >= 0 ? this.colOrder[ref.col] : -1, drow = ref.row >= 0 ? this.rowOrder[ref.row] : -1;
    return {
      context: drow >= 0 ? this.p.contexts[drow] : null,
      type: dcol >= 0 ? this.p.types[dcol] : null,
      value: drow >= 0 && dcol >= 0 ? this.p.matrix[drow][dcol] : null,
      colMeta: dcol >= 0 ? this.p.columnMetadata[this.p.types[dcol]] : null,
      rowMeta: drow >= 0 ? this.p.rowMetadata[this.p.contexts[drow]] : null,
      displayRow: ref.row, displayCol: ref.col,
    };
  }
  private syncProjectOrder(): void {
    if (!this.p) return;
    this.p.order = { rows: Array.from(this.rowOrder, (i) => this.p.contexts[i]), cols: Array.from(this.colOrder, (i) => this.p.types[i]) };
  }
  private syncFixedToMeta(): void {
    if (!this.p) return;
    this.p.contexts.forEach((c, i) => { const m = this.p.rowMetadata[c]; if (m) m.isFixed = this.rowFixed.has(i); });
    this.p.types.forEach((t, j) => { const m = this.p.columnMetadata[t]; if (m) m.isFixed = this.colFixed.has(j); });
  }
}

/* ── Helfer ── */
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", ""); const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [parseInt(n.slice(0, 2), 16) || 0, parseInt(n.slice(2, 4), 16) || 0, parseInt(n.slice(4, 6), 16) || 0];
}
function clip(s: string, n: number): string { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
