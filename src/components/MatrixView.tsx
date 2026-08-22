import { useEffect, useRef, useState, useCallback, type MutableRefObject, type PointerEvent as RPointerEvent } from "react";
import type { ProjectV2, FilterSettings } from "../core/model.js";
import { MatrixRenderer, type CellRef, type OrderSnapshot } from "../matrix/MatrixRenderer.js";
import { compute } from "../workers/client.js";
import { scoreCompute, nextScoreEpoch } from "../workers/scoreClient.js";
import { quality, type Quality } from "../seriation/metrics.js";
import { buildMissingMask } from "../core/missing.js";
import { Inspector } from "./Inspector.js";
import { AnnotationEditor } from "./AnnotationEditor.js";
import { annotationCount } from "../annotations/annotations.js";
import { useLink, activeCtx, activeType } from "../link.js";
import { emptyFilters, writeBackOrder, writeBackAnnotations, writeBackMissing } from "../core/filter.js";

// Perf-HUD-Typen aus dem Renderer abgeleitet, damit sie synchron bleiben.
type PerfStats = MatrixRenderer["perfStats"];
type BenchResult = Awaited<ReturnType<MatrixRenderer["benchmark"]>>;
import { METHOD_LABELS, METHOD_HISTORY, type SeriationMethod } from "../seriation/strategies.js";
import { readMatrixTheme, onThemeChange } from "../core/theme.js";
import { useT } from "../i18n/I18nContext.js";

/** Globale Shortcuts (Strg+Z, Umschalt+P) dürfen nicht in Textfelder
 *  eingreifen — sonst schluckt Umschalt+P das große „P" in Notizen/Suche und
 *  Strg+Z macht statt der Texteingabe die Matrix-Ordnung rückgängig. */
function isEditingTarget(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

export function MatrixView({ project, baseProject, filters, onFilters, focusOn = false, onFocusToggle, filterActive = false }: {
  project: ProjectV2;
  baseProject?: ProjectV2;
  filters?: FilterSettings;
  onFilters?: (f: FilterSettings) => void;
  focusOn?: boolean;
  onFocusToggle?: () => void;
  filterActive?: boolean;
}) {
  const glRef = useRef<HTMLCanvasElement>(null);
  const ovRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendRef = useRef<MatrixRenderer | null>(null);
  const undoRef = useRef<OrderSnapshot[]>([]);
  const redoRef = useRef<OrderSnapshot[]>([]);
  const dragPrev = useRef<OrderSnapshot | null>(null);
  const link = useLink();
  const linkRef = useRef(link); linkRef.current = link;
  const t = useT();

  const [sel, setSel] = useState<CellRef | null>(null);
  const [cell, setCell] = useState(18);
  const [q, setQ] = useState<Quality>(() => initialQuality(project));
  const [scoring, setScoring] = useState(false);
  const [prevTotal, setPrevTotal] = useState<number | null>(null);
  const [hist, setHist] = useState(0);
  const [supported, setSupported] = useState(true);
  const [backend, setBackend] = useState<"webgl2" | "canvas2d" | "none">("webgl2");
  // Perf-HUD — standardmäßig aus, kein Overhead solange geschlossen
  const [hud, setHud] = useState(false);
  const [perf, setPerf] = useState<PerfStats | null>(null);
  const [bench, setBench] = useState<BenchResult | null>(null);
  const [benching, setBenching] = useState(false);
  const [mode, setMode] = useState<"navigate" | "select">("navigate");
  const [areaCells, setAreaCells] = useState<Array<[number, number]>>([]);
  const [annVer, setAnnVer] = useState(0);
  const [seriating, setSeriating] = useState(false);
  const [method, setMethod] = useState<SeriationMethod>("centroid");
  const [caDim, setCaDim] = useState(0);
  const maxCaDim = Math.max(1, Math.min(4, Math.min(project.contexts.length, project.types.length) - 1));
  const projectRef = useRef(project); projectRef.current = project;
  // §8.5: der zuletzt bekannte Score (für „vorher"-Vergleich ohne Neuberechnung),
  // eine global eindeutige Matrix-Epoch und ein Veralterungs-Zähler (nur das
  // jüngste Async-Ergebnis darf die Oberfläche aktualisieren).
  const qRef = useRef<Quality>(q);
  const epochRef = useRef(0);
  const scoreReqRef = useRef(0);
  const setScore = useCallback((res: Quality) => { qRef.current = res; setQ(res); }, []);

  // §8.5: Score-Neuberechnung asynchron im Worker (früher synchron im Haupt-Thread,
  // bei 1.000×1.000 mehrsekündiger Freeze nach jedem Drop/Undo/Seriation). Liest
  // stets die aktuellen Werte über Refs, damit die einmalig verdrahteten Renderer-
  // Callbacks nicht auf einem veralteten Projekt hängen. Nur das jüngste Ergebnis
  // wird angezeigt (Veralterungs-Zähler); ein harter Worker-Fehler fällt synchron
  // zurück, damit die Anzeige nie „hängen" bleibt.
  const refreshScore = useCallback(() => {
    const r = rendRef.current; if (!r) return;
    const proj = projectRef.current;
    const myId = ++scoreReqRef.current;
    setScoring(true);
    scoreCompute.score(proj.matrix, epochRef.current, r.order, undefined, buildMissingMask(proj))
      .then((res) => { if (myId === scoreReqRef.current) { setScore(res); setScoring(false); } })
      .catch((e) => {
        if (myId !== scoreReqRef.current) return;
        if ((e as Error)?.name !== "AbortError") {
          try { setScore(quality(projectRef.current, r.order)); } catch { /* ignorieren */ }
        }
        setScoring(false);
      });
  }, [setScore]);
  const commit = useCallback((prev: OrderSnapshot) => {
    undoRef.current.push(prev); if (undoRef.current.length > 100) undoRef.current.shift();
    redoRef.current = []; setHist((h) => h + 1);
  }, []);

  // Editierbare gefilterte Sicht: in der Teilmenge erzeugte Ordnung/Fixierung
  // über die stabilen Namen ins Grundprojekt zurückspiegeln (bei ungefilterter
  // Ansicht ist baseProject === project → No-op).
  const writeBack = useCallback(() => {
    const base = baseProject, v = project;
    if (!base || base === v) return;
    base.order = { rows: writeBackOrder(base.order.rows, v.order.rows), cols: writeBackOrder(base.order.cols, v.order.cols) };
    for (const name of v.contexts) { const bm = base.rowMetadata[name], vm = v.rowMetadata[name]; if (bm && vm) bm.isFixed = vm.isFixed; }
    for (const name of v.types) { const bm = base.columnMetadata[name], vm = v.columnMetadata[name]; if (bm && vm) bm.isFixed = vm.isFixed; }
  }, [baseProject, project]);
  const writeBackRef = useRef(writeBack); writeBackRef.current = writeBack;

  async function doSeriate() {
    const r = rendRef.current; if (!r || seriating) return;
    const captured = project;
    // §8.5: „vorher" ist der bereits angezeigte Score — keine synchrone Neuberechnung.
    const prev = r.getSnapshot(); const before = qRef.current.total;
    const seed = 12345;
    setSeriating(true);
    try {
      const order = await compute.seriate(captured, { method, seed, caDim });
      // Projektwechsel während der Berechnung? Ergebnis verwerfen.
      const rr = rendRef.current; if (!rr || projectRef.current !== captured) return;
      rr.applySeriationOrder(order.rows, order.cols);
      commit(prev); setPrevTotal(before); writeBack();
      // §8.5: Ergebnis-Score asynchron (kein Freeze); History-Eintrag folgt, sobald bekannt.
      const myId = ++scoreReqRef.current;
      setScoring(true);
      const res = await scoreCompute.score(captured.matrix, epochRef.current, rr.order, undefined, buildMissingMask(captured));
      if (projectRef.current !== captured) return;
      const histMethod = method === "ca" ? `correspondence analysis seriation (CA-Dim ${caDim + 1})` : METHOD_HISTORY[method];
      (baseProject ?? captured).history.push({ method: histMethod, params: method === "ca" ? { method, caDim } : { method, seed }, timestamp: new Date().toISOString(), score: res.total });
      if (myId === scoreReqRef.current) { setScore(res); setScoring(false); }
    } catch (e) {
      if ((e as Error)?.name !== "AbortError") console.error(e);
      setScoring(false);
    } finally {
      setSeriating(false);
    }
  }
  function doUndo() { const r = rendRef.current; if (!r || !undoRef.current.length) return; redoRef.current.push(r.getSnapshot()); r.restore(undoRef.current.pop()!); setPrevTotal(null); setHist((h) => h + 1); refreshScore(); writeBack(); }
  function doRedo() { const r = rendRef.current; if (!r || !redoRef.current.length) return; undoRef.current.push(r.getSnapshot()); r.restore(redoRef.current.pop()!); setPrevTotal(null); setHist((h) => h + 1); refreshScore(); writeBack(); }
  function togglePin(axis: "row" | "col", displayPos: number) { const r = rendRef.current; if (!r) return; const prev = r.getSnapshot(); r.toggleFix(axis, displayPos); commit(prev); writeBack(); setSel((s) => (s ? { ...s } : s)); }
  function doFit() { rendRef.current?.fit(); }
  function onZoom(v: number) { setCell(v); rendRef.current?.setCell(v); }
  function switchMode(m: "navigate" | "select") { setMode(m); rendRef.current?.setMode(m); if (m === "navigate") { setAreaCells([]); rendRef.current?.clearArea(); } }
  function afterAnnotate() {
    if (baseProject && baseProject !== project) { writeBackAnnotations(baseProject, project); writeBackMissing(baseProject, project); }
    rendRef.current?.refresh(); setAnnVer((v) => v + 1);
    // §9.6: „nicht erfasst"-Markierungen können sich geändert haben → Fehlwert-Maske
    // neu übertragen (frische Epoch) und Score masken-bewusst neu berechnen.
    epochRef.current = nextScoreEpoch(); refreshScore();
  }

  useEffect(() => {
    if (!glRef.current || !ovRef.current) return;
    const r = new MatrixRenderer(glRef.current, ovRef.current, {
      onSelect: (ref) => { setSel(ref); const info = ref ? r.info(ref) : null; linkRef.current.setSel(info?.context ?? linkRef.current.selCtx, info?.type ?? linkRef.current.selType); },
      onHover: (ref) => { if (!ref) { linkRef.current.clearHover(); return; } const info = r.info(ref); linkRef.current.setHover(info.context, info.type); },
      onReorderStart: () => { dragPrev.current = r.getSnapshot(); },
      onReorder: () => { if (dragPrev.current) { commit(dragPrev.current); dragPrev.current = null; } refreshScore(); writeBackRef.current(); },
      onChange: () => { setHist((h) => h + 1); refreshScore(); },
      onAreaSelect: (cells) => setAreaCells(cells),
    });
    rendRef.current = r; setSupported(r.supported); setBackend(r.backend);
    r.setTheme(readMatrixTheme()); // aktuelle Hell/Dunkel-Farben
    const offTheme = onThemeChange(() => r.setTheme(readMatrixTheme()));
    if (import.meta.env.DEV) (window as unknown as { __app?: unknown }).__app = { renderer: r, undo: () => doUndo(), redo: () => doRedo(), seriate: () => doSeriate(), pin: (a: "row" | "col", p: number) => togglePin(a, p) };
    const ro = new ResizeObserver(() => r.resize()); if (wrapRef.current) ro.observe(wrapRef.current); r.resize();
    return () => { offTheme(); ro.disconnect(); r.destroy(); rendRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const r = rendRef.current; if (!r) return;
    r.setProject(project); r.resize();
    undoRef.current = []; redoRef.current = []; setHist(0);
    setSel(null); setPrevTotal(null);
    // §8.5: neue Matrix-Identität → frische Epoch (Worker sendet Matrix neu).
    epochRef.current = nextScoreEpoch();
    setScore(initialQuality(project)); // sofortiger Wert (bzw. Platzhalter bei großen Matrizen)
    refreshScore();                    // exakter Wert asynchron, ohne Freeze
    setCell(Math.round((r as unknown as { view: { cell: number } }).view.cell));
  }, [project, refreshScore, setScore]);

  // verlinkte Hervorhebung aus anderen Ansichten in die Matrix spiegeln
  useEffect(() => { rendRef.current?.setLinked(activeCtx(link), activeType(link)); }, [link]);

  // Achsentitel bei Sprachwechsel aktualisieren
  useEffect(() => { rendRef.current?.setAxisLabels(t("matrix.axis.context"), t("matrix.axis.type")); }, [t]);

  // Perf-HUD: Shift+P schaltet um
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (isEditingTarget(e)) return; if (e.shiftKey && (e.key === "P" || e.key === "p") && !e.repeat) { e.preventDefault(); setHud((v) => !v); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Messung nur solange das HUD offen ist (sonst null Overhead)
  useEffect(() => {
    const r = rendRef.current; if (!r || !hud) return;
    r.setProfiling(true);
    const id = window.setInterval(() => { const rr = rendRef.current; if (rr) setPerf(rr.perfStats); }, 250);
    return () => { window.clearInterval(id); rendRef.current?.setProfiling(false); setPerf(null); };
  }, [hud]);

  const runBench = useCallback(() => {
    const r = rendRef.current; if (!r || benching) return;
    setBenching(true); setBench(null);
    r.benchmark(2000).then((res) => { setBench(res); setBenching(false); });
  }, [benching]);

  // Gefilterte Ansicht ist vollständig editierbar; Änderungen (Umsortieren,
  // Seriation, Fixieren, Annotieren) wirken auf die Teilmenge und werden über die
  // stabilen Namen ins Grundprojekt zurückgeschrieben.
  useEffect(() => { rendRef.current?.setInteractive(true); }, [filterActive]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isEditingTarget(e)) return;
      const mod = e.ctrlKey || e.metaKey; if (!mod) return;
      if (e.key.toLowerCase() === "z" && !e.shiftKey) { e.preventDefault(); doUndo(); }
      else if ((e.key.toLowerCase() === "z" && e.shiftKey) || e.key.toLowerCase() === "y") { e.preventDefault(); doRedo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const up = prevTotal !== null && q.total > prevTotal;
  const canUndo = undoRef.current.length > 0, canRedo = redoRef.current.length > 0;

  return (
    <>
      <div className="sec-hdr">
        <h2>{t("matrix.title")}</h2>
        <span className="hint">{t("matrix.dragHint")}</span>
      </div>
      <div className="frow">
        <button className="btn" onClick={doSeriate} disabled={seriating}>{seriating ? t("matrix.seriating") : t("matrix.seriate")}</button>
        <select className="sl" value={method} onChange={(e) => setMethod(e.target.value as SeriationMethod)} disabled={seriating} title={t("matrix.methodTitle")}>
          {(Object.keys(METHOD_LABELS) as SeriationMethod[]).map((m) => <option key={m} value={m}>{t("matrix.method." + m)}</option>)}
        </select>
        {method === "ca" && (
          <select className="sl" value={Math.min(caDim, maxCaDim - 1)} onChange={(e) => setCaDim(+e.target.value)} disabled={seriating} title={t("matrix.caDimTitle")}>
            {Array.from({ length: maxCaDim }, (_, d) => d).map((d) => <option key={d} value={d}>{t("matrix.byDim", { n: d + 1 })}</option>)}
          </select>
        )}
        <button className="btn btn-ghost" onClick={doUndo} disabled={!canUndo} title={t("matrix.undoTitle")}>↶ {t("matrix.undo")}</button>
        <button className="btn btn-ghost" onClick={doRedo} disabled={!canRedo} title={t("matrix.redoTitle")}>↷ {t("matrix.redo")}</button>
        <button className="btn btn-ghost" onClick={doFit}>{t("matrix.fit")}</button>
        <span className="tb-label" style={{ marginLeft: ".4rem" }}>{t("matrix.modeLabel")}</span>
        <div className="seg">
          <button className={"seg-b" + (mode === "navigate" ? " on" : "")} onClick={() => switchMode("navigate")}>{t("matrix.mode.navigate")}</button>
          <button className={"seg-b" + (mode === "select" ? " on" : "")} onClick={() => switchMode("select")}>{t("matrix.mode.select")}</button>
        </div>
        <span className="tb-label">{t("matrix.cellSize")}</span>
        {/* Untergrenze identisch zur Klemmung im Renderer (setCell: 2..60) —
            sonst ist die kleinste Zellgröße über die Oberfläche nicht erreichbar. */}
        <input className="zoom" type="range" min={2} max={40} value={cell} onChange={(e) => onZoom(+e.target.value)} />
        <div className={"score" + (scoring ? " scoring" : "")} aria-busy={scoring}>
          <span className="sc">{t("matrix.score.concentration")}<b>{q.concentration.toFixed(3)}</b></span>
          <span className="sc" title={t("matrix.score.antiRobinsonTitle")}>{t("matrix.score.antiRobinson")}<b>{q.antiRobinson.toFixed(3)}</b></span>
          <span className="sc">{t("matrix.score.continuity")}<b>{q.continuity.toFixed(3)}</b></span>
          <span className={"sc" + (up ? " up" : "")}>{t("matrix.score.total")}<b>{q.total.toFixed(3)}</b>
            {scoring && <span className="sc-spin" role="status" aria-live="polite" title={t("matrix.score.updating")} aria-label={t("matrix.score.updating")} />}
          </span>
        </div>
        <button className={"btn btn-ghost perf-btn" + (hud ? " on" : "")} onClick={() => setHud((v) => !v)} title={t("perf.toggleTitle")} aria-label={t("perf.toggleTitle")} aria-pressed={hud}>⏱</button>
      </div>
      {baseProject && filters && onFilters && (
        <FilterBar base={baseProject} view={project} filters={filters} onFilters={onFilters}
          focusOn={focusOn} onFocusToggle={onFocusToggle} active={filterActive} link={link} />
      )}
      {filterActive && (
        <div className="filter-banner">{t("filter.banner")}</div>
      )}
      <div className="mx-grid">
        <div className="mx-wrap" ref={wrapRef}>
          <canvas className="gl" ref={glRef} onClick={(e) => rendRef.current?.click(e.nativeEvent)}
            role="img" aria-label={t("a11y.matrixLabel", { rows: project.contexts.length, cols: project.types.length })} />
          <canvas ref={ovRef} style={{ pointerEvents: "none" }} aria-hidden="true" />
          <div className="mx-hint">{
            backend === "none"
              ? t("matrix.hint.none")
              : (backend === "canvas2d" ? t("matrix.hint.canvas2d") : "") + t("matrix.hint.default")
          }</div>
          {supported && <Minimap project={project} rendRef={rendRef} refreshKey={hist} />}
          {hud && <PerfHud perf={perf} bench={bench} benching={benching} onBench={runBench} onClose={() => setHud(false)} t={t} />}
        </div>
        {mode === "select"
          ? <AnnotationEditor project={project} cells={areaCells} onApplied={afterAnnotate} onClear={afterAnnotate} />
          : <Inspector project={project} sel={sel} renderer={rendRef.current} onTogglePin={togglePin} />}
      </div>
      <Legend project={project} annCount={annotationCount(project)} annVer={annVer} />
    </>
  );
}

function Legend({ project, annCount, annVer }: { project: ProjectV2; annCount: number; annVer: number }) {
  void annVer;
  const t = useT();
  return (
    <div className="legend">
      <span className="tb-label">{t("matrix.legend.materials")}</span>
      {Object.entries(project.materialGroups).map(([name, color]) => (
        <span className="l" key={name}><span className="dot" style={{ background: color }} />{name}</span>
      ))}
      <span className="l" style={{ marginLeft: "auto" }}><span style={{ width: 6, height: 6, borderRadius: "50%", background: "#a81d26", display: "inline-block" }} /> {t("matrix.legend.fixed")}</span>
      <span className="l">{annCount} {t("matrix.legend.annotations")}</span>
    </div>
  );
}

function hexToRgb(hex: string): [number, number, number] {
  let h = (hex || "").replace("#", ""); if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h || "3a6ea5", 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Übersichts-Minimap: heruntergerechnete Dichtekarte der Matrix (in Anzeige-
 *  Reihenfolge, materialgefärbt) mit Viewport-Rechteck. Klicken/Ziehen zentriert
 *  das Sichtfenster — schnelle Navigation bei großen Matrizen (Spec §4.5). */
function Minimap({ project, rendRef, refreshKey }: { project: ProjectV2; rendRef: MutableRefObject<MatrixRenderer | null>; refreshKey: number }) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const densRef = useRef<HTMLCanvasElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const NR = project.contexts.length, NC = project.types.length;
  const boxW = 196;
  const boxH = Math.round(Math.max(64, Math.min(240, (boxW * NR) / Math.max(1, NC))));

  // Dichtekarte (offscreen) bei Daten-/Ordnungsänderung neu aufbauen
  useEffect(() => {
    if (NR < 2 || NC < 2) return;
    const r = rendRef.current;
    let ord: { rows: number[]; cols: number[] };
    if (r) { ord = r.order; }
    else {
      const ri = new Map(project.contexts.map((c, i) => [c, i] as const));
      const ci = new Map(project.types.map((t, j) => [t, j] as const));
      ord = { rows: project.order.rows.map((n) => ri.get(n) ?? 0), cols: project.order.cols.map((n) => ci.get(n) ?? 0) };
    }
    const bw = Math.min(NC, boxW), bh = Math.min(NR, boxH);
    const dens = document.createElement("canvas"); dens.width = bw; dens.height = bh;
    const dctx = dens.getContext("2d"); if (!dctx) return;
    const img = dctx.createImageData(bw, bh);
    const cnt = new Float32Array(bw * bh); let maxc = 1;
    const M = project.matrix;
    for (let i = 0; i < NR; i++) { const rr = ord.rows[i], py = Math.min(bh - 1, ((i * bh) / NR) | 0);
      for (let j = 0; j < NC; j++) { if (M[rr][ord.cols[j]]) { const px = Math.min(bw - 1, ((j * bw) / NC) | 0), idx = py * bw + px; cnt[idx]++; if (cnt[idx] > maxc) maxc = cnt[idx]; } } }
    const colColor: Array<[number, number, number]> = [];
    for (let px = 0; px < bw; px++) { const dc = Math.min(NC - 1, (((px + 0.5) * NC) / bw) | 0); const t = project.types[ord.cols[dc]]; colColor[px] = hexToRgb(project.columnMetadata[t]?.color || "#3a6ea5"); }
    for (let py = 0; py < bh; py++) for (let px = 0; px < bw; px++) { const idx = py * bw + px, o = idx * 4, cc = cnt[idx];
      if (cc > 0) { const [cr, cg, cb] = colColor[px]; img.data[o] = cr; img.data[o + 1] = cg; img.data[o + 2] = cb; img.data[o + 3] = Math.round(90 + 165 * Math.min(1, cc / maxc)); } else { img.data[o + 3] = 0; } }
    dctx.putImageData(img, 0, 0);
    densRef.current = dens;
  }, [project, refreshKey, NR, NC, boxH]);

  // Viewport-Rechteck jede Frame über der Dichtekarte zeichnen
  useEffect(() => {
    if (NR < 2 || NC < 2) return;
    let raf = 0;
    const draw = () => {
      const cv = canvasRef.current, dens = densRef.current;
      if (cv && dens) {
        const g = cv.getContext("2d");
        if (g) {
          g.clearRect(0, 0, boxW, boxH); g.imageSmoothingEnabled = false;
          g.drawImage(dens, 0, 0, boxW, boxH);
          const r = rendRef.current;
          if (r) { const v = r.getViewport();
            const x = (v.c0 / v.NC) * boxW, y = (v.r0 / v.NR) * boxH, w = (v.cols / v.NC) * boxW, h = (v.rows / v.NR) * boxH;
            g.fillStyle = "rgba(210,38,48,.10)"; g.fillRect(x, y, w, h);
            g.strokeStyle = "#d22630"; g.lineWidth = 1.5; g.strokeRect(x + 0.75, y + 0.75, Math.max(3, w - 1.5), Math.max(3, h - 1.5));
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [boxW, boxH, NR, NC]);

  const nav = (e: RPointerEvent<HTMLCanvasElement>) => {
    const r = rendRef.current, cv = canvasRef.current; if (!r || !cv) return;
    const rect = cv.getBoundingClientRect();
    const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    r.centerOnCell(fy * NR, fx * NC);
  };

  if (NR < 2 || NC < 2) return null;
  return (
    <div className="minimap" title={t("matrix.minimap")}>
      <canvas ref={canvasRef} width={boxW} height={boxH}
        onPointerDown={(e) => { setDragging(true); e.currentTarget.setPointerCapture(e.pointerId); nav(e); }}
        onPointerMove={(e) => { if (dragging) nav(e); }}
        onPointerUp={() => setDragging(false)} onPointerCancel={() => setDragging(false)} />
    </div>
  );
}

function FilterBar({ base, view, filters, onFilters, focusOn, onFocusToggle, active, link }: {
  base: ProjectV2; view: ProjectV2; filters: FilterSettings; onFilters: (f: FilterSettings) => void;
  focusOn: boolean; onFocusToggle?: () => void; active: boolean;
  link: { selCtx: string | null; selType: string | null };
}) {
  const t = useT();
  const groups = Object.entries(base.materialGroups);
  const toggleMat = (name: string) => {
    const has = filters.materials.includes(name);
    onFilters({ ...filters, materials: has ? filters.materials.filter((m) => m !== name) : [...filters.materials, name] });
  };
  const canFocus = !!(link.selCtx || link.selType);
  const focusTarget = [link.selCtx, link.selType].filter(Boolean).join(" · ");
  const reset = () => { onFilters(emptyFilters()); if (focusOn) onFocusToggle?.(); };

  const NRb = base.contexts.length, NCb = base.types.length;
  const rFrom = filters.rowRange ? filters.rowRange[0] + 1 : 1, rTo = filters.rowRange ? filters.rowRange[1] + 1 : NRb;
  const cFrom = filters.colRange ? filters.colRange[0] + 1 : 1, cTo = filters.colRange ? filters.colRange[1] + 1 : NCb;
  const setRow = (a: number, b: number) => { a = Math.max(1, Math.min(a || 1, NRb)); b = Math.max(a, Math.min(b || NRb, NRb)); onFilters({ ...filters, rowRange: a === 1 && b === NRb ? null : [a - 1, b - 1] }); };
  const setCol = (a: number, b: number) => { a = Math.max(1, Math.min(a || 1, NCb)); b = Math.max(a, Math.min(b || NCb, NCb)); onFilters({ ...filters, colRange: a === 1 && b === NCb ? null : [a - 1, b - 1] }); };

  return (
    <div className="filter-bar">
      <span className="tb-label">{t("filter.label")}</span>
      {groups.map(([name, color]) => {
        const on = filters.materials.includes(name);
        return (
          <button key={name} className={"fchip" + (on ? " on" : "")} onClick={() => toggleMat(name)} title={on ? t("filter.matHide", { name }) : t("filter.matIsolate", { name })}>
            <span className="dot" style={{ background: color }} />{name}
          </button>
        );
      })}
      <button className={"fchip" + (filters.hideEmptyRows ? " on" : "")} onClick={() => onFilters({ ...filters, hideEmptyRows: !filters.hideEmptyRows })}>{t("filter.hideEmptyRows")}</button>
      <button className={"fchip" + (filters.hideEmptyCols ? " on" : "")} onClick={() => onFilters({ ...filters, hideEmptyCols: !filters.hideEmptyCols })}>{t("filter.hideEmptyCols")}</button>
      <button className={"fchip" + (focusOn ? " on" : "")} onClick={onFocusToggle} disabled={!canFocus && !focusOn}
        title={canFocus ? t("filter.focusOn", { target: focusTarget }) : t("filter.focusHint")}>◎ {t("filter.focus")}</button>
      <span className="tb-label" title={t("filter.rangeTitle")}>{t("filter.rows")}</span>
      <input className="num" type="number" min={1} max={NRb} value={rFrom} onChange={(e) => setRow(+e.target.value, rTo)} />
      <span className="tb-label">–</span>
      <input className="num" type="number" min={1} max={NRb} value={rTo} onChange={(e) => setRow(rFrom, +e.target.value)} />
      <span className="tb-label" title={t("filter.rangeTitle")}>{t("filter.types")}</span>
      <input className="num" type="number" min={1} max={NCb} value={cFrom} onChange={(e) => setCol(+e.target.value, cTo)} />
      <span className="tb-label">–</span>
      <input className="num" type="number" min={1} max={NCb} value={cTo} onChange={(e) => setCol(cFrom, +e.target.value)} />
      {active && <button className="fchip reset" onClick={reset} title={t("filter.resetTitle")}>✕ {t("filter.resetBtn")}</button>}
      <span className="tb-label" style={{ marginLeft: "auto" }}>
        {active ? t("filter.summaryActive", { vc: view.contexts.length, bc: base.contexts.length, vt: view.types.length, bt: base.types.length }) : t("filter.summary", { c: base.contexts.length, ty: base.types.length })}
      </span>
    </div>
  );
}

/** Perf-HUD: Frame-/Draw-Zeiten, FPS und In-Browser-Benchmark. */
function PerfHud({ perf, bench, benching, onBench, onClose, t }: {
  perf: PerfStats | null; bench: BenchResult | null; benching: boolean;
  onBench: () => void; onClose: () => void; t: (k: string, v?: Record<string, string | number>) => string;
}) {
  const ms = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " ms";
  const fpsCls = (f: number) => (f >= 55 ? "ok" : f >= 30 ? "warn" : "bad");
  const msCls = (v: number) => (v <= 16 ? "ok" : v <= 33 ? "warn" : "bad");
  const num = (n: number) => n.toLocaleString("de-AT");
  return (
    <div className="perf-hud" role="status" aria-live="polite">
      <div className="perf-hd"><span>{t("perf.title")}</span><button className="perf-x" onClick={onClose} aria-label={t("perf.close")}>✕</button></div>
      <div className="perf-grid">
        <span>Backend</span><b>{perf?.backend ?? "—"}</b>
        <span>{t("perf.cells")}</span><b>{perf ? num(perf.cells) : "—"}</b>
        <span>{t("perf.visible")}</span><b>{perf ? num(perf.visibleCells) : "—"}</b>
        <span>FPS</span><b className={perf ? fpsCls(perf.fps) : ""}>{perf ? perf.fps.toFixed(0) : "—"}</b>
        <span>{t("perf.frame")} ø / p95</span><b className={perf ? msCls(perf.frameP95) : ""}>{perf ? `${ms(perf.frameAvg)} / ${ms(perf.frameP95)}` : "—"}</b>
        <span>{t("perf.draw")} ø / p95</span><b className={perf ? msCls(perf.drawP95) : ""}>{perf ? `${ms(perf.drawAvg)} / ${ms(perf.drawP95)}` : "—"}</b>
      </div>
      <button className="btn btn-ghost perf-run" onClick={onBench} disabled={benching}>{benching ? t("perf.running") : t("perf.benchmark")}</button>
      {bench && (
        <div className="perf-bench">
          <div className="perf-grid">
            <span>{t("perf.result")}</span><b className={fpsCls(bench.fps)}>{bench.fps.toFixed(0)} FPS</b>
            <span>{t("perf.frame")} ø / p95</span><b className={msCls(bench.frameP95)}>{ms(bench.frameAvg)} / {ms(bench.frameP95)}</b>
            <span>{t("perf.draw")} ø / p95</span><b className={msCls(bench.drawP95)}>{ms(bench.drawAvg)} / {ms(bench.drawP95)}</b>
            <span>{t("perf.frames")}</span><b>{bench.frames} · {num(bench.visibleCells)} {t("perf.cellsShort")}</b>
          </div>
        </div>
      )}
      <div className="perf-note">{t("perf.note")}</div>
    </div>
  );
}

function canonOrder(p: ProjectV2) {
  const rIdx = new Map(p.contexts.map((c, i) => [c, i] as const));
  const cIdx = new Map(p.types.map((t, j) => [t, j] as const));
  return { rows: p.order.rows.map((r) => rIdx.get(r) ?? 0), cols: p.order.cols.map((c) => cIdx.get(c) ?? 0) };
}

/**
 * Sofort-Score beim Projektwechsel (§8.5). Bei kleinen Matrizen (≤ 200×200, unter
 * dem 16-ms-Budget) synchron exakt; bei großen ein neutraler Platzhalter, damit
 * der Wechsel nicht blockiert — der exakte Wert kommt unmittelbar danach aus dem
 * Worker. So blockiert weder Erstladen noch Filterwechsel den Haupt-Thread.
 */
function initialQuality(p: ProjectV2): Quality {
  const cells = p.contexts.length * p.types.length;
  if (cells <= 40000) return quality(p, canonOrder(p));
  return { concentration: 0, antiRobinson: 0, continuity: 0, total: 0 };
}
