/**
 * Tests für die Score-Auslagerung (§8.5).
 *
 * Deckt drei Ebenen ab:
 *  1. Refactor-Äquivalenz: `qualityFromMatrix` liefert byte-identisch dasselbe
 *     wie `quality` (der Umbau darf die Zahlen nicht verändern).
 *  2. Worker-Kern (`handleScoreRequest`) inkl. Epoch-Matrix-Cache: Matrix nur
 *     einmal senden, Wiederverwendung bei gleicher Epoch, Cache-Miss → „stale".
 *  3. Client-Fallback (`scoreCompute.score`) im Nicht-Worker-Kontext: synchron,
 *     Ergebnis gleich `quality`, Gewichte werden durchgereicht.
 */
import { quality, qualityFromMatrix, DEFAULT_WEIGHTS } from "../seriation/metrics.js";
import { seriateCentroid } from "../seriation/centroid.js";
import type { ProjectV2 } from "../core/model.js";
import { createScoreCache, handleScoreRequest } from "./scoreCore.js";
import { scoreCompute } from "./scoreClient.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const same = (a: { concentration: number; antiRobinson: number; continuity: number; total: number }, b: typeof a) =>
  a.concentration === b.concentration && a.antiRobinson === b.antiRobinson && a.continuity === b.continuity && a.total === b.total;

function mk(M: number[][]): ProjectV2 {
  const NR = M.length, NC = M[0].length;
  const contexts = Array.from({ length: NR }, (_, i) => "G" + i), types = Array.from({ length: NC }, (_, j) => "T" + j);
  const columnMetadata: any = {}, rowMetadata: any = {};
  types.forEach(t => columnMetadata[t] = { name: t, materialGroup: "U", color: "#808080", isIndexType: false, isFixed: false, notes: "" });
  contexts.forEach(cx => rowMetadata[cx] = { name: cx, contextType: "", area: "", isFixed: false, notes: "" });
  return { schemaVersion: 2, name: "t", dataType: "frequency", contexts, types, matrix: M, columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { U: "#808080" }, contextTypes: [], order: { rows: [...contexts], cols: [...types] }, view: { vizStyle: "", cellSize: 1, showValues: true, showColors: true, showCertainty: false, showFragmentation: false }, filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false }, history: [] };
}
function band(NR: number, W: number): number[][] {
  const NC = NR + W - 1; const M: number[][] = [];
  for (let i = 0; i < NR; i++) { const row = new Array(NC).fill(0); for (let j = i; j < i + W; j++) row[j] = 1; M.push(row); }
  return M;
}
const canon = (p: ProjectV2) => {
  const ri = new Map(p.contexts.map((c, i) => [c, i] as const));
  const ci = new Map(p.types.map((t, j) => [t, j] as const));
  return { rows: p.order.rows.map((r) => ri.get(r) ?? 0), cols: p.order.cols.map((c) => ci.get(c) ?? 0) };
};

console.log("\n\x1b[1mScore-Auslagerung (§8.5)\x1b[0m\n");

// ── 1) Refactor-Äquivalenz quality ≡ qualityFromMatrix ──
{
  const cases: Array<{ p: ProjectV2; order: { rows: number[]; cols: number[] } }> = [];
  const p1 = mk(band(12, 4)); cases.push({ p: p1, order: canon(p1) });
  const p2 = mk([[3, 0, 1], [0, 2, 0], [1, 0, 4], [0, 5, 0]]); cases.push({ p: p2, order: canon(p2) });
  const p3 = mk(band(20, 6)); cases.push({ p: p3, order: seriateCentroid(p3, 15, 42) });
  const p4 = mk([[0, 0], [0, 0]]); cases.push({ p: p4, order: canon(p4) }); // leer → 0-Schutz
  let allEqual = true, worst = "";
  for (const { p, order } of cases) {
    const a = quality(p, order);
    const b = qualityFromMatrix(p.matrix, p.contexts.length, p.types.length, order);
    if (!same(a, b)) { allEqual = false; worst = `${a.total} vs ${b.total}`; }
  }
  c("quality ≡ qualityFromMatrix (identische Zahlen über 4 Matrizen)", allEqual, worst);
}

// Gewichte werden respektiert (verändern total, nicht die Teil-Kennzahlen)
{
  const p = mk(band(10, 3)); const order = canon(p);
  const base = qualityFromMatrix(p.matrix, 10, order.cols.length, order);
  const w = { concentration: 1, antiRobinson: 0, continuity: 0 };
  const only = qualityFromMatrix(p.matrix, 10, order.cols.length, order, w);
  c("Gewichte übersteuern total korrekt", Math.abs(only.total - base.concentration) < 1e-12 && only.concentration === base.concentration);
}

// ── 2) Worker-Kern + Epoch-Cache ──
{
  const p = mk(band(15, 5)); const order = seriateCentroid(p, 15, 7);
  const NC = p.types.length;
  const ref = qualityFromMatrix(p.matrix, 15, NC, order);
  const cache = createScoreCache();

  // (a) Erste Anfrage MIT Matrix → done, korrekt
  const r1 = handleScoreRequest(cache, { id: 1, epoch: 5, matrix: p.matrix, rows: order.rows, cols: order.cols });
  c("Kern: erste Anfrage mit Matrix → done", r1.type === "done" && same((r1 as any).result, ref));

  // (b) Zweite Anfrage OHNE Matrix, gleiche Epoch → Cache-Treffer, korrekt
  const order2 = canon(p);
  const ref2 = qualityFromMatrix(p.matrix, 15, NC, order2);
  const r2 = handleScoreRequest(cache, { id: 2, epoch: 5, rows: order2.rows, cols: order2.cols });
  c("Kern: zweite Anfrage ohne Matrix (gleiche Epoch) nutzt Cache", r2.type === "done" && same((r2 as any).result, ref2));

  // (c) Andere Epoch OHNE Matrix → stale
  const r3 = handleScoreRequest(cache, { id: 3, epoch: 6, rows: order.rows, cols: order.cols });
  c("Kern: neue Epoch ohne Matrix → stale", r3.type === "stale");

  // (d) Neue Epoch MIT Matrix → done, Cache aktualisiert
  const p2 = mk(band(15, 3)); const o4 = canon(p2);
  const ref4 = qualityFromMatrix(p2.matrix, 15, p2.types.length, o4);
  const r4 = handleScoreRequest(cache, { id: 4, epoch: 6, matrix: p2.matrix, rows: o4.rows, cols: o4.cols });
  c("Kern: neue Epoch mit Matrix → done + Cache-Update", r4.type === "done" && same((r4 as any).result, ref4) && cache.epoch === 6);

  // (e) Fehler im Kern wird als error gemeldet (defekte Ordnung erzwingt Zugriff auf undefinierte Zeile)
  const bad = handleScoreRequest(cache, { id: 5, epoch: 6, rows: [0, 1, 999], cols: o4.cols });
  c("Kern: Ausnahme → error-Antwort (kein Absturz)", bad.type === "error");
}

// ── 3) Client-Fallback ohne Worker (Node-/Testkontext) ──
{
  const noWorker = typeof (globalThis as any).Worker === "undefined";
  c("Testumgebung hat keinen Worker (synchroner Fallback aktiv)", noWorker);
}

async function clientTests() {
  const p = mk(band(18, 5)); const order = seriateCentroid(p, 15, 3);
  const ref = quality(p, order);
  const viaClient = await scoreCompute.score(p.matrix, 1, order);
  c("Client-Fallback: score() ≡ quality()", same(viaClient, ref), `${viaClient.total} vs ${ref.total}`);

  // Gewichte über den Client
  const w = { concentration: 0, antiRobinson: 0, continuity: 1 };
  const cont = await scoreCompute.score(p.matrix, 1, order, w);
  c("Client-Fallback: Gewichte werden durchgereicht", Math.abs(cont.total - ref.continuity) < 1e-12);

  // Mehrere rasch aufeinanderfolgende Aufrufe lösen alle auf (kein Hängen)
  const rs = await Promise.all([canon(p), order, canon(p)].map((o) => scoreCompute.score(p.matrix, 2, o)));
  c("Client-Fallback: mehrere Aufrufe lösen alle auf", rs.length === 3 && rs.every((r) => typeof r.total === "number"));

  console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
  if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
  else console.log("\x1b[32m✓ Score-Auslagerung (§8.5) korrekt.\x1b[0m");
}

void clientTests();
void DEFAULT_WEIGHTS;
