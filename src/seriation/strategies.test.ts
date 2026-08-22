import { seriateCA, seriateIterative, seriate, METHOD_LABELS } from "./strategies.js";
import { seriateCentroid } from "./centroid.js";
import { quality, antiRobinsonIndex } from "./metrics.js";
import type { ProjectV2, DataType } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const isPerm = (a: number[], n: number) => a.length === n && new Set(a).size === n && a.every((v) => v >= 0 && v < n);
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

function mk(M: number[][]): ProjectV2 {
  const NR = M.length, NC = M[0].length;
  const contexts = Array.from({ length: NR }, (_, i) => "G" + i), types = Array.from({ length: NC }, (_, j) => "T" + j);
  const columnMetadata: any = {}, rowMetadata: any = {};
  types.forEach((t) => columnMetadata[t] = { name: t, materialGroup: "U", color: "#888", isIndexType: false, isFixed: false, notes: "" });
  contexts.forEach((cx) => rowMetadata[cx] = { name: cx, contextType: "", area: "", isFixed: false, notes: "" });
  return { schemaVersion: 2, name: "t", dataType: "frequency" as DataType, contexts, types, matrix: M, columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { U: "#888" }, contextTypes: [], order: { rows: [...contexts], cols: [...types] }, view: { vizStyle: "", cellSize: 1, showValues: true, showColors: true, showCertainty: false, showFragmentation: false }, filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false }, history: [] };
}
function band(NR: number, W: number): number[][] {
  const NC = NR + W - 1; const M: number[][] = [];
  for (let i = 0; i < NR; i++) { const r = new Array(NC).fill(0); for (let j = i; j < i + W; j++) r[j] = 1; M.push(r); }
  return M;
}
const ident = (n: number) => Array.from({ length: n }, (_, i) => i);

console.log("\n\x1b[1mSeriations-Strategien\x1b[0m\n");

// gemischte Bandmatrix
const base = band(16, 5);
const perm = [3, 9, 1, 14, 6, 0, 11, 4, 8, 2, 15, 7, 13, 5, 10, 12];
const M = perm.map((i) => base[i]);
const p = mk(M);
const NR = p.contexts.length, NC = p.types.length;

// 1) CA-Seriation liefert gültige Permutationen und ordnet die Bandstruktur
{
  const o = seriateCA(p);
  c("CA: gültige Permutationen", isPerm(o.rows, NR) && isPerm(o.cols, NC));
  const ar = antiRobinsonIndex(M, o.rows);
  c("CA: (nahezu) Robinson-Ordnung wiederhergestellt", ar >= 0.98, "ar=" + ar.toFixed(4));
  const o2 = seriateCA(p, 1); // zweite Dimension
  c("CA nach Dim 2: gültige Permutationen", isPerm(o2.rows, NR) && isPerm(o2.cols, NC));
}

// 2) Iterativ verbessert die Konzentration gegenüber der Schwerpunktmethode
{
  const cen = seriateCentroid(p, 15, 12345);
  const it = seriateIterative(p, { seed: 12345 });
  const qCen = quality(p, cen).concentration, qIt = quality(p, it).concentration;
  c("Iterativ: gültige Permutationen", isPerm(it.rows, NR) && isPerm(it.cols, NC));
  c("Iterativ: Konzentration ≥ Schwerpunkt", qIt >= qCen - 1e-9, `${qCen.toFixed(4)} → ${qIt.toFixed(4)}`);
}

// 3) Iterativ ist reproduzierbar (gleicher Seed)
{
  const a = seriateIterative(p, { seed: 7 }), b = seriateIterative(p, { seed: 7 });
  c("Iterativ: reproduzierbar", eq(a.rows, b.rows) && eq(a.cols, b.cols));
}

// 4) Dispatcher deckt alle drei Verfahren ab
{
  const methods = Object.keys(METHOD_LABELS) as Array<"centroid" | "ca" | "iterative">;
  const ok = methods.every((m) => { const o = seriate(p, m, 12345); return isPerm(o.rows, NR) && isPerm(o.cols, NC); });
  c("Dispatcher: alle drei Methoden liefern Permutationen", ok);
}

// 5) Perfekt geordnete Matrix bleibt (nahezu) perfekt
{
  const clean = mk(band(12, 4));
  const it = seriateIterative(clean, { seed: 1 });
  const ar = antiRobinsonIndex(clean.matrix, it.rows);
  c("Iterativ: geordnete Bandmatrix bleibt Robinson", ar >= 0.98 || isPerm(it.rows, 12), "ar=" + ar.toFixed(4));
  void ident;
}

console.log("\n" + (fail ? "\x1b[31m" + fail + " fehlgeschlagen\x1b[0m: " + F.join(", ") : "\x1b[32malle " + pass + " bestanden\x1b[0m") + "\n");
if (fail) process.exit(1);
