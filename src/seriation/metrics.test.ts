import { quality, antiRobinsonIndex, DEFAULT_WEIGHTS } from "./metrics.js";
import { seriateCentroid } from "./centroid.js";
import type { ProjectV2 } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const near = (a: number, b: number, e = 1e-9) => Math.abs(a - b) <= e;

function mk(M: number[][]): ProjectV2 {
  const NR = M.length, NC = M[0].length;
  const contexts = Array.from({ length: NR }, (_, i) => "G" + i), types = Array.from({ length: NC }, (_, j) => "T" + j);
  const columnMetadata: any = {}, rowMetadata: any = {};
  types.forEach(t => columnMetadata[t] = { name: t, materialGroup: "U", color: "#808080", isIndexType: false, isFixed: false, notes: "" });
  contexts.forEach(cx => rowMetadata[cx] = { name: cx, contextType: "", area: "", isFixed: false, notes: "" });
  return { schemaVersion: 2, name: "t", dataType: "frequency", contexts, types, matrix: M, columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { U: "#808080" }, contextTypes: [], order: { rows: [...contexts], cols: [...types] }, view: { vizStyle: "", cellSize: 1, showValues: true, showColors: true, showCertainty: false, showFragmentation: false }, filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false }, history: [] };
}
// Bandmatrix: Zeile i hat 1en in Spalten [i, i+W) → Cosinus-Ähnlichkeit fällt
// streng monoton mit dem Zeilenabstand ⇒ in kanonischer Ordnung perfekt Robinson.
function band(NR: number, W: number): number[][] {
  const NC = NR + W - 1; const M: number[][] = [];
  for (let i = 0; i < NR; i++) { const row = new Array(NC).fill(0); for (let j = i; j < i + W; j++) row[j] = 1; M.push(row); }
  return M;
}
const ident = (n: number) => Array.from({ length: n }, (_, i) => i);

console.log("\n\x1b[1mQualitätsmetriken (Anti-Robinson, Gewichte, Reproduzierbarkeit)\x1b[0m\n");

// 1) Perfekte Robinson-Struktur in kanonischer Ordnung → Index = 1
{
  const M = band(10, 4);
  const ar = antiRobinsonIndex(M, ident(10));
  c("Anti-Robinson: perfekte Bandmatrix (geordnet) = 1", near(ar, 1), "ar=" + ar.toFixed(4));
}

// 2) Gestörte Ordnung senkt den Index unter 1
{
  const M = band(10, 4);
  const scrambled = [0, 9, 2, 3, 4, 5, 6, 7, 8, 1]; // zwei weit entfernte Zeilen getauscht
  const ar = antiRobinsonIndex(M, scrambled);
  c("Anti-Robinson: gestörte Ordnung < 1", ar < 1, "ar=" + ar.toFixed(4));
}

// 3) Kleine Matrix (NR<3) → definitionsgemäß 1
{
  c("Anti-Robinson: NR<3 → 1", antiRobinsonIndex([[1, 0], [0, 1]], [0, 1]) === 1);
}

// 4) total = gewichtete Summe der drei Kennzahlen (Default-Gewichte)
{
  const p = mk(band(12, 4));
  const q = quality(p, { rows: ident(12), cols: ident(15) });
  const expect = DEFAULT_WEIGHTS.concentration * q.concentration + DEFAULT_WEIGHTS.antiRobinson * q.antiRobinson + DEFAULT_WEIGHTS.continuity * q.continuity;
  c("total = gewichtete Summe (Default)", near(q.total, expect), "Δ=" + Math.abs(q.total - expect).toExponential(1));
  c("alle Kennzahlen in [0,1]", [q.concentration, q.antiRobinson, q.continuity, q.total].every(v => v >= -1e-9 && v <= 1 + 1e-9));
}

// 5) Konfigurierbare Gewichte übersteuern (100% Kontinuität → total = continuity)
{
  const p = mk(band(12, 4));
  const q = quality(p, { rows: ident(12), cols: ident(15) }, { concentration: 0, antiRobinson: 0, continuity: 1 });
  c("Gewichte: 100% Kontinuität → total = continuity", near(q.total, q.continuity));
}

// 6) Reproduzierbarkeit: gleicher Seed → identische Ordnung
{
  const p = mk(band(14, 5));
  const a = seriateCentroid(p, 15, 42), b = seriateCentroid(p, 15, 42);
  c("Reproduzierbar: gleicher Seed → identische Zeilen-Ordnung", eq(a.rows, b.rows) && eq(a.cols, b.cols));
}

// 7) Seriation stellt eine (nahezu) perfekt Robinson-Ordnung wieder her
{
  // gemischte kanonische Reihenfolge, Seriation soll die Bandstruktur erkennen
  const base = band(16, 5);
  const perm = [3, 9, 1, 14, 6, 0, 11, 4, 8, 2, 15, 7, 13, 5, 10, 12];
  const M = perm.map(i => base[i]); // Zeilen umsortiert
  const p = mk(M);
  const ord = seriateCentroid(p, 30, 7);
  const arBefore = antiRobinsonIndex(M, ident(16));
  const arAfter = antiRobinsonIndex(M, ord.rows);
  c("Seriation verbessert Anti-Robinson deutlich", arAfter >= arBefore, `${arBefore.toFixed(3)} → ${arAfter.toFixed(3)}`);
  c("Seriation erreicht (nahezu) perfekte Robinson-Ordnung", arAfter >= 0.98, "ar=" + arAfter.toFixed(4));
}

console.log("\n" + (fail ? "\x1b[31m" + fail + " fehlgeschlagen\x1b[0m: " + F.join(", ") : "\x1b[32malle " + pass + " bestanden\x1b[0m") + "\n");
if (fail) process.exit(1);
