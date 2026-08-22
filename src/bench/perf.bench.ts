/**
 * Performance-Benchmark gegen die §8-/§12.2-Budgets.
 *
 * Misst reproduzierbar die CPU-seitigen Kosten und vergleicht sie mit den
 * Akzeptanz-Budgets. Bewusst getrennt nach Belastbarkeit über Hardware hinweg:
 *
 *  A) ALGORITHMISCHE INVARIANTE (gatet den Exit-Code):
 *     - Live-Drag-Reorder (Modell-Op) muss sub-ms bleiben (< 16 ms mit großer
 *       Reserve). Reißt diese Grenze, liegt eine Komplexitätsregression vor
 *       (z. B. versehentlich O(n²)). Diese Grenze ist hardwarerobust.
 *
 *  B) WALL-CLOCK-BUDGETS (ausgewiesen mit Verdikt, aber NICHT gatend):
 *     - CA 500×500                  < 2000 ms  (§8, läuft im Worker)
 *     - Score-Neuberechnung 200×200 <   16 ms  (§8, einmalig nach Drop)
 *     Wall-Clock hängt von der Zielhardware ab; ein CPU-Container darf kein
 *     falsches CI-Fail erzeugen. Die Zahlen sind trotzdem aussagekräftig.
 *
 *  C) INFO: Datenaufbau 1.000×1.000, Seriation (drei Verfahren).
 *
 * GPU-Budget (1.000² @ 60 fps schwenk-/zoombar, Draw < 16 ms) ist hier nicht
 * messbar — dafür `MatrixRenderer.benchmark()` und das Perf-HUD im Browser.
 *
 * Zur Einordnung der Score-Zahlen: Die Neuberechnung ist O(NR²·NC). Gemessen wird
 * hier der reine Rechenkern (`quality`) — in der App läuft er im Score-Worker,
 * blockiert also die Oberfläche nicht. Die gemessene Zeit ist damit die *Latenz*
 * bis zur Aktualisierung der Score-Anzeige, nicht eine Blockade des Haupt-Threads.
 * Das 16-ms-Budget ist entsprechend als Reaktionsziel zu lesen, nicht als Frame-Budget.
 *
 * Aufruf:  npm run bench
 */
import { computeCA } from "../analysis/ca.js";
import { seriate } from "../seriation/strategies.js";
import { quality } from "../seriation/metrics.js";
import { moveFree } from "../matrix/orderModel.js";
import { makeSyntheticProject } from "./synth.js";
import type { ProjectV2 } from "../core/model.js";

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

interface Stat { min: number; median: number; mean: number; }
function timeit(fn: () => void, runs = 5, warmup = 1): Stat {
  for (let i = 0; i < warmup; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < runs; i++) { const t0 = now(); fn(); ts.push(now() - t0); }
  ts.sort((a, b) => a - b);
  return { min: ts[0], median: ts[(ts.length - 1) >> 1], mean: ts.reduce((a, b) => a + b, 0) / ts.length };
}

function buildRenderData(p: ProjectV2): number {
  const NR = p.contexts.length, NC = p.types.length;
  let vmax = 1;
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) { const v = p.matrix[i][j]; if (v > vmax) vmax = v; }
  const disp = new Uint8Array(NR * NC);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) { const v = p.matrix[i][j]; disp[i * NC + j] = v ? Math.max(28, Math.round((v / vmax) * 255)) : 0; }
  return disp.length;
}

type Kind = "invariant" | "wall" | "info";
interface Row { name: string; budget: number | null; measured: number; kind: Kind; }
const rows: Row[] = [];
const add = (name: string, measured: number, budget: number | null, kind: Kind) => rows.push({ name, budget, measured, kind });

console.log("\n\x1b[1mCombiTab v2 — Performance-Benchmark (§8/§12.2)\x1b[0m");
console.log("\x1b[2mUmgebung: " + (typeof navigator !== "undefined" && navigator.userAgent ? navigator.userAgent : "Node/tsx (CPU-only)") + "\x1b[0m\n");

// A) Invariante: Live-Drag-Reorder-Op
{
  const order = Array.from({ length: 1000 }, (_, i) => i);
  const fixed = new Set<number>();
  const s = timeit(() => { for (let k = 0; k < 50; k++) moveFree(order, fixed, (k * 7) % 1000, (k * 13 + 3) % 1000); }, 200, 20);
  add("Live-Drag-Reorder — je Op (Order 1000)", s.median / 50, 16, "invariant");
}

// B) Wall-Clock: CA 500×500
{
  const p = makeSyntheticProject(500, 500, { seed: 7 });
  const s = timeit(() => { computeCA(p, 4); }, 3, 1);
  add("CA 500×500 (Dim 1–4, im Worker)", s.median, 2000, "wall");
}

// B) Wall-Clock: Score-Neuberechnung 200×200 (einmalig nach Drop)
{
  const p = makeSyntheticProject(200, 200, { seed: 11 });
  const order = { rows: p.contexts.map((_, i) => i), cols: p.types.map((_, j) => j) };
  const s = timeit(() => { quality(p, order); }, 50, 5);
  add("Score-Neuberechnung 200×200 (nach Drop)", s.median, 16, "wall");
}

// C) Info
{
  const p = makeSyntheticProject(1000, 1000, { seed: 3 });
  add("Datenaufbau 1.000×1.000 (10⁶ Zellen)", timeit(() => buildRenderData(p), 5, 1).median, null, "info");
}
{
  const p = makeSyntheticProject(1000, 1000, { seed: 5 });
  const order = { rows: p.contexts.map((_, i) => i), cols: p.types.map((_, j) => j) };
  add("Score-Neuberechnung 1.000×1.000 (nach Drop)", timeit(() => quality(p, order), 3, 1).median, null, "info");
}
{
  const p = makeSyntheticProject(200, 200, { seed: 9 });
  for (const m of ["centroid", "ca", "iterative"] as const)
    add(`Seriation „${m}" 200×200 (im Worker)`, timeit(() => seriate(p, m, 12345, 0), 5, 1).median, null, "info");
}

// ── Ausgabe ──
const fmt = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - s.length));
let invariantFail = 0;
console.log(pad("Messung", 44) + pad("Budget", 13) + pad("Gemessen", 12) + "Verdikt");
console.log("─".repeat(84));
for (const r of rows) {
  const meas = fmt(r.measured) + " ms";
  let budget = "\x1b[2m—\x1b[0m", verdict = "\x1b[2m— (Info)\x1b[0m";
  if (r.budget != null) {
    budget = "< " + r.budget + " ms";
    const ok = r.measured < r.budget, reserve = r.budget / Math.max(r.measured, 1e-6);
    const rtxt = reserve >= 2 ? `${reserve.toFixed(reserve > 100 ? 0 : 1)}× Reserve` : ok ? "knapp" : `${(1 / reserve).toFixed(1)}× über Budget`;
    if (r.kind === "invariant") {
      if (!ok) invariantFail++;
      verdict = (ok ? "\x1b[32m✓ INVARIANTE\x1b[0m" : "\x1b[31m✗ INVARIANTE VERLETZT\x1b[0m") + `  (${rtxt})`;
    } else {
      verdict = (ok ? "\x1b[32m✓ im Budget\x1b[0m" : "\x1b[33m⚠ über Budget\x1b[0m") + `  (${rtxt})`;
    }
  }
  console.log(pad(r.name, 44) + pad(budget, 13 + 9) + pad(meas, 12) + verdict);
}
console.log("─".repeat(84));
console.log("\x1b[2mGPU-Budget (1.000² @ 60 fps, Draw < 16 ms): im Browser via Perf-HUD / renderer.benchmark().\x1b[0m");
console.log("\x1b[2mWall-Clock ist hardwareabhängig und gatet den Exit-Code nicht; nur die algorithmische Invariante tut es.\x1b[0m");
console.log("\x1b[2mScore-Neuberechnung: gemessen wird der Rechenkern (O(NR²·NC)). In der App läuft er im Score-Worker und\x1b[0m");
console.log("\x1b[2mblockiert die Oberfläche nicht — die Zahl ist die Latenz bis zur Score-Anzeige, kein Frame-Budget.\x1b[0m");

if (invariantFail > 0) { console.log(`\n\x1b[31m${invariantFail} algorithmische Invariante verletzt — Komplexitätsregression.\x1b[0m`); process.exit(1); }
console.log("\n\x1b[32mAlgorithmische Invarianten eingehalten.\x1b[0m");
