/**
 * „Nicht erfasst" vs. „strukturelle Absenz" (§9.6).
 */
import { isMissingToken, isMissing, setMissing, missingCount, clearAllMissing, effectiveValue, typePresence, contextPresence } from "./missing.js";
import { importCSV } from "./io/importTable.js";
import { filterProject } from "./filter.js";
import { toCSV } from "../export/exportTable.js";
import type { ProjectV2 } from "./model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mNicht erfasst vs. strukturelle Absenz (§9.6)\x1b[0m\n");

// ── Token-Erkennung ──
c("Sentinels ?, NA, n/a erkannt", isMissingToken("?") && isMissingToken("NA") && isMissingToken("n/a") && isMissingToken(" na "));
c("0, Zahl, leer, Text sind NICHT missing", !isMissingToken("0") && !isMissingToken("5") && !isMissingToken("") && !isMissingToken("x"));

// ── Helfer-Round-Trip ──
{
  const p: ProjectV2 = { schemaVersion: 2, name: "t", dataType: "frequency", contexts: ["A", "B"], types: ["x", "y", "z"],
    matrix: [[0, 3, 0], [1, 0, 0]], columnMetadata: {} as any, rowMetadata: {} as any, cellAnnotations: {},
    materialGroups: {}, contextTypes: [], order: { rows: ["A", "B"], cols: ["x", "y", "z"] },
    view: {} as any, filters: {} as any, history: [] };
  c("frisches Projekt hat kein missing", missingCount(p) === 0 && !isMissing(p, 0, 0));
  setMissing(p, [[0, 0], [1, 2]], true);
  c("setMissing markiert Zellen", isMissing(p, 0, 0) && isMissing(p, 1, 2) && missingCount(p) === 2);
  c("effectiveValue: missing → null, sonst Wert", effectiveValue(p, 0, 0) === null && effectiveValue(p, 0, 1) === 3);
  setMissing(p, [[0, 0]], false);
  c("setMissing(false) entfernt Markierung", !isMissing(p, 0, 0) && missingCount(p) === 1);
  clearAllMissing(p);
  c("clearAllMissing leert alles", missingCount(p) === 0 && p.missingCells === undefined);
}

// ── Präsenzstatistik unterscheidet 0 von nicht erfasst ──
{
  const p: ProjectV2 = { schemaVersion: 2, name: "t", dataType: "frequency", contexts: ["A", "B", "C"], types: ["x"],
    matrix: [[2], [0], [0]], columnMetadata: {} as any, rowMetadata: {} as any, cellAnnotations: {},
    materialGroups: {}, contextTypes: [], order: { rows: ["A", "B", "C"], cols: ["x"] }, view: {} as any, filters: {} as any, history: [] };
  setMissing(p, [[2, 0]], true); // C/x nicht erfasst
  const pr = typePresence(p, 0);
  c("typePresence trennt vorhanden/absent/nicht-erfasst", pr.present === 1 && pr.absent === 1 && pr.missing === 1, JSON.stringify(pr));
  const cp = contextPresence(p, 1); // B: x=0 → absent
  c("contextPresence zählt korrekt", cp.present === 0 && cp.absent === 1 && cp.missing === 0);
}

// ── Import-Sentinel: „?“ wird missing, „0“ bleibt Absenz, ohne Warnung ──
{
  const csv = "Context,x,y,z\nA,2,?,0\nB,0,0,NA";
  const { project, report } = importCSV(csv, { name: "imp" });
  const jx = project.types.indexOf("y"), jz = project.types.indexOf("z");
  c("„?“ wird als nicht erfasst importiert", isMissing(project, 0, jx) && project.matrix[0][jx] === 0);
  c("„NA“ wird als nicht erfasst importiert", isMissing(project, 1, jz));
  c("„0“ bleibt strukturelle Absenz", !isMissing(project, 0, jz) && project.matrix[0][jz] === 0);
  c("keine Nicht-numerisch-Warnung für Sentinels", !report.warnings.some((w) => w.includes("Nicht-numerischer")));
}

// ── CSV-Export schreibt Sentinel zurück (verlustfreier Round-Trip) ──
{
  const csv = "Context,x,y\nA,5,?\nB,?,0";
  const { project } = importCSV(csv, { name: "rt" });
  const out = toCSV(project);
  c("Export schreibt „?“ für nicht erfasst", out.includes("5,?") && /(^|\n)B,\?,0/.test(out.replace(/\r/g, "")), out.replace(/\r/g, "\\n"));
  // Round-Trip: erneut importieren → gleiche missing-Zellen
  const again = importCSV(out.replace(/\uFEFF/g, ""), { name: "rt2" }).project;
  c("Round-Trip erhält nicht-erfasst-Markierungen", missingCount(again) === missingCount(project) && missingCount(project) === 2);
}

// ── Filter remappt missing über Namen ──
{
  const csv = "Context,x,y,z\nA,1,?,0\nB,0,2,0\nC,?,0,3";
  const { project } = importCSV(csv, { name: "f" });
  const view = filterProject(project, { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: true } as any);
  // Nach Ausblenden leerer Spalten bleiben x,y,z ggf. teils erhalten; prüfe A/y bleibt missing
  const ai = view.contexts.indexOf("A"), yj = view.types.indexOf("y");
  c("Filter erhält nicht-erfasst-Markierung an neuen Indizes", ai >= 0 && yj >= 0 && isMissing(view, ai, yj));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
