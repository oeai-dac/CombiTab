/**
 * Tests der echten Fehlwert-Behandlung in den Rechenkernen (§9.6-Vertiefung).
 *
 * Prüft, dass „nicht erfasst" nicht mehr wie 0 behandelt wird:
 *  1. CA-Imputation (`imputeUnderIndependence`): Fixpunkt-Eigenschaft, beobachtete
 *     Zellen unberührt, vollständig fehlende Zeile bleibt 0, No-Mask = No-Op.
 *  2. CA masken-bewusst: `computeCA` mit Fehlwert ≡ `computeCA` auf der imputierten
 *     Matrix ohne Fehlwert (Verdrahtungsbeweis) und ≠ Behandlung als 0.
 *  3. Metrik: maskierte Kontinuität (Lücke durch „nicht erfasst" zählt nicht),
 *     maskierte Konzentration (nicht-null gespeicherter Fehlwert fällt raus),
 *     paarweise-vollständige Anti-Robinson-Ähnlichkeit.
 *  4. Score-Worker-Kern reicht die Maske durch.
 *  5. Regression: ohne Fehlwerte sind alle Kerne byte-identisch zum alten Pfad.
 */
import { quality, qualityFromMatrix, antiRobinsonIndex } from "../seriation/metrics.js";
import { computeCA, caRowScores, imputeUnderIndependence } from "./ca.js";
import { seriateCentroid } from "../seriation/centroid.js";
import { seriateCA } from "../seriation/strategies.js";
import { buildMissingMask } from "../core/missing.js";
import { handleScoreRequest, createScoreCache } from "../workers/scoreCore.js";
import type { ProjectV2 } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, e = 1e-9) => Math.abs(a - b) <= e;

function mk(M: number[][], missing?: Array<[number, number]>): ProjectV2 {
  const NR = M.length, NC = M[0].length;
  const contexts = Array.from({ length: NR }, (_, i) => "G" + i), types = Array.from({ length: NC }, (_, j) => "T" + j);
  const columnMetadata: any = {}, rowMetadata: any = {};
  types.forEach(t => columnMetadata[t] = { name: t, materialGroup: "U", color: "#808080", isIndexType: false, isFixed: false, notes: "" });
  contexts.forEach(cx => rowMetadata[cx] = { name: cx, contextType: "", area: "", isFixed: false, notes: "" });
  const p: ProjectV2 = { schemaVersion: 2, name: "t", dataType: "frequency", contexts, types, matrix: M, columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { U: "#808080" }, contextTypes: [], order: { rows: [...contexts], cols: [...types] }, view: { vizStyle: "", cellSize: 1, showValues: true, showColors: true, showCertainty: false, showFragmentation: false }, filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false }, history: [] };
  if (missing) { p.missingCells = {}; for (const [i, j] of missing) p.missingCells[`${i}:${j}`] = true; }
  return p;
}
const canon = (p: ProjectV2) => ({ rows: p.contexts.map((_, i) => i), cols: p.types.map((_, j) => j) });

console.log("\n\x1b[1mFehlwert-Behandlung in den Rechenkernen (§9.6-Vertiefung)\x1b[0m\n");

// ── 1) Imputation ──
{
  const M = [[4, 0, 2], [0, 3, 1], [2, 1, 5]];
  const mask = buildMissingMask(mk(M, [[0, 1], [2, 0]]))!; // (0,1) und (2,0) fehlen
  const X = imputeUnderIndependence(M, 3, 3, mask, { maxIter: 200, tol: 1e-12 });

  // beobachtete Zellen unberührt
  let obsOk = true;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (!mask[i * 3 + j] && X[i][j] !== M[i][j]) obsOk = false;
  c("Imputation lässt beobachtete Zellen unverändert", obsOk);

  // Fixpunkt: fehlende Zelle = R_i·C_j/T
  const R = X.map((r) => r.reduce((a, b) => a + b, 0));
  const C = [0, 1, 2].map((j) => X[0][j] + X[1][j] + X[2][j]);
  const T = R.reduce((a, b) => a + b, 0);
  let fixOk = true, worst = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (mask[i * 3 + j]) { const e = R[i] * C[j] / T; worst = Math.max(worst, Math.abs(e - X[i][j])); if (!near(e, X[i][j], 1e-7)) fixOk = false; }
  c("Imputation erfüllt Unabhängigkeits-Fixpunkt (E_ij = R_i·C_j/T)", fixOk, "maxΔ=" + worst.toExponential(1));

  // No-Mask = No-Op (gleiche Referenz zurück)
  c("Imputation ohne Maske ist No-Op (Original-Referenz)", imputeUnderIndependence(M, 3, 3, null) === M);

  // vollständig fehlende Zeile bleibt 0
  const M2 = [[1, 2], [3, 4]];
  const mask2 = buildMissingMask(mk(M2, [[0, 0], [0, 1]]))!; // Zeile 0 komplett fehlend
  const X2 = imputeUnderIndependence(M2, 2, 2, mask2, { maxIter: 200 });
  c("Vollständig fehlende Zeile bleibt 0 (kein Signal → kein Rateergebnis)", X2[0][0] === 0 && X2[0][1] === 0);
}

// ── 2) CA masken-bewusst ──
{
  const M = [[5, 1, 0, 0], [2, 4, 1, 0], [0, 2, 5, 1], [0, 0, 1, 6], [1, 0, 0, 3]];
  const pMiss = mk(M.map((r) => r.slice()), [[1, 2]]);       // eine Zelle „nicht erfasst"
  const mask = buildMissingMask(pMiss)!;
  const imputed = imputeUnderIndependence(M, 5, 4, mask, { maxIter: 300, tol: 1e-12 });
  const pImp = mk(imputed);                                    // gleiche Matrix, KEIN Fehlwert

  const caMiss = computeCA(pMiss, 3), caImp = computeCA(pImp, 3);
  let eqCoord = caMiss.eigenvalues.length === caImp.eigenvalues.length;
  for (let d = 0; d < caMiss.eigenvalues.length; d++) eqCoord &&= near(caMiss.eigenvalues[d], caImp.eigenvalues[d], 1e-9);
  c("computeCA(Fehlwert) ≡ computeCA(imputierte Matrix ohne Fehlwert)", eqCoord);

  // … und ≠ Behandlung als 0
  const pZero = mk(M.map((r) => r.slice())); pZero.matrix[1][2] = 0; // wie „strukturelle 0"
  const caZero = computeCA(pZero, 3);
  const diff = Math.abs(caMiss.eigenvalues[0] - caZero.eigenvalues[0]);
  c("CA mit Fehlwert ≠ CA mit 0-Behandlung", diff > 1e-6, "Δλ1=" + diff.toExponential(2));

  // caRowScores masken-bewusst konsistent mit computeCA-Verdrahtung
  const rs = caRowScores(M, 5, 4, 0, mask);
  const rsImp = caRowScores(imputed, 5, 4, 0);
  let rsEq = true; for (let i = 0; i < 5; i++) rsEq &&= near(Math.abs(rs[i]), Math.abs(rsImp[i]), 1e-6);
  c("caRowScores(Maske) ≡ caRowScores(imputiert)", rsEq);
}

// ── 3) Metrik-Maskierung ──
{
  // Kontinuität: Spalte präsent in Zeile 0 und 2, Zeile 1 dazwischen „nicht erfasst"
  const M = [[1], [0], [1]];
  const noMiss = qualityFromMatrix(M, 3, 1, canon(mk(M)));
  const withMiss = qualityFromMatrix(M, 3, 1, canon(mk(M)), undefined, buildMissingMask(mk(M, [[1, 0]])));
  c("Kontinuität ohne Maske = 2/3 (Lücke zählt)", near(noMiss.continuity, 2 / 3));
  c("Kontinuität mit Fehlwert in der Lücke = 1 (unbekannt ist keine Lücke)", near(withMiss.continuity, 1));

  // Konzentration: nicht-null Fehlwert abseits der Diagonale fällt aus der Gewichtung
  const M2 = [[3, 2], [0, 3]];
  const base = qualityFromMatrix(M2, 2, 2, canon(mk(M2)));
  const masked = qualityFromMatrix(M2, 2, 2, canon(mk(M2)), undefined, buildMissingMask(mk(M2, [[0, 1]])));
  c("Konzentration: maskierter Off-Diagonal-Fehlwert erhöht Konzentration", near(base.concentration, 0.75) && near(masked.concentration, 1));

  // Anti-Robinson paarweise-vollständig: ein verzerrender Ausreißer (r2 col0 = 9),
  // als „nicht erfasst" markiert, stellt die Robinson-Eigenschaft wieder her.
  const M3 = [[1, 1, 0], [0, 1, 1], [9, 0, 1]];
  const order = [0, 1, 2];
  const ar0 = antiRobinsonIndex(M3, order);
  const ar1 = antiRobinsonIndex(M3, order, buildMissingMask(mk(M3, [[2, 0]])), 3);
  c("Anti-Robinson paarweise-vollständig ignoriert maskierten Ausreißer", near(ar0, 0) && near(ar1, 1), `ar0=${ar0.toFixed(3)} ar1=${ar1.toFixed(3)}`);
}

// ── 4) Score-Worker-Kern reicht Maske durch ──
{
  const M = [[2, 0, 1], [0, 3, 0], [1, 0, 4], [0, 1, 0]];
  const p = mk(M, [[0, 2]]);
  const mask = buildMissingMask(p)!;
  const order = canon(p);
  const ref = quality(p, order); // masken-bewusst über den ProjectV2-Pfad
  const cache = createScoreCache();
  const res = handleScoreRequest(cache, { id: 1, epoch: 1, matrix: M, missing: mask, rows: order.rows, cols: order.cols });
  c("Score-Worker-Kern mit Maske ≡ quality(ProjectV2)", res.type === "done" && near((res as any).result.total, ref.total, 1e-12));
  // Cache behält die Maske über eine folgende Anfrage ohne Matrix
  const res2 = handleScoreRequest(cache, { id: 2, epoch: 1, rows: order.rows, cols: order.cols });
  c("Score-Worker-Kern behält Maske im Cache", res2.type === "done" && near((res2 as any).result.total, ref.total, 1e-12));
}

// ── 5) Regression: ohne Fehlwerte byte-identisch ──
{
  const M = [[5, 1, 0], [2, 4, 1], [0, 2, 5], [0, 0, 6]];
  const p = mk(M);
  const a = qualityFromMatrix(M, 4, 3, canon(p));
  const b = qualityFromMatrix(M, 4, 3, canon(p), undefined, null);
  c("Metrik ohne Maske: masked-Pfad ≡ unmasked", a.total === b.total && a.antiRobinson === b.antiRobinson && a.continuity === b.continuity);
  c("buildMissingMask ohne Fehlwerte → null (Schnellpfad)", buildMissingMask(p) === null);
  // Seriation läuft und liefert gültige Permutationen (mit und ohne Fehlwert)
  const s1 = seriateCentroid(p, 15, 1), s2 = seriateCA(p, 0);
  const okPerm = new Set(s1.rows).size === 4 && new Set(s2.cols).size === 3;
  c("Seriation liefert gültige Ordnungen", okPerm);
  // Zentroid mit 0-gespeichertem Fehlwert identisch zu ohne (Invarianz)
  const pMiss0 = mk(M.map((r) => r.slice()), [[3, 0]]); // M[3][0] ist bereits 0
  const s3 = seriateCentroid(pMiss0, 15, 1);
  c("Zentroid: 0-gespeicherter Fehlwert ändert die Ordnung nicht", JSON.stringify(s3.rows) === JSON.stringify(s1.rows));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
else console.log("\x1b[32m✓ Fehlwert-Behandlung (§9.6-Vertiefung) korrekt.\x1b[0m");
