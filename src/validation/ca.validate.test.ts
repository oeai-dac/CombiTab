/**
 * Referenz-Validierung der CA (§9.10) — gegen publizierte Ergebnisse, nicht nur
 * interne Konsistenz. Prüft den kanonischen Greenacre-„smoking"-Datensatz (Träg­heiten,
 * Anteile, Prinzipalkoordinaten) und die theoretische Seriationsrückgewinnung an einer
 * Petrie-/Robinson-Matrix (Hill 1974).
 */
import { computeCA } from "../analysis/ca.js";
import { SMOKE, referenceProject, petrieMatrix } from "./reference.js";
import { mulberry32 } from "../core/rng.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

console.log("\n\x1b[1mReferenz-Validierung der CA (§9.10)\x1b[0m\n");

// ── Greenacre „smoking" ──
{
  const p = referenceProject("smoke", SMOKE.rows, SMOKE.cols, SMOKE.matrix);
  const ca = computeCA(p, 3);

  c("Gesamtträgheit = publiziert (0.085190)", near(ca.totalInertia, SMOKE.totalInertia, 1e-3), ca.totalInertia.toFixed(6));
  c("Haupt­trägheit λ1 = 0.074759", near(ca.eigenvalues[0], SMOKE.eigenvalues[0], 1e-4), ca.eigenvalues[0].toFixed(6));
  c("Haupt­trägheit λ2 = 0.010017", near(ca.eigenvalues[1], SMOKE.eigenvalues[1], 1e-4), ca.eigenvalues[1].toFixed(6));
  c("Haupt­trägheit λ3 = 0.000414", near(ca.eigenvalues[2], SMOKE.eigenvalues[2], 1e-4), ca.eigenvalues[2].toFixed(6));
  c("Trägheitsanteil Dim1 ≈ 87.76 %", near(ca.inertiaPct[0] * 100, SMOKE.inertiaPct[0], 0.1), (ca.inertiaPct[0] * 100).toFixed(2));
  c("Trägheitsanteil Dim2 ≈ 11.76 %", near(ca.inertiaPct[1] * 100, SMOKE.inertiaPct[1], 0.1), (ca.inertiaPct[1] * 100).toFixed(2));

  // Koordinatenvergleich mit Vorzeichenausrichtung je Achse
  const cmpCoords = (label: string, keys: string[], appByIdx: number[][], ref: Record<string, [number, number]>) => {
    for (let d = 0; d < 2; d++) {
      // Vorzeichen der Achse an der Referenz ausrichten
      let dot = 0; keys.forEach((k, i) => (dot += appByIdx[i][d] * ref[k][d]));
      const sign = dot < 0 ? -1 : 1;
      let maxErr = 0;
      keys.forEach((k, i) => (maxErr = Math.max(maxErr, Math.abs(sign * appByIdx[i][d] - ref[k][d]))));
      c(`${label}: Dim${d + 1} stimmt mit publizierten Koordinaten (±0.002)`, maxErr <= 0.002, `maxErr=${maxErr.toFixed(4)}`);
    }
  };
  cmpCoords("Zeilen", SMOKE.rows, SMOKE.rows.map((_, i) => ca.rowCoords[i]), SMOKE.rowCoords);
  cmpCoords("Spalten", SMOKE.cols, SMOKE.cols.map((_, j) => ca.colCoords[j]), SMOKE.colCoords);

  // Publizierte qualitative Aussage: entlang Dim1 folgen die Rauchgrade none→heavy monoton.
  const d1 = SMOKE.cols.map((_, j) => ca.colCoords[j][0]);
  const asc = d1.every((v, i, a) => i === 0 || v > a[i - 1]);
  const desc = d1.every((v, i, a) => i === 0 || v < a[i - 1]);
  c("Rauchgrad-Gradient (none→heavy) ist entlang Dim1 monoton", asc || desc);
}

// ── Seriationsrückgewinnung an einer Petrie-Matrix (Hill 1974) ──
{
  const NR = 30, NC = 24;
  const M = petrieMatrix(NR, NC, 4);
  // Zeilen seed-basiert mischen; wahre Reihenfolge = ursprünglicher Index
  const perm = Array.from({ length: NR }, (_, i) => i);
  const rnd = mulberry32(123);
  for (let i = NR - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const rows = perm.map((tru) => `k${String(tru).padStart(3, "0")}`);
  const shuffled = perm.map((tru) => M[tru]);
  const cols = Array.from({ length: NC }, (_, j) => `t${j}`);
  const p = referenceProject("petrie", rows, cols, shuffled);

  const ca = computeCA(p, 2);
  // Spearman-Rangkorrelation zwischen CA-Dim1 und wahrer Reihenfolge (robust gg. Bindungen).
  const idx = rows.map((_, i) => i);
  const rankOfDim1 = new Array<number>(NR);
  idx.slice().sort((a, b) => ca.rowCoords[a][0] - ca.rowCoords[b][0]).forEach((i, rank) => (rankOfDim1[i] = rank));
  let sumd2 = 0; for (let i = 0; i < NR; i++) { const d = perm[i] - rankOfDim1[i]; sumd2 += d * d; }
  const rho = 1 - (6 * sumd2) / (NR * (NR * NR - 1));
  c("CA-Dim1 gewinnt die Seriationsreihenfolge zurück (|Spearman ρ| ≈ 1)", Math.abs(rho) > 0.99, `ρ=${rho.toFixed(4)}`);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
console.log("\x1b[32m✓ CA gegen publizierte Referenzen validiert.\x1b[0m");
