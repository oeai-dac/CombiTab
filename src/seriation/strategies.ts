/**
 * Seriations-Strategien (Spezifikation §5.1 — „alle drei Methoden").
 *
 * Alle Methoden liefern eine vollständige Anzeige-Ordnung (kanonische Indizes) für
 * Zeilen und Spalten. Fixierte Elemente werden **nicht** hier behandelt, sondern
 * einheitlich beim Anwenden über `orderModel.applySeriation`: fixierte Elemente
 * behalten ihre Position, die freien Positionen werden in der von der Methode
 * gelieferten Reihenfolge aufgefüllt. Dadurch ist jede Methode automatisch
 * fix-bewusst.
 */
import type { ProjectV2 } from "../core/model.js";
import { seriateCentroid, argsort, type Order } from "./centroid.js";
import { caRowScores } from "../analysis/ca.js";
import { buildMissingMask } from "../core/missing.js";

export type SeriationMethod = "centroid" | "ca" | "iterative";

export const METHOD_LABELS: Record<SeriationMethod, string> = {
  centroid: "Schwerpunktmethode",
  ca: "Korrespondenzanalyse",
  iterative: "Iterativ (Optimierung)",
};

/** Für die zitierfähige `history` (enthält engl. Tokens, die der Methods-Report erkennt). */
export const METHOD_HISTORY: Record<SeriationMethod, string> = {
  centroid: "reciprocal averaging (Schwerpunktmethode)",
  ca: "correspondence analysis seriation (CA-Dim 1)",
  iterative: "iterative seriation (Konzentrations-Optimierung)",
};

function transpose(M: number[][], NR: number, NC: number): number[][] {
  const T: number[][] = Array.from({ length: NC }, () => new Array<number>(NR).fill(0));
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) T[j][i] = M[i][j];
  return T;
}

/** Transponiert eine dichte Fehlwert-Maske (NR×NC → NC×NR), damit die
 *  Spalten-CA-Seriation dieselbe Imputation nutzt wie die Zeilen-CA. */
function transposeMask(mask: Uint8Array, NR: number, NC: number): Uint8Array {
  const T = new Uint8Array(NC * NR);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) if (mask[i * NC + j]) T[j * NR + i] = 1;
  return T;
}

/** Seriation über eine CA-Dimension (0-basiert; Zeilen und Spalten, exakt via SVD). */
export function seriateCA(p: ProjectV2, dim = 0): Order {
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;
  const mask = buildMissingMask(p); // §9.6: Fehlwerte werden in caRowScores imputiert
  const rs = caRowScores(M, NR, NC, dim, mask);
  const cs = caRowScores(transpose(M, NR, NC), NC, NR, dim, mask ? transposeMask(mask, NR, NC) : null);
  return { rows: argsort(rs), cols: argsort(cs) };
}

/**
 * Iterative Seriation: startet von der Schwerpunktmethode und verbessert die
 * Diagonal-Konzentration durch akzeptierende Nachbar-Vertauschungen (Hill-Climbing).
 * Die gewichtete Diagonaldistanz (WD) wird inkrementell gepflegt, jede
 * Vertauschung kostet nur O(NC) bzw. O(NR) — auch für größere Matrizen tragbar.
 * Deterministisch (Seed der Startlösung). Da nur WD-senkende Tausche akzeptiert
 * werden, gilt Konzentration(iterativ) ≥ Konzentration(Schwerpunkt).
 */
export function seriateIterative(p: ProjectV2, opts: { seed?: number; maxPasses?: number } = {}): Order {
  const seed = opts.seed ?? 12345, maxPasses = opts.maxPasses ?? 30;
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;
  const mask = buildMissingMask(p); // §9.6: nicht erfasste Zellen tragen nicht zur WD bei
  const miss = mask ? (i: number, j: number) => mask[i * NC + j] === 1 : () => false;
  const start = seriateCentroid(p, 15, seed);
  const rows = start.rows.slice(), cols = start.cols.slice();
  if (NR < 3 && NC < 3) return { rows, cols };

  const rN = NR > 1 ? NR - 1 : 1, cN = NC > 1 ? NC - 1 : 1;
  const rowY = (pos: number) => pos / rN;
  const colX = (pos: number) => pos / cN;

  const rowDeltaSwap = (i: number): number => {
    const a = rows[i], b = rows[i + 1], ya = rowY(i), yb = rowY(i + 1);
    let d = 0;
    for (let cj = 0; cj < NC; cj++) {
      const cx = colX(cj), c = cols[cj];
      const va = M[a][c]; if (va && !miss(a, c)) d += va * (Math.abs(yb - cx) - Math.abs(ya - cx));
      const vb = M[b][c]; if (vb && !miss(b, c)) d += vb * (Math.abs(ya - cx) - Math.abs(yb - cx));
    }
    return d;
  };
  const colDeltaSwap = (j: number): number => {
    const a = cols[j], b = cols[j + 1], xa = colX(j), xb = colX(j + 1);
    let d = 0;
    for (let ri = 0; ri < NR; ri++) {
      const ry = rowY(ri), r = rows[ri];
      const va = M[r][a]; if (va && !miss(r, a)) d += va * (Math.abs(ry - xb) - Math.abs(ry - xa));
      const vb = M[r][b]; if (vb && !miss(r, b)) d += vb * (Math.abs(ry - xa) - Math.abs(ry - xb));
    }
    return d;
  };

  for (let pass = 0; pass < maxPasses; pass++) {
    let improved = false;
    for (let i = 0; i < NR - 1; i++) { if (rowDeltaSwap(i) < -1e-12) { [rows[i], rows[i + 1]] = [rows[i + 1], rows[i]]; improved = true; } }
    for (let j = 0; j < NC - 1; j++) { if (colDeltaSwap(j) < -1e-12) { [cols[j], cols[j + 1]] = [cols[j + 1], cols[j]]; improved = true; } }
    if (!improved) break;
  }
  return { rows, cols };
}

/** Dispatcher über die drei Verfahren. `caDim` gilt nur für die CA-Seriation. */
export function seriate(p: ProjectV2, method: SeriationMethod, seed = 12345, caDim = 0): Order {
  switch (method) {
    case "ca": return seriateCA(p, caDim);
    case "iterative": return seriateIterative(p, { seed });
    default: return seriateCentroid(p, 15, seed);
  }
}
