/**
 * Korrespondenzanalyse (CA) — Spezifikation §7.2, mit den dort geforderten
 * konsistenten Koordinaten (Prinzipalkoordinaten für den Biplot).
 *
 *   P = X / T,  Zeilenmassen r,  Spaltenmassen c
 *   S_ij = (P_ij − r_i c_j) / sqrt(r_i c_j)              (standardisierte Residuen)
 *   S = U Σ Vᵀ  →  Zeilen-Prinzipalkoord.  F_ik = σ_k U_ik / sqrt(r_i)
 *                  Spalten-Prinzipalkoord. G_jk = σ_k V_jk / sqrt(c_j)
 *   Eigenwerte λ_k = σ_k²,  erklärte Trägheit = λ_k / Σλ
 *
 * Da S bereits um r cᵀ zentriert ist, entfällt die triviale Dimension; die obersten
 * Singulärvektoren sind direkt die CA-Achsen.
 */
import type { ProjectV2 } from "../core/model.js";
import { buildMissingMask } from "../core/missing.js";
import { svd } from "./svd.js";

export interface CAResult {
  rowCoords: number[][];   // NR × k (Kontexte)
  colCoords: number[][];   // NC × k (Typen)
  eigenvalues: number[];   // λ_k
  inertiaPct: number[];    // Anteil je Dimension
  totalInertia: number;
  k: number;               // Anzahl nutzbarer Dimensionen
}

/**
 * Fehlwert-Imputation für die CA (§9.6-Vertiefung).
 *
 * CA verlangt eine vollständige Kontingenztabelle; einzelne Zellen lassen sich
 * nicht wie in der Metrik paarweise maskieren. Als „nicht erfasst" markierte
 * Zellen werden daher iterativ mit ihrem **Erwartungswert unter Unabhängigkeit**
 * gefüllt: E_ij = R_i · C_j / T aus den jeweils aktuellen Randsummen. Da das
 * Füllen die Ränder verschiebt, wird bis zur Konvergenz iteriert (klassisches
 * EM/Nora-Chouteau-Schema mit Rang-0-Rekonstruktion).
 *
 * Bewusste, konservative Wahl im Sinne von §9.6: Die Rang-0-Rekonstruktion
 * **erfindet keine Assoziation** in unbekannten Zellen (sie legt sie auf das
 * Unabhängigkeitsmodell), statt über eine reduzierte-Rang-Rekonstruktion Struktur
 * zu unterstellen, die nicht beobachtet wurde. Vollständig fehlende Zeilen/Spalten
 * bleiben 0 (kein Signal → Ursprung in der CA) — ehrlich statt geraten.
 *
 * Gibt bei `mask === null` die Originalmatrix unverändert zurück (Schnellpfad).
 */
export function imputeUnderIndependence(
  M: number[][], NR: number, NC: number, mask: Uint8Array | null,
  opts: { maxIter?: number; tol?: number } = {},
): number[][] {
  if (!mask) return M;
  const maxIter = opts.maxIter ?? 100, tol = opts.tol ?? 1e-8;
  const X = M.map((row) => row.slice());
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) if (mask[i * NC + j]) X[i][j] = 0; // 0-Start
  const R = new Float64Array(NR), C = new Float64Array(NC);
  for (let iter = 0; iter < maxIter; iter++) {
    R.fill(0); C.fill(0); let T = 0;
    for (let i = 0; i < NR; i++) { const row = X[i]; for (let j = 0; j < NC; j++) { const v = row[j]; R[i] += v; C[j] += v; T += v; } }
    if (T <= 0) break;
    let maxDelta = 0;
    for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) if (mask[i * NC + j]) {
      const e = (R[i] * C[j]) / T;
      const d = Math.abs(e - X[i][j]); if (d > maxDelta) maxDelta = d;
      X[i][j] = e;
    }
    if (maxDelta <= tol * (T / (NR * NC) + 1)) break;
  }
  return X;
}

export function computeCA(p: ProjectV2, dims = 4): CAResult {  const NR = p.contexts.length, NC = p.types.length;
  const M = imputeUnderIndependence(p.matrix, NR, NC, buildMissingMask(p)); // §9.6: Fehlwerte imputieren (sonst Original)
  let T = 0;
  const r = new Float64Array(NR), c = new Float64Array(NC);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) { const v = M[i][j]; if (v) { r[i] += v; c[j] += v; T += v; } }
  if (T === 0) return { rowCoords: [], colCoords: [], eigenvalues: [], inertiaPct: [], totalInertia: 0, k: 0 };
  for (let i = 0; i < NR; i++) r[i] /= T;
  for (let j = 0; j < NC; j++) c[j] /= T;
  const sr = Float64Array.from(r, (x) => Math.sqrt(x) || 1e-12);
  const sc = Float64Array.from(c, (x) => Math.sqrt(x) || 1e-12);

  // Standardisierte Residualmatrix S (NR×NC), zeilen-major
  const S = new Float64Array(NR * NC);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) {
    const pij = M[i][j] / T;
    S[i * NC + j] = (pij - r[i] * c[j]) / (sr[i] * sc[j]);
  }

  const { U, s, V, k: kk } = svd(S, NR, NC);
  const totalInertia = Array.from(s).reduce((a, sig) => a + sig * sig, 0);

  // nur Dimensionen mit nennenswertem Singulärwert behalten
  const kMax = Math.min(dims, kk);
  let k = 0; for (let d = 0; d < kMax; d++) if (s[d] > 1e-9) k = d + 1;
  k = Math.max(1, k);

  const eigenvalues: number[] = [], inertiaPct: number[] = [];
  for (let d = 0; d < k; d++) { const lam = s[d] * s[d]; eigenvalues.push(lam); inertiaPct.push(totalInertia ? lam / totalInertia : 0); }

  const rowCoords: number[][] = Array.from({ length: NR }, () => new Array(k).fill(0));
  const colCoords: number[][] = Array.from({ length: NC }, () => new Array(k).fill(0));
  for (let d = 0; d < k; d++) {
    for (let i = 0; i < NR; i++) rowCoords[i][d] = (s[d] * U[i * kk + d]) / sr[i];
    for (let j = 0; j < NC; j++) colCoords[j][d] = (s[d] * V[j * kk + d]) / sc[j];
  }
  return { rowCoords, colCoords, eigenvalues, inertiaPct, totalInertia, k };
}

/** Schlanke Zeilen-Prinzipalkoordinaten einer Dimension (für Bootstrap-Wiederholungen).
 *  §9.6: Mit `mask` (kanonisch, `1` = „nicht erfasst") werden Fehlwerte vor der CA
 *  imputiert; ohne Maske unverändert (Schnellpfad, byte-identisch). */
export function caRowScores(M: number[][], NR: number, NC: number, dim = 0, mask?: Uint8Array | null): Float64Array {
  if (mask) M = imputeUnderIndependence(M, NR, NC, mask);
  let T = 0; const r = new Float64Array(NR), c = new Float64Array(NC);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) { const v = M[i][j]; if (v) { r[i] += v; c[j] += v; T += v; } }
  const out = new Float64Array(NR);
  if (T === 0) return out;
  for (let i = 0; i < NR; i++) r[i] /= T; for (let j = 0; j < NC; j++) c[j] /= T;
  const sr = Float64Array.from(r, (x) => Math.sqrt(x) || 1e-12), sc = Float64Array.from(c, (x) => Math.sqrt(x) || 1e-12);
  const S = new Float64Array(NR * NC);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) S[i * NC + j] = (M[i][j] / T - r[i] * c[j]) / (sr[i] * sc[j]);
  const { U, s, k } = svd(S, NR, NC);
  const d = Math.min(dim, k - 1);
  for (let i = 0; i < NR; i++) out[i] = (s[d] * U[i * k + d]) / sr[i];
  return out;
}
