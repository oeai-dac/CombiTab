/**
 * Bootstrap-Stabilität der Seriation (Spezifikation §9).
 *
 * Jede Zeile (Kontext) wird parametrisch neu gezogen: n_i Objekte werden
 * multinomial aus dem beobachteten Typprofil des Kontexts resampelt (Zeilensumme
 * bleibt erhalten, die Zusammensetzung variiert). Für jede Wiederholung wird die
 * CA-Dimension 1 berechnet, das Vorzeichen an die Referenz angeglichen und die
 * Kontexte gereiht. Über viele Wiederholungen ergibt sich je Kontext eine
 * Rang-Verteilung: enge Intervalle = chronologisch gut bestimmt, breite = unsicher.
 */
import type { ProjectV2 } from "../core/model.js";
import { caRowScores } from "./ca.js";
import { mulberry32 } from "../core/rng.js";

/** Re-Export für Abwärtskompatibilität: Bestehende Importe aus `./bootstrap.js`
 *  bleiben gültig, die Implementierung liegt jetzt zentral in `core/rng.ts`. */
export { mulberry32 };

export interface StabilityRow {
  context: string; refRank: number;
  mean: number; median: number; lo: number; hi: number; sd: number;
}
export interface StabilityResult { rows: StabilityRow[]; replicates: number; globalStability: number; }

export function bootstrapStability(p: ProjectV2, opts: { replicates?: number; rng?: () => number; onProgress?: (done: number, total: number) => void } = {}): StabilityResult {
  const B = opts.replicates ?? 200;
  const rng = opts.rng ?? mulberry32(12345);
  const onProgress = opts.onProgress;
  const progressStep = Math.max(1, Math.floor(B / 100)); // ~100 Meldungen
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;

  const ref = caRowScores(M, NR, NC, 0);
  const refRank = ranks(ref);

  // Vorbereitung der Zeilenprofile (kumulative Verteilungen) + Totalzahlen
  const rowTot = new Array<number>(NR).fill(0);
  const cum: number[][] = [];
  for (let i = 0; i < NR; i++) {
    let n = 0; for (let j = 0; j < NC; j++) n += M[i][j]; rowTot[i] = n;
    const cp: number[] = new Array(NC); let acc = 0;
    for (let j = 0; j < NC; j++) { acc += n ? M[i][j] / n : 0; cp[j] = acc; }
    cum.push(cp);
  }

  const rankSamples: number[][] = Array.from({ length: NR }, () => []);
  const R: number[][] = Array.from({ length: NR }, () => new Array<number>(NC).fill(0));
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < NR; i++) {
      const row = R[i]; row.fill(0);
      const n = rowTot[i], cp = cum[i];
      for (let k = 0; k < n; k++) { const u = rng(); let j = lowerBound(cp, u); if (j >= NC) j = NC - 1; row[j]++; }
    }
    const s = caRowScores(R, NR, NC, 0);
    if (corr(s, ref) < 0) for (let i = 0; i < NR; i++) s[i] = -s[i]; // Vorzeichen angleichen
    const rk = ranks(s);
    for (let i = 0; i < NR; i++) rankSamples[i].push(rk[i]);
    if (onProgress && ((b + 1) % progressStep === 0 || b + 1 === B)) onProgress(b + 1, B);
  }

  const rows: StabilityRow[] = [];
  let stabSum = 0;
  for (let i = 0; i < NR; i++) {
    const arr = rankSamples[i].slice().sort((a, b) => a - b);
    const lo = percentile(arr, 0.05), hi = percentile(arr, 0.95);
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const median = percentile(arr, 0.5);
    const sd = Math.sqrt(arr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / arr.length);
    rows.push({ context: p.contexts[i], refRank: refRank[i], mean, median, lo, hi, sd });
    stabSum += 1 - (hi - lo) / Math.max(1, NR - 1);
  }
  rows.sort((a, b) => a.refRank - b.refRank);
  return { rows, replicates: B, globalStability: stabSum / NR };
}

/* ── Helfer ── */
function ranks(a: Float64Array): number[] {
  const idx = Array.from(a.keys()).sort((i, j) => a[i] - a[j]);
  const r = new Array<number>(a.length); idx.forEach((v, i) => (r[v] = i)); return r;
}
function corr(a: Float64Array, b: Float64Array): number {
  const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; }
  return s / Math.sqrt(sa * sb || 1);
}
function lowerBound(cp: number[], u: number): number {
  let lo = 0, hi = cp.length; while (lo < hi) { const m = (lo + hi) >> 1; if (cp[m] < u) lo = m + 1; else hi = m; } return lo;
}
function percentile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = q * (sorted.length - 1), base = Math.floor(pos), rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}
