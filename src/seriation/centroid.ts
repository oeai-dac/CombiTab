/**
 * Seriation über die Schwerpunktmethode (reciprocal averaging).
 * Arbeitet auf der dichten Matrix eines ProjectV2 und liefert Anzeige-Ordnungen
 * (Arrays kanonischer Indizes) für Zeilen und Spalten.
 *
 * Fixierte Elemente (columnMetadata.isFixed / rowMetadata.isFixed) werden hier
 * bewusst noch nicht sonderbehandelt — das kommt in einer späteren Ausbaustufe;
 * die Grundmechanik ist unveraendert.
 */
import type { ProjectV2 } from "../core/model.js";
import { mulberry32 } from "../core/rng.js";
import { buildMissingMask } from "../core/missing.js";

export interface Order { rows: number[]; cols: number[]; }

/**
 * Seriation über die Schwerpunktmethode.
 *
 * Der Startvektor wird über einen seed-basierten PRNG (mulberry32) initialisiert
 * statt über `Math.random()` — dadurch ist der Lauf **reproduzierbar** (gleicher
 * Seed ⇒ identische Ordnung), passend zur Provenienz-/Methods-Anforderung §9.1.
 * Der Seed ist Teil der Signatur und wird in der `history` protokolliert.
 */
export function seriateCentroid(p: ProjectV2, iters = 15, seed = 12345): Order {
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;
  const mask = buildMissingMask(p); // §9.6: nicht erfasste Zellen aus dem Mittel nehmen
  const miss = mask ? (i: number, j: number) => mask[i * NC + j] === 1 : () => false;
  const rs = new Float64Array(NR), cs = new Float64Array(NC);
  const rw = new Float64Array(NR), cw = new Float64Array(NC);
  const rng = mulberry32(seed);
  for (let i = 0; i < NR; i++) rs[i] = rng();
  for (let i = 0; i < NR; i++) { let w = 0; for (let j = 0; j < NC; j++) if (!miss(i, j)) w += M[i][j]; rw[i] = w || 1; }
  for (let j = 0; j < NC; j++) { let w = 0; for (let i = 0; i < NR; i++) if (!miss(i, j)) w += M[i][j]; cw[j] = w || 1; }
  for (let it = 0; it < iters; it++) {
    cs.fill(0);
    for (let i = 0; i < NR; i++) { const ri = rs[i], row = M[i]; for (let j = 0; j < NC; j++) { const v = row[j]; if (v && !miss(i, j)) cs[j] += v * ri; } }
    for (let j = 0; j < NC; j++) cs[j] /= cw[j];
    rs.fill(0);
    for (let i = 0; i < NR; i++) { const row = M[i]; let s = 0; for (let j = 0; j < NC; j++) { const v = row[j]; if (v && !miss(i, j)) s += v * cs[j]; } rs[i] = s / rw[i]; }
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < NR; i++) { if (rs[i] < mn) mn = rs[i]; if (rs[i] > mx) mx = rs[i]; }
    const rg = (mx - mn) || 1;
    for (let i = 0; i < NR; i++) rs[i] = (rs[i] - mn) / rg;
  }
  return { rows: argsort(rs), cols: argsort(cs) };
}

export function argsort(a: Float64Array): number[] {
  return Array.from(a.keys()).sort((i, j) => a[i] - a[j]);
}
