/**
 * Qualitätsmetriken für eine gegebene Anzeige-Ordnung (Spezifikation §7.3).
 *
 * Drei sauber getrennte Kennzahlen, alle im Bereich [0, 1] (höher = besser):
 *  - `concentration` : Konzentration der Belegung um die Diagonale (gewichteter
 *    Abstand, wie v1, vektorisiert über die dichte Matrix).
 *  - `antiRobinson`  : **echter Anti-Robinson-Index** auf einer Zeilen-Ähnlich-
 *    keitsmatrix. Robinson-Eigenschaft: die Ähnlichkeit zweier Kontexte fällt
 *    monoton mit ihrem Abstand in der Ordnung. Der Index misst den Anteil der
 *    Nachbarschafts-Vergleiche, die dieser Eigenschaft genügen (1 = perfekt
 *    Robinson). Ersetzt die in v1 fälschlich als „Anti-Robinson" geführte
 *    reine Lücken-/Kontinuitätszahl (§2.3).
 *  - `continuity`    : mittlere Kontinuität je Typ (Anteil belegter Zeilen
 *    innerhalb der Belegungsspanne) — die bisherige Kennzahl, korrekt benannt.
 *
 * `total` ist eine transparente, **konfigurierbare** Gewichtung dieser drei
 * Kennzahlen (Default dokumentiert unten). Die Gewichte lassen sich pro Aufruf
 * übersteuern.
 *
 * Hinweis Performance: der Anti-Robinson-Index ist O(NR²·NC). Für die hier
 * typischen Matrizen (einige hundert Zeilen) ist das unkritisch; für sehr große
 * Matrizen wird die Metrikberechnung über {@link qualityFromMatrix} in den
 * Score-Web-Worker verlagert (§8.5), damit der Haupt-Thread nach Drop/Undo/
 * Seriation nicht mehr sekundenlang blockiert.
 */
import type { ProjectV2 } from "../core/model.js";
import { buildMissingMask } from "../core/missing.js";
import type { Order } from "./centroid.js";

export interface Quality {
  concentration: number;
  antiRobinson: number;
  continuity: number;
  total: number;
}

/** Gewichte der Gesamt-Score (Summe idealerweise 1). Default dokumentiert. */
export interface QualityWeights {
  concentration: number;
  antiRobinson: number;
  continuity: number;
}

export const DEFAULT_WEIGHTS: QualityWeights = {
  concentration: 0.40,
  antiRobinson: 0.35,
  continuity: 0.25,
};

export function quality(p: ProjectV2, order: Order, weights: QualityWeights = DEFAULT_WEIGHTS): Quality {
  return qualityFromMatrix(p.matrix, p.contexts.length, p.types.length, order, weights, buildMissingMask(p));
}

/**
 * Matrix-basierte Variante von {@link quality} (§8.5). Entkoppelt die reine
 * Score-Berechnung vom `ProjectV2`, damit sie im Web-Worker ausgeführt werden
 * kann, ohne das gesamte Projekt zu übertragen (es genügen Matrix, Dimensionen
 * und Ordnung). Das Ergebnis ist byte-identisch zu `quality()` — dieselbe Formel,
 * nur ohne den `ProjectV2`-Umweg. `quality()` delegiert hierher.
 *
 * §9.6-Vertiefung: `mask` (optional, `NR*NC` zeilen-major, `1` = „nicht erfasst")
 * maskiert Fehlwerte statt sie wie 0 zu behandeln. Ohne Maske (`null`/undefiniert)
 * bleibt der Schnellpfad byte-identisch. Maskiert gilt:
 *   - Konzentration: markierte Zellen zählen weder in Distanz noch Gewicht.
 *   - Kontinuität: markierte Zellen innerhalb der Belegungsspanne gelten als
 *     unbekannt (keine Lücke) — sie werden aus dem Nenner herausgerechnet.
 *   - Anti-Robinson: paarweise-vollständige Ähnlichkeit (siehe {@link antiRobinsonIndex}).
 */
export function qualityFromMatrix(
  M: number[][],
  NR: number,
  NC: number,
  order: Order,
  weights: QualityWeights = DEFAULT_WEIGHTS,
  mask?: Uint8Array | null,
): Quality {
  const rpos = new Int32Array(NR), cpos = new Int32Array(NC);
  order.rows.forEach((r, i) => (rpos[r] = i));
  order.cols.forEach((c, j) => (cpos[c] = j));

  // --- Konzentration um die Diagonale ---
  let wd = 0, tw = 0;
  for (let r = 0; r < NR; r++) for (let c = 0; c < NC; c++) {
    if (mask && mask[r * NC + c]) continue; // nicht erfasst → nicht bewertet
    const v = M[r][c];
    if (v) { wd += v * Math.abs(rpos[r] / (NR - 1 || 1) - cpos[c] / (NC - 1 || 1)); tw += v; }
  }
  const concentration = tw ? 1 - wd / tw : 0;

  // --- Kontinuität je Typ ---
  let cs = 0, cn = 0;
  for (let c = 0; c < NC; c++) {
    let first = -1, last = -1, n = 0, missInSpanFirstLast = 0;
    for (let i = 0; i < NR; i++) {
      const rr = order.rows[i];
      if (mask && mask[rr * NC + c]) continue; // markierte Zelle ist keine Präsenz
      const v = M[rr][c];
      if (v) { if (first < 0) first = i; last = i; n++; }
    }
    if (n > 0) {
      // markierte Zellen innerhalb [first,last] aus dem Nenner nehmen (unbekannt ≠ Lücke)
      if (mask) for (let i = first + 1; i < last; i++) if (mask[order.rows[i] * NC + c]) missInSpanFirstLast++;
      const denom = (last - first + 1) - missInSpanFirstLast;
      if (denom > 0) { cs += n / denom; cn++; }
    }
  }
  const continuity = cn ? cs / cn : 0;

  // --- Anti-Robinson-Index ---
  const antiRobinson = antiRobinsonIndex(M, order.rows, mask, NC);

  const total =
    weights.concentration * concentration +
    weights.antiRobinson * antiRobinson +
    weights.continuity * continuity;

  return { concentration, antiRobinson, continuity, total };
}

/**
 * Echter Anti-Robinson-Index auf der Zeilen-Ähnlichkeitsmatrix.
 *
 * `rowOrder` sind kanonische Zeilenindizes in Anzeige-Reihenfolge. Für jede
 * Zeile als Anker werden die Ähnlichkeiten zu den Nachbarn nach außen betrachtet;
 * die Robinson-Eigenschaft verlangt, dass sie nach außen nicht ansteigen. Der
 * Index ist der Anteil erfüllter Nachbarschafts-Vergleiche in [0, 1].
 * Eine perfekt Robinson-geordnete Matrix liefert 1.
 *
 * §9.6-Vertiefung: Mit `mask` (kanonisch, `1` = „nicht erfasst") wird die
 * Cosinus-Ähnlichkeit **paarweise-vollständig** gerechnet — je Zeilenpaar nur über
 * die Spalten, in denen **beide** Zeilen erfasst sind. So verzerrt ein fehlender
 * Wert die Ähnlichkeit nicht (statt als 0 einzugehen). Ohne Maske läuft der
 * schnelle Pfad mit vorab berechneten Normen (byte-identisch zu vorher).
 */
export function antiRobinsonIndex(M: number[][], rowOrder: number[], mask?: Uint8Array | null, ncHint?: number): number {
  const NR = rowOrder.length;
  if (NR < 3) return 1;
  const NC = ncHint ?? (M[0]?.length ?? 0);

  // Zeilenvektoren in Anzeige-Reihenfolge; kanonische Indizes für die Maske merken.
  const rows: number[][] = new Array(NR);
  const canon = new Int32Array(NR);
  for (let p = 0; p < NR; p++) { canon[p] = rowOrder[p]; rows[p] = M[rowOrder[p]]; }

  let sim: (a: number, b: number) => number;
  if (!mask) {
    // Schnellpfad: Normen einmal vorab.
    const norm = new Float64Array(NR);
    for (let p = 0; p < NR; p++) { const v = rows[p]; let s = 0; for (let j = 0; j < NC; j++) s += v[j] * v[j]; norm[p] = Math.sqrt(s) || 1; }
    sim = (a, b) => { const va = rows[a], vb = rows[b]; let d = 0; for (let j = 0; j < NC; j++) d += va[j] * vb[j]; return d / (norm[a] * norm[b]); };
  } else {
    // Paarweise-vollständig: Skalarprodukt und Normen nur über gemeinsam erfasste Spalten.
    sim = (a, b) => {
      const va = rows[a], vb = rows[b], ca = canon[a] * NC, cb = canon[b] * NC;
      let d = 0, na = 0, nb = 0;
      for (let j = 0; j < NC; j++) {
        if (mask[ca + j] || mask[cb + j]) continue; // in mindestens einer Zeile nicht erfasst
        const x = va[j], y = vb[j];
        d += x * y; na += x * x; nb += y * y;
      }
      const den = Math.sqrt(na) * Math.sqrt(nb);
      return den ? d / den : 0;
    };
  }

  const EPS = 1e-9;
  let ok = 0, tot = 0;
  for (let a = 0; a < NR; a++) {
    // nach links: sim(a, p) sollte >= sim(a, p-1) sein (näher am Anker = ähnlicher)
    for (let p = a - 1; p >= 1; p--) { tot++; if (sim(a, p) >= sim(a, p - 1) - EPS) ok++; }
    // nach rechts: sim(a, p) sollte >= sim(a, p+1) sein
    for (let p = a + 1; p < NR - 1; p++) { tot++; if (sim(a, p) >= sim(a, p + 1) - EPS) ok++; }
  }
  return tot ? ok / tot : 1;
}
