/**
 * Fix-bewusstes Ordnungsmodell (rein, testbar) — Grundlage für Seriation,
 * Drag-Reorder und Pin/Unpin mit fixierten Elementen.
 *
 * Begriffe:
 *  - `order`  : Anzeige→kanonisch (Array der Länge N, order[displayPos] = canonicalIndex)
 *  - `fixed`  : Set kanonischer Indizes, deren Anzeige-Position gesperrt ist.
 *
 * Invariante: Ein fixiertes Element behält seine absolute Anzeige-Position. Alle
 * Umordnungen betreffen nur die *freien* Positionen (die nicht von fixierten
 * Elementen belegt sind).
 */

/** Positionen (0..N-1), deren Element NICHT fixiert ist. */
export function freePositions(order: number[], fixed: Set<number>): number[] {
  const out: number[] = [];
  for (let p = 0; p < order.length; p++) if (!fixed.has(order[p])) out.push(p);
  return out;
}

/**
 * Seriation anwenden: fixierte Elemente bleiben, wo sie sind; die freien
 * Positionen werden mit den nicht-fixierten Elementen in `sortedCanonical`-
 * Reihenfolge aufgefüllt.
 */
export function applySeriation(order: number[], fixed: Set<number>, sortedCanonical: number[]): number[] {
  const sortedFree = sortedCanonical.filter((x) => !fixed.has(x));
  const result = order.slice();
  let k = 0;
  for (let p = 0; p < order.length; p++) if (!fixed.has(order[p])) result[p] = sortedFree[k++];
  return result;
}

/**
 * Ein *freies* Element von Anzeige-Position `fromPos` möglichst nah an
 * Anzeige-Position `toPos` verschieben — ausschließlich innerhalb der freien
 * Positionen; fixierte Slots bleiben unberührt. Fixierte Elemente sind nicht
 * verschiebbar (Rückgabe unverändert).
 */
export function moveFree(order: number[], fixed: Set<number>, fromPos: number, toPos: number): number[] {
  if (fixed.has(order[fromPos])) return order.slice();       // fixiert → nicht bewegbar
  const free = freePositions(order, fixed);
  const fi = free.indexOf(fromPos);
  if (fi < 0) return order.slice();
  // Ziel-Rang unter den freien Positionen: Anzahl freier Positionen ≤ toPos, minus 1
  let ti = 0;
  for (const p of free) { if (p <= toPos) ti++; else break; }
  ti = Math.max(0, Math.min(free.length - 1, ti - 1));
  if (ti === fi) return order.slice();
  const elems = free.map((p) => order[p]);
  const [moved] = elems.splice(fi, 1);
  elems.splice(ti, 0, moved);
  const result = order.slice();
  free.forEach((p, k) => (result[p] = elems[k]));
  return result;
}

/** Fixierung eines Elements an Anzeige-Position `pos` umschalten (Reihenfolge bleibt). */
export function toggleFixed(order: number[], fixed: Set<number>, pos: number): Set<number> {
  const canon = order[pos];
  const next = new Set(fixed);
  if (next.has(canon)) next.delete(canon); else next.add(canon);
  return next;
}
