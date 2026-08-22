/**
 * Filterung & Fokus-Modus (Spezifikation §5.1).
 *
 * Rein und testbar: `visibleIndices` bestimmt die sichtbaren kanonischen Zeilen-/
 * Spaltenindizes unter den `FilterSettings` plus optionalem Fokus; `filterProject`
 * erzeugt daraus eine eigenständige, gefilterte Teil-Sicht (`ProjectV2`).
 *
 * Die gefilterte Sicht ist als **read-only Ansicht** gedacht (§6.2, Trennung
 * Struktur/Rohdaten): sie wird von allen verlinkten Ansichten identisch konsumiert
 * (Matrix, CA, Ford, Stabilität), verändert aber das zugrunde liegende Projekt nicht.
 * Annotationen werden über die stabilen Kontext-/Typnamen auf die neuen Indizes
 * umgeschlüsselt.
 */
import type { ProjectV2, FilterSettings } from "./model.js";
import { annotationKey } from "./model.js";

export interface FocusSel { ctx?: string | null; type?: string | null; }

/** Leere (inaktive) Filtereinstellungen. */
export function emptyFilters(): FilterSettings {
  return { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false };
}

/** Ist überhaupt ein Filter/Fokus aktiv? */
export function filtersActive(f: FilterSettings, focus?: FocusSel | null): boolean {
  return (
    f.materials.length > 0 || !!f.rowRange || !!f.colRange ||
    f.hideEmptyRows || f.hideEmptyCols ||
    !!(focus && (focus.ctx || focus.type))
  );
}

/** Sichtbare kanonische Indizes (aufsteigend) unter Filtern + optionalem Fokus. */
export function visibleIndices(p: ProjectV2, f: FilterSettings, focus?: FocusSel | null): { rows: number[]; cols: number[] } {
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;
  let rows = new Set<number>(Array.from({ length: NR }, (_, i) => i));
  let cols = new Set<number>(Array.from({ length: NC }, (_, j) => j));

  // Material-Filter (Spalten): nur Typen der gewählten Materialgruppen
  if (f.materials.length) {
    const keep = new Set(f.materials);
    cols = new Set([...cols].filter((j) => keep.has(p.columnMetadata[p.types[j]]?.materialGroup ?? "")));
  }
  // Bereichsfilter über kanonische Indizes (inklusiv)
  if (f.rowRange) { const [a, b] = f.rowRange; rows = new Set([...rows].filter((i) => i >= a && i <= b)); }
  if (f.colRange) { const [a, b] = f.colRange; cols = new Set([...cols].filter((j) => j >= a && j <= b)); }

  // Fokus: auf die Nachbarschaft der Auswahl einschränken
  if (focus && (focus.ctx || focus.type)) {
    const ci = focus.ctx ? p.contexts.indexOf(focus.ctx) : -1;
    const tj = focus.type ? p.types.indexOf(focus.type) : -1;
    const fRows = new Set<number>(), fCols = new Set<number>();
    if (ci >= 0) { fRows.add(ci); for (let j = 0; j < NC; j++) if (M[ci][j]) fCols.add(j); }
    if (tj >= 0) { fCols.add(tj); for (let i = 0; i < NR; i++) if (M[i][tj]) fRows.add(i); }
    // Kontexte, die einen der Fokus-Typen tragen …
    if (fCols.size) for (let i = 0; i < NR; i++) { for (const j of fCols) if (M[i][j]) { fRows.add(i); break; } }
    // … und Typen, die in den Fokus-Kontexten vorkommen
    if (fRows.size) for (const i of fRows) for (let j = 0; j < NC; j++) if (M[i][j]) fCols.add(j);
    if (fRows.size) rows = new Set([...rows].filter((i) => fRows.has(i)));
    if (fCols.size) cols = new Set([...cols].filter((j) => fCols.has(j)));
  }

  // Leere ausblenden (gegenseitig konsistent, nach allen anderen Filtern)
  if (f.hideEmptyCols) cols = new Set([...cols].filter((j) => { for (const i of rows) if (M[i][j]) return true; return false; }));
  if (f.hideEmptyRows) rows = new Set([...rows].filter((i) => { for (const j of cols) if (M[i][j]) return true; return false; }));

  return { rows: [...rows].sort((a, b) => a - b), cols: [...cols].sort((a, b) => a - b) };
}

/** Erzeugt aus Projekt + Filtern eine eigenständige, gefilterte Teil-Sicht. */
export function filterProject(p: ProjectV2, f: FilterSettings, focus?: FocusSel | null): ProjectV2 {
  const { rows, cols } = visibleIndices(p, f, focus);
  const contexts = rows.map((i) => p.contexts[i]);
  const types = cols.map((j) => p.types[j]);
  const matrix = rows.map((i) => cols.map((j) => p.matrix[i][j]));

  const columnMetadata: ProjectV2["columnMetadata"] = {};
  types.forEach((t) => { if (p.columnMetadata[t]) columnMetadata[t] = p.columnMetadata[t]; });
  const rowMetadata: ProjectV2["rowMetadata"] = {};
  contexts.forEach((c) => { if (p.rowMetadata[c]) rowMetadata[c] = p.rowMetadata[c]; });

  // Annotationen über stabile Namen auf neue Indizes umschlüsseln
  const newRowIdx = new Map(contexts.map((c, i) => [c, i] as const));
  const newColIdx = new Map(types.map((t, j) => [t, j] as const));
  const cellAnnotations: ProjectV2["cellAnnotations"] = {};
  for (const a of Object.values(p.cellAnnotations)) {
    const i = newRowIdx.get(a.context), j = newColIdx.get(a.type);
    if (i != null && j != null) cellAnnotations[annotationKey(i, j)] = a;
  }

  // „Nicht erfasst"-Markierungen (§9.6) namensbasiert umschlüsseln
  const missingCells: ProjectV2["missingCells"] = {};
  if (p.missingCells) for (const key of Object.keys(p.missingCells)) {
    const [bi, bj] = key.split(":").map(Number);
    const i = newRowIdx.get(p.contexts[bi]), j = newColIdx.get(p.types[bj]);
    if (i != null && j != null) missingCells[annotationKey(i, j)] = true;
  }

  // Anzeige-Reihenfolge beibehalten, auf sichtbare Elemente beschränkt
  const order = {
    rows: p.order.rows.filter((c) => newRowIdx.has(c)),
    cols: p.order.cols.filter((t) => newColIdx.has(t)),
  };

  return { ...p, contexts, types, matrix, columnMetadata, rowMetadata, cellAnnotations, missingCells: Object.keys(missingCells).length ? missingCells : undefined, order };
}

/**
 * Rück-Schreiben einer in der gefilterten Sicht erzeugten Reihenfolge in die
 * Grund-Ordnung (editierbare gefilterte Sichten): sichtbare Elemente werden
 * in `visibleSeq`-Reihenfolge auf die Positionen gesetzt, die zuvor von sichtbaren
 * Elementen belegt waren; verborgene Elemente behalten ihre absolute Position.
 * `base` und `visibleSeq` sind Namens-Arrays (Kontext- bzw. Typ-IDs).
 */
export function writeBackOrder(base: string[], visibleSeq: string[]): string[] {
  const vis = new Set(visibleSeq);
  const result = base.slice();
  let k = 0;
  for (let p = 0; p < base.length; p++) if (vis.has(base[p])) result[p] = visibleSeq[k++];
  return result;
}

/**
 * Spiegelt die Zell-Annotationen einer gefilterten Sicht namensbasiert ins
 * Grundprojekt (editierbare gefilterte Sichten, Annotieren). Im sichtbaren Fenster
 * wird der Annotationsbestand des Grundprojekts exakt an die Sicht angeglichen
 * (Hinzufügen, Ändern, Löschen); Annotationen außerhalb des Fensters bleiben
 * unberührt. Mutiert `base` in-place.
 */
export function writeBackAnnotations(base: ProjectV2, view: ProjectV2): void {
  const visRows = new Set(view.contexts), visCols = new Set(view.types);
  const baseRowIdx = new Map(base.contexts.map((c, i) => [c, i] as const));
  const baseColIdx = new Map(base.types.map((t, j) => [t, j] as const));
  // 1) bisherige Annotationen im sichtbaren Fenster entfernen (löst Löschungen auf)
  for (const [key, a] of Object.entries(base.cellAnnotations)) {
    if (visRows.has(a.context) && visCols.has(a.type)) delete base.cellAnnotations[key];
  }
  // 2) aktuellen Stand der Sicht übernehmen (über die stabilen Namen neu verschlüsseln)
  for (const a of Object.values(view.cellAnnotations)) {
    const i = baseRowIdx.get(a.context), j = baseColIdx.get(a.type);
    if (i != null && j != null) base.cellAnnotations[annotationKey(i, j)] = a;
  }
}

/** Wie `writeBackAnnotations`, aber für die „Nicht erfasst"-Markierungen (§9.6). */
export function writeBackMissing(base: ProjectV2, view: ProjectV2): void {
  const visRows = new Set(view.contexts), visCols = new Set(view.types);
  const baseRowIdx = new Map(base.contexts.map((c, i) => [c, i] as const));
  const baseColIdx = new Map(base.types.map((t, j) => [t, j] as const));
  if (base.missingCells) for (const key of Object.keys(base.missingCells)) {
    const [bi, bj] = key.split(":").map(Number);
    if (visRows.has(base.contexts[bi]) && visCols.has(base.types[bj])) delete base.missingCells[key];
  }
  if (view.missingCells && Object.keys(view.missingCells).length) {
    base.missingCells = base.missingCells || {};
    for (const key of Object.keys(view.missingCells)) {
      const [vi, vj] = key.split(":").map(Number);
      const i = baseRowIdx.get(view.contexts[vi]), j = baseColIdx.get(view.types[vj]);
      if (i != null && j != null) base.missingCells[annotationKey(i, j)] = true;
    }
  }
  if (base.missingCells && Object.keys(base.missingCells).length === 0) delete base.missingCells;
}
