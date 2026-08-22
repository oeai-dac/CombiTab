/**
 * Zell-Annotationen (Spezifikation §5.1) — Hilfsfunktionen über ProjectV2.
 * Schlüssel ist der kanonische Index-Schlüssel "i:j"; die Annotation trägt zusätzlich
 * Kontext-/Typnamen und ist damit reorder-sicher.
 */
import type { ProjectV2, CellAnnotation } from "../core/model.js";
import { annotationKey } from "../core/model.js";

const FIELDS = ["certainty", "fragmentation", "countMin", "countMax", "inventoryNumbers", "notes"] as const;

export function getAnnotation(p: ProjectV2, i: number, j: number): CellAnnotation | undefined {
  return p.cellAnnotations[annotationKey(i, j)];
}

/** Felder mergen; leere Annotation (nur Kontext/Typ) wird entfernt. */
export function setAnnotation(p: ProjectV2, i: number, j: number, patch: Partial<CellAnnotation>): void {
  const key = annotationKey(i, j);
  const cur = p.cellAnnotations[key] ?? { context: p.contexts[i], type: p.types[j] };
  const next: CellAnnotation = { ...cur, ...patch, context: p.contexts[i], type: p.types[j] };
  for (const f of FIELDS) {
    const v = (next as unknown as Record<string, unknown>)[f];
    if (v === "" || v === undefined || v === null || (Array.isArray(v) && v.length === 0)) delete (next as unknown as Record<string, unknown>)[f];
  }
  if (isEmpty(next)) delete p.cellAnnotations[key];
  else p.cellAnnotations[key] = next;
}

export function deleteAnnotation(p: ProjectV2, i: number, j: number): void {
  delete p.cellAnnotations[annotationKey(i, j)];
}

/** Patch auf eine Menge von Zellen anwenden (Batch). */
export function applyToCells(p: ProjectV2, cells: Array<[number, number]>, patch: Partial<CellAnnotation>): void {
  for (const [i, j] of cells) setAnnotation(p, i, j, patch);
}

export function clearCells(p: ProjectV2, cells: Array<[number, number]>): void {
  for (const [i, j] of cells) deleteAnnotation(p, i, j);
}

/** Gemeinsamer Wert eines Feldes über alle Zellen, sonst undefined (für Vorbelegung des Editors). */
export function commonValue<K extends keyof CellAnnotation>(p: ProjectV2, cells: Array<[number, number]>, field: K): CellAnnotation[K] | undefined {
  let val: CellAnnotation[K] | undefined; let first = true;
  for (const [i, j] of cells) {
    const a = getAnnotation(p, i, j);
    const v = a ? a[field] : undefined;
    if (first) { val = v; first = false; }
    else if (JSON.stringify(v) !== JSON.stringify(val)) return undefined;
  }
  return val;
}

/**
 * Baut aus den Editor-Werten einen Batch-Patch mit „Touched"-Semantik:
 * Nur Felder, die der/die Nutzer:in tatsächlich angefasst hat, kommen in den Patch —
 * ein angefasstes leeres Feld löscht das Feld (explizit), ein NICHT angefasstes Feld
 * fehlt im Patch und bleibt damit in jeder Zelle unverändert. Vorher wurden bei
 * gemischter Auswahl leere (weil uneinheitliche) Felder als Löschung mitgesendet —
 * stiller Datenverlust.
 */
export function buildBatchPatch(values: {
  certainty: string; fragmentation: string; countMin: string; countMax: string; inv: string; notes: string;
}, touched: ReadonlySet<string>): Partial<CellAnnotation> {
  const patch: Partial<CellAnnotation> = {};
  if (touched.has("certainty")) patch.certainty = values.certainty || undefined;
  if (touched.has("fragmentation")) patch.fragmentation = values.fragmentation || undefined;
  if (touched.has("countMin")) patch.countMin = values.countMin.trim() === "" ? undefined : Number(values.countMin);
  if (touched.has("countMax")) patch.countMax = values.countMax.trim() === "" ? undefined : Number(values.countMax);
  if (touched.has("inv")) patch.inventoryNumbers = values.inv.trim() === "" ? undefined : values.inv.split(",").map((s) => s.trim()).filter(Boolean);
  if (touched.has("notes")) patch.notes = values.notes.trim() === "" ? undefined : values.notes.trim();
  return patch;
}

export function annotationCount(p: ProjectV2): number { return Object.keys(p.cellAnnotations).length; }

function isEmpty(a: CellAnnotation): boolean { return !FIELDS.some((f) => (a as unknown as Record<string, unknown>)[f] !== undefined); }

/** Ampelfarbe nach Sicherheit (für Zellmarker). */
export function certaintyColor(c?: string): string {
  return c === "certain" ? "#3f7a4f" : c === "uncertain" ? "#b5892a" : c === "questionable" ? "#b23a2a" : "#8b857c";
}
