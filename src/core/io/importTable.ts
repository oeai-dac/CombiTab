/**
 * Tabellen-Import → ProjectV2 (Spezifikation §5.2).
 *
 * Deckt die in der Spec geforderte Flexibilität ab:
 *  - Wide-Format (Kontexte × Typen) und Long-Format (Kontext, Typ, Anzahl je Zeile)
 *  - Transponieren (Zeilen/Spalten vertauscht)
 *  - Trennzeichen-Erkennung (über parseDelimited)
 *  - automatische Datentyp-Erkennung (presence/absence vs. Frequenz), übersteuerbar
 *  - Validierung (Duplikate, nicht-numerische Werte) mit Warnungsbericht
 *
 * Der Kern arbeitet auf einem bereits geparsten Zellraster `string[][]`, damit
 * dieselbe Logik CSV *und* XLSX bedient (siehe importCSV / importXLSX).
 */

import type {
  ProjectV2, DataType, ColumnMetadata, RowMetadata,
  VisualizationSettings, FilterSettings,
} from "../model.js";
import { parseDelimited, detectDelimiter, type Delimiter } from "./parseDelimited.js";
import { isMissingToken } from "../missing.js";

export interface ImportOptions {
  name?: string;
  delimiter?: Delimiter;                 // nur CSV; sonst Auto
  format?: "wide" | "long";              // Default "wide"
  transpose?: boolean;                   // Wide: Zeilen/Spalten tauschen
  hasHeader?: boolean;                   // Default true
  hasIndexColumn?: boolean;              // Wide: erste Spalte = Kontextnamen (Default true)
  dataType?: DataType | "auto";          // Default "auto"
  /** Long-Format: Spaltennamen oder 0-basierte Indizes für Kontext/Typ/Anzahl. */
  long?: { context?: string | number; type?: string | number; count?: string | number };
}

export interface ImportReport {
  rows: number; cols: number; filledCells: number;
  dataType: DataType; delimiter?: Delimiter; format: "wide" | "long";
  warnings: string[];
}
export interface ImportResult { project: ProjectV2; report: ImportReport; }

export class ImportError extends Error {}

/* ── Standard-Paletten (deckungsgleich mit v1, damit Speichern „nativ" aussieht) ── */
const DEFAULT_MATERIAL_GROUPS: Record<string, string> = {
  Unassigned: "#808080", Ceramic: "#CD853F", Metal: "#4682B4",
  Glass: "#20B2AA", "Bone/Antler": "#DEB887", Stone: "#696969", Organic: "#8B4513",
};
const DEFAULT_CONTEXT_TYPES = ["Unassigned", "Grave", "Pit", "Ditch", "Layer", "Posthole", "Well"];

/* ── Öffentliche Einstiegspunkte ── */
export function importCSV(text: string, opts: ImportOptions = {}): ImportResult {
  const delim = opts.delimiter ?? detectDelimiter(text);
  const grid = parseDelimited(text, delim);
  const res = importGrid(grid, opts);
  res.report.delimiter = delim;
  return res;
}

/** Kern: geparstes Zellraster → ProjectV2. */
export function importGrid(grid: string[][], opts: ImportOptions = {}): ImportResult {
  if (grid.length === 0) throw new ImportError("Leere Tabelle.");
  const warnings: string[] = [];
  const format = opts.format ?? "wide";
  const result = format === "long"
    ? buildFromLong(grid, opts, warnings)
    : buildFromWide(grid, opts, warnings);

  const { contexts, types, matrix, missing } = result;
  if (contexts.length === 0 || types.length === 0)
    throw new ImportError("Keine Daten: Kontexte oder Typen fehlen.");
  assertUnique(contexts, "Kontext");
  assertUnique(types, "Typ");

  // Datentyp erkennen
  let filled = 0, onlyBinary = true;
  for (const row of matrix) for (const v of row) { if (v !== 0) filled++; if (v !== 0 && v !== 1) onlyBinary = false; }
  let dataType: DataType;
  if (opts.dataType && opts.dataType !== "auto") dataType = opts.dataType;
  else { dataType = onlyBinary ? "presence_absence" : "frequency";
    warnings.push(`Datentyp automatisch erkannt: ${dataType === "presence_absence" ? "Präsenz/Absenz" : "Frequenz"}.`); }

  const project = assembleProject(contexts, types, matrix, dataType, opts.name, missing);
  return { project, report: { rows: contexts.length, cols: types.length, filledCells: filled, dataType, format, warnings } };
}

/* ── Wide-Format ── */
function buildFromWide(grid: string[][], opts: ImportOptions, warnings: string[]) {
  const hasHeader = opts.hasHeader ?? true;
  const hasIndex = opts.hasIndexColumn ?? true;

  let g = grid;
  if (opts.transpose) g = transpose(g);

  const headerRow = hasHeader ? g[0] : null;
  const bodyRows = hasHeader ? g.slice(1) : g;

  const types = headerRow
    ? headerRow.slice(hasIndex ? 1 : 0).map((s) => s.trim())
    : bodyRows[0].slice(hasIndex ? 1 : 0).map((_, j) => `Typ_${j + 1}`);

  const contexts: string[] = [];
  const matrix: number[][] = [];
  const missing = new Set<string>();
  bodyRows.forEach((r, i) => {
    if (r.length === 1 && r[0].trim() === "") return; // leere Zeile
    const ctx = hasIndex ? (r[0]?.trim() || `Kontext_${i + 1}`) : `Kontext_${i + 1}`;
    const ri = contexts.push(ctx) - 1;
    const cells = r.slice(hasIndex ? 1 : 0);
    const row = new Array<number>(types.length).fill(0);
    for (let j = 0; j < types.length; j++) {
      if (isMissingToken(cells[j])) { missing.add(`${ri}:${j}`); row[j] = 0; }  // „nicht erfasst" (§9.6)
      else row[j] = parseValue(cells[j], ctx, types[j], warnings);
    }
    matrix.push(row);
  });
  return { contexts, types, matrix, missing };
}

/* ── Long-Format (Kontext, Typ, Anzahl je Zeile) ── */
function buildFromLong(grid: string[][], opts: ImportOptions, warnings: string[]) {
  const hasHeader = opts.hasHeader ?? true;
  const header = hasHeader ? grid[0].map((s) => s.trim()) : null;
  const body = hasHeader ? grid.slice(1) : grid;
  const resolve = (spec: string | number | undefined, fallback: number): number => {
    if (spec === undefined) return fallback;
    if (typeof spec === "number") return spec;
    const idx = header ? header.indexOf(spec) : -1;
    if (idx < 0) throw new ImportError(`Long-Format: Spalte "${spec}" nicht gefunden.`);
    return idx;
  };
  const ci = resolve(opts.long?.context, 0);
  const ti = resolve(opts.long?.type, 1);
  const vi = resolve(opts.long?.count, 2);

  const ctxOrder: string[] = [], typeOrder: string[] = [];
  const ctxSet = new Map<string, number>(), typeSet = new Map<string, number>();
  const triples: Array<[number, number, number]> = [];
  const missing = new Set<string>();
  body.forEach((r, k) => {
    if (r.length === 1 && r[0].trim() === "") return;
    const ctx = (r[ci] ?? "").trim(), typ = (r[ti] ?? "").trim();
    if (!ctx || !typ) { warnings.push(`Zeile ${k + 1}: Kontext oder Typ leer — übersprungen.`); return; }
    if (!ctxSet.has(ctx)) { ctxSet.set(ctx, ctxOrder.length); ctxOrder.push(ctx); }
    if (!typeSet.has(typ)) { typeSet.set(typ, typeOrder.length); typeOrder.push(typ); }
    const ii = ctxSet.get(ctx)!, jj = typeSet.get(typ)!;
    if (isMissingToken(r[vi])) { missing.add(`${ii}:${jj}`); triples.push([ii, jj, 0]); }  // „nicht erfasst" (§9.6)
    else { const v = parseValue(r[vi], ctx, typ, warnings); if (v > 0) missing.delete(`${ii}:${jj}`); triples.push([ii, jj, v]); }
  });
  const matrix = ctxOrder.map(() => new Array<number>(typeOrder.length).fill(0));
  let dupes = 0;
  for (const [i, j, v] of triples) { if (matrix[i][j] !== 0) dupes++; matrix[i][j] += v; } // Duplikate summieren
  if (dupes) warnings.push(`${dupes} doppelte (Kontext, Typ)-Paare wurden aufsummiert.`);
  return { contexts: ctxOrder, types: typeOrder, matrix, missing };
}

/* ── ProjectV2 zusammensetzen (Defaults konsistent mit v1) ── */
function assembleProject(
  contexts: string[], types: string[], matrix: number[][],
  dataType: DataType, name?: string, missing?: Set<string>,
): ProjectV2 {
  const columnMetadata: Record<string, ColumnMetadata> = {};
  for (const t of types) columnMetadata[t] = {
    name: t, materialGroup: "Unassigned", color: "#808080", isIndexType: false, isFixed: false, notes: "",
  };
  const rowMetadata: Record<string, RowMetadata> = {};
  for (const c of contexts) rowMetadata[c] = {
    name: c, contextType: "Unassigned", area: "", isFixed: false, notes: "",
  };
  const view: VisualizationSettings = {
    vizStyle: "classic", cellSize: 0.4, showValues: true, showColors: true,
    showCertainty: false, showFragmentation: false,
  };
  const filters: FilterSettings = {
    materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false,
  };
  return {
    schemaVersion: 2,
    name: name ?? "Importiertes Projekt",
    dataType,
    contexts: [...contexts], types: [...types], matrix,
    columnMetadata, rowMetadata, cellAnnotations: {},
    materialGroups: { ...DEFAULT_MATERIAL_GROUPS },
    contextTypes: [...DEFAULT_CONTEXT_TYPES],
    order: { rows: [...contexts], cols: [...types] },
    missingCells: missing && missing.size ? Object.fromEntries([...missing].map((k) => [k, true as const])) : undefined,
    view, filters, history: [],
  };
}

/* ── Hilfsfunktionen ── */
function parseValue(raw: string | undefined, ctx: string, typ: string, warnings: string[]): number {
  const s = (raw ?? "").trim();
  if (s === "") return 0;
  const n = Number(s.replace(",", "."));   // "3,5" ebenso zulassen
  if (!Number.isFinite(n)) {
    if (warnings.length < 12) warnings.push(`Nicht-numerischer Wert "${s}" (${ctx} / ${typ}) → 0.`);
    return 0;
  }
  return n;
}
function transpose(g: string[][]): string[][] {
  const cols = Math.max(...g.map((r) => r.length));
  const out: string[][] = [];
  for (let j = 0; j < cols; j++) out.push(g.map((r) => r[j] ?? ""));
  return out;
}
function assertUnique(names: string[], label: string): void {
  const seen = new Set<string>();
  for (const n of names) { if (seen.has(n)) throw new ImportError(`Doppelter ${label}-Name: "${n}".`); seen.add(n); }
}
