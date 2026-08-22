/**
 * CombiTab v1 → v2 Migrationsadapter (Spezifikation §6.3).
 *
 * `migrateV1`  : liest ein v1-Projekt (JSON-Objekt) und liefert ein `ProjectV2`.
 * `dumpV1`     : Rückrichtung (v2 → v1). Dient dem Beweis der Verlustfreiheit:
 *                migrateV1 gefolgt von dumpV1 muss das Original exakt reproduzieren.
 *
 * Kernproblem (in §6.3 benannt): v1 referenziert Zell-Annotationen über den
 * Schlüssel `"{row}_{col}"`. Da sowohl Zeilen- (`Grave_001`) als auch Spalten-
 * namen (`Shield_boss_conical`) Unterstriche enthalten, ist ein naives Splitten
 * mehrdeutig. Wir lösen das über Präfix-Abgleich mit der bekannten Zeilenliste
 * (`matrix_index`): Für jeden Schlüssel wird der eindeutige Zeilenname gesucht,
 * dessen Präfix passt und dessen Rest ein gültiger Spaltenname ist.
 */

import type {
  ProjectV2, ColumnMetadata, RowMetadata, CellAnnotation,
  VisualizationSettings, FilterSettings, DataType,
} from "../model.js";
import { annotationKey } from "../model.js";

/* ── v1-Typen (so, wie sie in der JSON-Datei stehen) ── */
export interface ProjectV1 {
  name: string;
  data_type: string;
  matrix: Record<string, Record<string, number>>; // col -> (row -> value)
  matrix_index: string[];                          // Zeilennamen (kanonisch)
  column_metadata: Record<string, {
    name: string; material_group: string; color: string;
    is_index_type: boolean; is_fixed: boolean; notes: string;
  }>;
  row_metadata: Record<string, {
    name: string; context_type: string; area: string; is_fixed: boolean; notes: string;
  }>;
  cell_annotations: Record<string, {
    certainty?: string; fragmentation?: string;
    count_min?: number; count_max?: number;
    inventory_numbers?: string[]; notes?: string;
  }>;
  material_groups: Record<string, string>;
  context_types: string[];
  row_order: string[];
  col_order: string[];
  visualization_settings: {
    viz_style: string; cell_size: number;
    show_values: boolean; show_colors: boolean;
    show_certainty: boolean; show_fragmentation: boolean;
  };
  filter_settings: {
    filter_materials: string[];
    filter_row_range: [number, number] | null;
    filter_col_range: [number, number] | null;
    filter_hide_empty_rows: boolean;
    filter_hide_empty_cols: boolean;
  };
}

export class MigrationError extends Error {}

/** Zerlegt einen v1-Annotationsschlüssel eindeutig in [row, col]. */
export function splitAnnotationKey(
  key: string, rowNames: string[], colNameSet: Set<string>,
): [string, string] {
  const matches: Array<[string, string]> = [];
  for (const r of rowNames) {
    const pre = r + "_";
    if (key.startsWith(pre)) {
      const c = key.slice(pre.length);
      if (colNameSet.has(c)) matches.push([r, c]);
    }
  }
  if (matches.length === 1) return matches[0];
  if (matches.length === 0)
    throw new MigrationError(`Annotationsschlüssel nicht auflösbar: "${key}"`);
  throw new MigrationError(
    `Annotationsschlüssel mehrdeutig: "${key}" → ${matches.map(m => m.join("/")).join(", ")}`,
  );
}

/* ── Vorwärts: v1 → v2 ── */
export function migrateV1(v1: ProjectV1): ProjectV2 {
  requireKeys(v1, [
    "name", "data_type", "matrix", "matrix_index", "column_metadata",
    "row_metadata", "cell_annotations", "material_groups", "context_types",
    "row_order", "col_order", "visualization_settings", "filter_settings",
  ]);

  const contexts = [...v1.matrix_index];        // kanonische Zeilen-Identität/-Reihenfolge
  const types = Object.keys(v1.matrix);         // kanonische Spalten-Identität/-Reihenfolge
  const rowIndex = index(contexts), colIndex = index(types);
  const colNameSet = new Set(types);

  // Dichte, zeilen-major Matrix aus dem spalten-major v1-Dict
  const matrix: number[][] = contexts.map((r) =>
    types.map((c) => {
      const col = v1.matrix[c];
      const v = col ? col[r] : undefined;
      return typeof v === "number" ? v : 0;
    }),
  );

  const columnMetadata: Record<string, ColumnMetadata> = {};
  for (const c of types) {
    const m = v1.column_metadata[c];
    if (!m) throw new MigrationError(`Spalten-Metadaten fehlen für "${c}"`);
    columnMetadata[c] = {
      name: m.name, materialGroup: m.material_group, color: m.color,
      isIndexType: m.is_index_type, isFixed: m.is_fixed, notes: m.notes,
    };
  }

  const rowMetadata: Record<string, RowMetadata> = {};
  for (const r of contexts) {
    const m = v1.row_metadata[r];
    if (!m) throw new MigrationError(`Zeilen-Metadaten fehlen für "${r}"`);
    rowMetadata[r] = {
      name: m.name, contextType: m.context_type, area: m.area,
      isFixed: m.is_fixed, notes: m.notes,
    };
  }

  // Annotationen: mehrdeutigen Schlüssel auflösen → kanonischer Index-Schlüssel
  const cellAnnotations: Record<string, CellAnnotation> = {};
  for (const [key, a] of Object.entries(v1.cell_annotations)) {
    const [row, col] = splitAnnotationKey(key, contexts, colNameSet);
    const ann: CellAnnotation = { context: row, type: col };
    if (a.certainty !== undefined) ann.certainty = a.certainty;
    if (a.fragmentation !== undefined) ann.fragmentation = a.fragmentation;
    if (a.count_min !== undefined) ann.countMin = a.count_min;
    if (a.count_max !== undefined) ann.countMax = a.count_max;
    if (a.inventory_numbers !== undefined) ann.inventoryNumbers = a.inventory_numbers;
    if (a.notes !== undefined) ann.notes = a.notes;
    cellAnnotations[annotationKey(rowIndex.get(row)!, colIndex.get(col)!)] = ann;
  }

  const view: VisualizationSettings = {
    vizStyle: v1.visualization_settings.viz_style,
    cellSize: v1.visualization_settings.cell_size,
    showValues: v1.visualization_settings.show_values,
    showColors: v1.visualization_settings.show_colors,
    showCertainty: v1.visualization_settings.show_certainty,
    showFragmentation: v1.visualization_settings.show_fragmentation,
  };

  const filters: FilterSettings = {
    materials: [...v1.filter_settings.filter_materials],
    rowRange: v1.filter_settings.filter_row_range,
    colRange: v1.filter_settings.filter_col_range,
    hideEmptyRows: v1.filter_settings.filter_hide_empty_rows,
    hideEmptyCols: v1.filter_settings.filter_hide_empty_cols,
  };

  return {
    schemaVersion: 2,
    name: v1.name,
    dataType: v1.data_type as DataType,
    contexts, types, matrix,
    columnMetadata, rowMetadata, cellAnnotations,
    materialGroups: { ...v1.material_groups },
    contextTypes: [...v1.context_types],
    order: { rows: [...v1.row_order], cols: [...v1.col_order] },
    view, filters,
    history: [],   // v2-Neuerung; bei Migration leer
  };
}

/* ── Rückwärts: v2 → v1 (für den Verlustfreiheits-Beweis) ── */
export function dumpV1(p: ProjectV2): ProjectV1 {
  const { contexts, types } = p;

  const matrix: Record<string, Record<string, number>> = {};
  types.forEach((c, j) => {
    const col: Record<string, number> = {};
    contexts.forEach((r, i) => { col[r] = p.matrix[i][j]; });
    matrix[c] = col;
  });

  const column_metadata: ProjectV1["column_metadata"] = {};
  for (const c of types) {
    const m = p.columnMetadata[c];
    column_metadata[c] = {
      name: m.name, material_group: m.materialGroup, color: m.color,
      is_index_type: m.isIndexType, is_fixed: m.isFixed, notes: m.notes,
    };
  }

  const row_metadata: ProjectV1["row_metadata"] = {};
  for (const r of contexts) {
    const m = p.rowMetadata[r];
    row_metadata[r] = {
      name: m.name, context_type: m.contextType, area: m.area,
      is_fixed: m.isFixed, notes: m.notes,
    };
  }

  const cell_annotations: ProjectV1["cell_annotations"] = {};
  for (const a of Object.values(p.cellAnnotations)) {
    const out: ProjectV1["cell_annotations"][string] = {};
    if (a.certainty !== undefined) out.certainty = a.certainty;
    if (a.fragmentation !== undefined) out.fragmentation = a.fragmentation;
    if (a.countMin !== undefined) out.count_min = a.countMin;
    if (a.countMax !== undefined) out.count_max = a.countMax;
    if (a.inventoryNumbers !== undefined) out.inventory_numbers = a.inventoryNumbers;
    if (a.notes !== undefined) out.notes = a.notes;
    cell_annotations[`${a.context}_${a.type}`] = out;
  }

  return {
    name: p.name,
    data_type: p.dataType,
    matrix,
    matrix_index: [...contexts],
    column_metadata, row_metadata, cell_annotations,
    material_groups: { ...p.materialGroups },
    context_types: [...p.contextTypes],
    row_order: [...p.order.rows],
    col_order: [...p.order.cols],
    visualization_settings: {
      viz_style: p.view.vizStyle, cell_size: p.view.cellSize,
      show_values: p.view.showValues, show_colors: p.view.showColors,
      show_certainty: p.view.showCertainty, show_fragmentation: p.view.showFragmentation,
    },
    filter_settings: {
      filter_materials: [...p.filters.materials],
      filter_row_range: p.filters.rowRange,
      filter_col_range: p.filters.colRange,
      filter_hide_empty_rows: p.filters.hideEmptyRows,
      filter_hide_empty_cols: p.filters.hideEmptyCols,
    },
  };
}

/* ── Hilfsfunktionen ── */
function index(arr: string[]): Map<string, number> {
  const m = new Map<string, number>();
  arr.forEach((v, i) => m.set(v, i));
  return m;
}
function requireKeys(o: object, keys: string[]): void {
  for (const k of keys) if (!(k in o)) throw new MigrationError(`v1-Feld fehlt: "${k}"`);
}
