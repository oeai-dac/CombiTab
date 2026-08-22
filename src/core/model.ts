/**
 * CombiTab v2 — Kern-Datenmodell (framework-frei).
 *
 * Dieses Modul definiert das v2-Projektschema (schemaVersion: 2), wie in der
 * Spezifikation §6 beschrieben. Es ist bewusst frei von UI-/Framework-Abhängig-
 * keiten, damit es in Web-Workern, Tests und der App gleichermaßen läuft.
 *
 * Konventionen ggü. v1:
 *  - camelCase statt snake_case
 *  - eindeutige Identität über `contexts` (Zeilen) und `types` (Spalten) in
 *    kanonischer Reihenfolge; die *Anzeige*-Reihenfolge liegt getrennt in `order`
 *  - Zell-Annotationen werden über kanonische Indizes `"i:j"` referenziert
 *    (statt des mehrdeutigen v1-Schlüssels `"{row}_{col}"`) und tragen zusätzlich
 *    ihre Kontext-/Typ-Namen, sind also selbstbeschreibend.
 */

export type DataType = "presence_absence" | "frequency";

/** Sicherheits- und Fragmentierungsgrade sind offene Strings (Vorwärtskompatibilität),
 *  die bekannten Werte sind hier nur als Hinweis dokumentiert. */
export type Certainty = "certain" | "uncertain" | "questionable" | (string & {});
export type Fragmentation = "complete" | "fragmented" | "unknown" | (string & {});

export interface ColumnMetadata {
  name: string;
  materialGroup: string;
  color: string;          // Hex, z. B. "#CD853F"
  isIndexType: boolean;   // Leittyp
  isFixed: boolean;       // in der Seriation fixiert
  notes: string;
}

export interface RowMetadata {
  name: string;
  contextType: string;    // z. B. "Grave"
  area: string;
  isFixed: boolean;
  notes: string;
}

export interface CellAnnotation {
  context: string;        // Zeilen-ID (selbstbeschreibend, reorder-sicher)
  type: string;           // Spalten-ID
  certainty?: Certainty;
  fragmentation?: Fragmentation;
  countMin?: number;      // v1: count_min (optional)
  countMax?: number;      // v1: count_max (optional)
  inventoryNumbers?: string[]; // v1: inventory_numbers (optional)
  notes?: string;
}

export interface VisualizationSettings {
  vizStyle: string;       // v1: viz_style
  cellSize: number;
  showValues: boolean;
  showColors: boolean;
  showCertainty: boolean;
  showFragmentation: boolean;
}

export interface FilterSettings {
  materials: string[];                 // v1: filter_materials
  rowRange: [number, number] | null;   // v1: filter_row_range
  colRange: [number, number] | null;   // v1: filter_col_range
  hideEmptyRows: boolean;              // v1: filter_hide_empty_rows
  hideEmptyCols: boolean;              // v1: filter_hide_empty_cols
}

/** Provenienz-Eintrag (v2-Neuerung, §9). Bei Migration aus v1 leer. */
export interface AnalysisStep {
  method: string;
  params: Record<string, unknown>;
  timestamp: string;      // ISO
  score?: number;
}

export interface ProjectV2 {
  schemaVersion: 2;
  name: string;
  dataType: DataType;

  /** Kanonische Identität + Reihenfolge (aus v1 `matrix_index` bzw. Spaltenreihenfolge). */
  contexts: string[];     // Zeilen-IDs
  types: string[];        // Spalten-IDs

  /** Dichte Matrix, zeilen-major: matrix[i][j] = Wert von types[j] in contexts[i]. */
  matrix: number[][];

  columnMetadata: Record<string, ColumnMetadata>;
  rowMetadata: Record<string, RowMetadata>;

  /** Schlüssel: `"${rowIndex}:${colIndex}"` (kanonische Indizes). */
  cellAnnotations: Record<string, CellAnnotation>;

  /**
   * „Nicht erfasst" (fehlender Wert) vs. „strukturelle Absenz" (§9.6). Optional und
   * abwärtskompatibel: Schlüssel wie bei Annotationen (`"${i}:${j}"`); markierte Zellen
   * bedeuten „unbekannt/nicht erfasst" — im Gegensatz zur echten 0 (Typ war nicht da).
   */
  missingCells?: Record<string, true>;

  materialGroups: Record<string, string>; // Name -> Hexfarbe
  contextTypes: string[];

  /** Anzeige-Reihenfolge (kann von der kanonischen Identität abweichen). */
  order: { rows: string[]; cols: string[] };

  view: VisualizationSettings;
  filters: FilterSettings;

  history: AnalysisStep[];
}

/** Flache, typisierte Darstellung für Rechnen/Rendering (Int32Array, zeilen-major). */
export function toDense(p: ProjectV2): { data: Int32Array; nRows: number; nCols: number } {
  const nRows = p.contexts.length, nCols = p.types.length;
  const data = new Int32Array(nRows * nCols);
  for (let i = 0; i < nRows; i++) for (let j = 0; j < nCols; j++) data[i * nCols + j] = p.matrix[i][j] | 0;
  return { data, nRows, nCols };
}

/** Kanonischer Annotations-Schlüssel aus Zeilen-/Spaltenindex. */
export function annotationKey(rowIndex: number, colIndex: number): string {
  return `${rowIndex}:${colIndex}`;
}
