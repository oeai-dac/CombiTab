/**
 * Referenzdatensätze zur Validierung der Korrespondenzanalyse (§9.10).
 *
 * Die selbstgeschriebene CA wird gegen **publizierte** Ergebnisse geprüft, nicht nur
 * auf interne Konsistenz. Primäranker ist Greenacres kanonischer „smoking"-Datensatz
 * (auch als `ca::smoke` im R-Paket `ca` enthalten), dessen Haupt­trägheiten und
 * Prinzipalkoordinaten in der Literatur bis auf vier Dezimalen dokumentiert sind.
 *
 * Quelle: M. Greenacre, „Correspondence Analysis in Practice" (2. Aufl., 2007),
 * Kap. 8/13; identisch reproduziert vom R-Paket `ca` (Nenadic & Greenacre 2007).
 *
 * Vorzeichen von CA-Achsen sind konventionsabhängig (Eigenvektoren bis aufs Vorzeichen
 * bestimmt); der Test richtet jede Achse vor dem Vergleich am Vorzeichen aus.
 */
import type { ProjectV2, ColumnMetadata, RowMetadata } from "../core/model.js";

/** Baut ein minimales, gültiges ProjectV2 aus einer rohen Zähl-Matrix. */
export function referenceProject(name: string, rows: string[], cols: string[], matrix: number[][]): ProjectV2 {
  const columnMetadata: Record<string, ColumnMetadata> = {};
  cols.forEach((t) => (columnMetadata[t] = { name: t, materialGroup: "ref", color: "#888888", isIndexType: false, isFixed: false, notes: "" }));
  const rowMetadata: Record<string, RowMetadata> = {};
  rows.forEach((r) => (rowMetadata[r] = { name: r, contextType: "", area: "", isFixed: false, notes: "" }));
  return {
    schemaVersion: 2, name, dataType: "frequency",
    contexts: rows.slice(), types: cols.slice(), matrix: matrix.map((r) => r.slice()),
    columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { ref: "#888888" }, contextTypes: [],
    order: { rows: rows.slice(), cols: cols.slice() },
    view: { vizStyle: "classic", cellSize: 18, showValues: false, showColors: true, showCertainty: false, showFragmentation: false },
    filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false },
    history: [],
  };
}

/* ── Greenacre „smoking" (Personal × Rauchverhalten) ── */
export const SMOKE = {
  rows: ["SM", "JM", "SE", "JE", "SC"],
  cols: ["none", "light", "medium", "heavy"],
  matrix: [
    [4, 2, 3, 2],
    [4, 3, 7, 4],
    [25, 10, 12, 4],
    [18, 24, 33, 13],
    [10, 6, 7, 2],
  ],
  // Publizierte Referenzwerte:
  totalInertia: 0.085190,               // Summe der drei Haupt­trägheiten
  eigenvalues: [0.074759, 0.010017, 0.000414],
  inertiaPct: [87.76, 11.76, 0.48],
  // Prinzipalkoordinaten (Dim1, Dim2), publiziert (Greenacre / R `ca`).
  rowCoords: {
    SM: [-0.0658, 0.1938], JM: [0.2590, 0.2434], SE: [-0.3806, 0.0107], JE: [0.2330, -0.0577], SC: [-0.2011, -0.0789],
  } as Record<string, [number, number]>,
  colCoords: {
    none: [-0.3933, 0.0295], light: [0.0995, -0.1411], medium: [0.1963, -0.0074], heavy: [0.2938, 0.1978],
  } as Record<string, [number, number]>,
};

/**
 * Baut eine „perfekte" Petrie-/Robinson-Inzidenzmatrix: jeder Typ ist in einem
 * zusammenhängenden Laufbereich von Kontexten präsent; die korrekte Seriations­reihen­
 * folge ist 0..NR-1. Theoretisches Referenzergebnis (Hill 1974): CA-Dimension 1 stellt
 * genau diese Reihenfolge (bis auf Umkehrung) wieder her.
 */
export function petrieMatrix(NR: number, NC: number, width = 4): number[][] {
  const M: number[][] = Array.from({ length: NR }, () => new Array<number>(NC).fill(0));
  for (let j = 0; j < NC; j++) {
    const center = Math.round((j / Math.max(1, NC - 1)) * (NR - 1));
    for (let i = Math.max(0, center - width); i <= Math.min(NR - 1, center + width); i++) M[i][j] = 1;
  }
  return M;
}
