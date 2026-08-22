/**
 * Synthetische Testmatrizen (Performance-Profiling).
 *
 * Erzeugt reproduzierbar (seed-basiert) ein gültiges `ProjectV2` mit einer
 * seriationsfreundlichen Bandstruktur plus Rauschen — realistisch für die
 * Profilierung von CA, Seriation, Metriken und Datenaufbau an großen Matrizen
 * (bis 1.000×1.000). Framework-frei, nutzbar in Benchmark und Tests.
 */
import type { ProjectV2, ColumnMetadata, RowMetadata } from "../core/model.js";
import { mulberry32 } from "../core/rng.js";

const MATS = ["Keramik", "Metall", "Glas", "Bein", "Stein"] as const;
const COLORS = ["#CD853F", "#4682B4", "#20B2AA", "#B5892A", "#8B857C"] as const;

export interface SynthOpts {
  /** Bandbreite der Belegung um die Diagonale (Anteil 0..1 der Spalten). */
  band?: number;
  /** Rauschanteil: Wahrscheinlichkeit einer Belegung außerhalb des Bandes. */
  noise?: number;
  /** Häufigkeitsdaten (sonst Präsenz/Absenz). */
  frequency?: boolean;
  seed?: number;
}

/**
 * Baut eine NR×NC-Matrix, in der Typ j vor allem in Kontexten um die Diagonale
 * i ≈ (j/NC)·NR belegt ist — also bereits grob seriiert, mit etwas Rauschen.
 */
export function makeSyntheticProject(NR: number, NC: number, opts: SynthOpts = {}): ProjectV2 {
  const band = opts.band ?? 0.12, noise = opts.noise ?? 0.03, freq = opts.frequency ?? true;
  const rnd = mulberry32(opts.seed ?? 42);

  const contexts = Array.from({ length: NR }, (_, i) => `K${String(i + 1).padStart(4, "0")}`);
  const types = Array.from({ length: NC }, (_, j) => `T${String(j + 1).padStart(4, "0")}`);
  const halfBand = Math.max(1, Math.round(band * NR));

  const matrix: number[][] = new Array(NR);
  for (let i = 0; i < NR; i++) {
    const row = new Array<number>(NC).fill(0);
    for (let j = 0; j < NC; j++) {
      const center = (j / Math.max(1, NC - 1)) * (NR - 1);
      const d = Math.abs(i - center);
      if (d <= halfBand) {
        const w = 1 - d / (halfBand + 1); // Kern stärker belegt
        row[j] = freq ? 1 + Math.floor(rnd() * 6 * w) : (rnd() < 0.5 + 0.5 * w ? 1 : 0);
      } else if (rnd() < noise) {
        row[j] = freq ? 1 + Math.floor(rnd() * 2) : 1;
      }
    }
    matrix[i] = row;
  }

  const materialGroups: Record<string, string> = {};
  MATS.forEach((m, k) => (materialGroups[m] = COLORS[k]));

  const columnMetadata: Record<string, ColumnMetadata> = {};
  types.forEach((t, j) => {
    const g = j % MATS.length;
    columnMetadata[t] = { name: t, materialGroup: MATS[g], color: COLORS[g], isIndexType: false, isFixed: false, notes: "" };
  });
  const rowMetadata: Record<string, RowMetadata> = {};
  contexts.forEach((c) => (rowMetadata[c] = { name: c, contextType: "Grab", area: "", isFixed: false, notes: "" }));

  return {
    schemaVersion: 2,
    name: `synthetic-${NR}x${NC}`,
    dataType: freq ? "frequency" : "presence_absence",
    contexts, types, matrix,
    columnMetadata, rowMetadata,
    cellAnnotations: {},
    materialGroups,
    contextTypes: ["Grab"],
    order: { rows: contexts.slice(), cols: types.slice() },
    view: { vizStyle: "classic", cellSize: 18, showValues: false, showColors: true, showCertainty: false, showFragmentation: false },
    filters: { materials: [], rowRange: null, colRange: null, hideEmptyRows: false, hideEmptyCols: false },
    history: [],
  };
}
