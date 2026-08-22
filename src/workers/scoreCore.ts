/**
 * Reiner Kern des Score-Workers (§8.5).
 *
 * Die eigentliche Score-Neuberechnung (`qualityFromMatrix`, O(NR²·NC)) blockierte
 * bislang synchron den Haupt-Thread nach jedem Drop/Undo/Redo/Seriation — bei
 * 1.000×1.000 mehrere Sekunden Freeze (Profiling). Sie wird jetzt in einen
 * eigenen Web-Worker verlagert. Dieses Modul enthält die framework- **und**
 * worker-freie Logik, damit sie headless testbar ist: der Worker (`score.worker.ts`)
 * ist nur eine dünne Hülle, die den Cache-Zustand hält und Nachrichten durchreicht.
 *
 * Matrix-Cache: die Belegungsmatrix ändert sich (in v2) nicht durch Umsortieren,
 * Seriation oder Annotationen — nur beim Wechsel von Projekt/gefilterter Sicht.
 * Deshalb trägt jede Anfrage eine `epoch`; die volle Matrix wird nur mitgeschickt,
 * wenn sich die Epoch ändert. Folgeanfragen (nach jedem Drop) übertragen nur noch
 * die beiden Ordnungs-Arrays — kein wiederholtes Klonen der ganzen Matrix.
 */
import { qualityFromMatrix, DEFAULT_WEIGHTS, type Quality, type QualityWeights } from "../seriation/metrics.js";

export interface ScoreRequest {
  id: number;
  epoch: number;
  /** Nur gesetzt, wenn sich die Epoch gegenüber dem Client-Stand geändert hat. */
  matrix?: number[][];
  /** §9.6: Fehlwert-Maske (NR*NC, 1 = „nicht erfasst"), zusammen mit der Matrix
   *  übertragen. `null`/fehlend = keine Fehlwerte → Schnellpfad. */
  missing?: Uint8Array | null;
  rows: number[];
  cols: number[];
  weights?: QualityWeights;
}

export type ScoreResponse =
  | { id: number; type: "done"; result: Quality }
  /** Cache-Miss: der Worker hat für diese Epoch keine Matrix (z. B. nach Neustart
   *  des Workers). Der Client schickt die Matrix erneut und wiederholt die Anfrage. */
  | { id: number; type: "stale" }
  | { id: number; type: "error"; message: string };

/** Veränderlicher Cache-Zustand des Workers (eine Matrix + Maske, per Epoch geschlüsselt). */
export interface ScoreCacheState {
  epoch: number;
  matrix: number[][] | null;
  missing: Uint8Array | null;
}

export function createScoreCache(): ScoreCacheState {
  return { epoch: -1, matrix: null, missing: null };
}

/**
 * Verarbeitet eine Anfrage gegen den Cache und liefert die Antwort. Mutiert den
 * übergebenen Cache-Zustand (setzt Matrix/Maske/Epoch, wenn eine Matrix mitkam).
 */
export function handleScoreRequest(state: ScoreCacheState, req: ScoreRequest): ScoreResponse {
  try {
    if (req.matrix) {
      state.matrix = req.matrix;
      state.missing = req.missing ?? null;
      state.epoch = req.epoch;
    }
    if (state.matrix === null || state.epoch !== req.epoch) {
      // Der Client dachte, die Matrix sei gecacht — ist sie aber nicht.
      return { id: req.id, type: "stale" };
    }
    const NR = req.rows.length, NC = req.cols.length;
    const result = qualityFromMatrix(
      state.matrix,
      NR,
      NC,
      { rows: req.rows, cols: req.cols },
      req.weights ?? DEFAULT_WEIGHTS,
      state.missing,
    );
    return { id: req.id, type: "done", result };
  } catch (e) {
    return { id: req.id, type: "error", message: e instanceof Error ? e.message : String(e) };
  }
}
