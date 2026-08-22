/**
 * Client-Wrapper um den Score-Worker (§8.5).
 *
 * Anders als {@link compute} (ein Lauf gleichzeitig, cancelt Vorgänger) darf der
 * Score jederzeit laufen, ohne einen laufenden Bootstrap/CA/Seriation-Lauf zu
 * stören — deshalb ein eigener Worker. Anfragen werden per `id` gemultiplext;
 * jede Anfrage bekommt ihr eigenes Promise, das genau bei der zugehörigen Antwort
 * auflöst. Der Aufrufer entscheidet über Veralterung (nur das jeweils jüngste
 * Ergebnis in die Oberfläche schreiben).
 *
 * Matrix-Übertragung: die volle Matrix wird nur mitgeschickt, wenn sich die
 * `epoch` gegenüber dem zuletzt gesendeten Stand ändert. Meldet der Worker einen
 * Cache-Miss (`stale`, z. B. nach Neustart), wird die Matrix einmal erneut
 * gesendet und die Anfrage wiederholt.
 *
 * Fällt auf synchrone Ausführung im Haupt-Thread zurück, falls die Umgebung keine
 * Worker unterstützt (Test-/Embed-Kontexte). Der synchrone Pfad blockiert dann
 * zwar wieder — das ist die bewusste, ehrliche Grenze für Nicht-Worker-Umgebungen,
 * in denen ohnehin keine große Matrix interaktiv bearbeitet wird.
 */
import { qualityFromMatrix, DEFAULT_WEIGHTS, type Quality, type QualityWeights } from "../seriation/metrics.js";
import type { Order } from "../seriation/centroid.js";
import type { ScoreRequest, ScoreResponse } from "./scoreCore.js";

interface PendingScore {
  resolve: (q: Quality) => void;
  reject: (e: Error) => void;
  /** Kontext für den einmaligen Stale-Retry. */
  retry: () => void;
  retried: boolean;
}

class ScoreClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private lastEpochSent = -1;
  private pending = new Map<number, PendingScore>();

  private ensureWorker(): Worker | null {
    if (typeof Worker === "undefined") return null;
    if (!this.worker) {
      this.worker = new Worker(new URL("./score.worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (ev: MessageEvent<ScoreResponse>) => this.handle(ev.data);
      this.worker.onerror = (ev) => this.failAll(new Error(ev.message || "Score-Worker-Fehler"));
      // Neuer Worker hat leeren Cache → Matrix beim nächsten Lauf erneut senden.
      this.lastEpochSent = -1;
    }
    return this.worker;
  }

  private failAll(e: Error) {
    const all = [...this.pending.values()];
    this.pending.clear();
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    this.lastEpochSent = -1;
    for (const p of all) p.reject(e);
  }

  private handle(msg: ScoreResponse) {
    const p = this.pending.get(msg.id);
    if (!p) return;
    if (msg.type === "done") { this.pending.delete(msg.id); p.resolve(msg.result); return; }
    if (msg.type === "stale") {
      // Worker-Cache leer: einmal die Matrix erneut senden und wiederholen.
      this.pending.delete(msg.id);
      this.lastEpochSent = -1;
      if (!p.retried) p.retry();
      else p.reject(new Error("Score-Cache konnte nicht befüllt werden"));
      return;
    }
    // error
    this.pending.delete(msg.id);
    p.reject(new Error(msg.message));
  }

  /**
   * Fordert den Qualitäts-Score für `order` auf `matrix` an. `epoch` identifiziert
   * den Matrix-Inhalt: solange sie gleich bleibt, wird die Matrix nicht erneut
   * übertragen. `missing` (§9.6, kanonische Fehlwert-Maske) wird zusammen mit der
   * Matrix übertragen. Liefert stets ein auflösendes Promise (Veralterung entscheidet
   * der Aufrufer).
   */
  score(matrix: number[][], epoch: number, order: Order, weights: QualityWeights = DEFAULT_WEIGHTS, missing?: Uint8Array | null): Promise<Quality> {
    const worker = this.ensureWorker();
    if (!worker) {
      return new Promise<Quality>((resolve, reject) => {
        try { resolve(qualityFromMatrix(matrix, order.rows.length, order.cols.length, order, weights, missing)); }
        catch (e) { reject(e instanceof Error ? e : new Error(String(e))); }
      });
    }
    return new Promise<Quality>((resolve, reject) => {
      const send = (forceMatrix: boolean) => {
        const id = this.nextId++;
        const sendMatrix = forceMatrix || epoch !== this.lastEpochSent;
        if (sendMatrix) this.lastEpochSent = epoch;
        const entry: PendingScore = {
          resolve, reject, retried: forceMatrix,
          retry: () => send(true),
        };
        this.pending.set(id, entry);
        const req: ScoreRequest = {
          id, epoch, rows: order.rows, cols: order.cols, weights,
          ...(sendMatrix ? { matrix, missing: missing ?? null } : {}),
        };
        worker.postMessage(req);
      };
      send(false);
    });
  }

  /** Erzwingt erneutes Senden der Matrix beim nächsten Lauf (z. B. bei Matrix-Edit). */
  invalidate() { this.lastEpochSent = -1; }

  /** Beendet den Worker und verwirft alle offenen Anfragen (Aufräumen). */
  dispose() { this.failAll(new DOMException("disposed", "AbortError")); }
}

/** Prozessweit geteilte Instanz. */
export const scoreCompute = new ScoreClient();

/**
 * Global monotone Epoch-Vergabe. Der Score-Client cacht die Matrix per Epoch;
 * da sich alle `MatrixView`-Instanzen denselben Worker teilen, müssen Epochen
 * über Instanzgrenzen hinweg eindeutig sein — sonst hielte der Cache
 * fälschlich die Matrix einer anderen Sicht für aktuell. Jede neue Matrix-
 * Identität (Projekt-/Filterwechsel) zieht sich hier eine frische Epoch.
 */
let epochCounter = 0;
export function nextScoreEpoch(): number { return ++epochCounter; }
