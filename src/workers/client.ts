/**
 * Client-Wrapper um den Rechen-Worker.
 *
 * Kapselt die Worker-Instanz, vergibt Request-IDs, liefert ein Promise je Lauf und
 * leitet Fortschritt über einen Callback weiter (nur Bootstrap). Ein Lauf lässt sich
 * abbrechen (`cancel`): der Worker wird terminiert und neu erzeugt, das laufende
 * Promise mit einer `AbortError`-Meldung verworfen. Es gibt genau einen Lauf
 * gleichzeitig — ein neuer Aufruf bricht den vorherigen ab.
 *
 * Fällt automatisch auf synchrone Ausführung im Haupt-Thread zurück, falls die
 * Umgebung keine Worker unterstützt (z. B. bestimmte Test-/Embed-Kontexte).
 */
import type { ProjectV2 } from "../core/model.js";
import { bootstrapStability, type StabilityResult } from "../analysis/bootstrap.js";
import { computeCA, type CAResult } from "../analysis/ca.js";
import type { Order } from "../seriation/centroid.js";
import { seriate, type SeriationMethod } from "../seriation/strategies.js";
import { mulberry32 } from "../core/rng.js";
import type { ComputeRequest, ComputeResponse } from "./compute.worker.js";

type Result = StabilityResult | CAResult | Order;

class ComputeClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending: {
    id: number;
    resolve: (r: Result) => void;
    reject: (e: Error) => void;
    onProgress?: (done: number, total: number) => void;
  } | null = null;

  private ensureWorker(): Worker | null {
    if (typeof Worker === "undefined") return null;
    if (!this.worker) {
      this.worker = new Worker(new URL("./compute.worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (ev: MessageEvent<ComputeResponse>) => this.handle(ev.data);
      this.worker.onerror = (ev) => { const p = this.pending; this.pending = null; p?.reject(new Error(ev.message || "Worker-Fehler")); };
    }
    return this.worker;
  }

  private handle(msg: ComputeResponse) {
    const p = this.pending;
    if (!p || msg.id !== p.id) return; // veraltete/abgebrochene Anfrage ignorieren
    if (msg.type === "progress") { p.onProgress?.(msg.done, msg.total); return; }
    this.pending = null;
    if (msg.type === "done") p.resolve(msg.result);
    else p.reject(new Error(msg.message));
  }

  /** Bricht den laufenden Lauf ab (Worker neu erzeugen) und verwirft sein Promise. */
  cancel(reason = "Abgebrochen") {
    if (this.pending) { const p = this.pending; this.pending = null; p.reject(new DOMException(reason, "AbortError")); }
    if (this.worker) { this.worker.terminate(); this.worker = null; }
  }

  /** Generischer Lauf: postet die Anfrage, oder führt sie synchron aus (Fallback). */
  private run(
    build: (id: number) => ComputeRequest,
    sync: () => Result,
    onProgress?: (done: number, total: number) => void,
    signal?: AbortSignal,
  ): Promise<Result> {
    this.cancel(); // nur eine Anfrage gleichzeitig
    const worker = this.ensureWorker();

    if (!worker) {
      return new Promise<Result>((resolve, reject) => {
        try { resolve(sync()); } catch (e) { reject(e instanceof Error ? e : new Error(String(e))); }
      });
    }

    const id = this.nextId++;
    return new Promise<Result>((resolve, reject) => {
      this.pending = { id, resolve, reject, onProgress };
      if (signal) {
        if (signal.aborted) { this.cancel(); return; }
        signal.addEventListener("abort", () => this.cancel(), { once: true });
      }
      worker.postMessage(build(id));
    });
  }

  bootstrap(project: ProjectV2, args: { replicates: number; seed?: number; onProgress?: (done: number, total: number) => void; signal?: AbortSignal }): Promise<StabilityResult> {
    const seed = args.seed ?? 12345;
    return this.run(
      (id) => ({ id, kind: "bootstrap", project, replicates: args.replicates, seed }),
      () => bootstrapStability(project, { replicates: args.replicates, rng: mulberry32(seed), onProgress: args.onProgress }),
      args.onProgress,
      args.signal,
    ) as Promise<StabilityResult>;
  }

  ca(project: ProjectV2, dims = 5, signal?: AbortSignal): Promise<CAResult> {
    return this.run(
      (id) => ({ id, kind: "ca", project, dims }),
      () => computeCA(project, dims),
      undefined,
      signal,
    ) as Promise<CAResult>;
  }

  seriate(project: ProjectV2, args: { method?: SeriationMethod; seed?: number; caDim?: number; signal?: AbortSignal } = {}): Promise<Order> {
    const method = args.method ?? "centroid", seed = args.seed ?? 12345, caDim = args.caDim ?? 0;
    return this.run(
      (id) => ({ id, kind: "seriate", project, method, seed, caDim }),
      () => seriate(project, method, seed, caDim),
      undefined,
      args.signal,
    ) as Promise<Order>;
  }
}

/** Prozessweit geteilte Instanz. */
export const compute = new ComputeClient();
