/**
 * Rechen-Worker. Führt die schweren, blockierenden Kern-Verfahren außerhalb
 * des UI-Threads aus, damit die Oberfläche reaktiv bleibt (Spezifikation §3.2, §8.3).
 *
 * Der Kern (`analysis/*`, `seriation/*`, `core/rng`) ist framework-frei und läuft
 * unverändert hier im Worker. Ausgelagert sind:
 *   - "bootstrap" : Bootstrap-Stabilität (mit Fortschritt) — laufzeitstärkstes Verfahren
 *   - "ca"        : Korrespondenzanalyse (SVD)
 *   - "seriate"   : Seriation (Schwerpunktmethode)
 *
 * Protokoll (Haupt-Thread → Worker): eine `ComputeRequest` mit eindeutiger `id`.
 * Antworten (Worker → Haupt-Thread) tragen dieselbe `id`:
 *   { id, type: "progress", done, total }   (nur bootstrap)
 *   { id, type: "done", result }
 *   { id, type: "error", message }
 */
import type { ProjectV2 } from "../core/model.js";
import { bootstrapStability, type StabilityResult } from "../analysis/bootstrap.js";
import { computeCA, type CAResult } from "../analysis/ca.js";
import type { Order } from "../seriation/centroid.js";
import { seriate, type SeriationMethod } from "../seriation/strategies.js";
import { mulberry32 } from "../core/rng.js";

export type ComputeRequest =
  | { id: number; kind: "bootstrap"; project: ProjectV2; replicates: number; seed: number }
  | { id: number; kind: "ca"; project: ProjectV2; dims: number }
  | { id: number; kind: "seriate"; project: ProjectV2; method: SeriationMethod; seed: number; caDim: number };

export type ComputeResultMap = {
  bootstrap: StabilityResult;
  ca: CAResult;
  seriate: Order;
};

export type ComputeResponse =
  | { id: number; type: "progress"; done: number; total: number }
  | { id: number; type: "done"; result: StabilityResult | CAResult | Order }
  | { id: number; type: "error"; message: string };

const post = (msg: ComputeResponse) => (self as unknown as Worker).postMessage(msg);

self.onmessage = (ev: MessageEvent<ComputeRequest>) => {
  const req = ev.data;
  try {
    let result: StabilityResult | CAResult | Order;
    switch (req.kind) {
      case "bootstrap":
        result = bootstrapStability(req.project, {
          replicates: req.replicates,
          rng: mulberry32(req.seed),
          onProgress: (done, total) => post({ id: req.id, type: "progress", done, total }),
        });
        break;
      case "ca":
        result = computeCA(req.project, req.dims);
        break;
      case "seriate":
        result = seriate(req.project, req.method, req.seed, req.caDim);
        break;
      default:
        post({ id: (req as { id: number }).id, type: "error", message: "Unbekannte Anfrage" });
        return;
    }
    post({ id: req.id, type: "done", result });
  } catch (e) {
    post({ id: req.id, type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
