/**
 * Score-Worker (§8.5). Dünne Hülle um {@link handleScoreRequest}: hält den
 * Matrix-Cache über die Zeit und reicht jede Anfrage an den reinen Kern durch.
 * Getrennt vom Rechen-Worker (`compute.worker.ts`), damit ein Score-Lauf einen
 * laufenden Bootstrap/CA/Seriation nicht abbricht (und umgekehrt).
 *
 * Protokoll: eine {@link ScoreRequest} rein, eine {@link ScoreResponse} raus,
 * beide mit derselben `id`. Antworten werden sequentiell verarbeitet.
 */
import { createScoreCache, handleScoreRequest, type ScoreRequest, type ScoreResponse } from "./scoreCore.js";

const cache = createScoreCache();
const post = (msg: ScoreResponse) => (self as unknown as Worker).postMessage(msg);

self.onmessage = (ev: MessageEvent<ScoreRequest>) => {
  post(handleScoreRequest(cache, ev.data));
};
