/**
 * Selbsttest der Desktop-Fassung.
 *
 * Prüft im laufenden Fenster genau die Eigenschaften, die beim Wechsel vom
 * Browser ins Electron-Gehäuse brechen können — allen voran die Modul-Worker,
 * die unter file:// nicht laden würden und ohne die Korrespondenzanalyse,
 * Bootstrap und Score-Berechnung ausfielen.
 *
 * Aufruf:  npm run smoke
 * Beendet sich mit Code 0 (alles bestanden) oder 1 (mindestens ein Fehlschlag).
 */
import { readdir } from "node:fs/promises";
import { join } from "node:path";

const GREEN = "\x1b[32m", RED = "\x1b[31m", OFF = "\x1b[0m";

/** Findet den gehashten Dateinamen des Score-Workers im gebauten Bundle. */
async function findScoreWorker(webRoot) {
  const files = await readdir(join(webRoot, "assets"));
  const hit = files.find((f) => /^score\.worker-.*\.js$/.test(f));
  if (!hit) throw new Error("score.worker-*.js nicht in dist/assets/ gefunden");
  return `assets/${hit}`;
}

/** Im Renderer ausgeführt. Liefert eine Liste { name, ok, detail }. */
async function rendererProbe(scoreWorkerPath) {
  const checks = [];
  {
    const add = (name, ok, detail = "") =>
      checks.push({ name, ok: Boolean(ok), detail: String(detail) });

    add("Oberfläche gemountet", document.querySelector("#root")?.childElementCount > 0,
        `${document.querySelector("#root")?.childElementCount ?? 0} Kindknoten`);
    add("Desktop-Kennung im Fenster", Boolean(window.combitabDesktop),
        window.combitabDesktop ? `Electron ${window.combitabDesktop.electron}` : "fehlt");
    add("Origin ist app://", location.protocol === "app:", location.origin);
    add("Sicherer Kontext", window.isSecureContext === true, String(window.isSecureContext));
    add("CompressionStream (Teilen-Link)", typeof CompressionStream === "function");
    add("IndexedDB vorhanden (Autosave)", typeof indexedDB === "object" && indexedDB !== null);

    // WebGL2 — der Matrix-Renderer; ohne ihn greift der Canvas-2D-Ersatzpfad.
    let gl = null;
    try { gl = document.createElement("canvas").getContext("webgl2"); } catch { /* egal */ }
    add("WebGL2-Kontext", Boolean(gl), gl ? gl.getParameter(gl.VERSION) : "nicht verfügbar");

    // Eingebettete Schriften: müssen sich ohne Netz laden lassen. `check()`
    // allein taugt nicht — es meldet false, solange eine Schrift noch nicht
    // angefordert wurde. `load()` erzwingt den Abruf und beweist damit, dass
    // die gebündelten woff2-Dateien unter app:// tatsächlich erreichbar sind.
    const families = ["Outfit", "Cormorant Garamond", "JetBrains Mono"];
    const loaded = [];
    for (const f of families) {
      try {
        const faces = await document.fonts.load(`16px "${f}"`, "Aa");
        if (faces.length > 0) loaded.push(f);
      } catch { /* zählt als nicht geladen */ }
    }
    add("Lokale Schriften geladen", loaded.length === families.length,
        `${loaded.length}/${families.length}: ${loaded.join(", ") || "keine"}`);

    // Kein Verweis mehr auf einen Fremdserver irgendwo im Dokument.
    const external = [...document.querySelectorAll("link[href], script[src]")]
      .map((n) => n.getAttribute("href") || n.getAttribute("src"))
      .filter((u) => u && /^https?:/i.test(u));
    add("Keine externen Ressourcen", external.length === 0, external.join(", ") || "keine");

  }

  // Der eigentliche Prüfstein: ein Modul-Worker muss laden UND antworten.
  await new Promise((resolve) => {
    const add = (name, ok, detail = "") =>
      checks.push({ name, ok: Boolean(ok), detail: String(detail) });
    const finish = () => resolve();
    let settled = false;
    try {
      const w = new Worker(new URL(scoreWorkerPath, document.baseURI), { type: "module" });
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        add("Modul-Worker antwortet", false, "Zeitüberschreitung nach 8 s");
        w.terminate(); finish();
      }, 8000);
      w.onerror = (e) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        add("Modul-Worker antwortet", false, `Ladefehler: ${e.message || "unbekannt"}`);
        w.terminate(); finish();
      };
      w.onmessage = (ev) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const d = ev.data;
        add("Modul-Worker antwortet", d && d.id === 1 && d.type === "done",
            d ? `type=${d.type}` : "leere Antwort");
        w.terminate(); finish();
      };
      // Kleine, echte Anfrage nach dem ScoreRequest-Protokoll.
      w.postMessage({
        id: 1, epoch: 1,
        matrix: [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        rows: [0, 1, 2], cols: [0, 1, 2],
      });
    } catch (e) {
      settled = true;
      add("Modul-Worker antwortet", false, `Konstruktor warf: ${e.message}`);
      finish();
    }
  });

  return checks;
}

/** Führt den Selbsttest im gegebenen Fenster aus und beendet den Prozess. */
export async function runSmoke(app, win, webRoot) {
  let code = 1;
  try {
    const scoreWorkerPath = await findScoreWorker(webRoot);
    const checks = await win.webContents.executeJavaScript(
      `(${rendererProbe.toString()})(${JSON.stringify(scoreWorkerPath)})`,
      true,
    );

    let failed = 0;
    console.log("\nSelbsttest der Desktop-Fassung\n");
    for (const c of checks) {
      if (!c.ok) failed++;
      const mark = c.ok ? `${GREEN}OK  ${OFF}` : `${RED}FEHL${OFF}`;
      console.log(`  ${mark} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`);
    }
    console.log(
      failed === 0
        ? `\n${GREEN}Alle ${checks.length} Prüfungen bestanden.${OFF}\n`
        : `\n${RED}${failed} von ${checks.length} Prüfungen fehlgeschlagen.${OFF}\n`,
    );
    code = failed === 0 ? 0 : 1;
  } catch (e) {
    console.error("Selbsttest abgebrochen:", e?.message ?? e);
  }
  app.exit(code);
}
