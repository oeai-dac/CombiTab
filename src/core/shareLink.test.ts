/**
 * Teilbarer URL-State (§9.8) — Round-Trip und Größenverhalten.
 */
import { encodeShare, decodeShare, buildShareUrl, readShareFromHash, LINK_MAX, type ShareState } from "./shareLink.js";
import { makeSyntheticProject } from "../bench/synth.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mTeilbarer URL-State (§9.8)\x1b[0m\n");

(async () => {
  // ── Round-Trip erhält Projekt + Reihenfolge + UI ──
  {
    const p = makeSyntheticProject(20, 15, { seed: 3 });
    p.order.rows = p.contexts.slice().reverse(); // eine vom Kanon abweichende Seriation
    const state: ShareState = { project: p, ui: { tab: "ca" } };
    const frag = await encodeShare(state);
    c("Fragment beginnt mit Format-Flag (g/r)", frag[0] === "g" || frag[0] === "r");
    const back = await decodeShare(frag);
    c("Round-Trip liefert Projekt zurück", !!back && back.project.contexts.length === 20 && back.project.types.length === 15);
    c("Seriationsreihenfolge bleibt erhalten", !!back && JSON.stringify(back.project.order.rows) === JSON.stringify(p.order.rows));
    c("UI-Zustand (Tab) bleibt erhalten", !!back && back.ui.tab === "ca");
    c("Matrixwerte bleiben erhalten", !!back && JSON.stringify(back.project.matrix) === JSON.stringify(p.matrix));
  }

  // ── Kompression bringt spürbaren Gewinn ──
  {
    const p = makeSyntheticProject(60, 60, { seed: 1 });
    const state: ShareState = { project: p, ui: {} };
    const frag = await encodeShare(state);
    const rawLen = JSON.stringify(state).length;
    if (frag[0] === "g") c("gzip verkleinert deutlich (< 60% der JSON-Länge)", frag.length < rawLen * 0.6, `${frag.length} vs ${rawLen}`);
    else c("ohne CompressionStream: roh kodiert (Fallback greift)", frag[0] === "r");
  }

  // ── buildShareUrl + tooLong ──
  {
    const small = makeSyntheticProject(10, 10, { seed: 2 });
    const r1 = await buildShareUrl({ project: small, ui: {} }, "https://combitab.example", "/");
    c("kleine URL ist teilbar (nicht tooLong)", r1.url.startsWith("https://combitab.example/#s=") && !r1.tooLong, `${r1.url.length}`);

    const big = makeSyntheticProject(400, 400, { seed: 4 });
    const r2 = await buildShareUrl({ project: big, ui: {} }, "https://combitab.example", "/");
    c("großes Projekt wird als tooLong erkannt", r2.tooLong && r2.url.length > LINK_MAX, `${r2.url.length}`);
  }

  // ── readShareFromHash ──
  {
    const p = makeSyntheticProject(8, 8, { seed: 5 });
    const frag = await encodeShare({ project: p, ui: { tab: "ford" } });
    const fromHash = await readShareFromHash(`#s=${frag}`);
    c("liest Zustand aus #s=…-Hash", !!fromHash && fromHash.ui.tab === "ford" && fromHash.project.contexts.length === 8);
    c("ohne s-Parameter → null", (await readShareFromHash("#theme=dark")) === null);
    c("defektes Fragment → null (kein Absturz)", (await decodeShare("gNOT_VALID_base64!!")) === null);
  }

  console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
  if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
})();
