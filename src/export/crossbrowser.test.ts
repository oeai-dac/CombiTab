/**
 * Cross-Browser-Export-Tests (§11).
 *
 * Prüft die Eigenschaften, die Exportdateien in fremden Browsern und Programmen
 * zuverlässig öffnen lassen — unabhängig von der reinen Inhaltskorrektheit
 * (die `export.test.ts` abdeckt):
 *   - CSV mit UTF-8-BOM + CRLF (Excel/Windows-Kompatibilität, Umlaute)
 *   - SVG mit XML-Prolog + Namespace (eigenständiges Rendern)
 *   - PDF mit MediaBox (Seitenmaße, portables Öffnen)
 *   - portable Dateinamen aus beliebigen Projektnamen
 */
import { toCSV, toCSVForDownload, UTF8_BOM } from "./exportTable.js";
import { buildMatrixScene, sceneToSVG, sceneToPDF } from "./exportImage.js";
import { safeFilename } from "./download.js";
import { makeSyntheticProject } from "../bench/synth.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mCross-Browser-Export (§11)\x1b[0m\n");

const p = makeSyntheticProject(12, 8, { seed: 1 });

// ── CSV: BOM + CRLF ──
{
  const plain = toCSV(p), dl = toCSVForDownload(p);
  c("CSV (roh) hat KEIN BOM", plain.charCodeAt(0) !== 0xfeff);
  c("CSV (Download) beginnt mit UTF-8-BOM", dl.charCodeAt(0) === 0xfeff && dl.startsWith(UTF8_BOM));
  c("CSV nutzt CRLF-Zeilenenden (Excel)", plain.includes("\r\n") && !/[^\r]\n/.test(plain));
  c("CSV-Download = BOM + roher Inhalt", dl === UTF8_BOM + plain);
}

// ── CSV: Sonderzeichen korrekt maskiert (portables Parsen) ──
{
  const q = makeSyntheticProject(3, 3, { seed: 2 });
  q.contexts[0] = 'Grab "A", Nord'; q.order.rows = q.contexts.slice();
  const line = toCSV(q).split("\r\n")[1];
  c("CSV maskiert Komma/Anführungszeichen in Feldern", line.startsWith('"Grab ""A"", Nord"'), line.slice(0, 24));
}

// ── SVG: Prolog + Namespace + wohlgeformt ──
{
  const svg = sceneToSVG(buildMatrixScene(p));
  c("SVG hat XML-Prolog", svg.startsWith("<?xml"));
  c("SVG deklariert den SVG-Namespace", svg.includes('xmlns="http://www.w3.org/2000/svg"'));
  c("SVG ist geschlossen", svg.trimEnd().endsWith("</svg>"));
  c("SVG hat viewBox (skaliert sauber)", /viewBox="0 0 [\d.]+ [\d.]+"/.test(svg));
}

// ── PDF: Header/MediaBox/EOF ──
{
  const bytes = sceneToPDF(buildMatrixScene(p));
  const txt = new TextDecoder().decode(bytes);
  c("PDF-Header %PDF-1.x", txt.startsWith("%PDF-1."));
  c("PDF hat /MediaBox (Seitenmaße)", txt.includes("/MediaBox"));
  c("PDF endet mit %%EOF", txt.trimEnd().endsWith("%%EOF"));
}

// ── Dateinamen: portabel aus beliebigen Projektnamen ──
{
  c("Umlaute werden transliteriert", safeFilename("Gräberfeld Halbturn") === "Graberfeld_Halbturn", safeFilename("Gräberfeld Halbturn"));
  c("Pfadtrenner/Sonderzeichen entfernt", safeFilename("a/b\\c:d*e?") === "a_b_c_d_e");
  c("leerer/illegaler Name → Fallback", safeFilename("///") === "combitab" && safeFilename("") === "combitab");
  c("führende Punkte entfernt (keine versteckten Dateien)", !safeFilename("...heikel").startsWith("."));
  c("Länge gekappt", safeFilename("x".repeat(400)).length <= 120);
  c("normaler Name unverändert", safeFilename("Cemetery_2024") === "Cemetery_2024");
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
