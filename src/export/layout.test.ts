/**
 * Geometrie-Regressionstests des Bild-Exports.
 *
 * Hintergrund: In der ersten Fassung liefen die gedrehten Spaltenbeschriftungen
 * in die Matrix hinein (falsche Drehrichtung gegenüber dem Textanker), und die
 * Legende ging nicht in die Breite der Zeichenfläche ein, sodass die letzten
 * Materialgruppen rechts abgeschnitten wurden. Beide Fehler waren rein
 * geometrisch — also genau hier prüfbar, ohne Browser und ohne Bildvergleich.
 *
 * Geprüft wird die *Szene*, aus der SVG, PNG und PDF gemeinsam abgeleitet werden:
 *   1. Kein Text ragt über die Zeichenfläche hinaus.
 *   2. Kein Text überlappt eine gefüllte Fläche (Zelle oder Legenden-Farbfeld).
 *   3. Beides gilt auch bei vielen und bei sehr langen Bezeichnungen.
 */
import { buildMatrixScene, sceneToSVG, sceneToPDF, type Scene } from "./exportImage.js";
import { textWidth } from "./textMetrics.js";
import { importCSV } from "../core/io/importTable.js";
import { SAMPLE_CSV } from "../data/sample.js";
import { addMaterialGroup } from "../core/materialGroups.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mExport-Layout (Geometrie)\x1b[0m\n");

interface Box { x0: number; y0: number; x1: number; y1: number; }
const ASCENT = 0.78, DESCENT = 0.24; // Anteile der Schriftgröße, großzügig

/** Achsenparallele Hülle eines Textes — inklusive Drehung und Anker. */
function textBox(t: Scene["texts"][number]): Box {
  const w = textWidth(t.s, t.size), a = t.size * ASCENT, d = t.size * DESCENT;
  if (t.rot === -90) {
    // Grundlinie läuft nach oben; Versalhöhe nach links, Unterlängen nach rechts.
    const y1 = t.anchor === "end" ? t.y + w : t.y;
    return { x0: t.x - a, y0: y1 - w, x1: t.x + d, y1 };
  }
  const x0 = t.anchor === "end" ? t.x - w : t.x;
  return { x0, y0: t.y - a, x1: x0 + w, y1: t.y + d };
}
function overlaps(a: Box, b: Box, tol = 0.5): boolean {
  return a.x0 < b.x1 - tol && b.x0 < a.x1 - tol && a.y0 < b.y1 - tol && b.y0 < a.y1 - tol;
}

function checkScene(label: string, sc: Scene) {
  const boxes = sc.texts.map(textBox);
  const outside = boxes.filter((b) => b.x0 < -0.5 || b.y0 < -0.5 || b.x1 > sc.w + 0.5 || b.y1 > sc.h + 0.5);
  c(`${label}: kein Text ragt über die Zeichenfläche hinaus`, outside.length === 0,
    outside.length ? `${outside.length} von ${boxes.length}, z. B. x1=${outside[0].x1.toFixed(1)} > w=${sc.w}` : "");

  const rectsOut = sc.rects.filter((r) => r.x < -0.5 || r.y < -0.5 || r.x + r.w > sc.w + 0.5 || r.y + r.h > sc.h + 0.5);
  c(`${label}: keine Fläche ragt über die Zeichenfläche hinaus`, rectsOut.length === 0, `${rectsOut.length}`);

  let clashes = 0, sample = "";
  for (let i = 0; i < sc.texts.length; i++) {
    for (const r of sc.rects) {
      if (overlaps(boxes[i], { x0: r.x, y0: r.y, x1: r.x + r.w, y1: r.y + r.h })) {
        clashes++; if (!sample) sample = `„${sc.texts[i].s}"`;
        break;
      }
    }
  }
  c(`${label}: keine Beschriftung überlappt eine Fläche`, clashes === 0, clashes ? `${clashes}, z. B. ${sample}` : "");
}

/* ── Realer Demo-Datensatz ── */
const { project } = importCSV(SAMPLE_CSV, { name: "Layout" });
checkScene("Demo", buildMatrixScene(project));

/* ── Spaltenlabels müssen OBERHALB der Matrix liegen (Kern des alten Fehlers) ── */
{
  const sc = buildMatrixScene(project);
  const cells = sc.rects.filter((r) => Math.abs(r.w - 14.4) < 0.01);
  const matrixTop = Math.min(...cells.map((r) => r.y));
  const rotated = sc.texts.filter((t) => t.rot === -90);
  c("gedrehte Spaltenlabels vorhanden", rotated.length === project.types.length, `${rotated.length}`);
  const below = rotated.map(textBox).filter((b) => b.y1 > matrixTop + 0.5);
  c("gedrehte Spaltenlabels enden über der Matrixoberkante", below.length === 0,
    below.length ? `${below.length} reichen bis y=${Math.max(...below.map((b) => b.y1)).toFixed(1)} (Matrix ab ${matrixTop})` : "");
  const above = rotated.map(textBox).every((b) => b.y0 >= -0.5);
  c("gedrehte Spaltenlabels bleiben im oberen Rand", above);
}

/* ── Viele und lange Materialgruppen: Legende muss umbrechen, nicht abschneiden ── */
{
  const p2 = importCSV(SAMPLE_CSV, { name: "Legende" }).project;
  const before = Object.keys(p2.materialGroups).length;
  for (const n of ["Buntmetall", "Edelmetall (Silber, vergoldet)", "Bernstein und Gagat",
                   "Textil/Leder-Reste", "Geweih und Knochen", "Eisen (stark korrodiert)",
                   "Glasfluss und Email", "Keramik: Feinware", "Keramik: Grobware",
                   "Organische Reste (nicht bestimmbar)"]) addMaterialGroup(p2, n);
  c("Testgruppen angelegt", Object.keys(p2.materialGroups).length === before + 10);
  checkScene("viele Gruppen", buildMatrixScene(p2));
}

/* ── Sehr lange Kontext-/Typnamen werden gekürzt statt zu überlappen ── */
{
  const p3 = importCSV(SAMPLE_CSV, { name: "Lang" }).project;
  const longCtx = "Kontext " + "M".repeat(200);
  const longType = "Typ " + "W".repeat(200);
  p3.contexts[0] = longCtx; p3.order.rows[p3.order.rows.indexOf(p3.order.rows[0])] = longCtx;
  p3.order.rows[0] = longCtx;
  p3.rowMetadata[longCtx] = { name: longCtx, contextType: "", area: "", isFixed: false, notes: "" };
  const oldType = p3.types[0];
  p3.types[0] = longType; p3.order.cols[p3.order.cols.indexOf(oldType)] = longType;
  p3.columnMetadata[longType] = p3.columnMetadata[oldType];
  const sc = buildMatrixScene(p3);
  checkScene("überlange Namen", sc);
  c("überlanger Text wurde gekürzt", sc.texts.some((t) => t.s.endsWith("\u2026")));
  // Der Rand darf durch einen einzelnen Extremnamen nicht unbegrenzt wachsen:
  // Kürzung deckelt Zeilenrand und Spaltenrand (260 bzw. 280 plus Abstände).
  const cells = sc.rects.filter((r) => Math.abs(r.w - 14.4) < 0.01);
  const marginX = Math.min(...cells.map((r) => r.x)), marginY = Math.min(...cells.map((r) => r.y));
  c("Zeilenrand ist gedeckelt", marginX <= 272, `${marginX}`);
  c("Spaltenrand ist gedeckelt", marginY <= 292, `${marginY}`);
}

/* ── Ausgabeformate spiegeln die korrigierte Drehung ── */
{
  const sc = buildMatrixScene(project);
  const svg = sceneToSVG(sc);
  c("SVG: gedrehte Labels sind start-verankert",
    /text-anchor="start"[^>]*transform="rotate\(-90/.test(svg) || /transform="rotate\(-90[^"]*"/.test(svg) && !/text-anchor="end"[^>]*rotate\(-90/.test(svg));
  const pdf = new TextDecoder().decode(sceneToPDF(sc));
  c("PDF: Textmatrix läuft nach oben (0 1 -1 0)", pdf.includes("0 1 -1 0 "));
  c("PDF: keine abwärts laufende Textmatrix mehr", !pdf.includes("0 -1 1 0 "));
  c("PDF: MediaBox = Szenenmaß", pdf.includes(`/MediaBox [0 0 ${sc.w} ${sc.h}]`));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("FAIL: " + F.join(", ")); process.exit(1); }
console.log("\x1b[32m✓ Export-Layout korrekt.\x1b[0m");
