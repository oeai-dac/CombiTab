import { buildMatrixScene, sceneToSVG, sceneToPDF } from "./exportImage.js";
import { toCSV, toXLSX, sortedAoa } from "./exportTable.js";
import { toProjectJSONv2, toProjectJSONv1 } from "./exportProject.js";
import { migrateV1, type ProjectV1 } from "../core/io/migrateV1.js";
import { importCSV } from "../core/io/importTable.js";
import * as XLSX from "xlsx";
import { readFileSync, existsSync } from "node:fs";
import { SAMPLE_CSV } from "../data/sample.js";

// Reproduzierbar ohne externe Datei: liegt die Upload-CSV nicht vor, greift das
// eingebettete (identische) Demo-Sample. In der Arbeitsumgebung mit Datei unverändert.
const SAMPLE_PATH = "/mnt/user-data/uploads/large_cemetery_sample.csv";
const sampleCSV = existsSync(SAMPLE_PATH) ? readFileSync(SAMPLE_PATH, "utf8") : SAMPLE_CSV;

let pass=0,fail=0; const F:string[]=[];
function c(n:string,ok:boolean,d=""){ok?pass++:(fail++,F.push(n));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n+(d?" — "+d:""));}

console.log("\n\x1b[1mExport\x1b[0m\n");

// Projekt aus echter CSV, dann Reihenfolge leicht ändern (Kontext an Pos 0 ans Ende)
const csv = sampleCSV;
const { project } = importCSV(csv, { name: "Cemetery" });
// Anzeige-Ordnung modifizieren, um zu prüfen, dass Export der ORDNUNG folgt
project.order.rows = [...project.order.rows.slice(1), project.order.rows[0]];
project.order.cols = [...project.order.cols.slice(2), ...project.order.cols.slice(0,2)];
// eine Fixierung setzen (für Label-Färbung im Bild)
project.rowMetadata[project.contexts[3]].isFixed = true;

// ── CSV ──
{
  const out = toCSV(project);
  const lines = out.split("\r\n");
  c("CSV: 21 Zeilen (Kopf + 20)", lines.length === 21, `${lines.length}`);
  c("CSV: Kopf beginnt mit Context", lines[0].startsWith("Context,"));
  c("CSV: erste Datenzeile = erster Kontext der ORDNUNG", lines[1].startsWith(project.order.rows[0] + ","));
  c("CSV: Spaltenzahl = 101", lines[0].split(",").length === 101);
}
// ── XLSX round-trip ──
{
  const bytes = await toXLSX(project);
  c("XLSX: Bytes erzeugt (PK-Signatur)", bytes[0] === 0x50 && bytes[1] === 0x4b, `len=${bytes.length}`);
  const wb = XLSX.read(bytes, { type: "array" });
  const aoa = XLSX.utils.sheet_to_json<any[]>(wb.Sheets[wb.SheetNames[0]], { header: 1 });
  c("XLSX: Blatt 'Seriation'", wb.SheetNames[0] === "Seriation");
  c("XLSX: 21 Zeilen zurückgelesen", aoa.length === 21, `${aoa.length}`);
  c("XLSX: Wert an [1][1] stimmt mit sortedAoa überein", String(aoa[1][1]) === String(sortedAoa(project)[1][1]));
}
// ── SVG ──
{
  const svg = buildMatrixScene(project); const s = sceneToSVG(svg);
  c("SVG: wohlgeformter Wurzelknoten", s.startsWith("<?xml") && s.includes("<svg") && s.trimEnd().endsWith("</svg>"));
  c("SVG: enthält Zell-Rechtecke", (s.match(/<rect/g)||[]).length > 100);
  c("SVG: enthält Kontext-Label", s.includes(project.contexts[0]));
  c("SVG: rotierte Spaltenlabels", s.includes("rotate(-90"));
}
// ── PDF ──
{
  const scene = buildMatrixScene(project); const bytes = sceneToPDF(scene);
  const head = new TextDecoder().decode(bytes.slice(0,8));
  const tail = new TextDecoder().decode(bytes.slice(-6));
  c("PDF: %PDF-Header", head.startsWith("%PDF-1."));
  c("PDF: EOF-Marke", tail.includes("%%EOF"));
  c("PDF: enthält xref + Catalog", new TextDecoder().decode(bytes).includes("/Catalog") && new TextDecoder().decode(bytes).includes("xref"));
}
// ── Projekt v2 + v1 ──
{
  const j2 = toProjectJSONv2(project); const p2 = JSON.parse(j2);
  c("Projekt v2: schemaVersion 2", p2.schemaVersion === 2);
  c("Projekt v2: Ordnung enthalten", Array.isArray(p2.order.rows) && p2.order.rows[0] === project.order.rows[0]);
  const j1 = toProjectJSONv1(project); const p1 = JSON.parse(j1) as ProjectV1;
  c("Projekt v1: hat matrix_index + row_order", Array.isArray(p1.matrix_index) && Array.isArray(p1.row_order));
  // v1 → v2 re-migrierbar (Konsistenz)
  const back = migrateV1(p1);
  c("Projekt v1 re-migrierbar (konsistent)", back.contexts.length === project.contexts.length && back.types.length === project.types.length);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ Export korrekt.\x1b[0m");
