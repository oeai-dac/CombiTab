/**
 * Bild-Export der Kombinationstabelle (Spezifikation §5.2, §6).
 *
 * Es wird EINE auflösungsunabhängige „Szene" (Rechtecke + Beschriftungen) der
 * gesamten Matrix in aktueller Anzeige-Reihenfolge gebaut. Daraus werden SVG
 * (Vektor), PNG (rasterisiert) und PDF (Vektor, ohne Fremdbibliothek) abgeleitet.
 *
 * Layout-Grundsätze — alle drei Formate teilen sich diese eine Szene und sind
 * dadurch deckungsgleich:
 *
 *  - **Ränder folgen echten Textbreiten.** Breite der Zeilenbeschriftung und Höhe
 *    der Spaltenbeschriftung werden aus der Helvetica-Metrik berechnet, nicht aus
 *    einer pauschalen Zeichenbreite. Überschreitet eine Bezeichnung den Höchstrand,
 *    wird sie sichtbar gekürzt, statt in die Matrix hineinzulaufen.
 *  - **Gedrehte Spaltenlabels lesen von unten nach oben** und wachsen von der
 *    oberen Matrixkante aus *nach oben* in den reservierten Rand hinein.
 *  - **Die Legende bricht um** und geht in die Breite der Zeichenfläche ein, damit
 *    beliebig viele Materialgruppen vollständig sichtbar bleiben.
 */
import type { ProjectV2 } from "../core/model.js";
import { textWidth, truncateToWidth, maxTextWidth } from "./textMetrics.js";

interface Rect { x: number; y: number; w: number; h: number; c: [number, number, number]; }
interface Txt { x: number; y: number; s: string; size: number; c: [number, number, number]; anchor: "start" | "end"; rot: 0 | -90; }
export interface Scene { w: number; h: number; rects: Rect[]; texts: Txt[]; }

export interface ImageOptions { cell?: number; }

/* Layout-Konstanten (Punkt bzw. CSS-Pixel — bei den Vektorformaten identisch). */
const ROW_FONT = 9;          // Zeilenbeschriftung (Kontexte)
const COL_FONT = 8;          // Spaltenbeschriftung (Typen)
const LEGEND_FONT = 8;
const LABEL_GAP = 6;         // Abstand Beschriftung ↔ Matrixkante
const EDGE = 4;              // Außenrand links/oben
const RIGHT_MARGIN = 10;     // Platz für Unterlängen des letzten Spaltenlabels
const BOTTOM_MARGIN = 10;
const MAX_ROW_LABEL = 260;   // Höchstbreite der Zeilenbeschriftung
const MAX_COL_LABEL = 280;   // Höchstlänge der gedrehten Spaltenbeschriftung
const LEGEND_GAP = 16;       // Abstand Matrixunterkante ↔ erste Legendenzeile
const LEGEND_LINE = 14;      // Zeilenhöhe der Legende
const LEGEND_SWATCH = 9;
const LEGEND_TEXT_GAP = 4;
const LEGEND_ITEM_GAP = 16;

export function buildMatrixScene(p: ProjectV2, opts: ImageOptions = {}): Scene {
  const cell = opts.cell ?? 15;
  const NR = p.contexts.length, NC = p.types.length, M = p.matrix;
  const rIdx = new Map(p.contexts.map((c, i) => [c, i] as const));
  const cIdx = new Map(p.types.map((t, j) => [t, j] as const));
  const rowSeq = p.order.rows.map((c) => rIdx.get(c) ?? 0);
  const colSeq = p.order.cols.map((c) => cIdx.get(c) ?? 0);

  let vmax = 1; for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) if (M[i][j] > vmax) vmax = M[i][j];

  // Beschriftungen vorab kürzen, damit reservierter Rand und tatsächlich
  // gezeichneter Text garantiert zusammenpassen — die Szene ist in sich konsistent.
  const rowLabels = rowSeq.map((i) => truncateToWidth(p.contexts[i] ?? "", ROW_FONT, MAX_ROW_LABEL));
  const colLabels = colSeq.map((j) => truncateToWidth(p.types[j] ?? "", COL_FONT, MAX_COL_LABEL));

  const marginX = Math.ceil(EDGE + maxTextWidth(rowLabels, ROW_FONT) + LABEL_GAP);
  const marginY = Math.ceil(EDGE + maxTextWidth(colLabels, COL_FONT) + LABEL_GAP);

  const rects: Rect[] = [], texts: Txt[] = [];
  const white: [number, number, number] = [255, 255, 255];

  // Zellen
  for (let di = 0; di < rowSeq.length; di++) {
    const i = rowSeq[di];
    for (let dj = 0; dj < colSeq.length; dj++) {
      const j = colSeq[dj], v = M[i]?.[j] ?? 0;
      if (!v) continue;
      const base = hexToRgb(p.columnMetadata[p.types[j]]?.color ?? "#808080");
      const t = 0.28 + 0.72 * Math.min(1, v / vmax);
      rects.push({ x: marginX + dj * cell, y: marginY + di * cell, w: cell - 0.6, h: cell - 0.6, c: mix(white, base, t) });
    }
  }
  // Zeilenbeschriftung (rechtsbündig, links der Matrix)
  for (let di = 0; di < rowSeq.length; di++) {
    const i = rowSeq[di], fixed = p.rowMetadata[p.contexts[i]]?.isFixed;
    texts.push({
      x: marginX - LABEL_GAP, y: marginY + di * cell + cell / 2 + ROW_FONT * 0.35,
      s: rowLabels[di], size: ROW_FONT, c: fixed ? [168, 29, 38] : [60, 60, 60], anchor: "end", rot: 0,
    });
  }
  // Spaltenbeschriftung: um -90° gedreht, am oberen Matrixrand linksbündig
  // verankert — der Text wächst dadurch nach OBEN in den reservierten Rand.
  for (let dj = 0; dj < colSeq.length; dj++) {
    const j = colSeq[dj], fixed = p.columnMetadata[p.types[j]]?.isFixed;
    texts.push({
      x: marginX + dj * cell + cell / 2 + COL_FONT * 0.3, y: marginY - LABEL_GAP,
      s: colLabels[dj], size: COL_FONT, c: fixed ? [168, 29, 38] : [60, 60, 60], anchor: "start", rot: -90,
    });
  }

  const matrixW = colSeq.length * cell;
  const matrixBottom = marginY + rowSeq.length * cell;

  /* Legende (Materialgruppen) — umbrechend. Umbruchbreite ist mindestens die
     Matrixbreite; ein einzelner überbreiter Eintrag darf die Fläche verbreitern,
     damit nie etwas rechts abgeschnitten wird. */
  const items = Object.entries(p.materialGroups).map(([name, color]) => ({
    name, color,
    w: LEGEND_SWATCH + LEGEND_TEXT_GAP + textWidth(name, LEGEND_FONT),
  }));
  const wrapW = Math.max(matrixW, ...items.map((it) => it.w), 1);
  const lines: (typeof items)[] = [];
  let line: typeof items = [], lineW = 0;
  for (const it of items) {
    if (line.length && lineW + LEGEND_ITEM_GAP + it.w > wrapW) { lines.push(line); line = []; lineW = 0; }
    lineW += (line.length ? LEGEND_ITEM_GAP : 0) + it.w;
    line.push(it);
  }
  if (line.length) lines.push(line);

  let legendW = 0;
  lines.forEach((ln, k) => {
    let lx = marginX;
    const baseline = matrixBottom + LEGEND_GAP + k * LEGEND_LINE;
    for (const it of ln) {
      rects.push({ x: lx, y: baseline - LEGEND_SWATCH + 1, w: LEGEND_SWATCH, h: LEGEND_SWATCH, c: hexToRgb(it.color) });
      texts.push({ x: lx + LEGEND_SWATCH + LEGEND_TEXT_GAP, y: baseline, s: it.name, size: LEGEND_FONT, c: [90, 90, 90], anchor: "start", rot: 0 });
      lx += it.w + LEGEND_ITEM_GAP;
    }
    legendW = Math.max(legendW, lx - LEGEND_ITEM_GAP - marginX);
  });
  const legendH = lines.length
    ? LEGEND_GAP + lines.length * LEGEND_LINE - LEGEND_FONT + BOTTOM_MARGIN
    : BOTTOM_MARGIN;

  const contentW = Math.max(matrixW, legendW);
  const w = Math.ceil(marginX + contentW + RIGHT_MARGIN);
  const h = Math.ceil(matrixBottom + legendH);
  return { w, h, rects, texts };
}

/* ── SVG ── */
export function sceneToSVG(sc: Scene): string {
  const parts: string[] = [`<?xml version="1.0" encoding="UTF-8" standalone="no"?>`, `<svg xmlns="http://www.w3.org/2000/svg" width="${sc.w}" height="${sc.h}" viewBox="0 0 ${sc.w} ${sc.h}" font-family="Outfit, Helvetica, Arial, sans-serif">`];
  parts.push(`<rect width="${sc.w}" height="${sc.h}" fill="#ffffff"/>`);
  for (const r of sc.rects) parts.push(`<rect x="${f(r.x)}" y="${f(r.y)}" width="${f(r.w)}" height="${f(r.h)}" fill="${rgb(r.c)}"/>`);
  for (const t of sc.texts) {
    // rotate(-90) dreht gegen den Uhrzeigersinn: ein „start"-verankerter Text
    // wächst vom Ankerpunkt aus nach oben und liest von unten nach oben.
    const transform = t.rot === -90 ? ` transform="rotate(-90 ${f(t.x)} ${f(t.y)})"` : "";
    parts.push(`<text x="${f(t.x)}" y="${f(t.y)}" font-size="${t.size}" text-anchor="${t.anchor}" fill="${rgb(t.c)}"${transform}>${esc(t.s)}</text>`);
  }
  parts.push("</svg>");
  return parts.join("\n");
}

/* ── PNG (rasterisiert die Szene direkt auf ein Canvas) ── */
export async function sceneToPNG(sc: Scene, scale = 2): Promise<Blob> {
  const cv = document.createElement("canvas");
  cv.width = Math.ceil(sc.w * scale); cv.height = Math.ceil(sc.h * scale);
  const g = cv.getContext("2d")!; g.scale(scale, scale);
  g.fillStyle = "#ffffff"; g.fillRect(0, 0, sc.w, sc.h);
  for (const r of sc.rects) { g.fillStyle = rgb(r.c); g.fillRect(r.x, r.y, r.w, r.h); }
  g.textBaseline = "alphabetic";
  for (const t of sc.texts) {
    g.save(); g.font = `${t.size}px Outfit, Helvetica, Arial, sans-serif`; g.fillStyle = rgb(t.c);
    g.textAlign = t.anchor === "end" ? "right" : "left";
    g.translate(t.x, t.y); if (t.rot === -90) g.rotate(-Math.PI / 2); g.fillText(t.s, 0, 0); g.restore();
  }
  return await new Promise<Blob>((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error("toBlob"))), "image/png"));
}

/* ── PDF (Vektor, Base-14-Helvetica, ohne Fremdbibliothek) ── */
export function sceneToPDF(sc: Scene): Uint8Array {
  const H = sc.h;
  const body: string[] = [];
  let curColor = "";
  const setColor = (c: [number, number, number]) => { const s = `${(c[0] / 255).toFixed(3)} ${(c[1] / 255).toFixed(3)} ${(c[2] / 255).toFixed(3)}`; if (s !== curColor) { body.push(`${s} rg`); curColor = s; } };
  for (const r of sc.rects) { setColor(r.c); body.push(`${f(r.x)} ${f(H - r.y - r.h)} ${f(r.w)} ${f(r.h)} re f`); }
  for (const t of sc.texts) {
    setColor(t.c);
    const width = textWidth(t.s, t.size);
    const x = t.x, y = H - t.y; // PDF-Koordinaten: y zeigt nach oben
    body.push("BT", `/F1 ${t.size} Tf`);
    if (t.rot === -90) {
      // Textmatrix „0 1 -1 0": Vorschubrichtung +y (auf der Seite nach oben),
      // Versalhöhe nach links — deckungsgleich mit SVG-rotate(-90).
      const oy = t.anchor === "end" ? y - width : y;
      body.push(`0 1 -1 0 ${f(x)} ${f(oy)} Tm`);
    } else {
      const ox = t.anchor === "end" ? x - width : x;
      body.push(`1 0 0 1 ${f(ox)} ${f(y)} Tm`);
    }
    body.push(`(${pdfEsc(t.s)}) Tj`, "ET");
  }
  const content = body.join("\n");
  const objs: string[] = [];
  objs[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  objs[2] = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>";
  objs[3] = `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${f(sc.w)} ${f(sc.h)}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>`;
  objs[4] = `<< /Length ${byteLen(content)} >>\nstream\n${content}\nendstream`;
  objs[5] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";

  let pdf = "%PDF-1.4\n"; const offsets: number[] = [];
  for (let i = 1; i <= 5; i++) { offsets[i] = byteLen(pdf); pdf += `${i} 0 obj\n${objs[i]}\nendobj\n`; }
  const xrefPos = byteLen(pdf);
  pdf += `xref\n0 6\n0000000000 65535 f \n`;
  for (let i = 1; i <= 5; i++) pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  pdf += `trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefPos}\n%%EOF`;
  return new TextEncoder().encode(pdf);
}

/* ── Helfer ── */
function f(n: number): string { return (Math.round(n * 100) / 100).toString(); }
function rgb(c: [number, number, number]): string { return `rgb(${c[0]},${c[1]},${c[2]})`; }
function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)];
}
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", ""); const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  return [parseInt(n.slice(0, 2), 16) || 0, parseInt(n.slice(2, 4), 16) || 0, parseInt(n.slice(4, 6), 16) || 0];
}
function esc(s: string): string { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
/**
 * Zeichenketten für den PDF-Inhaltsstrom. Die Schrift ist WinAnsi-kodiert, daher
 * bleiben Umlaute als Oktal-Escape erhalten; typografische Sonderzeichen werden
 * auf ASCII-Äquivalente abgebildet, alles Übrige auf „?".
 */
function pdfEsc(s: string): string {
  return s
    .replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)")
    .replace(/\u2026/g, "...").replace(/[\u2013\u2014]/g, "-")
    .replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d\u201e]/g, '"')
    .replace(/[\u00a0-\u00ff]/g, (ch) => "\\" + ch.charCodeAt(0).toString(8).padStart(3, "0"))
    .replace(/[^\x20-\x7e]/g, "?");
}
function byteLen(s: string): number { return new TextEncoder().encode(s).length; }
