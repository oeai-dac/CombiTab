/**
 * Daten-Export der Matrix in aktueller Anzeige-Reihenfolge: CSV und XLSX.
 * Zeilen = Kontexte (Ordnung), Spalten = Typen (Ordnung), erste Spalte = Kontext-ID.
 */
import type { ProjectV2 } from "../core/model.js";
import { isMissing } from "../core/missing.js";

export function sortedAoa(p: ProjectV2): (string | number)[][] {
  const rIdx = new Map(p.contexts.map((c, i) => [c, i] as const));
  const cIdx = new Map(p.types.map((t, j) => [t, j] as const));
  const rowSeq = p.order.rows.map((c) => rIdx.get(c) ?? 0);
  const colSeq = p.order.cols.map((c) => cIdx.get(c) ?? 0);
  const header: (string | number)[] = ["Context", ...colSeq.map((j) => p.types[j])];
  const rows = rowSeq.map((i) => [p.contexts[i], ...colSeq.map((j) => (isMissing(p, i, j) ? "?" : p.matrix[i][j]))]);
  return [header, ...rows];
}

export function toCSV(p: ProjectV2): string {
  return sortedAoa(p).map((row) => row.map(csvCell).join(",")).join("\r\n");
}

/** UTF-8-Bytereihenfolge-Marke — Excel unter Windows liest UTF-8-CSV nur damit korrekt. */
export const UTF8_BOM = "\uFEFF";
/** CSV mit vorangestelltem BOM für den Download (Umlaute in Excel/Windows). */
export function toCSVForDownload(p: ProjectV2): string { return UTF8_BOM + toCSV(p); }
function csvCell(v: string | number): string {
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** v1.0: `xlsx` wird erst beim Export dynamisch geladen (Code-Splitting). */
export async function toXLSX(p: ProjectV2): Promise<Uint8Array> {
  const XLSX = await import("xlsx");
  const ws = XLSX.utils.aoa_to_sheet(sortedAoa(p));
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Seriation");
  return new Uint8Array(XLSX.write(wb, { type: "array", bookType: "xlsx" }) as ArrayBuffer);
}
