/**
 * XLSX-Import: dünne Brücke von SheetJS auf denselben Kern (`importGrid`).
 *
 * Bewusst getrennt gehalten, damit der Kern (parseDelimited/importTable) ohne
 * die xlsx-Abhängigkeit test- und nutzbar bleibt. In der App liest man die Datei
 * als ArrayBuffer (File API) und übergibt sie hier.
 *
 *   import { importXLSX } from "./core/io/importXLSX.js";
 *   const res = await importXLSX(await file.arrayBuffer(), { name: file.name });
 */

import { importGrid, type ImportOptions, type ImportResult } from "./importTable.js";

export interface XLSXImportOptions extends ImportOptions {
  /** Blattname oder 0-basierter Index; Default: erstes Blatt. */
  sheet?: string | number;
}

/** Konvertiert die Arbeitsmappe in ein Zellraster und importiert es.
 *  v1.0: `xlsx` wird erst hier dynamisch geladen (Code-Splitting) — die Bibliothek
 *  ist der größte Einzelposten im Bundle und wird nur beim XLSX-Import gebraucht. */
export async function importXLSX(data: ArrayBuffer | Uint8Array, opts: XLSXImportOptions = {}): Promise<ImportResult> {
  const XLSX = await import("xlsx");
  const wb = XLSX.read(data, { type: "array" });
  const sheetName = typeof opts.sheet === "string"
    ? opts.sheet
    : wb.SheetNames[typeof opts.sheet === "number" ? opts.sheet : 0];
  const sheet = wb.Sheets[sheetName];
  if (!sheet) throw new Error(`Arbeitsblatt nicht gefunden: ${String(opts.sheet ?? 0)}`);
  // header:1 → Array-of-Arrays; alle Zellen als Strings, leere als ""
  const grid = XLSX.utils.sheet_to_json<string[]>(sheet, { header: 1, raw: false, defval: "" });
  return importGrid(grid as unknown as string[][], opts);
}

/** Bequemer Reexport, damit App-Code nur ein Modul braucht. */
export { importCSV } from "./importTable.js";
