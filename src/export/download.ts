/**
 * Browser-Download-Helfer (Chrome/Firefox/Edge/Safari).
 *
 * - `safeFilename`: entschärft Projektnamen zu portablen Dateinamen (transliteriert
 *   Umlaute via NFKD, ersetzt unzulässige Zeichen, kappt die Länge).
 * - `canDownload`: Feature-Detect des <a download>-Attributs; ältere Browser
 *   erhalten einen `window.open`-Fallback statt eines stillen Fehlschlags.
 * - `downloadText` kann ein UTF-8-BOM voranstellen (für Excel-CSV auf Windows).
 */
export function safeFilename(name: string, fallback = "combitab"): string {
  let s = (name || "")
    .normalize("NFKD").replace(/[\u0300-\u036f]/g, "") // Umlaute → Basisbuchstabe (ä→a)
    .replace(/[^\w.\- ]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^[._]+|[._]+$/g, ""); // führende/anhängende Punkte/Unterstriche
  if (!s) s = fallback;
  return s.slice(0, 120);
}

/** Unterstützt der Browser den programmatischen Download (a[download])? */
export function canDownload(): boolean {
  return typeof document !== "undefined" && "download" in document.createElement("a");
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  if (!canDownload()) { window.open(url, "_blank", "noopener"); setTimeout(() => URL.revokeObjectURL(url), 4000); return; }
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.rel = "noopener"; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

export function downloadText(filename: string, text: string, mime = "text/plain", bom = false): void {
  const body = bom ? "\uFEFF" + text : text;
  downloadBlob(filename, new Blob([body], { type: mime + ";charset=utf-8" }));
}

export function downloadBytes(filename: string, bytes: Uint8Array, mime: string): void {
  downloadBlob(filename, new Blob([bytes as BlobPart], { type: mime }));
}
