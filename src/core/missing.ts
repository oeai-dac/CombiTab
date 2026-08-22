/**
 * „Nicht erfasst" (fehlender Wert) vs. „strukturelle Absenz" (§9.6).
 *
 * Additiv und abwärtskompatibel: `p.matrix[i][j] === 0` bleibt die echte Absenz
 * (der Typ war im Kontext nachweislich nicht vorhanden). Ein separates Set markiert
 * Zellen als „nicht erfasst/unbekannt". Beides darf nicht verwechselt werden — für
 * Präsenzstatistik und Export ist der Unterschied bedeutsam.
 */
import type { ProjectV2 } from "./model.js";
import { annotationKey } from "./model.js";

/** Erkennt in Importen übliche Marker für „nicht erfasst": `?`, `NA`, `N/A`. */
export function isMissingToken(raw: string | undefined): boolean {
  const s = (raw ?? "").trim();
  if (s === "?") return true;
  const l = s.toLowerCase();
  return l === "na" || l === "n/a";
}

export function isMissing(p: ProjectV2, i: number, j: number): boolean {
  return !!p.missingCells && p.missingCells[annotationKey(i, j)] === true;
}

/** Markiert/entfernt eine Menge kanonischer Zellen als „nicht erfasst". */
export function setMissing(p: ProjectV2, cells: Array<[number, number]>, missing: boolean): void {
  if (missing && !p.missingCells) p.missingCells = {};
  const m = p.missingCells;
  if (!m) return;
  for (const [i, j] of cells) {
    const k = annotationKey(i, j);
    if (missing) m[k] = true; else delete m[k];
  }
  if (Object.keys(m).length === 0) delete p.missingCells;
}

export function missingCount(p: ProjectV2): number {
  return p.missingCells ? Object.keys(p.missingCells).length : 0;
}

export function clearAllMissing(p: ProjectV2): void { delete p.missingCells; }

/** Anzahl als „nicht erfasst" markierter Zellen — schneller Vorabtest ohne Maske. */
export function hasMissing(p: ProjectV2): boolean {
  return !!p.missingCells && Object.keys(p.missingCells).length > 0;
}

/**
 * Dichte Fehlwert-Maske für die Rechenkerne (§9.6-Vertiefung): `NR*NC`-Bytes,
 * zeilen-major, `1` = „nicht erfasst". Liefert `null`, wenn keine Zelle markiert
 * ist — dann nehmen alle Kerne ihren unveränderten, byte-identischen Schnellpfad.
 * Die Maske liegt im kanonischen Indexraum (wie `p.matrix`), passt also direkt zu
 * den Ordnungs-Arrays (Permutationen kanonischer Indizes).
 */
export function buildMissingMask(p: ProjectV2): Uint8Array | null {
  const cells = p.missingCells;
  if (!cells) return null;
  const keys = Object.keys(cells);
  if (keys.length === 0) return null;
  const NR = p.contexts.length, NC = p.types.length;
  const mask = new Uint8Array(NR * NC);
  for (const k of keys) {
    const sep = k.indexOf(":");
    if (sep < 0) continue;
    const i = +k.slice(0, sep), j = +k.slice(sep + 1);
    if (Number.isInteger(i) && Number.isInteger(j) && i >= 0 && i < NR && j >= 0 && j < NC) mask[i * NC + j] = 1;
  }
  return mask;
}

/** Wert einer Zelle oder `null`, wenn sie als „nicht erfasst" markiert ist. */
export function effectiveValue(p: ProjectV2, i: number, j: number): number | null {
  return isMissing(p, i, j) ? null : p.matrix[i][j];
}

export interface Presence { present: number; absent: number; missing: number; }

/** Präsenzstatistik einer Spalte (eines Typs) über alle Kontexte. */
export function typePresence(p: ProjectV2, j: number): Presence {
  let present = 0, absent = 0, missing = 0;
  for (let i = 0; i < p.contexts.length; i++) {
    if (isMissing(p, i, j)) missing++;
    else if (p.matrix[i][j] > 0) present++;
    else absent++;
  }
  return { present, absent, missing };
}

/** Präsenzstatistik einer Zeile (eines Kontexts) über alle Typen. */
export function contextPresence(p: ProjectV2, i: number): Presence {
  let present = 0, absent = 0, missing = 0;
  for (let j = 0; j < p.types.length; j++) {
    if (isMissing(p, i, j)) missing++;
    else if (p.matrix[i][j] > 0) present++;
    else absent++;
  }
  return { present, absent, missing };
}
