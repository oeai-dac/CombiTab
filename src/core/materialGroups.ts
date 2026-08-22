/**
 * Verwaltung der Materialgruppen (anlegen, umbenennen, löschen).
 *
 * Materialgruppen sind der einzige Ort, an dem die Zellfarbe der Matrix definiert
 * wird: `columnMetadata[typ].materialGroup` verweist auf einen Schlüssel in
 * `project.materialGroups`, und `columnMetadata[typ].color` ist eine gespiegelte
 * Kopie der Gruppenfarbe (für den schnellen Zugriff im Renderer). Jede Änderung
 * an den Gruppen muss diese Spiegelung mitziehen — sonst fällt die Matrix auf
 * eine veraltete Farbe zurück.
 *
 * Das Modul ist bewusst framework-frei und mutiert das übergebene Projekt in
 * place (wie die übrigen Metadaten-Operationen), meldet aber über den Rückgabewert,
 * ob und was sich geändert hat.
 *
 * Wichtig: Die **Einfügereihenfolge** der Gruppen bestimmt die Reihenfolge in
 * Legende und Filterleiste. Beim Umbenennen wird sie daher bewusst erhalten,
 * statt den Eintrag ans Ende zu verschieben.
 */
import type { ProjectV2 } from "./model.js";
import { OKABE_ITO, deltaE } from "./palette.js";

export interface GroupResult {
  ok: boolean;
  /** Fehlerschlüssel für die Oberfläche (i18n), falls `ok === false`. */
  error?: "empty" | "duplicate" | "unknown" | "last";
  /** Endgültiger (getrimmter) Name bei Erfolg. */
  name?: string;
}

/** Vorhandene Gruppennamen in Anzeige-(Einfüge-)Reihenfolge. */
export function groupNames(p: ProjectV2): string[] {
  return Object.keys(p.materialGroups);
}

/**
 * Schlägt eine Farbe für eine neue Gruppe vor: aus der Okabe-Ito-Palette jene,
 * die von allen bereits vergebenen Farben perzeptuell am weitesten entfernt ist.
 * So bleibt die Matrix auch bei vielen selbst angelegten Gruppen lesbar.
 */
export function suggestGroupColor(p: ProjectV2): string {
  const used = Object.values(p.materialGroups);
  if (!used.length) return OKABE_ITO[0];
  let best = OKABE_ITO[0], bestDist = -1;
  for (const cand of OKABE_ITO) {
    let nearest = Infinity;
    for (const u of used) { const d = deltaE(cand, u); if (d < nearest) nearest = d; }
    if (nearest > bestDist) { bestDist = nearest; best = cand; }
  }
  return best;
}

/** Zählt die Typen je Materialgruppe. */
export function groupCounts(p: ProjectV2): Record<string, number> {
  const out: Record<string, number> = {};
  for (const g of Object.keys(p.materialGroups)) out[g] = 0;
  for (const ty of p.types) {
    const g = p.columnMetadata[ty]?.materialGroup;
    if (g) out[g] = (out[g] ?? 0) + 1;
  }
  return out;
}

/** Legt eine neue Materialgruppe an. Farbe optional, sonst automatisch vorgeschlagen. */
export function addMaterialGroup(p: ProjectV2, rawName: string, color?: string): GroupResult {
  const name = rawName.trim();
  if (!name) return { ok: false, error: "empty" };
  if (name in p.materialGroups) return { ok: false, error: "duplicate" };
  p.materialGroups[name] = normalizeHex(color ?? suggestGroupColor(p));
  return { ok: true, name };
}

/**
 * Benennt eine Gruppe um und zieht alle Verweise nach: Typzuweisungen und ein
 * eventuell aktiver Materialfilter im Projekt. Die Einfügereihenfolge bleibt erhalten.
 */
export function renameMaterialGroup(p: ProjectV2, oldName: string, rawNew: string): GroupResult {
  const next = rawNew.trim();
  if (!next) return { ok: false, error: "empty" };
  if (!(oldName in p.materialGroups)) return { ok: false, error: "unknown" };
  if (next === oldName) return { ok: true, name: next };
  if (next in p.materialGroups) return { ok: false, error: "duplicate" };

  // Reihenfolge erhalten: Record an Ort und Stelle neu aufbauen.
  const rebuilt: Record<string, string> = {};
  for (const [k, v] of Object.entries(p.materialGroups)) rebuilt[k === oldName ? next : k] = v;
  p.materialGroups = rebuilt;

  for (const ty of p.types) {
    const cm = p.columnMetadata[ty];
    if (cm && cm.materialGroup === oldName) cm.materialGroup = next;
  }
  if (p.filters?.materials?.length) {
    p.filters.materials = p.filters.materials.map((m) => (m === oldName ? next : m));
  }
  return { ok: true, name: next };
}

/**
 * Löscht eine Gruppe. Alle bisher zugewiesenen Typen wandern in `fallback`
 * (standardmäßig die erste verbleibende Gruppe) und übernehmen deren Farbe —
 * es entsteht also **keine** verwaiste Zuweisung. Die letzte verbleibende Gruppe
 * kann nicht gelöscht werden, damit jeder Typ stets eine gültige Farbe hat.
 */
export function removeMaterialGroup(p: ProjectV2, name: string, fallback?: string): GroupResult {
  if (!(name in p.materialGroups)) return { ok: false, error: "unknown" };
  const rest = Object.keys(p.materialGroups).filter((g) => g !== name);
  if (!rest.length) return { ok: false, error: "last" };
  const target = fallback && rest.includes(fallback) ? fallback : rest[0];

  delete p.materialGroups[name];
  const color = p.materialGroups[target];
  for (const ty of p.types) {
    const cm = p.columnMetadata[ty];
    if (cm && cm.materialGroup === name) { cm.materialGroup = target; cm.color = color; }
  }
  if (p.filters?.materials?.length) {
    p.filters.materials = p.filters.materials.filter((m) => m !== name);
  }
  return { ok: true, name: target };
}

/** Setzt die Farbe einer Gruppe und spiegelt sie auf alle zugewiesenen Typen. */
export function setMaterialGroupColor(p: ProjectV2, name: string, color: string): GroupResult {
  if (!(name in p.materialGroups)) return { ok: false, error: "unknown" };
  const hex = normalizeHex(color);
  p.materialGroups[name] = hex;
  for (const ty of p.types) {
    const cm = p.columnMetadata[ty];
    if (cm && cm.materialGroup === name) cm.color = hex;
  }
  return { ok: true, name };
}

/** Weist einem Typ eine Gruppe zu und übernimmt deren Farbe. */
export function assignTypeToGroup(p: ProjectV2, type: string, group: string): GroupResult {
  const cm = p.columnMetadata[type];
  if (!cm) return { ok: false, error: "unknown" };
  if (!(group in p.materialGroups)) return { ok: false, error: "unknown" };
  cm.materialGroup = group;
  cm.color = p.materialGroups[group];
  return { ok: true, name: group };
}

/** Entfernt Materialfilter, deren Gruppe es nicht (mehr) gibt. */
export function pruneMaterialFilter(materials: string[], p: ProjectV2): string[] {
  return materials.filter((m) => m in p.materialGroups);
}

function normalizeHex(c: string): string {
  const s = (c || "").trim();
  if (/^#[0-9a-fA-F]{6}$/.test(s)) return s.toLowerCase();
  if (/^#[0-9a-fA-F]{3}$/.test(s)) return ("#" + s.slice(1).split("").map((x) => x + x).join("")).toLowerCase();
  return "#808080";
}
