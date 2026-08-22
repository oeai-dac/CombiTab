/**
 * Tests der Materialgruppen-Verwaltung.
 *
 * Schwerpunkt: keine verwaisten Zuweisungen. Jeder Typ muss nach jeder Operation
 * auf eine existierende Gruppe zeigen und deren Farbe tragen — sonst fällt die
 * Matrix auf eine veraltete oder gar keine Farbe zurück.
 */
import { importCSV } from "./io/importTable.js";
import { SAMPLE_CSV } from "../data/sample.js";
import {
  addMaterialGroup, renameMaterialGroup, removeMaterialGroup, setMaterialGroupColor,
  assignTypeToGroup, suggestGroupColor, groupCounts, groupNames, pruneMaterialFilter,
} from "./materialGroups.js";
import { deltaE } from "./palette.js";
import { visibleIndices, emptyFilters } from "./filter.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mMaterialgruppen\x1b[0m\n");

const fresh = () => importCSV(SAMPLE_CSV, { name: "MG" }).project;

/** Invariante: jeder Typ zeigt auf eine existierende Gruppe und trägt deren Farbe. */
function consistent(p: ReturnType<typeof fresh>): boolean {
  return p.types.every((ty) => {
    const cm = p.columnMetadata[ty];
    const g = cm?.materialGroup;
    return !!g && g in p.materialGroups && cm.color === p.materialGroups[g];
  });
}

/* ── Anlegen ── */
{
  const p = fresh();
  const n0 = groupNames(p).length;
  const r = addMaterialGroup(p, "  Buntmetall  ");
  c("Anlegen liefert getrimmten Namen", r.ok && r.name === "Buntmetall");
  c("Gruppe ist enthalten", "Buntmetall" in p.materialGroups);
  c("Gruppenzahl +1", groupNames(p).length === n0 + 1);
  c("Farbe ist gültiges Hex", /^#[0-9a-f]{6}$/.test(p.materialGroups["Buntmetall"]));
  c("Duplikat wird abgelehnt", addMaterialGroup(p, "Buntmetall").error === "duplicate");
  c("Leerer Name wird abgelehnt", addMaterialGroup(p, "   ").error === "empty");
  c("neue Gruppe hat noch keine Typen", (groupCounts(p)["Buntmetall"] ?? -1) === 0);
  c("explizite Farbe wird übernommen", addMaterialGroup(p, "Gagat", "#123456").ok && p.materialGroups["Gagat"] === "#123456");
  c("Kurz-Hex wird expandiert", addMaterialGroup(p, "Kurz", "#abc").ok && p.materialGroups["Kurz"] === "#aabbcc");
}

/* ── Farbvorschlag ── */
{
  const p = fresh();
  const suggested = suggestGroupColor(p);
  const nearest = Math.min(...Object.values(p.materialGroups).map((u) => deltaE(suggested, u)));
  c("Farbvorschlag ist von bestehenden Farben abgesetzt", nearest > 12, `ΔE=${nearest.toFixed(1)}`);
}

/* ── Zuweisen ── */
{
  const p = fresh();
  addMaterialGroup(p, "Buntmetall", "#0072b2");
  const ty = p.types[0];
  c("Zuweisung gelingt", assignTypeToGroup(p, ty, "Buntmetall").ok);
  c("Gruppe gesetzt", p.columnMetadata[ty].materialGroup === "Buntmetall");
  c("Farbe gespiegelt", p.columnMetadata[ty].color === "#0072b2");
  c("Zuweisung an unbekannte Gruppe scheitert", assignTypeToGroup(p, ty, "Gibtsnicht").error === "unknown");
  c("Invariante hält", consistent(p));
}

/* ── Umbenennen ── */
{
  const p = fresh();
  const first = groupNames(p)[0];
  const order = groupNames(p);
  const n = groupCounts(p)[first];
  c("Umbenennen gelingt", renameMaterialGroup(p, first, "Neuer Name").ok);
  c("alter Name ist weg", !(first in p.materialGroups));
  c("Typzuweisungen wandern mit", groupCounts(p)["Neuer Name"] === n);
  c("Reihenfolge bleibt erhalten", groupNames(p)[0] === "Neuer Name" && groupNames(p).slice(1).join() === order.slice(1).join());
  c("Invariante hält", consistent(p));
  c("Umbenennen auf sich selbst ist ein No-op", renameMaterialGroup(p, "Neuer Name", "Neuer Name").ok);
  const second = groupNames(p)[1];
  c("Umbenennen auf bestehenden Namen scheitert", renameMaterialGroup(p, "Neuer Name", second).error === "duplicate");
  c("unbekannte Gruppe scheitert", renameMaterialGroup(p, "Gibtsnicht", "X").error === "unknown");
}

/* ── Umbenennen zieht den Projektfilter mit ── */
{
  const p = fresh();
  const first = groupNames(p)[0];
  p.filters.materials = [first];
  renameMaterialGroup(p, first, "Umbenannt");
  c("Projektfilter folgt der Umbenennung", p.filters.materials.join() === "Umbenannt");
}

/* ── Löschen ── */
{
  const p = fresh();
  const [g0, g1] = groupNames(p);
  const n0 = groupCounts(p)[g0], n1 = groupCounts(p)[g1];
  const res = removeMaterialGroup(p, g0);
  c("Löschen gelingt", res.ok);
  c("Gruppe ist weg", !(g0 in p.materialGroups));
  c("Typen wandern zur Ersatzgruppe", groupCounts(p)[g1] === n0 + n1, `${groupCounts(p)[g1]} statt ${n0 + n1}`);
  c("keine verwaiste Zuweisung", consistent(p));
  c("unbekannte Gruppe scheitert", removeMaterialGroup(p, "Gibtsnicht").error === "unknown");
}
{
  const p = fresh();
  const names = groupNames(p);
  const target = names[2];
  removeMaterialGroup(p, names[0], target);
  c("ausdrückliche Ersatzgruppe wird genutzt", p.types.filter((ty) => p.columnMetadata[ty].materialGroup === target).length > 0);
  c("Invariante hält", consistent(p));
}
{
  const p = fresh();
  for (const g of groupNames(p).slice(1)) removeMaterialGroup(p, g);
  c("eine Gruppe bleibt übrig", groupNames(p).length === 1);
  c("letzte Gruppe ist nicht löschbar", removeMaterialGroup(p, groupNames(p)[0]).error === "last");
  c("Invariante hält auch danach", consistent(p));
}

/* ── Farbe setzen ── */
{
  const p = fresh();
  const g = groupNames(p)[0];
  setMaterialGroupColor(p, g, "#D55E00");
  c("Gruppenfarbe normalisiert", p.materialGroups[g] === "#d55e00");
  c("Farbe auf alle Typen gespiegelt", p.types.filter((ty) => p.columnMetadata[ty].materialGroup === g).every((ty) => p.columnMetadata[ty].color === "#d55e00"));
  c("ungültige Farbe fällt auf Grau zurück", setMaterialGroupColor(p, g, "rot").ok && p.materialGroups[g] === "#808080");
}

/* ── Filter-Abgleich ── */
{
  const p = fresh();
  const [g0, g1] = groupNames(p);
  c("Filter bleibt bei existierender Gruppe", pruneMaterialFilter([g0, g1], p).length === 2);
  removeMaterialGroup(p, g0);
  c("Filter verliert gelöschte Gruppe", pruneMaterialFilter([g0, g1], p).join() === g1);
  c("Projektfilter wurde beim Löschen mitbereinigt", !(p.filters.materials ?? []).includes(g0));
}

/* ── Neue Gruppe ist im Matrixfilter wirksam ── */
{
  const p = fresh();
  addMaterialGroup(p, "Buntmetall", "#0072b2");
  const ty = p.types[1];
  assignTypeToGroup(p, ty, "Buntmetall");
  const vis = visibleIndices(p, { ...emptyFilters(), materials: ["Buntmetall"] });
  c("Filter auf neue Gruppe zeigt genau deren Typen", vis.cols.length === 1 && p.types[vis.cols[0]] === ty);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("FAIL: " + F.join(", ")); process.exit(1); }
console.log("\x1b[32m✓ Materialgruppen korrekt.\x1b[0m");
