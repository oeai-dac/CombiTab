/**
 * CIDOC-CRM / Linked-Open-Data-Export (§9.7, mit CRMarchaeo-Vertiefung) — Turtle
 * und JSON-LD aus einer gemeinsamen Zwischenrepräsentation (IR), damit beide
 * Serialisierungen garantiert dieselben Aussagen tragen.
 *
 * Modellierung (bewusst konservativ, dokumentiert, CRMarchaeo v2.x):
 *   - Datensatz                → dcat:Dataset
 *   - Fundstelle (Projektname) → crm:E27_Site                      (ein Ort für alles)
 *   - Kontext (Grab/Befund)    → crmarchaeo:A2_Stratigraphic_Volume_Unit
 *        crm:P2_has_type                        → Kontexttyp (E55_Type)
 *        crm:P53_has_former_or_current_location → Fundstelle
 *        ctb:seriationPosition                  → Rang in der Seriationsreihenfolge
 *   - Grabungseinheit          → crmarchaeo:A1_Excavation_Processing_Unit
 *        crmarchaeo:AP5_removed_part_or_all_of  → Kontext (A2)
 *        crm:P7_took_place_at                   → Fundstelle
 *   - Typ                      → crm:E55_Type   (+ Material als P127_has_broader_term, Leittyp-Notiz)
 *   - Materialgruppe           → crm:E57_Material
 *   - Fund (Objektidentität)   → crm:E22_Human-Made_Object
 *        crm:P2_has_type                        → Typ
 *      und je Kontext: A2 crmarchaeo:AP21_contains → Fund   (Verkürzung des A7-Embedding-Pfads)
 *
 * WESENTLICHE VERTIEFUNG ggü. §9.7-Erstfassung:
 *   1. Kontexte sind jetzt **stratigrafische Einheiten** (A2), nicht mehr E27_Site —
 *      ein Grab ist kein „Site". Die Fundstelle ist der eine E27_Site.
 *   2. **Echte Objektidentitäten** statt aggregierter Zählerdimension: eine Zelle mit
 *      Häufigkeit n erzeugt n einzelne E22-Objekte (jeweils crm:P2_has_type), im
 *      Kontext über crmarchaeo:AP21_contains verortet. Für sehr große n gibt es ein
 *      dokumentiertes Limit mit Aggregat-Rückfall (E54-Dimension), damit der Graph
 *      nicht unbegrenzt wächst — ehrlich benannt statt stillschweigend.
 *   3. **Grabungseinheiten** (A1) mit AP5_removed_part_or_all_of.
 *
 * EHRLICHE GRENZE: Das Datenmodell erfasst **keine beobachtete Stratigraphie**
 * (keine Harris-Matrix-Kanten). Daher werden **keine** crmarchaeo:AP13-Relationen
 * behauptet. Exportiert wird nur die **seriationsabgeleitete relative Ordnung**
 * (ctb:seriationPosition) — ausdrücklich als inferiert markiert und ohne behauptete
 * Richtung (Seriation liefert eine Sequenz, keine absolute Zeitrichtung). AP13 ist
 * der dokumentierte Erweiterungspunkt, sobald stratigrafische Beobachtungen vorliegen.
 *
 * „Nicht erfasst" (§9.6) und echte Absenz (0) erzeugen KEINEN Fund — Absenz wird nicht
 * behauptet, Fehlwerte werden nicht erfunden.
 */
import type { ProjectV2 } from "../core/model.js";
import { isMissing } from "../core/missing.js";

export interface RdfOptions {
  base?: string;
  /** Echte Objektidentitäten je Fund (Default true). Bei false: Aggregat mit Anzahl-Dimension. */
  objectIdentities?: boolean;
  /** Obergrenze einzelner Objektidentitäten je Zelle, darüber Aggregat-Rückfall (Default 1000). */
  maxObjectsPerCell?: number;
  /** Grabungseinheiten (A1) mit AP5 emittieren (Default true). */
  excavationUnits?: boolean;
  /** Seriationsabgeleitete relative Ordnung als ctb:seriationPosition (Default true). */
  seriationSequence?: boolean;
}

const DEFAULT_BASE = "https://combitab.example/id/";

const PREFIXES: Record<string, string> = {
  crm: "http://www.cidoc-crm.org/cidoc-crm/",
  crmarchaeo: "http://www.cidoc-crm.org/extensions/crmarchaeo/",
  rdfs: "http://www.w3.org/2000/01/rdf-schema#",
  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
  xsd: "http://www.w3.org/2001/XMLSchema#",
  skos: "http://www.w3.org/2004/02/skos/core#",
  dcat: "http://www.w3.org/ns/dcat#",
  dct: "http://purl.org/dc/terms/",
};

/* ── Zwischenrepräsentation ──────────────────────────────────────────────── */

type Term =
  | { iri: string }                                   // relativer IRI (rendert <iri> bzw. {"@id"})
  | { lit: string; dt?: string }                      // Literal, optional mit Datentyp
  | { bnode: RdfNode };                               // Blank Node (verschachtelt)
interface RdfNode { id?: string; types: string[]; props: Array<[string, Term | Term[]]>; }

const iri = (id: string): Term => ({ iri: id });
const lit = (s: string, dt?: string): Term => ({ lit: String(s), dt });
const named = (m: Record<string, string>) => Object.keys(m).filter((k) => k && k !== "Unassigned");

/** Turtle-String-Literal maskieren. */
function esc(s: string): string {
  return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t") + '"';
}

/* ── IR aufbauen ─────────────────────────────────────────────────────────── */

function buildGraph(p: ProjectV2, opts: RdfOptions): RdfNode[] {
  const objectIdentities = opts.objectIdentities ?? true;
  const maxObjects = opts.maxObjectsPerCell ?? 1000;
  const excavationUnits = opts.excavationUnits ?? true;
  const seriationSequence = opts.seriationSequence ?? true;

  const nodes: RdfNode[] = [];
  const today = new Date().toISOString().slice(0, 10);

  // Seriations-Rang je Kontext aus der Anzeigereihenfolge (relative Ordnung, keine Richtung)
  const rankOf = (cx: string, i: number): number => {
    const pos = p.order?.rows ? p.order.rows.indexOf(cx) : -1;
    return pos >= 0 ? pos : i;
  };

  // Datensatz
  nodes.push({ id: "dataset", types: ["dcat:Dataset"], props: [
    ["dct:title", lit(p.name)],
    ["dct:created", lit(today, "xsd:date")],
    ["rdfs:comment", lit("Aus CombiTab v2 exportiert (CIDOC-CRM / CRMarchaeo, §9.7-Vertiefung).")],
  ]});

  // Fundstelle (ein Site)
  nodes.push({ id: "site", types: ["crm:E27_Site"], props: [["rdfs:label", lit(p.name)]] });

  // Custom-Property für die Seriationsposition (dokumentiert)
  if (seriationSequence) {
    nodes.push({ id: "seriationPosition", types: ["rdf:Property"], props: [
      ["rdfs:label", lit("seriation position")],
      ["rdfs:comment", lit("Rang des Kontexts in der seriationsabgeleiteten relativen Ordnung. Inferiert, nicht beobachtet; die absolute Zeitrichtung ist durch Seriation nicht bestimmt.")],
    ]});
  }

  // Materialgruppen → E57_Material
  const matIdx = new Map<string, number>();
  named(p.materialGroups).forEach((m, k) => { matIdx.set(m, k); nodes.push({ id: `material_${k}`, types: ["crm:E57_Material"], props: [["rdfs:label", lit(m)]] }); });

  // Kontexttypen → E55_Type (nur verwendete)
  const usedCtypes = new Set<string>();
  for (const cx of p.contexts) { const ct = p.rowMetadata[cx]?.contextType; if (ct && ct !== "Unassigned") usedCtypes.add(ct); }
  const ctypeIdx = new Map<string, number>();
  [...usedCtypes].forEach((ct, k) => { ctypeIdx.set(ct, k); nodes.push({ id: `ctype_${k}`, types: ["crm:E55_Type"], props: [["rdfs:label", lit(ct)]] }); });

  // Typen → E55_Type
  p.types.forEach((t, j) => {
    const cm = p.columnMetadata[t];
    const props: Array<[string, Term | Term[]]> = [["rdfs:label", lit(t)]];
    const mk = cm && matIdx.get(cm.materialGroup);
    if (mk != null) props.push(["crm:P127_has_broader_term", iri(`material_${mk}`)]);
    if (cm?.isIndexType) props.push(["skos:note", lit("Leittyp")]);
    nodes.push({ id: `type_${j}`, types: ["crm:E55_Type"], props });
  });

  // Kontexte → A2_Stratigraphic_Volume_Unit; contains-Liste wird später gefüllt
  const containsByCtx: Term[][] = p.contexts.map(() => []);
  const ctxNodes: RdfNode[] = p.contexts.map((cx, i) => {
    const rm = p.rowMetadata[cx];
    const props: Array<[string, Term | Term[]]> = [["rdfs:label", lit(cx)]];
    const ck = rm && ctypeIdx.get(rm.contextType);
    if (ck != null) props.push(["crm:P2_has_type", iri(`ctype_${ck}`)]);
    props.push(["crm:P53_has_former_or_current_location", iri("site")]);
    if (rm?.area) props.push(["rdfs:comment", lit("Areal: " + rm.area)]);
    if (seriationSequence) props.push(["ctb:seriationPosition", lit(String(rankOf(cx, i)), "xsd:integer")]);
    const node: RdfNode = { id: `context_${i}`, types: ["crmarchaeo:A2_Stratigraphic_Volume_Unit"], props };
    return node;
  });

  // Grabungseinheiten → A1_Excavation_Processing_Unit
  const excNodes: RdfNode[] = excavationUnits ? p.contexts.map((cx, i) => ({
    id: `exc_${i}`, types: ["crmarchaeo:A1_Excavation_Processing_Unit"], props: [
      ["rdfs:label", lit("Grabungseinheit " + cx)],
      ["crmarchaeo:AP5_removed_part_or_all_of", iri(`context_${i}`)],
      ["crm:P7_took_place_at", iri("site")],
    ],
  })) : [];

  // Funde: echte Objektidentitäten (oder Aggregat-Rückfall)
  const findNodes: RdfNode[] = [];
  let objectCount = 0, cellCount = 0, aggregated = 0;
  for (let i = 0; i < p.contexts.length; i++) for (let j = 0; j < p.types.length; j++) {
    const n = p.matrix[i][j];
    if (n <= 0 || isMissing(p, i, j)) continue;
    cellCount++;
    if (objectIdentities && n <= maxObjects) {
      for (let k = 0; k < n; k++) {
        const id = `find_${i}_${j}_${k}`;
        findNodes.push({ id, types: ["crm:E22_Human-Made_Object"], props: [["crm:P2_has_type", iri(`type_${j}`)]] });
        containsByCtx[i].push(iri(id));
        objectCount++;
      }
    } else {
      // Aggregat-Rückfall: ein Knoten mit Anzahl-Dimension, ehrlich als Aggregat markiert
      const id = `find_${i}_${j}`;
      const dim: Term = { bnode: { types: ["crm:E54_Dimension"], props: [
        ["crm:P90_has_value", lit(String(n), "xsd:integer")],
        ["crm:P91_has_unit", iri("unit_count")],
      ]}};
      findNodes.push({ id, types: ["crm:E22_Human-Made_Object"], props: [
        ["crm:P2_has_type", iri(`type_${j}`)],
        ["crm:P43_has_dimension", dim],
        ["skos:note", lit("Aggregat: " + n + " Objekte (über Identitäts-Limit oder Aggregatmodus)")],
      ]});
      containsByCtx[i].push(iri(id));
      aggregated++;
    }
  }

  // Maßeinheit nur bei Aggregat-Nutzung
  if (aggregated > 0) nodes.push({ id: "unit_count", types: ["crm:E58_Measurement_Unit"], props: [["rdfs:label", lit("count")]] });

  // contains-Kanten an die Kontexte hängen
  ctxNodes.forEach((node, i) => { if (containsByCtx[i].length) node.props.push(["crmarchaeo:AP21_contains", containsByCtx[i]]); });

  nodes.push(...ctxNodes, ...excNodes, ...findNodes);

  // Zusammenfassung als Kommentar am Datensatz (nur informativ; wird beim Rendern ausgegeben)
  (nodes as any)._summary = { contexts: p.contexts.length, types: p.types.length, cells: cellCount, objects: objectCount, aggregated };
  return nodes;
}

/* ── Renderer ────────────────────────────────────────────────────────────── */

function termTurtle(t: Term): string {
  if ("iri" in t) return `<${t.iri}>`;
  if ("lit" in t) return esc(t.lit) + (t.dt ? "^^" + t.dt : "");
  // bnode
  const inner = t.bnode.types.map((ty) => `a ${ty}`).concat(
    t.bnode.props.map(([pr, v]) => `${pr} ${Array.isArray(v) ? v.map(termTurtle).join(", ") : termTurtle(v)}`),
  ).join(" ; ");
  return `[ ${inner} ]`;
}

function nodeTurtle(n: RdfNode): string {
  const head = `<${n.id}> a ${n.types.join(", ")}`;
  const parts = n.props.map(([pr, v]) => `${pr} ${Array.isArray(v) ? v.map(termTurtle).join(", ") : termTurtle(v)}`);
  if (parts.length === 0) return head + " .";
  return head + " ;\n    " + parts.join(" ;\n    ") + " .";
}

export function toTurtle(p: ProjectV2, opts: RdfOptions = {}): string {
  const base = opts.base ?? DEFAULT_BASE;
  const nodes = buildGraph(p, opts);
  const s = (nodes as any)._summary;
  const L: string[] = [];
  L.push(`@base <${base}> .`);
  for (const [pre, ns] of Object.entries(PREFIXES)) L.push(`@prefix ${pre}: <${ns}> .`);
  L.push(`@prefix : <${base}> .`);
  L.push(`@prefix ctb: <${base}> .`);
  L.push("");
  for (const n of nodes) L.push(nodeTurtle(n));
  L.push("");
  L.push(`# ${s.contexts} Kontexte, ${s.types} Typen, ${s.cells} belegte Zellen, ${s.objects} Objektidentitäten` + (s.aggregated ? `, ${s.aggregated} aggregiert` : "") + ".");
  return L.join("\n") + "\n";
}

function termJSONLD(t: Term): unknown {
  if ("iri" in t) return { "@id": t.iri };
  if ("lit" in t) return t.dt ? { "@value": t.lit, "@type": t.dt } : t.lit;
  const o: Record<string, unknown> = { "@type": t.bnode.types.length === 1 ? t.bnode.types[0] : t.bnode.types };
  for (const [pr, v] of t.bnode.props) o[pr] = Array.isArray(v) ? v.map(termJSONLD) : termJSONLD(v);
  return o;
}

export function toJSONLD(p: ProjectV2, opts: RdfOptions = {}): string {
  const base = opts.base ?? DEFAULT_BASE;
  const nodes = buildGraph(p, opts);
  const context: Record<string, unknown> = { "@base": base, ...PREFIXES, ctb: base };
  const graph = nodes.map((n) => {
    const o: Record<string, unknown> = { "@id": n.id, "@type": n.types.length === 1 ? n.types[0] : n.types };
    for (const [pr, v] of n.props) o[pr] = Array.isArray(v) ? v.map(termJSONLD) : termJSONLD(v);
    return o;
  });
  return JSON.stringify({ "@context": context, "@graph": graph }, null, 2);
}
