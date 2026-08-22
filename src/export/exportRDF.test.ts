/**
 * CIDOC-CRM / CRMarchaeo-LOD-Export (§9.7-Vertiefung).
 */
import { toTurtle, toJSONLD } from "./exportRDF.js";
import { importCSV } from "../core/io/importTable.js";
import type { ProjectV2 } from "../core/model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mCIDOC-CRM / CRMarchaeo-LOD-Export (§9.7-Vertiefung)\x1b[0m\n");

// Kleines Projekt mit Material, Kontexttyp und einer nicht erfassten Zelle
const csv = "Context,x,y,z\nA,2,?,0\nB,1,3,0\nC,0,0,4";
const { project } = importCSV(csv, { name: "Fundstelle" });
project.materialGroups = { Keramik: "#CD853F", Metall: "#4682B4" };
project.columnMetadata["x"].materialGroup = "Keramik"; project.columnMetadata["x"].isIndexType = true;
project.columnMetadata["y"].materialGroup = "Metall";
project.columnMetadata["z"].materialGroup = "Keramik";
project.contextTypes = ["Körpergrab"];
project.rowMetadata["A"].contextType = "Körpergrab"; project.rowMetadata["A"].area = "Areal 1";

// belegte, erfasste Zellen: A/x(2), B/x(1), B/y(3), C/z(4) → 4 Zellen, 10 Objektidentitäten
const EXPECTED_OBJECTS = 10;

// ── Turtle: CRMarchaeo-Grundgerüst ──
{
  const ttl = toTurtle(project);
  c("Turtle: CIDOC-CRM-Präfix", ttl.includes("@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> ."));
  c("Turtle: CRMarchaeo-Präfix", ttl.includes("@prefix crmarchaeo: <http://www.cidoc-crm.org/extensions/crmarchaeo/> ."));
  c("Turtle: genau eine Fundstelle als E27_Site", (ttl.match(/a crm:E27_Site/g) || []).length === 1);
  c("Turtle: Kontexte als A2_Stratigraphic_Volume_Unit", (ttl.match(/a crmarchaeo:A2_Stratigraphic_Volume_Unit/g) || []).length === 3);
  c("Turtle: Grabungseinheiten als A1_Excavation_Processing_Unit", (ttl.match(/a crmarchaeo:A1_Excavation_Processing_Unit/g) || []).length === 3);
  c("Turtle: AP5_removed_part_or_all_of verknüpft Grabung↔Kontext", (ttl.match(/crmarchaeo:AP5_removed_part_or_all_of/g) || []).length === 3);
  c("Turtle: Typen als E55_Type", (ttl.match(/<type_\d+> a crm:E55_Type/g) || []).length === 3);
  c("Turtle: Materialien als E57_Material", (ttl.match(/a crm:E57_Material/g) || []).length === 2);
  c("Turtle: Typ hat Material als broader term", ttl.includes("crm:P127_has_broader_term"));
  c("Turtle: Leittyp als skos:note", ttl.includes('skos:note "Leittyp"'));
  c("Turtle: Kontexttyp verknüpft (P2_has_type)", ttl.includes("crm:P2_has_type <ctype_0>"));
  c("Turtle: Kontext am Fundort (P53 → site)", ttl.includes("crm:P53_has_former_or_current_location <site>"));
  c("Turtle: Seriationsposition am Kontext", ttl.includes("ctb:seriationPosition"));
}

// ── Turtle: echte Objektidentitäten ──
{
  const ttl = toTurtle(project);
  const objs = (ttl.match(/a crm:E22_Human-Made_Object/g) || []).length;
  c(`Turtle: ${EXPECTED_OBJECTS} Objektidentitäten (n je Zelle einzeln)`, objs === EXPECTED_OBJECTS, `${objs}`);
  c("Turtle: C/z=4 erzeugt vier Objekte find_2_2_0..3", ttl.includes("<find_2_2_0>") && ttl.includes("<find_2_2_3>") && !ttl.includes("<find_2_2_4>"));
  c("Turtle: A2 enthält Funde (AP21_contains)", ttl.includes("crmarchaeo:AP21_contains"));
  c("Turtle: keine aggregierte Zähler-Dimension im Identitätsmodus", !ttl.includes("crm:P90_has_value"));
  c("Turtle: nicht erfasste Zelle A/y erzeugt keinen Fund", !ttl.includes("<find_0_1_0>") && !ttl.includes("<find_0_1>"));
}

// ── Turtle: Aggregat-Rückfall über Limit ──
{
  const ttl = toTurtle(project, { maxObjectsPerCell: 2 });
  // n<=2 bleibt Identität (A/x=2, B/x=1), n>2 wird Aggregat (B/y=3, C/z=4)
  c("Aggregat-Rückfall: kleine Zellen bleiben Identitäten", ttl.includes("<find_0_0_0>") && ttl.includes("<find_0_0_1>"));
  c("Aggregat-Rückfall: große Zelle B/y wird Aggregat mit Anzahl 3", /<find_1_1>[\s\S]*?crm:P90_has_value "3"\^\^xsd:integer/.test(ttl));
  c("Aggregat-Rückfall: Maßeinheit deklariert", ttl.includes("a crm:E58_Measurement_Unit"));
  c("Aggregat-Rückfall: als Aggregat markiert (skos:note)", ttl.includes("skos:note") && /Aggregat/.test(ttl));
}

// ── objectIdentities:false → durchgängig Aggregat ──
{
  const ttl = toTurtle(project, { objectIdentities: false });
  c("Aggregatmodus: keine Objektidentitäten (find_i_j_k)", !/<find_\d+_\d+_\d+>/.test(ttl));
  c("Aggregatmodus: vier Aggregat-Funde", (ttl.match(/a crm:E22_Human-Made_Object/g) || []).length === 4);
}

// ── Escaping ──
{
  const p2: ProjectV2 = { ...project, name: 'Grab "Nord"\tX' };
  const ttl = toTurtle(p2);
  c("Turtle: Sonderzeichen im Titel maskiert", ttl.includes('dct:title "Grab \\"Nord\\"\\tX"'));
}

// ── JSON-LD ──
{
  const jsonld = toJSONLD(project);
  let obj: any = null; let ok = true;
  try { obj = JSON.parse(jsonld); } catch { ok = false; }
  c("JSON-LD ist gültiges JSON", ok && !!obj);
  c("JSON-LD @context mit crm + crmarchaeo", obj && obj["@context"].crm === "http://www.cidoc-crm.org/cidoc-crm/" && obj["@context"].crmarchaeo === "http://www.cidoc-crm.org/extensions/crmarchaeo/");
  c("JSON-LD hat @graph als Array", obj && Array.isArray(obj["@graph"]));
  const objs = obj["@graph"].filter((n: any) => n["@type"] === "crm:E22_Human-Made_Object");
  c(`JSON-LD: ${EXPECTED_OBJECTS} Objektidentitäten`, objs.length === EXPECTED_OBJECTS, `${objs.length}`);
  const one = objs.find((n: any) => n["@id"] === "find_2_2_0"); // C/z, erstes Objekt
  c("JSON-LD: Objekt trägt Typ", !!one && one["crm:P2_has_type"]["@id"] === "type_2");
  const ctx2 = obj["@graph"].find((n: any) => n["@id"] === "context_2");
  c("JSON-LD: Kontext ist A2 und enthält Funde", !!ctx2 && ctx2["@type"] === "crmarchaeo:A2_Stratigraphic_Volume_Unit" && Array.isArray(ctx2["crmarchaeo:AP21_contains"]) && ctx2["crmarchaeo:AP21_contains"].length === 4);
  c("JSON-LD: kein Fund für nicht erfasste Zelle", !obj["@graph"].some((n: any) => String(n["@id"]).startsWith("find_0_1")));
  const site = obj["@graph"].filter((n: any) => n["@type"] === "crm:E27_Site");
  c("JSON-LD: genau eine Fundstelle (E27_Site)", site.length === 1);
  const exc = obj["@graph"].filter((n: any) => n["@type"] === "crmarchaeo:A1_Excavation_Processing_Unit");
  c("JSON-LD: drei Grabungseinheiten", exc.length === 3 && exc[0]["crmarchaeo:AP5_removed_part_or_all_of"]["@id"].startsWith("context_"));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
else console.log("\x1b[32m✓ CRMarchaeo-Export korrekt.\x1b[0m");
