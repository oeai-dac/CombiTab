import { visibleIndices, filterProject, filtersActive, emptyFilters, writeBackOrder, writeBackAnnotations } from "./filter.js";
import { annotationKey, type ProjectV2, type FilterSettings } from "./model.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

// 4 Kontexte × 5 Typen; T4 leer, G3 leer. Material: T0,T1,T4→A ; T2,T3→B
function mk(): ProjectV2 {
  const M = [
    [1, 1, 0, 0, 0], // G0
    [0, 1, 1, 0, 0], // G1
    [0, 0, 1, 1, 0], // G2
    [0, 0, 0, 0, 0], // G3 (leer)
  ];
  const mats = ["A", "A", "B", "B", "A"];
  const contexts = ["G0", "G1", "G2", "G3"], types = ["T0", "T1", "T2", "T3", "T4"];
  const columnMetadata: any = {}, rowMetadata: any = {};
  types.forEach((t, j) => columnMetadata[t] = { name: t, materialGroup: mats[j], color: "#888", isIndexType: false, isFixed: false, notes: "" });
  contexts.forEach((cx) => rowMetadata[cx] = { name: cx, contextType: "", area: "", isFixed: false, notes: "" });
  return { schemaVersion: 2, name: "t", dataType: "presence_absence", contexts, types, matrix: M, columnMetadata, rowMetadata, cellAnnotations: {}, materialGroups: { A: "#a00", B: "#0a0" }, contextTypes: [], order: { rows: [...contexts], cols: [...types] }, view: { vizStyle: "", cellSize: 1, showValues: true, showColors: true, showCertainty: false, showFragmentation: false }, filters: emptyFilters(), history: [] };
}
const F0 = (o: Partial<FilterSettings> = {}): FilterSettings => ({ ...emptyFilters(), ...o });

console.log("\n\x1b[1mFilterung & Fokus\x1b[0m\n");

// 1) Ohne Filter: alles sichtbar, nicht aktiv
{
  c("emptyFilters ⇒ inaktiv", filtersActive(emptyFilters()) === false);
  const v = visibleIndices(mk(), emptyFilters());
  c("ohne Filter: alle Zeilen/Spalten", eq(v.rows, [0, 1, 2, 3]) && eq(v.cols, [0, 1, 2, 3, 4]));
}

// 2) Material-Filter isoliert Spalten
{
  const v = visibleIndices(mk(), F0({ materials: ["A"] }));
  c("Material A ⇒ Spalten {0,1,4}", eq(v.cols, [0, 1, 4]));
  c("Material-Filter ist aktiv", filtersActive(F0({ materials: ["A"] })));
}

// 3) leere Spalten ausblenden (T4)
{
  const v = visibleIndices(mk(), F0({ hideEmptyCols: true }));
  c("hideEmptyCols entfernt T4", eq(v.cols, [0, 1, 2, 3]));
}

// 4) leere Zeilen ausblenden (G3)
{
  const v = visibleIndices(mk(), F0({ hideEmptyRows: true }));
  c("hideEmptyRows entfernt G3", eq(v.rows, [0, 1, 2]));
}

// 5) Material A + leere Zeilen: G2/G3 fallen weg
{
  const v = visibleIndices(mk(), F0({ materials: ["A"], hideEmptyRows: true }));
  c("Material A + hideEmptyRows ⇒ Zeilen {0,1}", eq(v.rows, [0, 1]) && eq(v.cols, [0, 1, 4]));
}

// 6) Bereichsfilter
{
  const v = visibleIndices(mk(), F0({ rowRange: [1, 2] }));
  c("rowRange [1,2]", eq(v.rows, [1, 2]));
}

// 7) Fokus auf G0 ⇒ Nachbarschaft
{
  const v = visibleIndices(mk(), emptyFilters(), { ctx: "G0" });
  c("Fokus G0 ⇒ Zeilen {0,1}, Spalten {0,1,2}", eq(v.rows, [0, 1]) && eq(v.cols, [0, 1, 2]), `rows=${v.rows} cols=${v.cols}`);
}

// 8) filterProject: Teilmatrix + Annotations-Remapping über Namen
{
  const p = mk();
  p.cellAnnotations[annotationKey(1, 2)] = { context: "G1", type: "T2", notes: "x" }; // G1×T2
  const fp = filterProject(p, F0({ materials: ["B"] })); // Spalten T2,T3
  c("filterProject: Typen = [T2,T3]", eq(fp.types.length ? [fp.types.indexOf("T2"), fp.types.indexOf("T3")] : [-1], [0, 1]));
  c("filterProject: Teilmatrix korrekt", fp.matrix[1][0] === 1 && fp.matrix[0][0] === 0); // G1×T2=1, G0×T2=0
  c("filterProject: Annotation über Namen umgeschlüsselt", !!fp.cellAnnotations[annotationKey(1, 0)]);
  c("filterProject: order nur sichtbare Typen", eq(fp.order.cols.map((t) => fp.types.indexOf(t)), [0, 1]));
}

// 9) writeBackOrder: sichtbare Neuordnung zurückschreiben, verborgene bleiben
{
  const base = ["A", "B", "C", "D", "E"];
  // sichtbar {A,C,E} an Basispositionen 0,2,4; neu geordnet zu E,A,C
  const out = writeBackOrder(base, ["E", "A", "C"]);
  c("writeBackOrder: sichtbare umsortiert, verborgene fix", out.join("") === "EBADC", out.join(""));
  const id = writeBackOrder(base, ["A", "B", "C", "D", "E"]);
  c("writeBackOrder: Identität bei voller Sicht", id.join("") === "ABCDE");
}

// 10) writeBackAnnotations: Sicht-Annotationen namensbasiert ins Grundprojekt
{
  const p = mk();
  // vorhandene Basis-Annotationen: eine außerhalb (G0×T0), eine innerhalb (G2×T3) des Fensters
  p.cellAnnotations[annotationKey(0, 0)] = { context: "G0", type: "T0", notes: "außen" };
  p.cellAnnotations[annotationKey(2, 3)] = { context: "G2", type: "T3", notes: "wird gelöscht" };
  // gefilterte Sicht auf Material B (Spalten T2,T3), alle Zeilen sichtbar
  const view = filterProject(p, F0({ materials: ["B"] }));
  const vr = view.contexts.indexOf("G1"), vc = view.types.indexOf("T2");
  view.cellAnnotations[annotationKey(vr, vc)] = { context: "G1", type: "T2", notes: "neu" };
  // Nutzer löscht die kopierte G2×T3-Annotation in der Sicht
  delete view.cellAnnotations[annotationKey(view.contexts.indexOf("G2"), view.types.indexOf("T3"))];
  writeBackAnnotations(p, view);
  c("writeBackAnn: neue Annotation im Grundprojekt", p.cellAnnotations[annotationKey(1, 2)]?.notes === "neu");
  c("writeBackAnn: Annotation außerhalb des Fensters bleibt", p.cellAnnotations[annotationKey(0, 0)]?.notes === "außen");
  c("writeBackAnn: gelöschte (nicht in Sicht) entfernt", !p.cellAnnotations[annotationKey(2, 3)]);
}

console.log("\n" + (fail ? "\x1b[31m" + fail + " fehlgeschlagen\x1b[0m: " + F.join(", ") : "\x1b[32malle " + pass + " bestanden\x1b[0m") + "\n");
if (fail) process.exit(1);
