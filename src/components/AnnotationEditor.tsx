import { useEffect, useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import { applyToCells, clearCells, commonValue, buildBatchPatch } from "../annotations/annotations.js";
import { setMissing, isMissing } from "../core/missing.js";
import { useT } from "../i18n/I18nContext.js";

const CERTAINTY = ["certain", "uncertain", "questionable"] as const;
const FRAG = ["complete", "fragmented"] as const;

export function AnnotationEditor({ project, cells, onApplied, onClear }: {
  project: ProjectV2; cells: Array<[number, number]>; onApplied: () => void; onClear: () => void;
}) {
  const t = useT();
  const [certainty, setCertainty] = useState<string>("");
  const [fragmentation, setFragmentation] = useState<string>("");
  const [countMin, setCountMin] = useState<string>("");
  const [countMax, setCountMax] = useState<string>("");
  const [inv, setInv] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  // Nur tatsächlich angefasste Felder werden beim „Setzen" angewendet —
  // uneinheitliche (leer vorbelegte) Felder einer gemischten Auswahl bleiben sonst erhalten.
  const [touched, setTouched] = useState<Set<string>>(new Set());
  const touch = (f: string) => setTouched((s) => (s.has(f) ? s : new Set(s).add(f)));

  // Vorbelegung aus gemeinsamen Werten der Auswahl
  useEffect(() => {
    setTouched(new Set());
    setCertainty((commonValue(project, cells, "certainty") as string) ?? "");
    setFragmentation((commonValue(project, cells, "fragmentation") as string) ?? "");
    const cmin = commonValue(project, cells, "countMin"); setCountMin(cmin != null ? String(cmin) : "");
    const cmax = commonValue(project, cells, "countMax"); setCountMax(cmax != null ? String(cmax) : "");
    const iv = commonValue(project, cells, "inventoryNumbers") as string[] | undefined; setInv(iv ? iv.join(", ") : "");
    setNotes((commonValue(project, cells, "notes") as string) ?? "");
  }, [project, cells]);

  if (!cells.length) return <aside className="det"><div className="det-empty">{t("annot.emptyA")}<br />{t("annot.emptyB")}</div></aside>;

  function apply() {
    const patch = buildBatchPatch({ certainty, fragmentation, countMin, countMax, inv, notes }, touched);
    if (Object.keys(patch).length === 0) return; // nichts angefasst → nichts ändern
    applyToCells(project, cells, patch);
    onApplied();
  }
  function clearAll() { clearCells(project, cells); onClear(); }
  function markMissing(v: boolean) { setMissing(project, cells, v); onApplied(); }
  const missingInSel = cells.filter(([i, j]) => isMissing(project, i, j)).length;

  const seg = (field: string, val: string, set: (v: string) => void, opts: readonly string[]) => (
    <div style={{ display: "flex", gap: ".3rem", flexWrap: "wrap" }}>
      {opts.map((o) => (
        <button key={o} className={"chip-btn" + (val === o ? " on" : "")} onClick={() => { touch(field); set(val === o ? "" : o); }}>{t("annot." + o)}</button>
      ))}
    </div>
  );

  return (
    <aside className="det">
      <div className="det-head">
        <h3 style={{ fontSize: "1.25rem" }}>{t("annot.cells", { n: cells.length })}</h3>
        <div className="sub">{t("annot.batch")}</div>
      </div>
      <div className="blk" style={{ marginTop: ".3rem" }}>{t("annot.certainty")}</div>
      {seg("certainty", certainty, setCertainty, CERTAINTY)}
      <div className="blk">{t("annot.fragmentation")}</div>
      {seg("fragmentation", fragmentation, setFragmentation, FRAG)}
      <div className="blk">{t("annot.countRange")}</div>
      <div style={{ display: "flex", gap: ".4rem", alignItems: "center" }}>
        <input className="ann-in" type="number" placeholder="min" value={countMin} onChange={(e) => { touch("countMin"); setCountMin(e.target.value); }} style={{ width: 80 }} />
        <span style={{ color: "var(--tx3)" }}>–</span>
        <input className="ann-in" type="number" placeholder="max" value={countMax} onChange={(e) => { touch("countMax"); setCountMax(e.target.value); }} style={{ width: 80 }} />
      </div>
      <div className="blk">{t("annot.inventory")}</div>
      <input className="ann-in" placeholder="INV-1, INV-2 …" value={inv} onChange={(e) => { touch("inv"); setInv(e.target.value); }} style={{ width: "100%" }} />
      <div className="blk">{t("annot.notes")}</div>
      <textarea className="ann-in" rows={3} value={notes} onChange={(e) => { touch("notes"); setNotes(e.target.value); }} style={{ width: "100%", resize: "vertical" }} />
      <div style={{ display: "flex", gap: ".5rem", marginTop: ".9rem" }}>
        <button className="btn" onClick={apply}>{t("annot.apply")}</button>
        <button className="btn btn-ghost" onClick={clearAll}>{t("annot.clearAll")}</button>
      </div>
      <div style={{ fontSize: ".72rem", color: "var(--tx3)", marginTop: ".6rem" }}>{t("annot.footer", { n: cells.length })}</div>

      <div className="blk">{t("missing.section")}</div>
      <div style={{ fontSize: ".76rem", color: "var(--tx3)", marginBottom: ".4rem" }}>{t("missing.hint")}{missingInSel > 0 && ` · ${t("missing.inSel", { n: missingInSel })}`}</div>
      <div style={{ display: "flex", gap: ".5rem" }}>
        <button className="btn btn-ghost" onClick={() => markMissing(true)}>{t("missing.mark")}</button>
        <button className="btn btn-ghost" onClick={() => markMissing(false)}>{t("missing.unmark")}</button>
      </div>
    </aside>
  );
}
