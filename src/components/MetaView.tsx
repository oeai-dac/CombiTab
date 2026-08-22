import { useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import { useT } from "../i18n/I18nContext.js";
import { assignCvdSafePalette, simulateCVD, indistinctPairs, type CVD } from "../core/palette.js";
import {
  addMaterialGroup, renameMaterialGroup, removeMaterialGroup, setMaterialGroupColor,
  assignTypeToGroup, suggestGroupColor, groupCounts,
} from "../core/materialGroups.js";

export function MetaView({ project }: { project: ProjectV2 }) {
  const t = useT();
  const [, setVer] = useState(0);
  const [q, setQ] = useState("");
  const [cvd, setCvd] = useState<CVD | "normal">("normal");
  const [newName, setNewName] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const bump = () => setVer((v) => v + 1);
  const groups = Object.keys(project.materialGroups);
  const counts = groupCounts(project);

  const errMsg = (code?: string) =>
    code === "duplicate" ? t("meta.errDuplicate") : code === "last" ? t("meta.errLast") : t("meta.errEmpty");

  function applyCvdPalette() {
    const map = assignCvdSafePalette(groups);
    for (const g of groups) setMaterialGroupColor(project, g, map[g]);
    bump();
  }

  function addGroup() {
    const res = addMaterialGroup(project, newName);
    if (!res.ok) { setErr(errMsg(res.error)); return; }
    setNewName(""); setErr(null); bump();
  }

  function commitRename(oldName: string) {
    const res = renameMaterialGroup(project, oldName, editValue);
    if (!res.ok) { setErr(errMsg(res.error)); return; }
    setEditing(null); setErr(null); bump();
  }

  function deleteGroup(name: string) {
    const rest = groups.filter((g) => g !== name);
    if (!rest.length) { setErr(t("meta.errLast")); return; }
    const n = counts[name] ?? 0;
    if (n > 0 && !window.confirm(t("meta.deleteConfirm", { name, n, target: rest[0] }))) return;
    const res = removeMaterialGroup(project, name);
    if (!res.ok) { setErr(errMsg(res.error)); return; }
    if (editing === name) setEditing(null);
    setErr(null); bump();
  }

  const shown = project.types.filter((ty) => ty.toLowerCase().includes(q.toLowerCase()));

  return (
    <>
      <div className="sec-hdr">
        <h2>{t("meta.title")}</h2>
        <span className="hint">{t("meta.hint")}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: ".9rem", flex: 1, overflow: "hidden", paddingBottom: "1rem", minHeight: 0 }}>
        <aside className="det" style={{ overflowY: "auto" }}>
          <div className="blk" style={{ marginTop: 0 }}>{t("meta.materials")}</div>
          {groups.map((name) => (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: ".4rem", marginBottom: ".45rem" }}>
              <input type="color" value={toHex(project.materialGroups[name])}
                onChange={(e) => { setMaterialGroupColor(project, name, e.target.value); bump(); }}
                aria-label={t("meta.color")}
                style={{ width: 26, height: 26, border: "1px solid var(--bd2)", borderRadius: 4, background: "none", padding: 0, flexShrink: 0 }} />
              {cvd !== "normal" && <span title={t("meta.cvdPreview")} style={{ width: 20, height: 20, borderRadius: 4, border: "1px solid var(--bd2)", background: simulateCVD(toHex(project.materialGroups[name]), cvd), flexShrink: 0 }} />}
              {editing === name ? (
                <>
                  <input className="ann-in" autoFocus value={editValue} onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") { e.preventDefault(); commitRename(name); }
                      if (e.key === "Escape") { e.preventDefault(); setEditing(null); setErr(null); }
                    }}
                    style={{ flex: 1, minWidth: 0, fontSize: ".8rem" }} />
                  <button className="btn btn-ghost" style={{ padding: ".15rem .35rem", fontSize: ".72rem" }}
                    onClick={() => commitRename(name)} title={t("meta.renameSave")}>✓</button>
                  <button className="btn btn-ghost" style={{ padding: ".15rem .35rem", fontSize: ".72rem" }}
                    onClick={() => { setEditing(null); setErr(null); }} title={t("meta.renameCancel")}>✕</button>
                </>
              ) : (
                <>
                  <span style={{ flex: 1, fontSize: ".84rem", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={name}>{name}</span>
                  <span className="badge">{counts[name] ?? 0}</span>
                  <button className="btn btn-ghost" style={{ padding: ".15rem .35rem", fontSize: ".72rem" }}
                    onClick={() => { setEditing(name); setEditValue(name); setErr(null); }}
                    title={t("meta.rename", { name })} aria-label={t("meta.rename", { name })}>✎</button>
                  <button className="btn btn-ghost" style={{ padding: ".15rem .35rem", fontSize: ".72rem" }}
                    onClick={() => deleteGroup(name)} disabled={groups.length < 2}
                    title={t("meta.deleteGroup", { name })} aria-label={t("meta.deleteGroup", { name })}>🗑</button>
                </>
              )}
            </div>
          ))}

          <div style={{ display: "flex", alignItems: "center", gap: ".4rem", marginTop: ".55rem" }}>
            <span style={{ width: 26, height: 26, borderRadius: 4, border: "1px dashed var(--bd2)", background: suggestGroupColor(project), flexShrink: 0 }} aria-hidden="true" />
            <input className="ann-in" value={newName} onChange={(e) => { setNewName(e.target.value); setErr(null); }}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addGroup(); } }}
              placeholder={t("meta.newGroupName")} aria-label={t("meta.addMaterial")}
              style={{ flex: 1, minWidth: 0, fontSize: ".8rem" }} />
            <button className="btn" style={{ padding: ".2rem .5rem", fontSize: ".75rem" }} onClick={addGroup}>{t("meta.addBtn")}</button>
          </div>
          {err && <div role="alert" style={{ fontSize: ".72rem", color: "var(--accent2)", marginTop: ".35rem" }}>{err}</div>}
          <div style={{ fontSize: ".72rem", color: "var(--tx3)", marginTop: ".55rem" }}>{t("meta.groupsHint")}</div>
          <div style={{ fontSize: ".72rem", color: "var(--tx3)", marginTop: ".3rem" }}>{t("meta.colorHint")}</div>

          <div className="blk">{t("meta.cvdTitle")}</div>
          <button className="btn btn-ghost" style={{ width: "100%", fontSize: ".78rem" }} onClick={applyCvdPalette}>{t("meta.cvdApply")}</button>
          <div style={{ display: "flex", alignItems: "center", gap: ".4rem", marginTop: ".5rem" }}>
            <span className="tb-label">{t("meta.cvdPreviewLabel")}</span>
            <select className="sl" value={cvd} onChange={(e) => setCvd(e.target.value as CVD | "normal")} style={{ flex: 1 }}>
              <option value="normal">{t("meta.cvdNormal")}</option>
              <option value="deuteranopia">{t("meta.cvdDeuter")}</option>
              <option value="protanopia">{t("meta.cvdProt")}</option>
              <option value="tritanopia">{t("meta.cvdTrit")}</option>
            </select>
          </div>
          {(() => {
            const bad = indistinctPairs(project.materialGroups, cvd === "normal" ? null : cvd, 12);
            if (bad.length === 0) return <div style={{ fontSize: ".72rem", color: "#3f7a4f", marginTop: ".5rem" }}>{t("meta.cvdOk")}</div>;
            return (
              <div style={{ fontSize: ".72rem", color: "var(--accent2)", marginTop: ".5rem" }}>
                {t("meta.cvdWarn")}
                <ul style={{ margin: ".3rem 0 0", paddingLeft: "1.1rem", color: "var(--tx2)" }}>
                  {bad.slice(0, 5).map((p) => <li key={p.a + p.b}>{p.a} ↔ {p.b} <span style={{ color: "var(--tx3)" }}>(ΔE {p.deltaE.toFixed(0)})</span></li>)}
                </ul>
              </div>
            );
          })()}
        </aside>

        <div className="restab-wrap" style={{ flex: 1, overflow: "auto", border: "1px solid var(--bd)", borderRadius: "var(--r)", background: "var(--bg2)" }}>
          <div style={{ position: "sticky", top: 0, background: "var(--bg2)", padding: ".6rem .7rem", borderBottom: "1px solid var(--bd)", zIndex: 1 }}>
            <input className="ann-in" placeholder={t("meta.search")} value={q} onChange={(e) => setQ(e.target.value)} style={{ width: "100%" }} />
          </div>
          <table className="metatab">
            <thead><tr><th>{t("meta.type")}</th><th>{t("meta.assign")}</th><th>{t("meta.color")}</th><th>{t("meta.leadType")}</th></tr></thead>
            <tbody>
              {shown.map((ty) => {
                const cm = project.columnMetadata[ty];
                return (
                  <tr key={ty}>
                    <td style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: ".78rem" }}>{ty}</td>
                    <td>
                      <select className="sl" value={cm.materialGroup}
                        onChange={(e) => { assignTypeToGroup(project, ty, e.target.value); bump(); }}>
                        {!groups.includes(cm.materialGroup) && <option value={cm.materialGroup}>{cm.materialGroup}</option>}
                        {groups.map((g) => <option key={g} value={g}>{g}</option>)}
                      </select>
                    </td>
                    <td><span style={{ display: "inline-block", width: 16, height: 16, borderRadius: 3, background: cm.color, border: "1px solid var(--bd2)" }} /></td>
                    <td><input type="checkbox" checked={cm.isIndexType} onChange={() => { cm.isIndexType = !cm.isIndexType; bump(); }} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function toHex(c: string): string {
  if (/^#[0-9a-fA-F]{6}$/.test(c)) return c;
  if (/^#[0-9a-fA-F]{3}$/.test(c)) return "#" + c.slice(1).split("").map((x) => x + x).join("");
  return "#808080";
}
