import type { ProjectV2 } from "../core/model.js";
import type { MatrixRenderer, CellRef } from "../matrix/MatrixRenderer.js";
import { isMissing, typePresence, contextPresence } from "../core/missing.js";
import { useT } from "../i18n/I18nContext.js";

export function Inspector({ project, sel, renderer, onTogglePin }: {
  project: ProjectV2; sel: CellRef | null; renderer: MatrixRenderer | null;
  onTogglePin: (axis: "row" | "col", displayPos: number) => void;
}) {
  const t = useT();
  if (!sel || !renderer) return <aside className="det"><div className="det-empty">{t("inspector.empty")}</div></aside>;
  const info = renderer.info(sel);
  const ctag = (mg: string, color: string) => <span className="ctag" style={{ color, borderColor: color }}>{mg}</span>;
  const fixedBadge = <span className="badge" style={{ borderColor: "rgba(210,38,48,.4)", color: "#a81d26" }}>{t("inspector.fixed")}</span>;

  const rowFixed = sel.row >= 0 && renderer.isFixedAt("row", sel.row);
  const colFixed = sel.col >= 0 && renderer.isFixedAt("col", sel.col);
  const pinBtn = (axis: "row" | "col", pos: number, fixed: boolean, label: string) => (
    <button className="btn btn-ghost" style={{ fontSize: ".76rem", padding: ".3rem .7rem" }} onClick={() => onTogglePin(axis, pos)}>
      {fixed ? "◉ " : "○ "}{fixed ? t("inspector.unpinLabel", { label }) : t("inspector.pinLabel", { label })}
    </button>
  );

  if (sel.kind === "cell" && info.context && info.type) {
    const cm = info.colMeta!;
    return (
      <aside className="det">
        <div className="det-head">
          <h3>{info.context}</h3>
          <div className="sub">{info.type}</div>
          <div className="det-meta">{ctag(cm.materialGroup, cm.color)}<span className="badge">{t("inspector.rowcol", { r: info.displayRow + 1, c: info.displayCol + 1 })}</span>{(rowFixed || colFixed) && fixedBadge}{isMissing(project, project.contexts.indexOf(info.context), project.types.indexOf(info.type)) && <span className="badge" style={{ borderColor: "rgba(120,120,120,.5)", color: "var(--tx2)" }}>{t("missing.badge")}</span>}</div>
        </div>
        <dl className="kv">
          <dt>{t("inspector.count")}</dt><dd className="mono">{isMissing(project, project.contexts.indexOf(info.context), project.types.indexOf(info.type)) ? t("missing.badge") : info.value}</dd>
          <dt>{t("inspector.material")}</dt><dd>{cm.materialGroup}</dd>
          <dt>{t("inspector.leadType")}</dt><dd>{cm.isIndexType ? t("inspector.yes") : "—"}</dd>
        </dl>
        <div className="blk">{t("inspector.context")}</div>
        <dl className="kv">
          <dt>{t("inspector.contextType")}</dt><dd>{info.rowMeta?.contextType || "—"}</dd>
          <dt>{t("inspector.area")}</dt><dd>{info.rowMeta?.area || "—"}</dd>
        </dl>
        <div className="blk">{t("inspector.pinBlock")}</div>
        <div style={{ display: "flex", gap: ".4rem", flexWrap: "wrap" }}>
          {pinBtn("row", sel.row, rowFixed, t("inspector.context"))}
          {pinBtn("col", sel.col, colFixed, t("inspector.type"))}
        </div>
      </aside>
    );
  }
  if (sel.kind === "row" && info.context) {
    const rm = info.rowMeta!;
    return (
      <aside className="det">
        <div className="det-head"><h3>{info.context}</h3><div className="sub">{t("inspector.rowPos", { r: info.displayRow + 1, n: project.contexts.length })}</div>
          <div className="det-meta"><span className="badge">{rm.contextType || t("inspector.context")}</span>{rowFixed && fixedBadge}</div></div>
        <dl className="kv"><dt>{t("inspector.area")}</dt><dd>{rm.area || "—"}</dd><dt>{t("inspector.notes")}</dt><dd>{rm.notes || "—"}</dd></dl>
        {(() => { const pr = contextPresence(project, project.contexts.indexOf(info.context)); return (
          <><div className="blk">{t("missing.presence")}</div><dl className="kv"><dt>{t("missing.present")}</dt><dd className="mono">{pr.present}</dd><dt>{t("missing.absent")}</dt><dd className="mono">{pr.absent}</dd><dt>{t("missing.badge")}</dt><dd className="mono">{pr.missing}</dd></dl></>
        ); })()}
        <div className="blk">{t("inspector.action")}</div>
        <div style={{ display: "flex", gap: ".4rem", flexWrap: "wrap" }}>{pinBtn("row", sel.row, rowFixed, t("inspector.context"))}</div>
        <div style={{ fontSize: ".76rem", color: "var(--tx3)", marginTop: ".5rem" }}>{t("inspector.pinRowHint")}</div>
      </aside>
    );
  }
  if (sel.kind === "col" && info.type) {
    const cm = info.colMeta!;
    return (
      <aside className="det">
        <div className="det-head"><h3 style={{ fontSize: "1.2rem" }}>{info.type}</h3><div className="sub">{t("inspector.colPos", { c: info.displayCol + 1, n: project.types.length })}</div>
          <div className="det-meta">{ctag(cm.materialGroup, cm.color)}{cm.isIndexType && <span className="badge">{t("inspector.leadType")}</span>}{colFixed && fixedBadge}</div></div>
        <dl className="kv"><dt>{t("inspector.material")}</dt><dd>{cm.materialGroup}</dd><dt>{t("inspector.color")}</dt><dd className="mono">{cm.color}</dd></dl>
        {(() => { const pr = typePresence(project, project.types.indexOf(info.type)); return (
          <><div className="blk">{t("missing.presence")}</div><dl className="kv"><dt>{t("missing.present")}</dt><dd className="mono">{pr.present}</dd><dt>{t("missing.absent")}</dt><dd className="mono">{pr.absent}</dd><dt>{t("missing.badge")}</dt><dd className="mono">{pr.missing}</dd></dl></>
        ); })()}
        <div className="blk">{t("inspector.action")}</div>
        <div style={{ display: "flex", gap: ".4rem", flexWrap: "wrap" }}>{pinBtn("col", sel.col, colFixed, t("inspector.type"))}</div>
        <div style={{ fontSize: ".76rem", color: "var(--tx3)", marginTop: ".5rem" }}>{t("inspector.pinColHint")}</div>
      </aside>
    );
  }
  return <aside className="det"><div className="det-empty">—</div></aside>;
}
