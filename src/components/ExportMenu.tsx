import { useEffect, useRef, useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import { buildMatrixScene, sceneToSVG, sceneToPNG, sceneToPDF } from "../export/exportImage.js";
import { toCSVForDownload, toXLSX } from "../export/exportTable.js";
import { toProjectJSONv2, toProjectJSONv1 } from "../export/exportProject.js";
import { toTurtle, toJSONLD } from "../export/exportRDF.js";
import { generateMethods } from "../analysis/methods.js";
import { getStability } from "../analysis/stabilityStore.js";
import { downloadText, downloadBytes, downloadBlob, safeFilename } from "../export/download.js";
import { useT } from "../i18n/I18nContext.js";

export function ExportMenu({ project }: { project: ProjectV2 }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  // v1.0: kurzes Erfolgs-Feedback nach dem Export (vorher schloss sich nur das Menü)
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(toastTimer.current), []);
  function showToast(msg: string) {
    setToast(msg);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 2200);
  }
  const ref = useRef<HTMLDivElement>(null);
  const base = safeFilename(project.name);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  async function run(kind: string) {
    setBusy(true);
    try {
      switch (kind) {
        case "png": { const scene = buildMatrixScene(project); const blob = await sceneToPNG(scene, 2); downloadBlob(`${base}_matrix.png`, blob); break; }
        case "svg": downloadText(`${base}_matrix.svg`, sceneToSVG(buildMatrixScene(project)), "image/svg+xml"); break;
        case "pdf": downloadBytes(`${base}_matrix.pdf`, sceneToPDF(buildMatrixScene(project)), "application/pdf"); break;
        case "csv": downloadText(`${base}_seriation.csv`, toCSVForDownload(project), "text/csv"); break;
        case "xlsx": downloadBytes(`${base}_seriation.xlsx`, await toXLSX(project), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); break;
        case "json2": downloadText(`${base}.combitab.json`, toProjectJSONv2(project), "application/json"); break;
        case "json1": downloadText(`${base}.v1.json`, toProjectJSONv1(project), "application/json"); break;
        case "methods": downloadText(`${base}_methods.md`, generateMethods(project, { stability: getStability(project) }), "text/markdown"); break;
        case "ttl": downloadText(`${base}.ttl`, toTurtle(project), "text/turtle"); break;
        case "jsonld": downloadText(`${base}.jsonld`, toJSONLD(project), "application/ld+json"); break;
      }
      showToast(t("export.done"));
    } catch (err) {
      showToast(t("export.failed") + (err instanceof Error && err.message ? ": " + err.message : ""));
    } finally { setBusy(false); setOpen(false); }
  }

  const item = (kind: string, label: string, sub: string) => (
    <button className="exp-item" onClick={() => run(kind)} disabled={busy}>
      <span>{label}</span><span className="exp-sub">{sub}</span>
    </button>
  );

  return (
    <div className="exp" ref={ref}>
      {toast && <div className="toast" role="status">{toast}</div>}
      <button className="file-btn" onClick={() => setOpen((o) => !o)} disabled={busy}>{busy ? t("export.busy") : t("export.button")}</button>
      {open && (
        <div className="exp-menu">
          <div className="exp-grp">{t("export.grpImage")}</div>
          {item("png", "PNG", t("export.pngSub"))}
          {item("svg", "SVG", t("export.vector"))}
          {item("pdf", "PDF", t("export.vector"))}
          <div className="exp-grp">{t("export.grpData")}</div>
          {item("csv", "CSV", t("export.csvSub"))}
          {item("xlsx", "XLSX", t("export.xlsxSub"))}
          <div className="exp-grp">{t("export.grpProject")}</div>
          {item("json2", t("export.projFile"), t("export.projV2Sub"))}
          {item("json1", t("export.projV1"), t("export.projV1Sub"))}
          <div className="exp-grp">{t("export.grpScience")}</div>
          {item("methods", t("export.methodsLabel"), t("export.methodsSub"))}
          <div className="exp-grp">{t("export.grpLod")}</div>
          {item("ttl", "Turtle", t("export.ttlSub"))}
          {item("jsonld", "JSON-LD", t("export.jsonldSub"))}
        </div>
      )}
    </div>
  );
}
