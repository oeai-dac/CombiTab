import { useEffect, useMemo, useRef, useState } from "react";
import "./theme.css";
import type { ProjectV2, FilterSettings } from "./core/model.js";
import { migrateV1, type ProjectV1 } from "./core/io/migrateV1.js";
import { importCSV } from "./core/io/importTable.js";
import { importXLSX } from "./core/io/importXLSX.js";
import { SAMPLE_CSV } from "./data/sample.js";
import { Shell, type TabId } from "./components/Shell.js";
import { MatrixView } from "./components/MatrixView.js";
import { CAView } from "./components/CAView.js";
import { FordView } from "./components/FordView.js";
import { MetaView } from "./components/MetaView.js";
import { StabilityView } from "./components/StabilityView.js";
import { LinkProvider, useLink } from "./link.js";
import { filterProject, filtersActive, emptyFilters } from "./core/filter.js";
import { pruneMaterialFilter } from "./core/materialGroups.js";
import { saveAutosave, loadAutosave, clearAutosave, type AutosaveRecord } from "./core/autosave.js";
import { buildShareUrl, readShareFromHash } from "./core/shareLink.js";
import { useI18n } from "./i18n/I18nContext.js";

export default function App() {
  const { t, lang } = useI18n();
  const [project, setProject] = useState<ProjectV2 | null>(null);
  const [tab, setTab] = useState<TabId>("matrix");
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [restore, setRestore] = useState<AutosaveRecord | null>(null);
  const [installEvt, setInstallEvt] = useState<{ prompt: () => Promise<unknown> } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const projectRef = useRef<ProjectV2 | null>(null); projectRef.current = project;
  const dirtyRef = useRef(false);
  const pristineRef = useRef<string | null>(null);
  const DEMO_NAME = "Early-Medieval-Cemetery-Sample";

  // Test-/Debug-Hook: aktuelles Projekt
  useEffect(() => { if (import.meta.env.DEV) (globalThis as unknown as { __project?: unknown }).__project = project; }, [project]);

  // Initialisierung: ein geteilter Link (§9.8) hat Vorrang vor Autosave und Demo.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const shared = await readShareFromHash(location.hash);
      if (cancelled) return;
      if (shared) {
        dirtyRef.current = true;
        setProject(shared.project);
        if (shared.ui?.tab) setTab(shared.ui.tab as TabId);
        return; // kein Demo, kein Wiederherstellungs-Banner
      }
      try { const { project } = importCSV(SAMPLE_CSV, { name: DEMO_NAME }); pristineRef.current = JSON.stringify(project); if (!cancelled) setProject(project); }
      catch (e) { if (!cancelled) setError(String(e)); }
      const rec = await loadAutosave();
      if (!cancelled && rec) setRestore(rec);
    })();
    return () => { cancelled = true; };
  }, []);

  // Aktuelle Sitzung sichern — beim Verlassen/Ausblenden der Seite; das unberührte
  // Demo wird übersprungen, ein bearbeitetes oder importiertes Projekt gesichert.
  useEffect(() => {
    const save = () => {
      const p = projectRef.current; if (!p) return;
      if (p.name === DEMO_NAME && !dirtyRef.current && JSON.stringify(p) === pristineRef.current) return;
      void saveAutosave(p);
    };
    const onVis = () => { if (document.visibilityState === "hidden") save(); };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("pagehide", save);
    return () => { document.removeEventListener("visibilitychange", onVis); window.removeEventListener("pagehide", save); };
  }, []);

  // Installations-Aufforderung abfangen (PWA)
  useEffect(() => {
    const onPrompt = (e: Event) => { e.preventDefault(); setInstallEvt(e as unknown as { prompt: () => Promise<unknown> }); };
    window.addEventListener("beforeinstallprompt", onPrompt);
    const onInstalled = () => setInstallEvt(null);
    window.addEventListener("appinstalled", onInstalled);
    return () => { window.removeEventListener("beforeinstallprompt", onPrompt); window.removeEventListener("appinstalled", onInstalled); };
  }, []);

  function doRestore() { if (restore) { dirtyRef.current = true; setProject(restore.project); setTab("matrix"); setRestore(null); } }
  function dismissRestore() { setRestore(null); void clearAutosave(); }

  // Teilbarer Link (§9.8)
  const [toast, setToast] = useState<string | null>(null);
  useEffect(() => { if (!toast) return; const id = window.setTimeout(() => setToast(null), 3500); return () => window.clearTimeout(id); }, [toast]);
  async function share() {
    const p = projectRef.current; if (!p) return;
    const { url, tooLong } = await buildShareUrl({ project: p, ui: { tab } }, location.origin, location.pathname);
    if (tooLong) { setToast(t("share.tooLarge")); return; }
    const frag = url.slice(url.indexOf("#"));
    try { await navigator.clipboard.writeText(url); setToast(t("share.copied")); }
    catch { location.hash = frag.slice(1); setToast(t("share.copyManual")); }
  }

  async function loadFile(file: File) {
    setError(null);
    try {
      const name = file.name.replace(/\.[^.]+$/, "");
      if (/\.json$/i.test(file.name)) {
        const raw = JSON.parse(await file.text());
        const p: ProjectV2 = raw.schemaVersion === 2 ? raw : migrateV1(raw as ProjectV1);
        setProject(p);
      } else if (/\.(xlsx|xls)$/i.test(file.name)) {
        setProject((await importXLSX(await file.arrayBuffer(), { name })).project);
      } else {
        setProject(importCSV(await file.text(), { name }).project);
      }
      setTab("matrix");
      dirtyRef.current = true;
    } catch (e) { setError(`Import fehlgeschlagen: ${e instanceof Error ? e.message : String(e)}`); }
  }

  return (
    <LinkProvider>
    <a className="skip-link" href="#main">{t("a11y.skip")}</a>
    <div className="app"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) loadFile(f); }}>
      <Shell project={project} tab={tab} onTab={setTab} onPickFile={() => fileInput.current?.click()} onShare={share} />
      <input ref={fileInput} type="file" accept=".json,.csv,.tsv,.xlsx,.xls" style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); e.target.value = ""; }} />
      <main id="main" className="content" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {restore && (
          <div className="restore-bar" role="region" aria-label={t("autosave.title")}>
            <span>{t("autosave.restoreQ", { name: restore.name, time: new Date(restore.savedAt).toLocaleString(lang === "de" ? "de-AT" : "en-GB") })}</span>
            <div className="restore-actions">
              <button className="btn" onClick={doRestore}>{t("autosave.restore")}</button>
              <button className="btn btn-ghost" onClick={dismissRestore}>{t("autosave.dismiss")}</button>
            </div>
          </div>
        )}
        {error && <div className="placeholder" role="alert" style={{ color: "var(--accent2)", flex: "initial", marginBottom: ".8rem" }}>{error}</div>}
        {!project ? <div className="placeholder">{t("app.loading")}</div> : <Workspace project={project} tab={tab} />}
      </main>
      {installEvt && (
        <button className="install-chip" onClick={() => { void installEvt.prompt(); setInstallEvt(null); }}>
          <span aria-hidden="true">⬇</span> {t("pwa.install")}
        </button>
      )}
      {dragging && <div className="drop-overlay">{t("app.drop")}</div>}
      {toast && <div className="toast" role="status" aria-live="polite">{toast}</div>}
    </div>
    </LinkProvider>
  );
}

/** Innerhalb des LinkProviders: hält Filter-/Fokus-Zustand, leitet die gefilterte
 *  Sicht an alle Ansichten weiter. Der Fokus richtet sich nach der aktuellen
 *  verlinkten Auswahl. Die gefilterte Sicht ist read-only (Bearbeiten pausiert). */
function Workspace({ project, tab }: { project: ProjectV2; tab: TabId }) {
  const link = useLink();
  const [filters, setFilters] = useState<FilterSettings>(() => project.filters ?? emptyFilters());
  const [focusOn, setFocusOn] = useState(false);
  useEffect(() => { setFilters(project.filters ?? emptyFilters()); setFocusOn(false); }, [project]);
  // Filterzustand ins Projekt zurückschreiben, damit er — wie in §9.8 zugesagt —
  // im Teilen-Link, Autosave und Projekt-Export enthalten ist. Nur bei tatsächlicher
  // Änderung, um den Pristine-Vergleich der Demo nicht zu stören.
  useEffect(() => {
    if (JSON.stringify(project.filters ?? emptyFilters()) !== JSON.stringify(filters)) project.filters = filters;
  }, [project, filters]);
  // Materialgruppen können in der Metadaten-Ansicht angelegt, umbenannt oder
  // gelöscht werden. Ein Filter auf eine inzwischen entfernte Gruppe würde die
  // Matrix leerfiltern, ohne dass es dafür noch einen abwählbaren Chip gäbe —
  // deshalb wird der Materialfilter beim Ansichtswechsel gegen die aktuellen
  // Gruppen abgeglichen.
  useEffect(() => {
    setFilters((f) => {
      if (!f.materials.length) return f;
      const kept = pruneMaterialFilter(f.materials, project);
      return kept.length === f.materials.length ? f : { ...f, materials: kept };
    });
  }, [tab, project]);

  const selCtx = link.selCtx, selType = link.selType;
  const deriveView = () => {
    const focusSel = focusOn ? { ctx: selCtx, type: selType } : null;
    return filtersActive(filters, focusSel) ? filterProject(project, filters, focusSel) : project;
  };
  // Matrix-Sicht bleibt beim Editieren stabil (der Renderer aktualisiert sie
  // in-place; Rück-Schreiben pflegt das Grundprojekt). Analyse-Tabs leiten frisch
  // ab, damit sie nach Tab-Wechsel die zurückgeschriebenen Änderungen zeigen.
  const matrixView = useMemo(deriveView, [project, filters, focusOn, selCtx, selType]);
  const active = matrixView !== project;
  const k = project.name + project.contexts.length;

  switch (tab) {
    case "matrix":
      return <MatrixView key={"m" + k} project={matrixView} baseProject={project} filters={filters} onFilters={setFilters}
        focusOn={focusOn} onFocusToggle={() => setFocusOn((o) => !o)} filterActive={active} />;
    case "ca": return <CAView key={"c" + k} project={deriveView()} />;
    case "ford": return <FordView key={"f" + k} project={deriveView()} />;
    case "stability": return <StabilityView key={"s" + k} project={deriveView()} />;
    default: return <MetaView key={"me" + k} project={project} />; // Metadaten immer am Grundprojekt
  }
}

