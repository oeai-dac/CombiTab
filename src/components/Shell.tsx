import { useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import { ExportMenu } from "./ExportMenu.js";
import { currentTheme, toggleTheme, type Theme } from "../core/theme.js";
import { useI18n, useT } from "../i18n/I18nContext.js";
import type { Lang } from "../i18n/i18n.js";

function ThemeToggle() {
  const t = useT();
  const [theme, setTheme] = useState<Theme>(currentTheme());
  const dark = theme === "dark";
  return (
    <button
      className="sl theme-toggle"
      onClick={() => setTheme(toggleTheme())}
      aria-label={dark ? t("header.theme.toLight") : t("header.theme.toDark")}
      title={dark ? t("header.theme.light") : t("header.theme.dark")}
      aria-pressed={dark}
    >
      <span aria-hidden="true">{dark ? "☀" : "☾"}</span>
    </button>
  );
}

function LangToggle() {
  const { lang, setLang, t } = useI18n();
  const next: Lang = lang === "de" ? "en" : "de";
  return (
    <button
      className="sl lang-toggle"
      onClick={() => setLang(next)}
      aria-label={next === "en" ? t("header.lang.toEnglish") : t("header.lang.toGerman")}
      title={t("header.lang.label")}
    >
      <span aria-hidden="true">{lang.toUpperCase()}</span>
    </button>
  );
}

const LOGO = (
  <svg viewBox="0 0 40 40" aria-hidden="true">
    <rect x="1" y="1" width="11" height="11" rx="2" fill="#d22630" />
    <rect x="14" y="1" width="11" height="11" rx="2" fill="#d22630" opacity=".5" />
    <rect x="27" y="1" width="11" height="11" rx="2" fill="#e2dfda" />
    <rect x="1" y="14" width="11" height="11" rx="2" fill="#d22630" opacity=".5" />
    <rect x="14" y="14" width="11" height="11" rx="2" fill="#d22630" />
    <rect x="27" y="14" width="11" height="11" rx="2" fill="#d22630" opacity=".5" />
    <rect x="1" y="27" width="11" height="11" rx="2" fill="#e2dfda" />
    <rect x="14" y="27" width="11" height="11" rx="2" fill="#d22630" opacity=".5" />
    <rect x="27" y="27" width="11" height="11" rx="2" fill="#d22630" />
  </svg>
);

export type TabId = "matrix" | "ca" | "ford" | "stability" | "meta";
const TABS: Array<{ id: TabId; labelKey: string; icon: string }> = [
  { id: "matrix", labelKey: "tab.matrix", icon: "▦" },
  { id: "ca", labelKey: "tab.ca", icon: "◆" },
  { id: "ford", labelKey: "tab.ford", icon: "⧫" },
  { id: "stability", labelKey: "tab.stability", icon: "⁛" },
  { id: "meta", labelKey: "tab.meta", icon: "⌥" },
];

export function Shell({ project, tab, onTab, onPickFile, onShare }: {
  project: ProjectV2 | null; tab: TabId; onTab: (t: TabId) => void; onPickFile: () => void; onShare?: () => void;
}) {
  const { lang, t } = useI18n();
  const locale = lang === "de" ? "de-AT" : "en-GB";
  return (
    <>
      <header className="hdr">
        <div className="hdr-logo">{LOGO}</div>
        <div className="hdr-title">
          <h1>CombiTab{project ? ` · ${project.name}` : ""}</h1>
          <p>{t("app.subtitle")}</p>
        </div>
        <span className="badge p2">v2.0</span>
        {project && <span className="badge">{project.contexts.length} × {project.types.length}</span>}
        {project && <ExportMenu project={project} />}
        {project && onShare && <button className="sl share-btn" onClick={onShare} title={t("share.title")} aria-label={t("share.button")}><span aria-hidden="true">⇗</span> {t("share.button")}</button>}
        <LangToggle />
        <ThemeToggle />
        <button className="file-btn" onClick={onPickFile}>{t("header.loadFile")}</button>
      </header>
      <nav className="tabs" role="tablist" aria-label={t("a11y.tabsLabel")}>
        {TABS.map((tb) => (
          <button key={tb.id} id={`tab-${tb.id}`} className={"tab" + (tab === tb.id ? " on" : "")} onClick={() => onTab(tb.id)}
            role="tab" aria-selected={tab === tb.id} aria-controls="main" tabIndex={tab === tb.id ? 0 : -1}
            onKeyDown={(e) => {
              const i = TABS.findIndex((x) => x.id === tab);
              let ni = -1;
              if (e.key === "ArrowRight") ni = (i + 1) % TABS.length;
              else if (e.key === "ArrowLeft") ni = (i - 1 + TABS.length) % TABS.length;
              else if (e.key === "Home") ni = 0;
              else if (e.key === "End") ni = TABS.length - 1;
              if (ni >= 0) { e.preventDefault(); const id = TABS[ni].id; onTab(id); requestAnimationFrame(() => document.getElementById(`tab-${id}`)?.focus()); }
            }}>
            <span aria-hidden="true">{tb.icon}</span> {t(tb.labelKey)}
            {tb.id === "matrix" && project && <span className="tab-cnt">{(project.contexts.length * project.types.length).toLocaleString(locale)}</span>}
            {tb.id === "ca" && project && <span className="tab-cnt">{project.types.length}</span>}
          </button>
        ))}
      </nav>
    </>
  );
}
