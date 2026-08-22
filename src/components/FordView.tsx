import { useEffect, useMemo, useRef } from "react";
import type { ProjectV2 } from "../core/model.js";
import { useLink, activeCtx, activeType, useThemeTick } from "../link.js";
import { readPlotColors } from "../core/theme.js";
import { useT } from "../i18n/I18nContext.js";

const TOP_TYPES = 30;

export function FordView({ project }: { project: ProjectV2 }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const link = useLink();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const themeTick = useThemeTick();
  const t = useT();

  // Anzeige-Reihenfolge aus project.order; Top-Typen nach Gesamtsumme
  const model = useMemo(() => {
    const rIdx = new Map(project.contexts.map((c, i) => [c, i] as const));
    const cIdx = new Map(project.types.map((t, j) => [t, j] as const));
    const rowSeq = project.order.rows.map((c) => rIdx.get(c) ?? 0);
    const colSeqAll = project.order.cols.map((t) => cIdx.get(t) ?? 0);
    const totals = colSeqAll.map((j) => ({ j, s: project.contexts.reduce((a, _c, i) => a + project.matrix[i][j], 0) }));
    const top = new Set(totals.slice().sort((a, b) => b.s - a.s).slice(0, TOP_TYPES).map((x) => x.j));
    const colSeq = colSeqAll.filter((j) => top.has(j));
    const rowTot = rowSeq.map((i) => Math.max(1, project.types.reduce((a, _t, j) => a + project.matrix[i][j], 0)));
    return { rowSeq, colSeq, rowTot };
  }, [project]);

  useEffect(() => {
    const cv = ref.current, w = wrap.current; if (!cv || !w) return;
    const g = cv.getContext("2d")!;
    const LX = 88, TY = 128;
    let geom = { colW: 0, rh: 0 };

    const draw = () => {
      const pc = readPlotColors();
      const W = w.clientWidth, H = w.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const { rowSeq, colSeq, rowTot } = model;
      const colW = (W - LX - 16) / Math.max(1, colSeq.length), rh = (H - TY - 12) / Math.max(1, rowSeq.length);
      geom = { colW, rh };
      const aCtx = activeCtx(link), aType = activeType(link);

      // aktive Zeile/Spalte hervorheben
      colSeq.forEach((j, k) => { if (project.types[j] === aType) { g.fillStyle = "rgba(210,38,48,.08)"; g.fillRect(LX + k * colW, TY, colW, rowSeq.length * rh); } });
      rowSeq.forEach((i, r) => { if (project.contexts[i] === aCtx) { g.fillStyle = "rgba(210,38,48,.08)"; g.fillRect(LX, TY + r * rh, colSeq.length * colW, rh); } });

      // Spaltenköpfe (vertikal)
      g.font = '500 8px "JetBrains Mono",monospace';
      colSeq.forEach((j, k) => {
        const x = LX + k * colW + colW / 2; g.save(); g.translate(x + 3, TY - 8); g.rotate(-Math.PI / 2);
        g.fillStyle = project.types[j] === aType ? pc.active : pc.label; g.fillText(clip(project.types[j], 16), 0, 0); g.restore();
      });
      // Zeilen
      g.textBaseline = "middle";
      rowSeq.forEach((i, r) => {
        g.font = '500 9px "JetBrains Mono",monospace'; g.fillStyle = project.contexts[i] === aCtx ? pc.active : pc.label;
        g.fillText(clip(project.contexts[i], 11), 8, TY + r * rh + rh / 2);
        colSeq.forEach((j, k) => {
          const v = project.matrix[i][j]; if (!v) return;
          const frac = v / rowTot[r]; const bw = Math.max(2, frac * colW * 4.2);
          const x = LX + k * colW + colW / 2;
          g.fillStyle = withAlpha(project.columnMetadata[project.types[j]]?.color ?? "#808080", 0.35 + 0.6 * Math.min(1, frac * 3));
          g.fillRect(x - bw / 2, TY + r * rh + 2, bw, rh - 4);
        });
      });
      g.fillStyle = pc.dim; g.font = '600 10px Outfit';
      g.fillText(t("ford.axisTitle"), 8, 20);
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(w);

    const onMove = (ev: MouseEvent) => {
      const rect = cv.getBoundingClientRect(); const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const { rowSeq, colSeq } = model; const { colW, rh } = geom;
      const r = Math.floor((my - TY) / rh), k = Math.floor((mx - LX) / colW);
      const ctx = my >= TY && r >= 0 && r < rowSeq.length ? project.contexts[rowSeq[r]] : null;
      const typ = mx >= LX && k >= 0 && k < colSeq.length ? project.types[colSeq[k]] : null;
      if (ctx || typ) link.setHover(ctx, typ); else link.clearHover();
    };
    const onLeave = () => link.clearHover();
    const onClick = (ev: MouseEvent) => {
      const rect = cv.getBoundingClientRect(); const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const { rowSeq, colSeq } = model; const { colW, rh } = geom;
      const r = Math.floor((my - TY) / rh), k = Math.floor((mx - LX) / colW);
      const ctx = my >= TY && r >= 0 && r < rowSeq.length ? project.contexts[rowSeq[r]] : link.selCtx;
      const typ = mx >= LX && k >= 0 && k < colSeq.length ? project.types[colSeq[k]] : link.selType;
      link.setSel(ctx, typ);
    };
    cv.addEventListener("mousemove", onMove); cv.addEventListener("mouseleave", onLeave); cv.addEventListener("click", onClick);
    return () => { ro.disconnect(); cv.removeEventListener("mousemove", onMove); cv.removeEventListener("mouseleave", onLeave); cv.removeEventListener("click", onClick); };
  }, [model, link, project, dpr, themeTick, t]);

  return (
    <>
      <div className="sec-hdr">
        <h2>{t("ford.title")}</h2>
        <span className="hint">{t("ford.hint")}</span>
        <span className="badge">{model.colSeq.length} Typen · {model.rowSeq.length} Kontexte</span>
      </div>
      <div className="mx-wrap" ref={wrap} style={{ flex: 1, marginBottom: "1rem" }}><canvas ref={ref} style={{ position: "absolute", inset: 0 }} /></div>
    </>
  );
}

function clip(s: string, n: number): string { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function withAlpha(hex: string, a: number): string {
  const h = hex.replace("#", ""); const r = parseInt(h.slice(0, 2), 16) || 0, g = parseInt(h.slice(2, 4), 16) || 0, b = parseInt(h.slice(4, 6), 16) || 0;
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}
