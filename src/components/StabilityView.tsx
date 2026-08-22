import { useEffect, useRef, useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import type { StabilityResult } from "../analysis/bootstrap.js";
import { compute } from "../workers/client.js";
import { setStability, getStability } from "../analysis/stabilityStore.js";
import { useLink, activeCtx, useThemeTick } from "../link.js";
import { useT } from "../i18n/I18nContext.js";
import { readPlotColors } from "../core/theme.js";

export function StabilityView({ project }: { project: ProjectV2 }) {
  const [result, setResult] = useState<StabilityResult | null>(() => getStability(project) ?? null);
  const [reps, setReps] = useState(200);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1
  const t = useT();

  function run() {
    setBusy(true); setProgress(0);
    compute
      .bootstrap(project, { replicates: reps, seed: 12345, onProgress: (done, total) => setProgress(done / total) })
      .then((res) => { setStability(project, res); setResult(res); })
      .catch((e) => { if ((e as Error)?.name !== "AbortError") console.error(e); })
      .finally(() => setBusy(false));
  }

  function cancel() { compute.cancel(); setBusy(false); }

  // Laufenden Lauf bei Projektwechsel / Verlassen der Ansicht abbrechen.
  useEffect(() => () => compute.cancel(), [project]);

  return (
    <>
      <div className="sec-hdr">
        <h2>{t("stability.title")}</h2>
        <span className="hint">{t("stability.hintTop")}</span>
        {result && <span className="badge">{t("stability.badge", { v: result.globalStability.toFixed(3) })}</span>}
      </div>
      <div className="frow">
        <select className="sl" value={reps} onChange={(e) => setReps(+e.target.value)} disabled={busy}>
          <option value={100}>{t("stability.reps", { n: 100 })}</option>
          <option value={200}>{t("stability.reps", { n: 200 })}</option>
          <option value={500}>{t("stability.reps", { n: 500 })}</option>
        </select>
        <button className="btn" onClick={run} disabled={busy}>{busy ? t("stability.computing") : result ? t("stability.recompute") : t("stability.run")}</button>
        {busy && <button className="btn btn-ghost" onClick={cancel}>{t("stability.cancel")}</button>}
        {busy && (
          <span className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <i style={{ width: `${Math.round(progress * 100)}%` }} />
            <em>{Math.round(progress * 100)}&nbsp;%</em>
          </span>
        )}
        {!busy && result && <span className="tb-label">{t("stability.summary", { n: project.contexts.length, r: result.replicates })}</span>}
      </div>
      {result ? <Caterpillar project={project} result={result} />
        : <div className="placeholder">{t("stability.emptyA")}<br />{t("stability.emptyB")}</div>}
    </>
  );
}

function Caterpillar({ project, result }: { project: ProjectV2; result: StabilityResult }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const themeTick = useThemeTick();
  const t = useT();
  const wrap = useRef<HTMLDivElement>(null);
  const link = useLink();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  useEffect(() => {
    const cv = ref.current, w = wrap.current; if (!cv || !w) return;
    const g = cv.getContext("2d")!;
    const N = result.rows.length;
    let rowY: number[] = [];

    const draw = () => {
      const pc = readPlotColors();
      const W = w.clientWidth, H = w.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const LX = 92, TY = 16, BX = 24, rh = (H - TY - 28) / N;
      const X = (rank: number) => LX + (rank / Math.max(1, N - 1)) * (W - LX - BX);
      rowY = [];
      const aCtx = activeCtx(link);
      // X-Achse
      g.strokeStyle = pc.grid; g.lineWidth = 1;
      g.beginPath(); g.moveTo(LX, H - 24); g.lineTo(W - BX, H - 24); g.stroke();
      g.fillStyle = pc.dim; g.font = "10px Outfit"; g.textAlign = "center";
      g.fillText(t("stability.axis"), (LX + W - BX) / 2, H - 8);
      g.textAlign = "left";

      result.rows.forEach((r, i) => {
        const y = TY + i * rh + rh / 2; rowY.push(y);
        const width = r.hi - r.lo;
        const stab = 1 - width / Math.max(1, N - 1);
        const col = stab > 0.85 ? "#3f7a4f" : stab > 0.6 ? "#b5892a" : "#b23a2a";
        const hot = r.context === aCtx;
        if (hot) { g.fillStyle = "rgba(210,38,48,.08)"; g.fillRect(0, y - rh / 2, W, rh); }
        // Label
        g.fillStyle = hot ? pc.active : pc.label; g.font = '500 9px "JetBrains Mono",monospace'; g.textBaseline = "middle";
        g.fillText(clip(r.context, 12), 8, y);
        // Intervall lo–hi
        g.strokeStyle = col; g.lineWidth = hot ? 3 : 2; g.beginPath(); g.moveTo(X(r.lo), y); g.lineTo(X(r.hi), y); g.stroke();
        // Whisker-Enden
        g.beginPath(); g.moveTo(X(r.lo), y - 3); g.lineTo(X(r.lo), y + 3); g.moveTo(X(r.hi), y - 3); g.lineTo(X(r.hi), y + 3); g.stroke();
        // Median
        g.fillStyle = col; g.beginPath(); g.arc(X(r.median), y, hot ? 4 : 3, 0, 7); g.fill();
        // Referenzrang (Kreuz)
        g.strokeStyle = pc.text; g.lineWidth = 1; const rx = X(r.refRank);
        g.beginPath(); g.moveTo(rx, y - 4); g.lineTo(rx, y + 4); g.stroke();
      });
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(w);
    const onMove = (ev: MouseEvent) => {
      const rect = cv.getBoundingClientRect(); const my = ev.clientY - rect.top;
      let best = -1, bd = 1e9; rowY.forEach((y, i) => { const d = Math.abs(y - my); if (d < bd) { bd = d; best = i; } });
      if (best >= 0 && bd < 14) link.setHover(result.rows[best].context, null); else link.clearHover();
    };
    const onLeave = () => link.clearHover();
    cv.addEventListener("mousemove", onMove); cv.addEventListener("mouseleave", onLeave);
    return () => { ro.disconnect(); cv.removeEventListener("mousemove", onMove); cv.removeEventListener("mouseleave", onLeave); };
  }, [project, result, link, dpr, themeTick, t]);

  return (
    <div className="mx-grid" style={{ gridTemplateColumns: "1fr 210px" }}>
      <div className="mx-wrap" ref={wrap}><canvas ref={ref} style={{ position: "absolute", inset: 0 }} /></div>
      <aside className="det">
        <div className="blk" style={{ marginTop: 0 }}>{t("stability.readAid")}</div>
        <div style={{ fontSize: ".78rem", color: "var(--tx2)", lineHeight: 1.6 }}>
          <p style={{ marginBottom: ".5rem" }}>{t("stability.legendBars")}</p>
          <p style={{ marginBottom: ".5rem" }}><span style={{ color: "#3f7a4f" }}>■</span> {t("stability.legend.tight")} · <span style={{ color: "#b5892a" }}>■</span> {t("stability.legend.medium")} · <span style={{ color: "#b23a2a" }}>■</span> {t("stability.legend.uncertain")}.</p>
          <p>{t("stability.legendIntervals")}</p>
        </div>
      </aside>
    </div>
  );
}

function clip(s: string, n: number): string { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
