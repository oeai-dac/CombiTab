import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectV2 } from "../core/model.js";
import type { CAResult } from "../analysis/ca.js";
import { compute } from "../workers/client.js";
import { useLink, activeCtx, activeType, useThemeTick } from "../link.js";
import { readPlotColors } from "../core/theme.js";
import { useT } from "../i18n/I18nContext.js";

export function CAView({ project }: { project: ProjectV2 }) {
  const [ca, setCa] = useState<CAResult | null>(null);
  const [dimX, setDimX] = useState(0);
  const [dimY, setDimY] = useState(1);
  const t = useT();
  const tooSmall = project.contexts.length < 3 || project.types.length < 3;
  useEffect(() => {
    if (tooSmall) { setCa(null); return; }
    let alive = true;
    setCa(null);
    compute.ca(project, 5)
      .then((r) => { if (alive) setCa(r); })
      .catch((e) => { if ((e as Error)?.name !== "AbortError") console.error(e); });
    return () => { alive = false; };
  }, [project, tooSmall]);

  const k = ca?.k ?? 0;
  const dx = Math.min(dimX, Math.max(0, k - 1));
  const dy = Math.min(dimY, Math.max(0, k - 1));
  const dimOpts = () => Array.from({ length: k }, (_, d) => d);

  return (
    <>
      <div className="sec-hdr">
        <h2>{t("ca.title")}</h2>
        <span className="hint">{t("ca.hint")}</span>
        <span className="badge">{t("ca.exactSvd")}</span>
      </div>
      {tooSmall ? (
        <div className="placeholder">{t("ca.tooSmall")}</div>
      ) : ca ? (
        <>
          <div className="frow">
            <span className="tb-label">{t("ca.axes")}</span>
            <span className="tb-label">X</span>
            <select className="sl" value={dx} onChange={(e) => setDimX(+e.target.value)}>
              {dimOpts().map((d) => <option key={d} value={d}>{t("ca.dimOpt", { n: d + 1, pct: (ca.inertiaPct[d] * 100).toFixed(1) })}</option>)}
            </select>
            <span className="tb-label">Y</span>
            <select className="sl" value={dy} onChange={(e) => setDimY(+e.target.value)}>
              {dimOpts().map((d) => <option key={d} value={d}>{t("ca.dimOpt", { n: d + 1, pct: (ca.inertiaPct[d] * 100).toFixed(1) })}</option>)}
            </select>
            {dx === dy && <span className="tb-label" style={{ color: "var(--accent2)" }}>{t("ca.sameAxis")}</span>}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: ".9rem", flex: 1, overflow: "hidden", paddingBottom: "1rem", minHeight: 0 }}>
            <Biplot project={project} ca={ca} dimX={dx} dimY={dy} />
            <Scree ca={ca} dimX={dx} dimY={dy} />
          </div>
        </>
      ) : (
        <div className="placeholder">{t("ca.loading")}</div>
      )}
    </>
  );
}

type CA = CAResult;

function Biplot({ project, ca, dimX, dimY }: { project: ProjectV2; ca: CA; dimX: number; dimY: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const themeTick = useThemeTick();
  const t = useT();
  const link = useLink();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  // Punktliste (kanonisch) + Bildschirm-Cache für Hit-Test
  const pts = useMemo(() => {
    const arr: Array<{ name: string; kind: "ctx" | "type"; x: number; y: number; color: string }> = [];
    if (ca.k >= 1) {
      const dx = Math.min(dimX, ca.k - 1), dy = Math.min(dimY, ca.k - 1);
      project.contexts.forEach((c, i) => arr.push({ name: c, kind: "ctx", x: ca.rowCoords[i][dx], y: ca.rowCoords[i][dy] ?? 0, color: "#d22630" }));
      project.types.forEach((t, j) => arr.push({ name: t, kind: "type", x: ca.colCoords[j][dx], y: ca.colCoords[j][dy] ?? 0, color: project.columnMetadata[t]?.color ?? "#808080" }));
    }
    return arr;
  }, [project, ca, dimX, dimY]);

  useEffect(() => {
    const cv = ref.current, w = wrap.current; if (!cv || !w) return;
    const g = cv.getContext("2d")!;
    let screen: Array<{ sx: number; sy: number; p: (typeof pts)[number] }> = [];

    const draw = () => {
      const pc = readPlotColors();
      const W = w.clientWidth, H = w.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      if (!pts.length) { g.fillStyle = pc.dim; g.font = '14px Outfit'; g.fillText(t("ca.none"), 20, 30); return; }
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
      const pad = 46; const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const sx = (v: number) => pad + ((v - minX) / (maxX - minX || 1)) * (W - 2 * pad);
      const sy = (v: number) => H - pad - ((v - minY) / (maxY - minY || 1)) * (H - 2 * pad);
      // Achsen (Nulllinien)
      g.strokeStyle = pc.grid; g.lineWidth = 1; g.beginPath();
      g.moveTo(sx(0), pad - 10); g.lineTo(sx(0), H - pad + 10); g.moveTo(pad - 10, sy(0)); g.lineTo(W - pad + 10, sy(0)); g.stroke();
      g.fillStyle = pc.dim; g.font = '600 10px Outfit';
      g.fillText(t("ca.dimOpt", { n: dimX + 1, pct: (ca.inertiaPct[dimX] * 100).toFixed(1) }), W - pad - 92, sy(0) - 6);
      g.save(); g.translate(sx(0) + 6, pad + 4); g.fillText(t("ca.dimOpt", { n: dimY + 1, pct: ((ca.inertiaPct[dimY] ?? 0) * 100).toFixed(1) }), 0, 0); g.restore();

      const aCtx = activeCtx(link), aType = activeType(link);
      screen = [];
      // Typen zuerst (kleiner), dann Kontexte
      for (const p of pts) {
        const X = sx(p.x), Y = sy(p.y); screen.push({ sx: X, sy: Y, p });
        const hot = (p.kind === "ctx" && p.name === aCtx) || (p.kind === "type" && p.name === aType);
        g.globalAlpha = p.kind === "type" ? (hot ? 1 : 0.7) : 1;
        g.fillStyle = p.color;
        g.beginPath(); g.arc(X, Y, hot ? 6.5 : p.kind === "ctx" ? 4.5 : 3.2, 0, 7); g.fill();
        if (hot) { g.globalAlpha = 1; g.strokeStyle = pc.text; g.lineWidth = 1.5; g.stroke(); }
      }
      g.globalAlpha = 1;
      // Labels für aktive Punkte
      g.font = '600 11px "JetBrains Mono", monospace'; g.fillStyle = pc.text;
      for (const s of screen) {
        const hot = (s.p.kind === "ctx" && s.p.name === aCtx) || (s.p.kind === "type" && s.p.name === aType);
        if (hot) g.fillText(s.p.name, s.sx + 9, s.sy + 3);
      }
      (cv as unknown as { _screen: typeof screen })._screen = screen;
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(w);

    const pick = (ev: MouseEvent) => {
      const rect = cv.getBoundingClientRect(); const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const sc = (cv as unknown as { _screen: typeof screen })._screen || [];
      let best: (typeof sc)[number] | null = null, bd = 12 * 12;
      for (const s of sc) { const d = (s.sx - mx) ** 2 + (s.sy - my) ** 2; if (d < bd) { bd = d; best = s; } }
      return best;
    };
    const onMove = (ev: MouseEvent) => { const b = pick(ev); if (b) link.setHover(b.p.kind === "ctx" ? b.p.name : null, b.p.kind === "type" ? b.p.name : null); else link.clearHover(); };
    const onLeave = () => link.clearHover();
    const onClick = (ev: MouseEvent) => { const b = pick(ev); if (b) link.setSel(b.p.kind === "ctx" ? b.p.name : link.selCtx, b.p.kind === "type" ? b.p.name : link.selType); };
    cv.addEventListener("mousemove", onMove); cv.addEventListener("mouseleave", onLeave); cv.addEventListener("click", onClick);
    return () => { ro.disconnect(); cv.removeEventListener("mousemove", onMove); cv.removeEventListener("mouseleave", onLeave); cv.removeEventListener("click", onClick); };
  }, [pts, ca, link, dpr, dimX, dimY, themeTick, t]);

  return <div className="mx-wrap" ref={wrap}><canvas ref={ref} style={{ position: "absolute", inset: 0 }} /></div>;
}

function Scree({ ca, dimX, dimY }: { ca: CA; dimX: number; dimY: number }) {
  const t = useT();
  return (
    <aside className="det">
      <div className="blk" style={{ marginTop: 0 }}>{t("ca.explainedInertia")}</div>
      <div style={{ fontSize: ".76rem", color: "var(--tx3)", marginBottom: ".6rem" }}>{t("ca.totalInertia", { v: ca.totalInertia.toFixed(4) })}</div>
      {ca.inertiaPct.map((pct, d) => {
        const axis = d === dimX ? "X" : d === dimY ? "Y" : null;
        return (
          <div key={d} style={{ marginBottom: ".5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: ".74rem", marginBottom: ".15rem" }}>
              <span style={{ fontWeight: axis ? 600 : 400 }}>{t("ca.dimN", { n: d + 1 })}{axis && <span style={{ color: "var(--accent2)", marginLeft: ".3rem" }}>◂ {axis}</span>}</span>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "var(--tx2)" }}>{(pct * 100).toFixed(1)} %</span>
            </div>
            <div style={{ height: 10, background: "var(--bg4)", borderRadius: 3 }}>
              <div style={{ width: `${pct * 100}%`, height: "100%", background: axis ? "var(--accent)" : "var(--bd2)", borderRadius: 3 }} />
            </div>
          </div>
        );
      })}
      <div style={{ fontSize: ".72rem", color: "var(--tx3)", marginTop: ".8rem" }}>{t("ca.screeHint")}</div>
    </aside>
  );
}
