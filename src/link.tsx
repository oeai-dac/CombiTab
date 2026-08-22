import { createContext, useContext, useMemo, useState, useEffect, type ReactNode } from "react";
import { onThemeChange } from "./core/theme.js";

/** Erhöht sich bei jedem Hell/Dunkel-Wechsel — als useEffect-Dependency zum Neuzeichnen von Canvas-Plots. */
export function useThemeTick(): number {
  const [tick, setTick] = useState(0);
  useEffect(() => onThemeChange(() => setTick((t) => t + 1)), []);
  return tick;
}

/** Verlinkte Auswahl/Hover über alle Ansichten hinweg — nach kanonischer Identität (Name). */
export interface LinkState {
  hoverCtx: string | null; hoverType: string | null;
  selCtx: string | null; selType: string | null;
}
interface LinkApi extends LinkState {
  setHover: (ctx: string | null, type: string | null) => void;
  setSel: (ctx: string | null, type: string | null) => void;
  clearHover: () => void;
}
const Ctx = createContext<LinkApi | null>(null);

export function LinkProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<LinkState>({ hoverCtx: null, hoverType: null, selCtx: null, selType: null });
  const api = useMemo<LinkApi>(() => ({
    ...s,
    setHover: (hoverCtx, hoverType) => setS((p) => ({ ...p, hoverCtx, hoverType })),
    setSel: (selCtx, selType) => setS((p) => ({ ...p, selCtx, selType })),
    clearHover: () => setS((p) => ({ ...p, hoverCtx: null, hoverType: null })),
  }), [s]);
  (globalThis as unknown as { __link?: LinkApi }).__link = api;
  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useLink(): LinkApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useLink außerhalb von LinkProvider");
  return v;
}

/** Aktiv hervorgehobener Kontext/Typ (Auswahl hat Vorrang vor Hover). */
export function activeCtx(l: LinkState): string | null { return l.hoverCtx ?? l.selCtx; }
export function activeType(l: LinkState): string | null { return l.hoverType ?? l.selType; }
