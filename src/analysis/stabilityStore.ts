/** Schlichter Cache des letzten Bootstrap-Ergebnisses, damit der Methods-Export
 *  es einbeziehen kann. Schlüssel = Projektname + Dimensionen. */
import type { StabilityResult } from "./bootstrap.js";
import type { ProjectV2 } from "../core/model.js";

let last: { key: string; result: StabilityResult } | null = null;
const keyOf = (p: ProjectV2) => `${p.name}|${p.contexts.length}x${p.types.length}`;

export function setStability(p: ProjectV2, result: StabilityResult): void { last = { key: keyOf(p), result }; }
export function getStability(p: ProjectV2): StabilityResult | undefined { return last && last.key === keyOf(p) ? last.result : undefined; }
