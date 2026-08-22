/**
 * Zitierfähiger „Methods"-Absatz (Spezifikation §9, Provenienz).
 * Fasst Datensatz, Seriationsverfahren, Qualitätsmaße, CA-Trägheit, Fixierungen,
 * Annotationen und — falls vorhanden — die Bootstrap-Stabilität in DE und EN zusammen.
 */
import type { ProjectV2 } from "../core/model.js";
import { computeCA } from "./ca.js";
import { quality } from "../seriation/metrics.js";
import type { StabilityResult } from "./bootstrap.js";

const VERSION = "2.0";

export interface MethodsOptions { stability?: StabilityResult; }

export function generateMethods(p: ProjectV2, opts: MethodsOptions = {}): string {
  const NR = p.contexts.length, NC = p.types.length;
  let filled = 0; for (let i = 0; i < NR; i++) for (let j = 0; j < NC; j++) if (p.matrix[i][j]) filled++;
  const density = ((filled / (NR * NC)) * 100).toFixed(1);
  const dataType = p.dataType === "presence_absence" ? { de: "Präsenz/Absenz", en: "presence/absence" } : { de: "Frequenzen (Stückzahlen)", en: "frequencies (counts)" };

  const ord = canonOrder(p);
  const q = quality(p, ord);
  const ca = computeCA(p, 4);
  const inertia = ca.inertiaPct.slice(0, 2).map((x) => (x * 100).toFixed(1));
  const totalInertia = ca.totalInertia.toFixed(4);

  const fixedRows = p.contexts.filter((c) => p.rowMetadata[c]?.isFixed).length;
  const fixedCols = p.types.filter((t) => p.columnMetadata[t]?.isFixed).length;
  const annCount = Object.keys(p.cellAnnotations).length;

  const seriationSteps = p.history.filter((h) => /serialis|seriat|centroid|reciprocal|correspond/i.test(h.method));
  const lastSeriation = seriationSteps[seriationSteps.length - 1];
  const method = lastSeriation?.method ?? "reciprocal averaging (Schwerpunktmethode)";

  const stab = opts.stability;
  const wellConstrained = stab ? stab.rows.filter((r) => r.hi - r.lo <= 2).length : 0;

  const de = [
    `## Methoden`,
    ``,
    `Die Kombinationstabelle umfasst ${NR} Kontexte und ${NC} Typen (${dataType.de}; ${filled} belegte Zellen, Belegungsdichte ${density}\u00A0%). ` +
    `Die chronologische Ordnung wurde mittels ${method} ermittelt${lastSeriation?.score != null ? ` (Gesamt-Qualität ${lastSeriation.score.toFixed(3)})` : ""}. ` +
    `Die resultierende Anordnung erreicht eine Diagonal-Konzentration von ${q.concentration.toFixed(3)}, einen Anti-Robinson-Index von ${q.antiRobinson.toFixed(3)} und eine mittlere Typ-Kontinuität von ${q.continuity.toFixed(3)}.`,
    ``,
    `Zur Absicherung wurde eine Korrespondenzanalyse durchgeführt; die erste Achse erklärt ${inertia[0]}\u00A0% der Gesamtträgheit (${totalInertia}), die zweite ${inertia[1] ?? "–"}\u00A0%.` +
    (fixedRows + fixedCols > 0 ? ` ${fixedRows} Kontexte und ${fixedCols} Typen wurden als Fixpunkte vorgegeben.` : "") +
    (annCount > 0 ? ` ${annCount} Zellen tragen Annotationen (Sicherheit/Fragmentierung/Inventar).` : ""),
    ``,
    stab
      ? `Die Robustheit der Ordnung wurde über einen parametrischen Bootstrap (${stab.replicates} Wiederholungen; multinomiale Neuziehung der Zeilenprofile, Rangbildung über CA-Dimension\u00A01) geprüft. Die mittlere Positions-Stabilität beträgt ${stab.globalStability.toFixed(3)}; ${wellConstrained} von ${NR} Kontexten sind eng bestimmt (90\u00A0%-Rangintervall \u2264 2 Positionen).`
      : `Eine Bootstrap-Stabilitätsanalyse wurde für diesen Bericht nicht einbezogen.`,
    ``,
    `Auswertung mit CombiTab v${VERSION} (Chr. Gugl, OeAI/ÖAW; Coding-Unterstützung Anthropic Claude), MIT-Lizenz.`,
  ].join("\n");

  const en = [
    `## Methods`,
    ``,
    `The combination table comprises ${NR} contexts and ${NC} types (${dataType.en}; ${filled} occupied cells, ${density}\u00A0% density). ` +
    `Chronological ordering was obtained by ${method}${lastSeriation?.score != null ? ` (overall quality ${lastSeriation.score.toFixed(3)})` : ""}. ` +
    `The resulting arrangement attains a diagonal concentration of ${q.concentration.toFixed(3)}, an anti-Robinson index of ${q.antiRobinson.toFixed(3)}, and a mean per-type continuity of ${q.continuity.toFixed(3)}.`,
    ``,
    `Correspondence analysis was used for validation; the first axis accounts for ${inertia[0]}\u00A0% of the total inertia (${totalInertia}), the second for ${inertia[1] ?? "–"}\u00A0%.` +
    (fixedRows + fixedCols > 0 ? ` ${fixedRows} contexts and ${fixedCols} types were constrained as fixed points.` : "") +
    (annCount > 0 ? ` ${annCount} cells carry annotations (certainty/fragmentation/inventory).` : ""),
    ``,
    stab
      ? `Ordering robustness was assessed by a parametric bootstrap (${stab.replicates} replicates; multinomial resampling of row profiles, ranking via CA dimension\u00A01). Mean positional stability is ${stab.globalStability.toFixed(3)}; ${wellConstrained} of ${NR} contexts are tightly constrained (90\u00A0% rank interval \u2264 2 positions).`
      : `A bootstrap stability analysis was not included in this report.`,
    ``,
    `Analysis performed with CombiTab v${VERSION} (Chr. Gugl, OeAI/ÖAW; coding assistance by Anthropic Claude), MIT license.`,
  ].join("\n");

  return `# CombiTab — Analysebericht: ${p.name}\n\n${de}\n\n---\n\n${en}\n`;
}

function canonOrder(p: ProjectV2) {
  const rIdx = new Map(p.contexts.map((c, i) => [c, i] as const));
  const cIdx = new Map(p.types.map((t, j) => [t, j] as const));
  return { rows: p.order.rows.map((r) => rIdx.get(r) ?? 0), cols: p.order.cols.map((c) => cIdx.get(c) ?? 0) };
}
