import { makeSyntheticProject } from "./synth.js";
import { quality } from "../seriation/metrics.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mSynthetischer Projekt-Generator\x1b[0m\n");

const p = makeSyntheticProject(40, 30, { seed: 1 });
c("Dimensionen stimmen", p.contexts.length === 40 && p.types.length === 30 && p.matrix.length === 40 && p.matrix[0].length === 30);
c("gültige ProjectV2-Grundstruktur", p.schemaVersion === 2 && !!p.columnMetadata[p.types[0]] && !!p.rowMetadata[p.contexts[0]] && p.order.rows.length === 40);
c("jeder Typ einer Materialgruppe mit Farbe zugewiesen", p.types.every((t) => { const cm = p.columnMetadata[t]; return cm && cm.materialGroup && /^#[0-9A-Fa-f]{6}$/.test(cm.color); }));

// Reproduzierbarkeit
const a = makeSyntheticProject(20, 20, { seed: 5 }), b = makeSyntheticProject(20, 20, { seed: 5 });
c("gleicher Seed → identische Matrix", JSON.stringify(a.matrix) === JSON.stringify(b.matrix));
const cc = makeSyntheticProject(20, 20, { seed: 6 });
c("anderer Seed → andere Matrix", JSON.stringify(a.matrix) !== JSON.stringify(cc.matrix));

// Bandstruktur: kanonische (diagonale) Ordnung ist gut Robinson-geordnet
{
  const q = makeSyntheticProject(60, 60, { seed: 2, noise: 0.01 });
  const order = { rows: q.contexts.map((_, i) => i), cols: q.types.map((_, j) => j) };
  const ar = quality(q, order).antiRobinson;
  c("Bandstruktur ist bereits grob seriiert (Anti-Robinson hoch)", ar > 0.8, `ar=${ar.toFixed(3)}`);
}

// nicht leer
{
  const q = makeSyntheticProject(50, 50, { seed: 4 });
  let nz = 0; for (const row of q.matrix) for (const v of row) if (v > 0) nz++;
  c("Matrix ist überwiegend belegt, aber nicht voll", nz > 500 && nz < 50 * 50, `nz=${nz}`);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
