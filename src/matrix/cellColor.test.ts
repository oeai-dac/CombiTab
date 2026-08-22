import { mixCell } from "./cellColor.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const eq = (v: [number, number, number], e: [number, number, number]) => v[0] === e[0] && v[1] === e[1] && v[2] === e[2];

console.log("\n\x1b[1mZell-Farbmodell (Canvas-2D == Shader)\x1b[0m\n");

const LIGHT: [number, number, number] = [0.965, 0.957, 0.949];
const DARK: [number, number, number] = [0.137, 0.125, 0.125];
const CERAMIC: [number, number, number] = [205, 133, 63]; // #CD853F

// Shader-Formel: mix(bg, base, 0.25 + 0.75*clamp(v)); t=1 → reine Materialfarbe
c("v=1 → reine Materialfarbe (Hintergrund irrelevant)", eq(mixCell(LIGHT, CERAMIC, 1), CERAMIC));
c("v=1 identisch in Hell und Dunkel (nur Sättigung tieffrequenter Zellen unterscheidet sich)",
  eq(mixCell(LIGHT, CERAMIC, 1), mixCell(DARK, CERAMIC, 1)));

// t = 0.25 bei v=0
c("v=0, weißer Hintergrund, schwarze Basis → 0.75·255 = 191", eq(mixCell([1, 1, 1], [0, 0, 0], 0), [191, 191, 191]));

// Clamping
c("v>1 wird auf 1 geklemmt", eq(mixCell(LIGHT, CERAMIC, 2.5), mixCell(LIGHT, CERAMIC, 1)));
c("v<0 wird auf 0 geklemmt", eq(mixCell(LIGHT, CERAMIC, -3), mixCell(LIGHT, CERAMIC, 0)));

// Hintergrund wirkt bei tiefer Frequenz: hell vs. dunkel unterscheiden sich
{
  const l = mixCell(LIGHT, CERAMIC, 0.2), d = mixCell(DARK, CERAMIC, 0.2);
  c("tieffrequente Zelle: Hell heller als Dunkel", l[0] > d[0] && l[1] > d[1] && l[2] > d[2], `hell=${l} dunkel=${d}`);
}

// Explizite Formelprobe: v=0.5 → t = 0.625
{
  const t = 0.625;
  const exp: [number, number, number] = [
    (1 * 255 * (1 - t) + 200 * t) | 0,
    (0.5 * 255 * (1 - t) + 100 * t) | 0,
    (0 * 255 * (1 - t) + 50 * t) | 0,
  ];
  c("v=0.5 entspricht mix(bg,col,0.625)", eq(mixCell([1, 0.5, 0], [200, 100, 50], 0.5), exp), `${mixCell([1, 0.5, 0], [200, 100, 50], 0.5)}`);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
