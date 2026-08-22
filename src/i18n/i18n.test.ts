import { translate, dictionaries, LANGS } from "./i18n.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }

console.log("\n\x1b[1mi18n (DE/EN)\x1b[0m\n");

// Grundübersetzung
c("DE liefert deutschen Text", translate("de", "matrix.seriate") === "Seriieren");
c("EN liefert englischen Text", translate("en", "matrix.seriate") === "Seriate");

// Interpolation
c("Interpolation {n}", translate("de", "annot.cells", { n: 5 }) === "5 Zellen");
c("Interpolation mehrerer Variablen", translate("en", "filter.summary", { c: 3, ty: 7 }) === "3 contexts · 7 types");
c("Interpolation numerisch → String", translate("en", "annot.cells", { n: 12 }) === "12 cells");

// Fallback
c("unbekannter Schlüssel fällt auf den Schlüssel zurück", translate("de", "does.not.exist") === "does.not.exist");
c("fehlende EN-Übersetzung fällt auf DE zurück", (() => {
  // simulierter Fall: temporär EN-Eintrag entfernen
  const k = "matrix.seriate"; const bak = dictionaries.en[k]; delete dictionaries.en[k];
  const got = translate("en", k); dictionaries.en[k] = bak; return got === "Seriieren";
})());

// Parität: gleiche Schlüsselmengen in beiden Sprachen
const de = Object.keys(dictionaries.de).sort();
const en = Object.keys(dictionaries.en).sort();
const onlyDe = de.filter((k) => !(k in dictionaries.en));
const onlyEn = en.filter((k) => !(k in dictionaries.de));
c("keine nur-DE-Schlüssel", onlyDe.length === 0, onlyDe.join(", "));
c("keine nur-EN-Schlüssel", onlyEn.length === 0, onlyEn.join(", "));
c("gleiche Schlüsselanzahl", de.length === en.length, `de=${de.length} en=${en.length}`);

// keine leeren Werte
for (const l of LANGS) {
  const empties = Object.entries(dictionaries[l]).filter(([, v]) => !v.trim()).map(([k]) => k);
  c(`keine leeren Werte (${l})`, empties.length === 0, empties.join(", "));
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }
