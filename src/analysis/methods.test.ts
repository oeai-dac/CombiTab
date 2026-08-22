import { generateMethods } from "./methods.js";
import { bootstrapStability, mulberry32 } from "./bootstrap.js";
import { importCSV } from "../core/io/importTable.js";
import { readFileSync, existsSync } from "node:fs";
import { SAMPLE_CSV } from "../data/sample.js";

const SAMPLE_PATH = "/mnt/user-data/uploads/large_cemetery_sample.csv";
const sampleCSV = existsSync(SAMPLE_PATH) ? readFileSync(SAMPLE_PATH, "utf8") : SAMPLE_CSV;

let pass=0,fail=0; const F:string[]=[];
function c(n:string,ok:boolean,d=""){ok?pass++:(fail++,F.push(n));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n+(d?" — "+d:""));}

console.log("\n\x1b[1mMethods-Export\x1b[0m\n");
const { project } = importCSV(sampleCSV, { name: "Cemetery" });
project.history.push({ method: "reciprocal averaging (Schwerpunktmethode)", params: { iterations: 15 }, timestamp: new Date().toISOString(), score: 0.72 });
project.rowMetadata[project.contexts[0]].isFixed = true;

const md = generateMethods(project);
c("enthält DE- und EN-Abschnitt", md.includes("## Methoden") && md.includes("## Methods"));
c("nennt Dimensionen", md.includes("20 Kontexte") && md.includes("100 Typen") && md.includes("20 contexts"));
c("nennt Seriationsmethode + Qualität", md.includes("Schwerpunktmethode") && md.includes("0.720"));
c("nennt CA-Trägheit", /erklärt \d+\.\d+\u00A0?% der Gesamtträgheit/.test(md) && /first axis accounts for/.test(md));
c("nennt Fixpunkt", md.includes("1 Kontexte") && md.includes("1 contexts"));
c("Zitat + Lizenz", md.includes("CombiTab v2.0") && md.includes("MIT"));
c("ohne Bootstrap: Hinweis", md.includes("nicht einbezogen") && md.includes("not included"));

const stab = bootstrapStability(project, { replicates: 60, rng: mulberry32(1) });
const md2 = generateMethods(project, { stability: stab });
c("mit Bootstrap: Wiederholungen + Stabilität", md2.includes("60 Wiederholungen") && /Positions-Stabilität beträgt \d/.test(md2) && md2.includes("replicates"));

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ Methods korrekt.\x1b[0m");
