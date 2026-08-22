import { bootstrapStability, mulberry32 } from "./bootstrap.js";
import type { ProjectV2 } from "../core/model.js";

let pass=0,fail=0; const F:string[]=[];
function c(n:string,ok:boolean,d=""){ok?pass++:(fail++,F.push(n));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n+(d?" — "+d:""));}

function mk(M:number[][]):ProjectV2{
  const NR=M.length,NC=M[0].length;
  const contexts=Array.from({length:NR},(_,i)=>"G"+i),types=Array.from({length:NC},(_,j)=>"T"+j);
  const columnMetadata:any={},rowMetadata:any={};
  types.forEach(t=>columnMetadata[t]={name:t,materialGroup:"U",color:"#808080",isIndexType:false,isFixed:false,notes:""});
  contexts.forEach(cx=>rowMetadata[cx]={name:cx,contextType:"",area:"",isFixed:false,notes:""});
  return {schemaVersion:2,name:"t",dataType:"frequency",contexts,types,matrix:M,columnMetadata,rowMetadata,cellAnnotations:{},materialGroups:{U:"#808080"},contextTypes:[],order:{rows:[...contexts],cols:[...types]},view:{vizStyle:"",cellSize:1,showValues:true,showColors:true,showCertainty:false,showFragmentation:false},filters:{materials:[],rowRange:null,colRange:null,hideEmptyRows:false,hideEmptyCols:false},history:[]};
}
// starke Gradientenstruktur (Gauß-Band, hohe Zählungen → stabil)
function gradient(NR:number,NC:number,scale:number){
  const M:number[][]=[];const sig=0.12;
  for(let i=0;i<NR;i++){const ti=i/(NR-1);const row:number[]=[];
    for(let j=0;j<NC;j++){const tj=(j+0.5)/NC;const d=ti-tj;const p=Math.exp(-d*d/(2*sig*sig));row.push(Math.round(p*scale));}
    M.push(row);}
  return M;
}

console.log("\n\x1b[1mBootstrap-Stabilität\x1b[0m\n");
{
  const p=mk(gradient(15,40,30)); // hohe Zählungen
  const res=bootstrapStability(p,{replicates:120,rng:mulberry32(7)});
  c("liefert eine Zeile je Kontext", res.rows.length===15);
  c("nach refRank sortiert", res.rows.every((r,i,a)=>i===0||a[i-1].refRank<=r.refRank));
  c("Intervalle lo ≤ median ≤ hi", res.rows.every(r=>r.lo<=r.median+1e-9 && r.median<=r.hi+1e-9));
  c("globale Stabilität in [0,1]", res.globalStability>=0 && res.globalStability<=1, res.globalStability.toFixed(3));
  c("starke Struktur → hohe Stabilität (>0.7)", res.globalStability>0.7, res.globalStability.toFixed(3));
  // Reproduzierbarkeit mit gleichem Seed
  const res2=bootstrapStability(p,{replicates:120,rng:mulberry32(7)});
  c("reproduzierbar (gleicher Seed)", JSON.stringify(res.rows)===JSON.stringify(res2.rows));
}
{
  // reines Rauschen → breitere Intervalle, geringere Stabilität als Struktur
  const NR=15,NC=40; const M:number[][]=[];
  const rng=mulberry32(3);
  for(let i=0;i<NR;i++){const row:number[]=[];for(let j=0;j<NC;j++)row.push(rng()<0.3?1+Math.floor(rng()*3):0);M.push(row);}
  const noise=bootstrapStability(mk(M),{replicates:120,rng:mulberry32(9)});
  const struct=bootstrapStability(mk(gradient(15,40,30)),{replicates:120,rng:mulberry32(9)});
  c("Struktur stabiler als Rauschen", struct.globalStability > noise.globalStability, `struct=${struct.globalStability.toFixed(3)} noise=${noise.globalStability.toFixed(3)}`);
}
console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ Bootstrap korrekt.\x1b[0m");
