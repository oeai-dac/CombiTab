import { svd } from "./svd.js";
import { computeCA } from "./ca.js";
import type { ProjectV2 } from "../core/model.js";

let pass=0,fail=0; const F:string[]=[];
function c(n:string,ok:boolean,d=""){ok?pass++:(fail++,F.push(n+(d?` (${d})`:"")));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n+(d?" — "+d:""));}

console.log("\n\x1b[1mSVD + CA\x1b[0m\n");

// ── SVD-Rekonstruktion + Orthonormalität (m>n und m<n) ──
function testSVD(m:number,n:number){
  const M=new Float64Array(m*n); for(let i=0;i<m*n;i++)M[i]=Math.sin(i*1.7)+0.3*Math.cos(i*0.9);
  const {U,s,V,k}=svd(M,m,n);
  // Rekonstruktion U diag(s) Vᵀ
  let err=0;
  for(let i=0;i<m;i++)for(let j=0;j<n;j++){let acc=0;for(let d=0;d<k;d++)acc+=U[i*k+d]*s[d]*V[j*k+d];err=Math.max(err,Math.abs(acc-M[i*n+j]));}
  // Orthonormalität von V-Spalten
  let orth=0;for(let a=0;a<k;a++)for(let b=0;b<k;b++){let dot=0;for(let j=0;j<n;j++)dot+=V[j*k+a]*V[j*k+b];orth=Math.max(orth,Math.abs(dot-(a===b?1:0)));}
  c(`SVD ${m}×${n}: Rekonstruktion exakt`, err<1e-8, `maxErr=${err.toExponential(1)}`);
  c(`SVD ${m}×${n}: V orthonormal`, orth<1e-8, `maxErr=${orth.toExponential(1)}`);
}
testSVD(6,4); testSVD(4,6);

// ── CA auf synthetischer Gradientenstruktur ──
function mkProject(NR:number,NC:number):{p:ProjectV2,tRow:number[]}{
  const M:number[][]=[],tRow:number[]=[],tCol:number[]=[];
  for(let i=0;i<NR;i++)tRow.push(Math.random());
  for(let j=0;j<NC;j++)tCol.push((j+0.5)/NC);
  const sig=0.12;
  for(let i=0;i<NR;i++){const row:number[]=[];for(let j=0;j<NC;j++){const d=tRow[i]-tCol[j];const pr=Math.exp(-d*d/(2*sig*sig));row.push(pr>0.2?1+Math.floor(pr*6):0);}M.push(row);}
  const contexts=Array.from({length:NR},(_,i)=>"G"+i), types=Array.from({length:NC},(_,j)=>"T"+j);
  const columnMetadata:any={},rowMetadata:any={};
  types.forEach(t=>columnMetadata[t]={name:t,materialGroup:"Unassigned",color:"#808080",isIndexType:false,isFixed:false,notes:""});
  contexts.forEach(cx=>rowMetadata[cx]={name:cx,contextType:"",area:"",isFixed:false,notes:""});
  const p:ProjectV2={schemaVersion:2,name:"t",dataType:"frequency",contexts,types,matrix:M,columnMetadata,rowMetadata,cellAnnotations:{},materialGroups:{Unassigned:"#808080"},contextTypes:[],order:{rows:[...contexts],cols:[...types]},view:{vizStyle:"",cellSize:1,showValues:true,showColors:true,showCertainty:false,showFragmentation:false},filters:{materials:[],rowRange:null,colRange:null,hideEmptyRows:false,hideEmptyCols:false},history:[]};
  return {p,tRow};
}
function spearmanAbs(x:number[],y:number[]){const n=x.length;const rk=(a:number[])=>{const idx=a.map((_,i)=>i).sort((p,q)=>a[p]-a[q]);const r=new Array(n);idx.forEach((v,i)=>r[v]=i);return r;};const rx=rk(x),ry=rk(y);let mx=0,my=0;for(let i=0;i<n;i++){mx+=rx[i];my+=ry[i];}mx/=n;my/=n;let sxy=0,sxx=0,syy=0;for(let i=0;i<n;i++){const a=rx[i]-mx,b=ry[i]-my;sxy+=a*b;sxx+=a*a;syy+=b*b;}return Math.abs(sxy/Math.sqrt(sxx*syy||1));}

{
  const {p,tRow}=mkProject(20,60);
  const ca=computeCA(p,4);
  c("CA: liefert ≥2 Dimensionen", ca.k>=2, `k=${ca.k}`);
  // Trägheitsanteile summieren ≤1 und sind absteigend
  const sum=ca.inertiaPct.reduce((a,b)=>a+b,0);
  c("CA: Trägheitsanteile ≤ 1 und absteigend", sum<=1.0001 && ca.inertiaPct.every((v,i,a)=>i===0||a[i-1]>=v-1e-9), `Σ=${sum.toFixed(3)}`);
  // Dim-1 der Zeilen rekonstruiert die eingepflanzte Chronologie
  const dim1=ca.rowCoords.map(r=>r[0]);
  const rho=spearmanAbs(dim1,tRow);
  c("CA: Dim 1 rekonstruiert Gradient (|ρ|)", rho>0.9, `ρ=${rho.toFixed(3)}`);
  // Zeilen-Prinzipalkoordinaten: gewichteter Schwerpunkt ~ 0
  let T=0;const rmass:number[]=new Array(p.contexts.length).fill(0);
  for(let i=0;i<20;i++)for(let j=0;j<60;j++){const v=p.matrix[i][j];rmass[i]+=v;T+=v;}
  let wmean=0;for(let i=0;i<20;i++)wmean+=(rmass[i]/T)*dim1[i];
  c("CA: gewichteter Schwerpunkt der Dim 1 ≈ 0", Math.abs(wmean)<1e-6, `${wmean.toExponential(1)}`);
}

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ SVD & CA korrekt.\x1b[0m");
