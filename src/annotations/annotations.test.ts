import { setAnnotation, getAnnotation, applyToCells, clearCells, commonValue, annotationCount, deleteAnnotation, buildBatchPatch } from "./annotations.js";
import { annotationKey } from "../core/model.js";
import type { ProjectV2 } from "../core/model.js";

let pass=0,fail=0; const F:string[]=[];
function c(n:string,ok:boolean,d=""){ok?pass++:(fail++,F.push(n));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n+(d?" — "+d:""));}

function mk():ProjectV2{
  const contexts=["G0","G1","G2"],types=["T0","T1"];
  const columnMetadata:any={},rowMetadata:any={};
  types.forEach(t=>columnMetadata[t]={name:t,materialGroup:"Unassigned",color:"#808080",isIndexType:false,isFixed:false,notes:""});
  contexts.forEach(cx=>rowMetadata[cx]={name:cx,contextType:"",area:"",isFixed:false,notes:""});
  return {schemaVersion:2,name:"t",dataType:"frequency",contexts,types,matrix:[[1,0],[0,2],[3,1]],columnMetadata,rowMetadata,cellAnnotations:{},materialGroups:{Unassigned:"#808080"},contextTypes:[],order:{rows:[...contexts],cols:[...types]},view:{vizStyle:"",cellSize:1,showValues:true,showColors:true,showCertainty:false,showFragmentation:false},filters:{materials:[],rowRange:null,colRange:null,hideEmptyRows:false,hideEmptyCols:false},history:[]};
}

console.log("\n\x1b[1mAnnotationen\x1b[0m\n");
{
  const p=mk();
  setAnnotation(p,1,0,{certainty:"certain",countMin:2,countMax:4});
  const a=getAnnotation(p,1,0)!;
  c("setzt Felder + Kontext/Typ", a.certainty==="certain" && a.countMin===2 && a.context==="G1" && a.type==="T0");
  c("kanonischer Schlüssel", !!p.cellAnnotations[annotationKey(1,0)]);
  // merge
  setAnnotation(p,1,0,{fragmentation:"fragmented"});
  c("merge behält bestehende Felder", getAnnotation(p,1,0)!.certainty==="certain" && getAnnotation(p,1,0)!.fragmentation==="fragmented");
  // leeren einzelner Felder
  setAnnotation(p,1,0,{certainty:"",countMin:undefined,countMax:undefined} as any);
  const a2=getAnnotation(p,1,0)!;
  c("leeres Feld wird entfernt, Rest bleibt", a2.certainty===undefined && a2.countMin===undefined && a2.fragmentation==="fragmented");
  // vollständig leeren → Annotation gelöscht
  setAnnotation(p,1,0,{fragmentation:""} as any);
  c("leere Annotation wird gelöscht", getAnnotation(p,1,0)===undefined && annotationCount(p)===0);
}
{
  const p=mk();
  const cells:Array<[number,number]>=[[0,0],[1,0],[2,1]];
  applyToCells(p,cells,{certainty:"uncertain"});
  c("Batch: 3 Annotationen gesetzt", annotationCount(p)===3);
  c("Batch: commonValue = uncertain", commonValue(p,cells,"certainty")==="uncertain");
  setAnnotation(p,0,0,{certainty:"certain"});
  c("commonValue undefined bei Abweichung", commonValue(p,cells,"certainty")===undefined);
  clearCells(p,cells);
  c("clearCells entfernt alle", annotationCount(p)===0);
}
{
  const p=mk();
  setAnnotation(p,2,1,{inventoryNumbers:["INV-1","INV-2"],notes:"zwei Scherben"});
  const a=getAnnotation(p,2,1)!;
  c("inventoryNumbers + notes", a.inventoryNumbers?.length===2 && a.notes==="zwei Scherben");
  setAnnotation(p,2,1,{inventoryNumbers:[]} as any);
  c("leeres Array entfernt inventoryNumbers", getAnnotation(p,2,1)!.inventoryNumbers===undefined);
  deleteAnnotation(p,2,1);
  c("deleteAnnotation", annotationCount(p)===0);
}
{
  // v1.0-Regression: Batch auf gemischte Auswahl darf unberührte Felder NICHT löschen.
  const p=mk();
  setAnnotation(p,0,0,{certainty:"certain",fragmentation:"complete",notes:"Bronzefibel"});
  setAnnotation(p,0,1,{certainty:"uncertain",inventoryNumbers:["INV-7"]});
  const vals={certainty:"",fragmentation:"",countMin:"",countMax:"",inv:"",notes:"Auswahlnotiz"};
  const patch=buildBatchPatch(vals,new Set(["notes"]));
  c("buildBatchPatch: nur angefasste Felder im Patch", Object.keys(patch).length===1 && patch.notes==="Auswahlnotiz");
  applyToCells(p,[[0,0],[0,1]],patch);
  const a=getAnnotation(p,0,0)!, b=getAnnotation(p,0,1)!;
  c("gemischte Auswahl: certainty/fragmentation bleiben erhalten", a.certainty==="certain" && a.fragmentation==="complete" && b.certainty==="uncertain");
  c("gemischte Auswahl: inventoryNumbers bleiben erhalten", b.inventoryNumbers?.[0]==="INV-7");
  c("gemischte Auswahl: Notiz überall gesetzt", a.notes==="Auswahlnotiz" && b.notes==="Auswahlnotiz");
  // Explizites Löschen bleibt möglich: Feld angefasst und geleert.
  const del=buildBatchPatch({...vals,notes:""},new Set(["notes"]));
  applyToCells(p,[[0,0]],del);
  c("angefasst+leer löscht das Feld weiterhin", getAnnotation(p,0,0)!.notes===undefined && getAnnotation(p,0,0)!.certainty==="certain");
  const noop=buildBatchPatch(vals,new Set());
  c("nichts angefasst → leerer Patch", Object.keys(noop).length===0);
}
console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ Annotationen korrekt.\x1b[0m");
