import { applySeriation, moveFree, toggleFixed, freePositions } from "./orderModel.js";
let pass=0,fail=0; const F:string[]=[];
const eq=(a:number[],b:number[])=>a.length===b.length&&a.every((v,i)=>v===b[i]);
function c(n:string,ok:boolean){ok?pass++:(fail++,F.push(n));console.log((ok?"  \x1b[32m✓\x1b[0m ":"  \x1b[31m✗\x1b[0m ")+n);}

console.log("\n\x1b[1morderModel\x1b[0m\n");
// order display->canonical = [0,1,2,3,4]; fix canonical 2 (at pos 2)
{
  const order=[0,1,2,3,4]; const fixed=new Set([2]);
  const sorted=[4,3,2,1,0]; // gewünschte Gesamtordnung
  const r=applySeriation(order,fixed,sorted);
  c("Seriation: fixiertes Element bleibt an Position 2", r[2]===2);
  c("Seriation: freie Slots mit sortierten freien Elementen gefüllt", eq(r,[4,3,2,1,0].map((x,i)=>i===2?2:x)) && eq(r.filter((_,i)=>i!==2),[4,3,1,0]));
}
// moveFree respektiert Fixierung
{
  const order=[0,1,2,3,4]; const fixed=new Set([2]);
  // ziehe Element an pos0 (canon 0) nach pos4
  const r=moveFree(order,fixed,0,4);
  c("Drag: fixiertes Element (pos2=2) unverändert", r[2]===2);
  c("Drag: freies Element wandert, Reihenfolge der freien plausibel", r.includes(0) && r.length===5 && r[2]===2);
  // fixiertes Element ist nicht bewegbar
  const r2=moveFree(order,fixed,2,0);
  c("Drag: fixiertes Element nicht verschiebbar (unverändert)", eq(r2,order));
}
// toggleFixed
{
  const order=[5,6,7]; let fixed=new Set<number>();
  fixed=toggleFixed(order,fixed,1); c("Pin: canon 6 fixiert", fixed.has(6));
  fixed=toggleFixed(order,fixed,1); c("Unpin: canon 6 gelöst", !fixed.has(6));
}
// freePositions
{ c("freePositions überspringt fixierte", eq(freePositions([0,1,2],new Set([1])),[0,2])); }
// mehrere fixierte behalten alle ihre Positionen nach Seriation
{
  const order=[0,1,2,3,4,5]; const fixed=new Set([1,4]);
  const r=applySeriation(order,fixed,[5,4,3,2,1,0]);
  c("Seriation: mehrere Fixierte behalten Positionen (1@1, 4@4)", r[1]===1 && r[4]===4);
  c("Seriation: freie in sortierter Reihenfolge", eq([r[0],r[2],r[3],r[5]],[5,3,2,0]));
}
console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if(fail){console.log("FAIL: "+F.join(", "));process.exit(1);}
console.log("\x1b[32m✓ orderModel korrekt.\x1b[0m");
