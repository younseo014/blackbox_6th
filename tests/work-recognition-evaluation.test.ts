import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { generateEventMotion } from "../app/demo-motion.ts";

test("offline evaluator separates reference recordings and never uses truth to choose predictions", () => {
  const dir=mkdtempSync(join(tmpdir(),"mg-recognition-eval-"));
  try {
    const save=(name:string,value:unknown)=>writeFileSync(join(dir,name),JSON.stringify(value));
    save("ref-a.json",{session:{id:"ref-a"},frames:generateEventMotion("normal_task",1)});
    save("ref-b.json",{session:{id:"ref-b"},frames:generateEventMotion("high_reach",1)});
    const observed=generateEventMotion("normal_task",3); save("observed.json",{session:{id:"evaluation"},frames:observed});
    const base={name:"generated-regression",split:"regression",recording:"observed.json",expectedLabels:["A","B"],
      references:[{label:"A",recording:"ref-a.json",startMs:0,endMs:observed.at(-1)![0],labelSource:"manual"},
        {label:"B",recording:"ref-b.json",startMs:0,endMs:observed.at(-1)![0],labelSource:"manual"}]};
    const run=(expected:unknown,suffix:string,overlap=false)=>{
      save("cases.json",[{...base,...(overlap?{references:[{label:"A",recording:"observed.json",startMs:0,endMs:observed.at(-1)![0]}]}:{}),expected}]);
      return spawnSync(process.execPath,["--import","tsx","scripts/evaluate-work-recognition.ts","--cases",join(dir,"cases.json"),"--output",join(dir,suffix)],{encoding:"utf8",timeout:30000});
    };
    const a=run([{label:"A",startMs:0,endMs:observed.at(-1)![0]}],"a.json");assert.equal(a.status,0,a.stderr);
    const b=run([{label:"B",startMs:0,endMs:observed.at(-1)![0]}],"b.json");assert.equal(b.status,0,b.stderr);
    const first=JSON.parse(readFileSync(join(dir,"a.json"),"utf8")), second=JSON.parse(readFileSync(join(dir,"b.json"),"utf8"));
    assert.deepEqual(first.results[0].predictions,second.results[0].predictions);
    assert.ok(first.results[0].predictions.length>0);
    assert.equal(first.results[0].split,"regression");
    const leaked=run(null,"bad.json",true);assert.notEqual(leaked.status,0);assert.match(leaked.stderr,/Training\/evaluation recording overlap/);
  } finally {rmSync(dir,{recursive:true,force:true});}
});
