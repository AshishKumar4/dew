import sys
sys.path[:0]=['src','tests','tools']
import kernel_liveness as k
import kernel_compile as c
import test_recurrence_mosaic as t
index,mode=int(sys.argv[1]),sys.argv[2]
if mode=='candidate':
 old=k._Pass._streamed
 def stream(self,op,name):
  if k._elementwise(op,name):
   deps={a for o in op.operands for a in self.atoms.of.get(o,()) if a is not None}
   keys={self.run[p] for p in self._read(op.operands) if p in self.run}
   if len(keys)!=1 and len(deps)>len(self._lanes(op.results[0].type)):return False
  return old(self,op,name)
 k._Pass._streamed=stream
cases=list(t.compiled_cases('sm90',{},[],{},only=t.GATE_GROUPS[index]))[:1]
records=c.compile_mosaic_kernels(cases,'sm90',least=True)
for record in records:
 print('K CASE',index,mode,record['name'],flush=True)
 for name,model in record.get('lowered_models',{}).items():
  print('K MODEL',mode,name,model.get('registers'),flush=True)
 if 'error' in record: print('K ERROR',record['error'],flush=True)
 for row in record.get('evidence',[]):
  print('K EVIDENCE',mode,row,flush=True)
 for kernel in record.get('kernels',[]):
  name=kernel.get('kernel'); need=kernel.get('need')
  caps=need.get('caps',[]) if isinstance(need,dict) else []
  print('K PTXAS',mode,name,'used',kernel.get('registers'),'least',kernel.get('least'),'caps',[(x['cap'],x['registers'],x['spilled']) for x in caps],flush=True)
 for line in c.model_failures(records):print('K FINDING',mode,line,flush=True)
