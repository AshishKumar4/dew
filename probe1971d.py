import sys
sys.path[:0]=['src','tests','tools']
import kernel_liveness as k
import test_recurrence_mosaic as t
from kernel_compile import lowered_for
from dew.nn.kernels import lowered
index,names=int(sys.argv[1]),sys.argv[2].split(',')
case=next(iter(t.compiled_cases('sm90',{},[],{},only=t.GATE_GROUPS[index])))
fn,args=case.call();calls=lowered.kernel_calls(lowered_for(fn,args,'sm90'))
old=k._Pass._streamed
for call in calls:
 if call.kind!='mosaic_gpu' or not any(n in(call.launch or call.name) for n in names):continue
 print('K CASE',index,call.launch or call.name,flush=True)
 for mode in ['baseline','runsafe','runsafe2']:
  def stream(self,op,name):
   if mode!='baseline' and k._elementwise(op,name):
    deps={a for o in op.operands for a in self.atoms.of.get(o,()) if a is not None}
    products=self._read(op.operands)
    keys={self.run[p] for p in products if p in self.run}
    guard=len(keys)!=1
    if mode=='runsafe2' and not products and not self._read_past_products(op):guard=False
    limit=len(self._lanes(op.results[0].type))
    if guard and len(deps)>limit:return False
   return old(self,op,name)
  k._Pass._streamed=stream
  for n,p,(b,r) in k.kernel_peaks(call.ir,k.Compiler((12,9))):
   if p.threads:print('K MODE',mode,p.total,p.where,b[:3],flush=True)
 k._Pass._streamed=old
