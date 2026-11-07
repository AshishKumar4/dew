import sys,collections
sys.path[:0]=['src','tests','tools']
import kernel_liveness as k
import test_recurrence_mosaic as t
from kernel_compile import lowered_for
from dew.nn.kernels import lowered
case=next(iter(t.compiled_cases('sm90',{},[],{},only=t.GATE_GROUPS[0])))
fn,args=case.call()
call=next(c for c in lowered.kernel_calls(lowered_for(fn,args,'sm90')) if 'solved' in(c.launch or c.name))
old=k._Pass.define; captured=[]
def define(self,op):
 old(self,op)
 if op.name=='arith.addf' and op.results:
  deps=self.atoms.of.get(op.results[0],[]);words=len(self._lanes(op.results[0].type))
  if len(deps)>words and any('!llvm.struct' in self.atoms.origin[a] for a in deps if a is not None):
   captured.append((len(deps),str(op),[(str(o.owner)[:230],len(self.atoms.of.get(o,[]))) for o in op.operands]))
 for r in op.results:
  deps=self.atoms.of.get(r,[])
  if len(deps)>len(self._lanes(r.type))*4 and len(captured)<2:
   print('K FIRST',str(op)[:190],len(deps),flush=True)
k._Pass.define=define
peaks=k.kernel_peaks(call.ir,k.Compiler((12,9)))
for row in captured[:3]+captured[len(captured)//2:len(captured)//2+2]+captured[-2:]:
 print('K ADD',row[0],row[1][:230],flush=True)
 for o,l in row[2]:print('K OPERAND',l,o,flush=True)
print('K COUNT',len(captured),flush=True)
