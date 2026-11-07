import sys
sys.path[:0]=['src','tests','tools']
import kernel_liveness as k
import pytest
mode=sys.argv[1]
if mode=='candidate':
 old=k._Pass._streamed
 def stream(self,op,name):
  if k._elementwise(op,name):
   deps={a for o in op.operands for a in self.atoms.of.get(o,()) if a is not None}
   keys={self.run[p] for p in self._read(op.operands) if p in self.run}
   if len(keys)!=1 and len(deps)>len(self._lanes(op.results[0].type)):return False
  return old(self,op,name)
 k._Pass._streamed=stream
class Results:
 def __init__(self):self.counts={};self.failed=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.failed:
   self.counts[report.outcome]=self.counts.get(report.outcome,0)+1
   if report.failed:self.failed.append((report.nodeid,str(report.longrepr)[-350:].replace('\n',' | ')))
r=Results()
status=pytest.main(['-q','-p','no:cacheprovider',*sys.argv[2:]],plugins=[r])
print('K TESTS',mode,'exit',status,r.counts,flush=True)
for node,text in r.failed:print('K FAILED',mode,node,text,flush=True)
sys.exit(status)
