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
   if len(deps)>len(self._lanes(op.results[0].type)):return False
  return old(self,op,name)
 k._Pass._streamed=stream
class Results:
 def __init__(self):self.counts={}
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.failed:
   self.counts[report.outcome]=self.counts.get(report.outcome,0)+1
   if report.failed:print('K FAILED',mode,report.nodeid,str(report.longrepr)[-900:].replace('\n',' | '),flush=True)
r=Results()
status=pytest.main(['-q','-p','no:cacheprovider',*sys.argv[2:]],plugins=[r])
print('K TESTS',mode,'exit',status,r.counts,flush=True)
sys.exit(status)
