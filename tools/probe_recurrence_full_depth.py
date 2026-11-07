import sys,inspect,tempfile,pathlib
sys.path[:0]=['src','tests','tools']
import test_recurrence_mosaic as t
import kernel_compile as kc
from dew.nn.kernels import recurrence_algebra as alg
source=inspect.getsource(alg.key_decayed_pairs_back)
source=source.replace('d_pair.rows(place, sub).T, down_rows.rows(place, sub)',
 'd_pair.T, lambda down_rows=down_rows: jnp.where(within, down_rows.value(), 0.0)')
exec(source,alg.__dict__)
target,index=sys.argv[1],int(sys.argv[2])
for case in t.compiled_cases(target,{},[],{},only=t.GATE_GROUPS[index]):
 print('K CASE',case.name,flush=True)
 if 'compile' in sys.argv:
  record={'name':case.name}
  with tempfile.TemporaryDirectory() as d:
   kc._compiled(case,target,record,d)
   if 'error' in record:print('K FAIL',record['error'].splitlines()[0],flush=True)
   else:print('K COMPILED',flush=True)
   for p in pathlib.Path(d).glob('*.ptx'):
    if 'key_pairs_back' in p.stem:print('K PTXAS',p.stem,kc.ptxas_read(kc._ptxas_text(p.read_text(),target,kc.RELOCATABLE)[0]),flush=True)
 else:
  f,a=case.call()
  try:kc.lowered_for(f,a,target);print('K OK',flush=True)
  except Exception as e:print('K FAIL',str(e).splitlines()[0],flush=True)
