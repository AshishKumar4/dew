import sys,tempfile,pathlib
sys.path[:0]=['src','tests','tools']
import jax
from jax.experimental.pallas import mosaic_gpu as pg
import test_recurrence_mosaic as t
import kernel_compile as kc
from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates.recurrence import key_backward as kb
from dew.nn.kernels import recurrence_algebra as algebra
original=kb._key_pairs_staged
def staged(call,keys):
 tiles,landing=original(call,keys)
 def narrow(tile):
  swizzle=mosaic.swizzle_for(tile.dtype,algebra.SUB)
  return pg.SMEM(tile.shape,tile.dtype,transforms=mosaic.swizzle_transforms(tile.dtype,swizzle))
 pairs,squares,gated,*other=tiles
 return (pairs,jax.tree.map(narrow,squares),gated,*other),landing
kb._key_pairs_staged=staged
target,index=sys.argv[1],int(sys.argv[2])
for case in t.compiled_cases(target,{},[],{},only=t.GATE_GROUPS[index]):
 print('K CASE',case.name,flush=True)
 record={'name':case.name}
 with tempfile.TemporaryDirectory() as d:
  kc._compiled(case,target,record,d)
  if 'error' in record:print('K FAIL',record['error'].splitlines()[0],flush=True)
  else:print('K COMPILED',flush=True)
  for p in pathlib.Path(d).glob('*.ptx'):
   if 'key_pairs_back' in p.stem:print('K PTXAS',p.stem,kc.ptxas_read(kc._ptxas_text(p.read_text(),target,kc.RELOCATABLE)[0]),flush=True)
