import sys; sys.path[:0]=["src","tests","."]
import jax
import jax.numpy as j
from dew.algebra import scan
from dew.nn.precision import compensated_add,vanishing
from test_recurrence_chunk import test_the_portable_rule_takes_forward_mode_at_its_public_seam
mode=sys.argv[1] if len(sys.argv)>1 else "base"
if "peel" in mode:
  original_chunk=scan._chunk
  def peeled(form,held,h,entering,to_last,whole,between,*values):
    out,final=original_chunk(form,held,h,entering,to_last,whole,between,*values)
    count=entering.shape[0]
    B,U,s,C,Q,R=values[:6]
    # In this fixture every real token has a nonzero write; padding has B=0.
    last=j.max(j.where(j.any(B!=0,axis=-1),j.arange(count),0))
    previous=j.maximum(last-1,0)
    ending=j.where(j.arange(count)[:,None]<last,between[previous],0.)
    _,prefix=original_chunk(form,held,h,entering,ending,entering[previous],between,*values)
    prefix=j.where(last>0,prefix,h)
    decay=j.where(last>0,between[last,previous],entering[0])
    decayed=prefix*decay[:,None]
    final=decayed+B[last,:,None]*(U[last]-s[last]*(C[last]@decayed))[None,:]
    return out,final
  scan._chunk=peeled
if "exp" in mode:
  def value(z,base2):
    gone=z<=vanishing(z.dtype,base2)
    high,low=scan._compensated(j.where(gone,0.,z),restarts=gone)
    segments=j.cumsum(gone.astype(j.int32),axis=0)
    count=z.shape[0]
    joined=(j.arange(count)[:,None]>=j.arange(count)[None,:])[:,:,None]&(segments[:,None]==segments[None,:])
    started,ending=segments==0,segments==segments[-1]
    def exp(hi,lo,mask):
      return j.where(mask,scan._exp(base2,j.where(mask,hi,0.))*scan._exp(base2,j.where(mask,lo,0.)),0.)
    gaps=compensated_add((high[:,None],low[:,None]),(-high[None,:],-low[None,:]))
    last=compensated_add((high[-1],low[-1]),(-high,-low))
    return scan._Decays(exp(high,low,started),exp(*last,ending),exp(high[-1],low[-1],started[-1]),exp(*gaps,joined))
  scan._decays_value=value
if "half" in mode:
  original=scan._paired
  def paired(l,r,b,issue=None,role="scores",strict=False):
    if b.shape[-1]!=1:return original(l,r,b,issue,role,strict=strict)
    issue=issue or scan._issued
    half=l.shape[-1]//2
    joined=j.where(j.tril(j.ones(b.shape[:2],bool),-1)[:,:,None],b,0.) if strict else b
    active=j.any(joined!=0,axis=-1)
    product=issue(role,l[:,:half],r[:,:half].T)+issue(role,l[:,half:],r[:,half:].T)
    return j.where(active,product,0.)*joined[...,0]
  scan._paired=paired
try:
  test_the_portable_rule_takes_forward_mode_at_its_public_seam(True)
  print("K",mode,"PASS",flush=True)
except Exception as e:
  print("K",mode,type(e).__name__,str(e).replace("\n"," ")[:4000],flush=True)
if "metrics" in mode:
  import numpy as np
  from reference_error import row_errors,without_ynn
  from test_recurrence_chunk import portable_draw,portable_rule
  from dew.nn.kernels.recurrence_chunk import recurrent_delta_rule
  def fn(*drawn):
    args,directions,starts=drawn[:5],drawn[5:10],drawn[10]
    return jax.jvp(lambda *x:portable_rule(*x,starts=starts),args,directions)[0][1]
  def ref(*drawn):
    args,directions,starts=drawn[:5],drawn[5:10],drawn[10]
    return jax.jvp(lambda *x:recurrent_delta_rule(*x,starts=starts),args,directions)[0][1]
  mine,theirs,truth=without_ynn(fn),without_ynn(ref),without_ynn(ref)
  worst=[(0.,None),(0.,None)]
  for seed in range(40):
    operands,starts,directions=portable_draw(seed,1,True)
    given=(*operands,*directions,starts)
    with jax.enable_x64():
      widened=tuple(x.astype(j.float64) if j.issubdtype(x.dtype,j.inexact) else x for x in given)
      t=truth(*widened)
    for i,x in enumerate((mine(*given),theirs(*given))):
      sizes,errors=row_errors(x,t)
      ratio=np.max(np.where(sizes>0,errors/sizes,0.))
      if ratio>worst[i][0]:worst[i]=(float(ratio),seed)
  print("K",mode,"worst",worst,flush=True)
