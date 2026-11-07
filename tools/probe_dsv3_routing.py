"""Scratch reproduction for ledger 4734; run from the checkout on armada.

Arguments, one task per line:
  measure|0|512             original errors and expert-choice strata
  gate|0                   original paired sizing, unresolved at 512
  force|512                replay float64 expert choices in both bf16 arms
  truthmeasure|0|4096       compare Dew/reference truth margins and bf16 movement
  truthmeasure|1048576|4096 independent pilot block
  truthgate|0.0625          Dew-float64 clearing_selections, sizing and check 1

The threshold replacement below is instrumentation only: the product helper
currently fixes the float32 MARGIN and needs an explicit margin argument for bf16.
All scores include the correction bias. Choice comparisons ignore top-k order.
"""
import os,sys,json
sys.path[:0]=[os.getcwd()+'/src',os.getcwd()+'/tests',os.getcwd()+'/tools']
from lane_environment import configure_lane
configure_lane(os.environ)
import numpy as np,torch,jax,jax.numpy as jnp
import transformers
from pathlib import Path
from interop_support import fixture_ids,transformers_model
from torch_precision import float64,float64_twin
from reference_error import as_given,seeds_for,assert_as_exact
from arm_arithmetic import summed_as_run,widened
from dew.interop import Pretrained
from selection_margins import top_margin,clearing,capture,margin
from dataclasses import replace
import selection_margins
from interop_support import clearing_selections
from flax import linen as nn
from dew.nn.moe import Router
p=Path('tests/fixtures/hf/deepseek-v3-tiny')
print('K START',sys.argv,flush=True)
print('K VERSIONS',torch.__version__,transformers.__version__,jax.__version__,flush=True)
torch.set_num_threads(1)
source=transformers_model(p)
narrow=transformers_model(p,dtype='bfloat16')
wide=float64_twin(source)
loaded=Pretrained.load(str(p),dtype='bfloat16',attention_impl='reference')
def dew_forward(ids):return loaded.model.apply(loaded.variables,ids,mutable=['router'])
dew=jax.jit(dew_forward)
plain=jax.jit(lambda ids:loaded.model.apply(loaded.variables,ids).astype(jnp.float32))
with jax.enable_x64():
 dew_truth=replace(loaded,model=loaded.model.clone(dtype=jnp.float64,attention_impl='reference'),variables=widened(loaded.variables))
 truth_capture=capture(dew_truth.model)
truth_cache={}
def true_selections(ids):
 key=ids.tobytes()
 if key not in truth_cache:
  with jax.enable_x64():
   captured=truth_capture(dew_truth.variables,jnp.asarray(ids))
   margins=margin(dew_truth.model_config,captured,'dsv3 truth')
  groups=np.asarray(captured['intermediates']['layers_1']['mlp']['gate']['group_scores'][0]).reshape(-1,4)
  truth_cache[key]=(margins,groups)
 return truth_cache[key]
raw=fixture_ids(p)
routers={}
def hook(name):
 def save(mod,inputs,output):
  logits,weights,ids=output
  scores=logits.sigmoid()+mod.e_score_correction_bias
  groups=scores.reshape(-1,4,2).topk(2,dim=-1)[0].sum(-1)
  routers[name]=(ids.detach().numpy(),scores.detach().double().numpy(),groups.detach().double().numpy())
 return save
narrow.model.layers[1].mlp.gate.register_forward_hook(hook('n'))
wide.model.layers[1].mlp.gate.register_forward_hook(hook('w'))
forced=[None]
def force_torch(mod,inputs,output):
 if forced[0] is None:return output
 logits=output[0];ids=torch.from_numpy(forced[0]).long()
 weights=logits.sigmoid().gather(1,ids)
 weights=weights/(weights.sum(-1,keepdim=True)+1e-20)
 return logits,weights*mod.routed_scaling_factor,ids
narrow.model.layers[1].mlp.gate.register_forward_hook(force_torch)
def forced_dew(ids,choices):
 def intercept(next_fun,args,kwargs,context):
  if isinstance(context.module,Router) and context.method_name=='_selected':return choices.reshape(2,12,4)
  return next_fun(*args,**kwargs)
 with nn.intercept_methods(intercept):
  return loaded.model.apply(loaded.variables,ids).astype(jnp.float32)
dew_fixed=jax.jit(forced_dew)
cache={}
def evaluate(ids,ours=True):
 key=ids.tobytes()
 if key not in cache:
  with torch.no_grad():
   tokens=torch.from_numpy(ids.astype(np.int64))
   n=narrow(tokens,use_cache=False).logits.float().numpy()
   with float64(): w=wide(tokens,use_cache=False).logits.double().numpy()
  ni,ns,ng=routers['n'];wi,ws,wg=routers['w']
  nm=float(top_margin(ng,2).min());wm=float(top_margin(wg,2).min())
  vals={'n':n,'w':w,'ni':ni,'wi':wi,'ns':ns,'ws':ws,'ng':ng,'wg':wg,'nm':nm,'wm':wm,'ne':float(top_margin(ns,4).min()),'we':float(top_margin(ws,4).min()),'perturb':float(np.max(np.abs(ng-wg)/np.max(np.abs(wg),-1,keepdims=True)))}
  cache[key]=vals
 vals=cache[key]
 if ours and 'd' not in vals:
  with summed_as_run(): d,col=dew(jnp.asarray(ids));dp=plain(jnp.asarray(ids))
  d=np.asarray(d,np.float32)
  vals['d']=d;vals['same']=bool(np.array_equal(d,np.asarray(dp)))
  vals['di']=np.asarray(col['router']['layers_1']['mlp']['gate']['indices'][0]).reshape(-1,4)
  ds=np.asarray(col['router']['layers_1']['mlp']['gate']['scores'][0]).reshape(-1,8)
  bias=np.asarray(loaded.variables['moe']['layers_1']['mlp']['gate']['e_score_correction_bias'])
  vals['dg']=(ds+bias).reshape(-1,4,2).sum(-1)
 return vals

def record(seed):
 v=evaluate(raw(seed)[0]);wi=np.sort(v['wi'],-1)
 return {'seed':seed,'df':int(np.any(np.sort(v['di'],-1)!=wi,axis=-1).sum()),'nf':int(np.any(np.sort(v['ni'],-1)!=wi,axis=-1).sum()),'dm':float(np.mean((v['d'].astype(float)-v['w'])**2)),'nmse':float(np.mean((v['n'].astype(float)-v['w'])**2)),**{k:v[k] for k in ('nm','wm','ne','we','perturb','same')}}
mode=sys.argv[1]
if mode=='truthmeasure':
 first=int(sys.argv[2]);count=int(sys.argv[3]);found=[]
 for seed in range(first,first+count):
  ids=raw(seed)[0];v=evaluate(ids);tm,tg=true_selections(ids)
  scale=np.max(np.abs(v['wg']),-1,keepdims=True)
  ng=v['ng']-v['wg'];dg=v['dg']-v['wg']
  found.append((abs(tm-v['wm']),np.max(np.abs(tg-v['wg'])),np.max(np.abs(ng)/scale),np.max(np.abs(dg)/scale),np.max(np.ptp(ng,-1)/scale[:,0]),np.max(np.ptp(dg,-1)/scale[:,0]),tm,bool(np.any(np.sort(v['di'],-1)!=np.sort(v['wi'],-1))),bool(np.any(np.sort(v['ni'],-1)!=np.sort(v['wi'],-1)))))
 arr=np.asarray(found)
 print('K TRUTHMAX',first,count,arr[:,:6].max(0).tolist(),flush=True)
 for threshold in [.0078125,.015625,.03125,.0625,.125]:
  s=arr[arr[:,6]>threshold]
  print('K TRUTHMARGIN',threshold,'count',len(s),'flips',s[:,7:].sum(0).tolist(),flush=True)
elif mode=='force':
 count=int(sys.argv[2]);rows=[]
 for seed in range(count):
  ids=raw(seed)[0];v=evaluate(ids)
  forced[0]=v['wi']
  with torch.no_grad():n=narrow(torch.from_numpy(ids.astype(np.int64)),use_cache=False).logits.float().numpy()
  forced[0]=None
  with summed_as_run():d=np.asarray(dew_fixed(jnp.asarray(ids),jnp.asarray(v['wi'])))
  rows.append((float(np.mean((v['d'].astype(float)-v['w'])**2)),float(np.mean((v['n'].astype(float)-v['w'])**2)),float(np.mean((d.astype(float)-v['w'])**2)),float(np.mean((n.astype(float)-v['w'])**2))))
 arr=np.asarray(rows)
 print('K FORCED',count,'original RMS',np.sqrt(arr[:,:2].mean(0)).tolist(),'forced RMS',np.sqrt(arr[:,2:].mean(0)).tolist(),'original ratio',np.sqrt(arr[:,0].sum()/arr[:,1].sum()),'forced ratio',np.sqrt(arr[:,2].sum()/arr[:,3].sum()),flush=True)
elif mode=='measure':
 first=int(sys.argv[2]);count=int(sys.argv[3]);rows=[record(i) for i in range(first,first+count)]
 print('K SAME',all(r['same'] for r in rows),flush=True)
 for dk,nk in [(0,0),(1,0),(0,1),(1,1)]:
  s=[r for r in rows if bool(r['df'])==bool(dk) and bool(r['nf'])==bool(nk)]
  if s:print('K STRATUM',dk,nk,'count',len(s),'rms',np.sqrt(np.mean([r['dm'] for r in s])),np.sqrt(np.mean([r['nmse'] for r in s])),'ratio',np.sqrt(sum(r['dm'] for r in s)/sum(r['nmse'] for r in s)),flush=True)
 print('K TOTAL',len(rows),'ratio',np.sqrt(sum(r['dm'] for r in rows)/sum(r['nmse'] for r in rows)),'flipped_tokens',sum(r['df'] for r in rows),sum(r['nf'] for r in rows),flush=True)
 for key in ['nm','wm','ne','we','perturb']:
  print('K QUANTILE',key,np.quantile([r[key] for r in rows],[0,.1,.5,.9,.99,1]).tolist(),flush=True)
 for threshold in [1e-5,.001,.002,.003,.004,.005,.0075,.01,.015,.02,.03,.04]:
  s=[r for r in rows if min(r['nm'],r['wm'])>threshold]
  if s:print('K MARGIN',threshold,'count',len(s),'flips',sum(bool(r['df']) for r in s),sum(bool(r['nf']) for r in s),'ratio',np.sqrt(sum(r['dm'] for r in s)/sum(r['nmse'] for r in s)),flush=True)
 s=[r for r in rows if r['df'] or r['nf']]
 print('K ANYFLIP',len(s),'ratio',np.sqrt(sum(r['dm'] for r in s)/sum(r['nmse'] for r in s)),'share squared error',sum(r['dm'] for r in s)/sum(r['dm'] for r in rows),sum(r['nmse'] for r in s)/sum(r['nmse'] for r in rows),flush=True)
 if s:print('K MAX-FLIP-MARGIN',max(min(r['nm'],r['wm']) for r in s),flush=True)
 for r in sorted(rows,key=lambda r:r['dm']+r['nmse'],reverse=True)[:10]:print('K EXTREME',json.dumps(r),flush=True)
 # Each draw's flip classification and RMS errors, compact evidence.
 for r in [r for r in rows if r['df'] or r['nf']][:5]:
  print('K FLIPDRAW',r['seed'],r['df'],r['nf'],r['dm'],r['nmse'],r['nm'],r['wm'],flush=True)
else:
 threshold=float(sys.argv[2]);accepted=[];attempts=[0]
 def clears(inputs):
  attempts[0]+=1;v=evaluate(inputs[0],ours=False)
  ok=min(v['nm'],v['wm'])>threshold
  if ok:accepted.append(inputs[0].tobytes())
  return ok
 if mode=='truthgate':
  selection_margins.MARGIN=threshold
  def counted(seed):
   attempts[0]+=1
   return raw(seed)
  initial=clearing_selections(dew_truth,counted,'dsv3 truth',tries=256)
  def draw(seed):
   with jax.enable_x64():inputs=initial(seed)
   accepted.append(inputs[0].tobytes())
   return inputs
 else:draw=clearing(raw,clears,tries=64)
 @as_given
 def ref(ids):return evaluate(ids,ours=False)['n']
 @as_given
 def truth(ids):return evaluate(ids,ours=False)['w']
 def candidate(ids):return loaded.model.apply(loaded.variables,ids).astype(jnp.float32)
 try:
  seeds=seeds_for(ref,draw,'dsv3 margin '+str(threshold),truth_fn=truth,dew_fn=candidate)
  print('K SIZED',threshold,seeds,'attempts',attempts[0],flush=True)
  assert_as_exact(candidate,ref,draw,'dsv3 margin '+str(threshold),seeds=seeds,truth_fn=truth)
  print('K GREEN',threshold,seeds,'attempts',attempts[0],flush=True)
 except Exception as e:print('K FAIL',type(e).__name__,str(e),flush=True)
 vals=[evaluate(np.frombuffer(key,np.int32).reshape(2,12)) for key in set(accepted)]
 print('K ACCEPTED',len(vals),'attempts',attempts[0],'flips',sum(np.any(np.sort(v['di'],-1)!=np.sort(v['wi'],-1)) for v in vals),sum(np.any(np.sort(v['ni'],-1)!=np.sort(v['wi'],-1)) for v in vals),flush=True)
 print('K RATIO',np.sqrt(sum(np.sum((v['d'].astype(float)-v['w'])**2) for v in vals)/sum(np.sum((v['n'].astype(float)-v['w'])**2) for v in vals)),flush=True)
