import sys; sys.path[:0]=["src","tests","."]
import os, inspect, urllib.request, json, time
if sys.argv[1].startswith(('gate','stagesgate')):
 os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
 os.environ['XLA_FLAGS']=os.environ.get('XLA_FLAGS','')+' --xla_allow_excess_precision=false'
print('K START',sys.argv,flush=True)
try:
 r=urllib.request.urlopen('https://huggingface.co/AntonV/mamba2-130m-hf/resolve/05e8773fc4ac1cd067e8a18a5c45372ce5178405/config.json',timeout=30)
 print('K NETWORK',r.status, len(r.read()),flush=True)
except Exception as e:
 print('K NETWORK_FAIL',repr(e),flush=True); raise
import numpy as np, torch, jax, jax.numpy as jnp
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.mamba2 import modeling_mamba2 as mm
from tools.torch_precision import float64, float64_twin
from dew.interop import Pretrained
from dew.nn.mixers.mamba2 import Mamba2
from ssd_support import mamba2_layer_params
name='AntonV/mamba2-130m-hf';rev='05e8773fc4ac1cd067e8a18a5c45372ce5178405'
torch.set_num_threads(1 if sys.argv[1].startswith(('gate','stagesgate')) else 2)
jax.config.update('jax_enable_x64',True)
mode=sys.argv[1]
direct=mode.endswith('_direct')
if direct:mode=mode[:-7]
if mode.startswith(('gate','stagesgate')):
 from dew.nn.precision import FP32_SUMS
 jax.config.update('jax_default_matmul_precision',FP32_SUMS)
 if mode in ('gate','gate_actual'):mode='control'
if mode.startswith('stageswide'):os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
if mode=='source':
 for obj in [getattr(mm.Mamba2Mixer,'torch_forward',mm.Mamba2Mixer.forward),mm.mamba2_chunk_scan,mm.segment_sum]:
  print('K SOURCE',obj.__name__,inspect.getfile(obj),flush=True)
  for n,line in enumerate(inspect.getsource(obj).splitlines(),inspect.getsourcelines(obj)[1]):print('K SRC',n,line,flush=True)
 sys.exit()
text=('The Cascade Range runs from northern California through Oregon and Washington into '
'British Columbia, and its volcanoes include Mount Rainier, Mount Hood and Mount St. '
'Helens, which erupted in 1980. The capital of France is')
ids=AutoTokenizer.from_pretrained(name,revision=rev)(text,return_tensors='np')['input_ids'];tokens=np.unique(ids)
print('K TOKENS',ids.shape,flush=True)
source=AutoModelForCausalLM.from_pretrained(name,revision=rev,dtype=torch.float32).eval()
twin=float64_twin(source)
print('K MODELS',jax.devices(),next(source.parameters()).device,flush=True)
def rms(x): return float(np.sqrt(np.mean(np.square(np.asarray(x,dtype=np.float64)))))
def emit(label,a,b,t):
 a=np.asarray(a);b=np.asarray(b);t=np.asarray(t)
 ea,eb=rms(a-t),rms(b-t)
 print('K ERROR',label,'dew',ea,'torch',eb,'ratio',ea/eb if eb else 0,'maxdelta',np.max(np.abs(a-b)),flush=True)
if mode in ('control','fixed','control_dyn','dense4','densewide','denseplain','wide_cpu'):
 if mode=='wide_cpu':
  from dew.nn import precision
  os.environ[precision.WIDE_CPU_ARITHMETIC]='1'
 if mode in ('dense4','densewide','denseplain'):
  import dew.nn.mixers.mamba2 as dm
  original_dot=dm.float32_dot_general
  def altered(lhs,rhs,dimension_numbers,precision=None,preferred_element_type=None,**kwargs):
   if mode=='denseplain':return original_dot(lhs,rhs,dimension_numbers,precision=precision,preferred_element_type=preferred_element_type,**kwargs)
   ((lc,rc),(lb,rb))=dimension_numbers
   if mode=='densewide':
    return jax.lax.dot_general(lhs.astype(jnp.float64),rhs.astype(jnp.float64),dimension_numbers,precision=precision,**kwargs).astype(jnp.result_type(lhs,rhs))
   assert len(lc)==len(rc)==1
   ll=jnp.split(lhs,4,axis=lc[0]);rr=jnp.split(rhs,4,axis=rc[0])
   terms=[jax.lax.dot_general(a,b,dimension_numbers,precision=precision,**kwargs) for a,b in zip(ll,rr)]
   return (terms[0]+terms[1])+(terms[2]+terms[3])
  dm.float32_dot_general=altered
 if mode=='fixed':
  import dew.nn.kernels.recurrence_algebra as ra
  from dew.algebra import forms
  print('K FORMS',forms,flush=True)
 print('K LOAD_START',time.time(),flush=True)
 if direct:
  from types import SimpleNamespace
  from dew.interop import mamba2 as adapter
  from dew.interop.hf_decoders import translate_weights
  config=adapter.config_from_hf(source.config.to_dict())
  model=config.value.clone(dtype=jnp.float32,attention_impl='xla')
  variables=translate_weights({k:v.detach().numpy() for k,v in source.state_dict().items()},config,'mamba2')
  loaded=SimpleNamespace(model=model,variables=variables)
 else:loaded=Pretrained.load(name,revision=rev,dtype='float32',param_dtype='float32',attention_impl='xla')
 print('K LOAD_DONE',time.time(),flush=True)
 print('K ARITHMETIC',os.environ.get('DEW_WIDE_CPU_ARITHMETIC'),jax.config.jax_default_matmul_precision,os.environ.get('XLA_FLAGS'),flush=True)
 actual=sys.argv[1]=='gate_actual'
 run=jax.jit(lambda x:loaded.model.apply(loaded.variables,x)) if actual else jax.jit(loaded.model.apply)
 errors=[]
 for seed in range(16):
  drawn=np.random.default_rng([242,seed]).choice(tokens,size=ids.shape).astype(np.int32)
  with torch.no_grad():
   b=source(torch.from_numpy(drawn.astype(np.int64)),use_cache=False).logits.numpy()
   with float64():t=twin(torch.from_numpy(drawn.astype(np.int64)),use_cache=False).logits.numpy()
  a=np.asarray(run(jnp.asarray(drawn)) if actual else run(loaded.variables,jnp.asarray(drawn)))
  emit(str(seed),a,b,t)
  errors.append((rms(a-t),rms(b-t)))
 print('K POOLED',float(np.linalg.norm(np.asarray(errors)[:,0])/np.linalg.norm(np.asarray(errors)[:,1])),flush=True)
 sys.exit()
if mode=='layers' or mode.startswith('stages'):
 drawn=np.random.default_rng([242,0]).choice(tokens,size=ids.shape).astype(np.int32)
 saved={}
 def hook(i):
  def f(m,args,out):saved[i]=(args[0].detach().numpy(),out.detach().numpy())
  return f
 handles=[l.mixer.register_forward_hook(hook(i)) for i,l in enumerate(twin.backbone.layers)]
 with torch.no_grad(),float64():twin(torch.from_numpy(drawn.astype(np.int64)),use_cache=False)
 for h in handles:h.remove()
 cfg=source.config
 print('K CONFIG',cfg.to_dict(),flush=True)
 model=Mamba2(emb_features=cfg.hidden_size,num_heads=cfg.num_heads,head_dim=cfg.head_dim,state_size=cfg.state_size,n_groups=cfg.n_groups,conv_kernel=cfg.conv_kernel,chunk_size=cfg.chunk_size,use_bias=cfg.use_bias,use_conv_bias=cfg.use_conv_bias,norm_eps=cfg.layer_norm_epsilon,dtype=jnp.float32)
 run=jax.jit(model.apply)
 for i,l in enumerate(source.backbone.layers):
  if mode.startswith('stages') and mode.split(':')[1]!='all' and i!=int(mode.split(':')[1]):continue
  x,t=saved[i]
  with torch.no_grad():b=getattr(l.mixer,'torch_forward',l.mixer.forward)(torch.from_numpy(x.astype(np.float32))).numpy()
  p=mamba2_layer_params({k:v.detach().numpy() for k,v in l.mixer.state_dict().items()})
  a=np.asarray(run(p,jnp.asarray(x,dtype=jnp.float32)))
  emit(str(i),a,b,t)
  if mode.startswith('stages'):
   import torch.nn.functional as F
   from dew.nn.linear import causal_conv1d
   from dew.nn.mixers.mamba2 import chunk_ssd
   from dew.nn.kernels import recurrence_chunk as rc, recurrence_algebra as ra
   narrow=l.mixer;wide=twin.backbone.layers[i].mixer
   p32=p
   def ja(v):return jnp.asarray(v.detach().numpy() if isinstance(v,torch.Tensor) else v,dtype=jnp.float32)
   def ta(v):return torch.from_numpy(np.asarray(v,dtype=np.float32).copy())
   def dwmethod(fn,v):return np.asarray(jax.jit(lambda z:model.apply(p32,z,method=fn))(ja(v)))
   with torch.no_grad():
    tx=torch.from_numpy(x)
    pt=wide.in_proj(tx);pn=narrow.in_proj(ta(x))
    emit('in_proj',dwmethod(lambda m,z:m.in_proj(z),tx),pn.numpy(),pt.numpy())
    def dots(label,inp,weight,truth,reference):
     xx=ja(inp);ww=ja(weight).T
     rounded_truth=np.asarray(xx,dtype=np.float64)@np.asarray(ww,dtype=np.float64)
     for blocks in (1,2,4,8,16,32):
      def product(x,w):
       xs=jnp.split(x,blocks,axis=-1);ws=jnp.split(w,blocks,axis=0)
       terms=[jnp.matmul(a,b,precision=jax.lax.Precision.HIGHEST) for a,b in zip(xs,ws)]
       while len(terms)>1:terms=[a+b for a,b in zip(terms[::2],terms[1::2])]
       return terms[0]
      out=np.asarray(jax.jit(product)(xx,ww))
      emit(label+'-blocks-'+str(blocks),out,reference,truth)
      print('K DOTINTRINSIC',label,blocks,rms(out-rounded_truth),rms(reference-rounded_truth),flush=True)
     if label=='in_proj' and jax.default_backend()=='cpu':
      for chains in (1,2,4,8):
       def accumulated(x,w):
        xx=jnp.moveaxis(x,-1,0).reshape(-1,chains,*x.shape[:-1])
        ww=w.reshape(-1,chains,w.shape[-1])
        def one(acc,parts):
         v,k=parts
         return acc+v[...,None]*k[:,None,None,:],None
        final,_=jax.lax.scan(one,jnp.zeros((chains,*x.shape[:-1],w.shape[-1]),jnp.float32),(xx,ww))
        while final.shape[0]>1:final=final[::2]+final[1::2]
        return final[0]
       got=np.asarray(jax.jit(accumulated)(xx,ww))
       base=np.asarray(jax.jit(lambda x,w:x@w)(xx,ww))
       print('K DOTCHAINS',chains,'native-delta',rms(got-base),'equal',np.mean(got==base),'error',rms(got-rounded_truth),flush=True)
    dots('in_proj',tx,narrow.in_proj.weight,pt.numpy(),pn.numpy())
    gate,mixed,raw=pt.split([wide.intermediate_size,wide.conv_dim,wide.num_heads],dim=-1)
    ct=F.silu(wide.conv1d(mixed.transpose(1,2))[:,:,:mixed.shape[1]]).transpose(1,2)
    cn=F.silu(narrow.conv1d(ta(mixed.numpy()).transpose(1,2))[:,:,:mixed.shape[1]]).transpose(1,2)
    ca=jax.jit(lambda z:causal_conv1d(z,p32['params']['conv1d']['weight'].squeeze(1),bias=p32['params']['conv1d']['bias']))(ja(mixed))
    emit('conv1d',ca,cn.numpy(),ct.numpy())
    at=-torch.exp(wide.A_log);an=-torch.exp(narrow.A_log)
    aa=-jnp.exp(p32['params']['A_log']);emit('A',aa,an.numpy(),at.numpy())
    dt=F.softplus(raw+wide.dt_bias)
    dn=F.softplus(ta(raw.numpy())+narrow.dt_bias)
    da=jax.nn.softplus(ja(raw)+p32['params']['dt_bias']);emit('dt',da,dn.numpy(),dt.numpy())
    xt,bt,ctt=ct.split([wide.intermediate_size,wide.n_groups*wide.ssm_state_size,wide.n_groups*wide.ssm_state_size],dim=-1)
    xt=xt.reshape(*xt.shape[:2],wide.num_heads,wide.head_dim)
    bt=bt.reshape(*bt.shape[:2],wide.n_groups,wide.ssm_state_size)
    ctt=ctt.reshape(*ctt.shape[:2],wide.n_groups,wide.ssm_state_size)
    captured={}
    def tracer(frame,event,arg):
     if event=='return' and frame.f_code.co_name=='mamba2_chunk_scan':
      captured.update(frame.f_locals)
     return tracer
    def ts(x,dt,a,b,c,d):
     sys.settrace(tracer)
     try:return mm.mamba2_chunk_scan(x,dt,a,b,c,chunk_size=wide.chunk_size,D=d,return_final_states=True)
     finally:sys.settrace(None)
    with float64():yt,ft=ts(xt,dt,at,bt,ctt,wide.D)
    exact={k:v.detach().numpy() for k,v in captured.items() if isinstance(v,torch.Tensor)}
    yn,fn=ts(*(ta(v.numpy()) for v in (xt,dt,at,bt,ctt,wide.D)))
    torch32={k:v.detach().numpy() for k,v in captured.items() if isinstance(v,torch.Tensor)}
    print('K CAPTURED',list(exact),flush=True)
    xa,dta,aaa,ba,cca,dda=(ja(v) for v in (xt,dt,at,bt,ctt,wide.D))
    ya,fa=jax.jit(lambda *v:chunk_ssd(*v,chunk_size=wide.chunk_size))(xa,dta,aaa,ba,cca,dda)
    emit('SSD-output',ya,yn.numpy(),yt.numpy());emit('SSD-final',fa,fn.numpy(),ft.numpy())
    print('K SSD_ROUTE',rc.scan_kernel(cca,ba,xa,aaa*dta,dta,chunk_size=wide.chunk_size),flush=True)
    if jax.default_backend()=='gpu':
     from dew.nn.mixers.mamba2 import recurrent_ssd
     ryo,ryf=jax.jit(recurrent_ssd)(xa,dta,aaa,ba,cca,dda)
     emit('SSD-recurrent-output',ryo,yn.numpy(),yt.numpy())
     for size in (16,32,64,128):
      sy,sf=jax.jit(lambda *v:chunk_ssd(*v,chunk_size=size))(xa,dta,aaa,ba,cca,dda)
      emit('SSD-chunk-'+str(size),sy,yn.numpy(),yt.numpy())
     if hasattr(rc,'chunk_delta_rule'):
      heads=wide.num_heads//wide.n_groups
      pyo,pyf=jax.jit(lambda q,k,v,g,b:rc.chunk_delta_rule(q,k,v,g[...,None],b,None,wide.chunk_size,delta=False,scale=1.0,dtype=jnp.float32))(jnp.repeat(cca,heads,axis=2),jnp.repeat(ba,heads,axis=2),xa,aaa*dta,dta)
      pyo=pyo+xa*dda[...,None]
      emit('SSD-portable-output',pyo,yn.numpy(),yt.numpy())
    ng=wide.num_heads//wide.n_groups
    bb,cc=(jnp.repeat(v,ng,axis=2) for v in (ba,cca))
    if hasattr(rc,'chunked'):
     st=jax.jit(lambda q,k,v,g,b:rc.chunked(q,k,v,g,b,wide.chunk_size,delta=False,scale=1.0))(cc,bb,xa,(aaa*dta)[...,None],dta)
     # Torch [b,nchunk,l,s,h] versus portable [b,h,nchunk,l,s].
     def tl(v):return np.asarray(v).transpose(0,2,3,4,1)
     emit('SSD-decay-L',tl(st.pairs[...,0]),exact['L'].transpose(0,2,3,4,1) if False else torch32['L'].transpose(0,2,3,4,1),exact['L'].transpose(0,2,3,4,1))
     scores=np.asarray(ra.paired('scores',st.query,st.key,st.pairs))
     emit('SSD-weighted-scores',tl(scores),torch32['M'],exact['M'])
     diag=np.asarray(ra.remembered(jnp.asarray(scores),st.u,name='local')).transpose(0,2,3,1,4)
     emit('SSD-intra-output',diag,torch32['Y_diag'],exact['Y_diag'])
     zero=jnp.zeros((xa.shape[0],wide.num_heads,wide.ssm_state_size,wide.head_dim),jnp.float32)
     entered,corrected,final=rc.xla_chunk_states(st.key,st.w,st.u,st.grown[...,-1:,:],st.last,zero,st.kept)
     emit('SSD-chunk-state',np.asarray(final).swapaxes(-1,-2),torch32['final_state'],exact['final_state'])
     print('K INTERCHUNK',st.key.shape[2],'entering-rms',rms(entered),'torch-Yoff-rms',rms(torch32['Y_off']),flush=True)
    with float64():gt=wide.norm(yt.reshape(*yt.shape[:2],-1),gate)
    gn=narrow.norm(ta(yt.reshape(*yt.shape[:2],-1).numpy()),ta(gate.numpy()))
    ga=jax.jit(lambda z,g:model.apply(p32,z,g,method=lambda m,v,g:m.norm(v,g)))(ja(yt.reshape(*yt.shape[:2],-1)),ja(gate))
    emit('gated-norm',ga,gn.numpy(),gt.numpy())
    ot=wide.out_proj(gt);on=narrow.out_proj(ta(gt.numpy()))
    oa=dwmethod(lambda m,z:m.out_proj(z),gt)
    emit('out_proj',oa,on.numpy(),ot.numpy())
    dots('out_proj',gt,narrow.out_proj.weight,ot.numpy(),on.numpy())
 print('K DONE',flush=True)
