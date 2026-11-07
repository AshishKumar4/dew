"""Instrumentation for ledger 4741; run on armada with KIND red|green|seeded.

`red` reproduces the recorder's altered quantizer; `green` restores native
Tensor.to only inside it and demands exact agreement with CT's fp32 QDQ.
`seeded` also compares complete model truths and runs check 1 over eight
draws. This file changes no product code or fixtures.
"""

import sys,json,contextlib
sys.path[:0]=['tools','tests','.']
import lane_environment
import numpy as np, torch, jax, jax.numpy as jnp
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.torch_precision import float64,float64_twin
from compressed_tensors.quantization import QuantizationArgs,QuantizationStatus
import compressed_tensors.quantization.lifecycle.forward as f
import compressed_tensors.quantization.lifecycle.forward_helpers as fh
from dew.quantization import InputQuantizer,nvfp4_input_qdq
from dew.interop import Pretrained
from float64_terms import float64_forward
from transformers import AutoModelForCausalLM
jax.config.update('jax_enable_x64',True);torch.set_num_threads(1)
kind=sys.argv[1];d=Path('tests/fixtures/codecs/nvfp4_qdq')/kind;r=np.load(d/'reference.npz');name='model.layers.0.self_attn.q_proj'
args=QuantizationArgs(**json.load(open(d/'config.json'))['quantization_config']['config_groups']['group_0']['input_activations'])
x=torch.from_numpy(r[name+'/inputs']);g=torch.from_numpy(r[name+'/global']);m=SimpleNamespace(quantization_status=QuantizationStatus.COMPRESSED,input_global_scale=g)
direct_float=torch.Tensor.float; direct_to=torch.Tensor.to
original_fake=f.fake_quantize;original_round=fh.round_to_quantized_type_args
held={}
def fake(**kw):
 for k in ['x','scale','zero_point','global_scale']:held[k]=kw[k].double().numpy().copy()
 held['scale_dtype']=str(kw['scale'].dtype)
 return original_fake(**kw)
def rounded(*a,**kw):
 held['quotient']=kw['tensor'].double().numpy().copy();held['quotient_dtype']=str(kw['tensor'].dtype)
 y=original_round(*a,**kw);held['codes']=y.double().numpy().copy();held['codes_dtype']=str(y.dtype);return y
def qdq(mode):
 with torch.no_grad(),patch.object(f,'fake_quantize',fake),patch.object(fh,'round_to_quantized_type_args',rounded):
  with float64() if mode else contextlib.nullcontext():
   with patch.object(torch.Tensor,'to',direct_to) if mode==2 else contextlib.nullcontext():y=f.forward_quantize(m,x,'input',args)
 return y.numpy(),dict(held)
def cmp(n,a,b):
 a=np.asarray(a);b=np.asarray(b);print('K',kind,n,'dtype',a.dtype,b.dtype,'rms',np.sqrt(np.mean((a.astype(float)-b.astype(float))**2)),'max',np.max(abs(a.astype(float)-b.astype(float))),'neq',np.sum(a!=b))
ys=[];hs=[]
for mode in range(3):
 y,h=qdq(mode);ys.append(y);hs.append(h);print('K',kind,'mode',mode,{k:v for k,v in h.items() if k.endswith('dtype')})
for k in ['x','scale','zero_point','global_scale','quotient','codes']:cmp('stage-'+k,hs[0][k],hs[1][k])
for mode in (1,2):cmp('qdq-normal-mode'+str(mode),ys[0],ys[mode])
if len(sys.argv)>2 and sys.argv[2] in ('red','green'):
 selected=1 if sys.argv[2]=='red' else 2
 np.testing.assert_array_equal(ys[0],ys[selected])
 print('K',kind,'quantizer regression GREEN')
 raise SystemExit(0)
# Show native dequantized product versus the widened dequantized product for equal codes/scales.
effective=hs[1]['scale']/hs[1]['global_scale']
if kind=='unrounded':effective=effective.astype(np.float32).astype(float)
codes=hs[1]['codes'];effective=effective[...,None]
cmp('widened-is-f64-product',ys[1],(codes.astype(float)*effective.astype(float)).reshape(x.shape))
s=InputQuantizer.nvfp4(g.item(),kind=='e4m3');dew=np.asarray(nvfp4_input_qdq(jnp.asarray(x.numpy(),jnp.float64),s));cmp('dew-ct-normal',dew,ys[0]);cmp('dew-ct-repaired',dew,ys[2])
# Full fixture's truth, with identical readers and one toggled cast patch.
model=AutoModelForCausalLM.from_pretrained(d,dtype=torch.float32,attn_implementation='eager').eval();torch.nn.Module.float(model)
exact=float64_twin(model,lambda:AutoModelForCausalLM.from_pretrained(d,dtype=torch.float64,attn_implementation='eager'))
for net in (model,exact):
 for mod in net.modules():
  if hasattr(mod,'quantization_status'):mod.quantization_status=QuantizationStatus.COMPRESSED
original=f.forward_quantize
mode=1
def quantize_wide(module,value,base_name,args):
 overall=module.input_global_scale;module.input_global_scale=torch.nn.Parameter(direct_float(overall),requires_grad=False)
 try:
  with patch.object(torch.Tensor,'to',direct_to) if mode==2 else contextlib.nullcontext():
   result=original(module,direct_float(value),base_name,args).double()
  return result
 finally:module.input_global_scale=overall
ids=torch.from_numpy(r['ids']).long();full=[]
for mode in (1,2):
 with float64(),torch.no_grad(),patch.object(f,'forward_quantize',quantize_wide):full.append(exact(ids,use_cache=False).logits.numpy())
loaded=Pretrained.load(d,dtype='float32',attention_impl='reference');dewtruth,_=float64_forward(loaded.model,loaded.variables,r['ids'])
if isinstance(dewtruth,dict):dewtruth=dewtruth['logits']
cmp('existing-record-v-original-truth',r['logits_f64'],full[0]);cmp('dewtruth-v-original',dewtruth,full[0]);cmp('dewtruth-v-repaired',dewtruth,full[1]);cmp('reference-f32-v-repairedtruth',r['logits'],full[1])
if len(sys.argv)>2 and sys.argv[2]=='seeded':
 from reference_error import assert_as_exact,as_given
 from dew.registry import ModelConfig
 wide=ModelConfig.from_model(loaded.model).with_dtype('float64').build()
 variables=jax.tree.map(lambda leaf:jnp.asarray(leaf,jnp.float64) if jnp.issubdtype(jnp.asarray(leaf).dtype,jnp.floating) else leaf,loaded.variables)
 dew32=jax.jit(lambda ids:loaded.model.apply(loaded.variables,ids))
 dew64=jax.jit(lambda ids:wide.apply(variables,ids))
 def draw(seed):return (np.random.default_rng([47,seed]).integers(0,96,r['ids'].shape).astype(np.int32),)
 def narrow(ids):
  with torch.no_grad():return model(torch.from_numpy(np.asarray(ids)).long(),use_cache=False).logits.numpy()
 largest=0.
 def truth(ids):
  global largest,mode
  mode=2
  with float64(),torch.no_grad(),patch.object(f,'forward_quantize',quantize_wide):
   truth=exact(torch.from_numpy(np.asarray(ids)).long(),use_cache=False).logits.numpy()
  largest=max(largest,float(np.sqrt(np.mean((np.asarray(dew64(ids))-truth)**2))))
  return truth
 assert_as_exact(as_given(dew32),as_given(narrow),draw,'codec '+kind,seeds=8,truth_fn=as_given(truth),terms=False)
 print('K',kind,'seeded check1 GREEN','largest Dew/CT float64 truth RMS',largest)
