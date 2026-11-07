"""Scratch check-1 reproduction: STEPS BATCH WIDTH MOST FIRST MODE.

MODE is old or wide (sizing and a verdict on disjoint seeds); summary modes
old-summary and wide-summary keep each step's tail measurements instead.
Run only through the armada probe runner, from the repository root.
"""
import os, sys, json, time
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/"tests"),str(Path.cwd()/"tools/reference_runs")]
import lane_environment
import numpy as np
import jax, jax.numpy as jnp
import torch
import torch.nn.functional as F
import contextlib
sys.path.insert(0,str(Path.cwd()/"tools"))
from torch_precision import float64
from transformers import AutoModelForCausalLM
from common import warmup_cosine
from arm_arithmetic import summed_as_run
from dew.batch import TEXT_KEY
from dew.interop import Pretrained
from dew.objectives.lm import LMObjective
from dew.optim import Cosine, OptimConfig, ParamGroup
from dew.training import MeshSpec, Trainer
from check1_statistics import _interval, _sized, _log_ratios
steps,batch,width,count,first=map(int,sys.argv[1:6])
mode=sys.argv[6]
wide=mode.startswith("wide")
def oracle_scope(dtype):
    return float64() if wide and dtype==torch.float64 else contextlib.nullcontext()
torch.set_num_threads(1)
root=Path.cwd()/"tests/fixtures/hf/qwen3-tiny"
options=dict(lr_peak=1e-2,lr_init=1e-3,lr_end=1e-3,warmup=2)
schedule=Cosine(peak_value=1e-2,warmup_steps=2,end_value=1e-3,init_value=1e-3)
solver=OptimConfig(optimizer="adamw",b1=.9,b2=.95,optimizer_opts={"eps":1e-8},learning_rate=schedule,weight_decay=.1,clip_grads=1.,param_groups=(ParamGroup("all",("*",)),)).build(steps)
pre=Pretrained.load(str(root),dtype="float32",attention_impl="xla")
trainer=Trainer(LMObjective(pre,width-1,ema_decay=None),solver,key=jax.random.key(0),mesh=MeshSpec(),checkpoints=None,tracker=None)
with summed_as_run():
    initial,_,_=trainer.place()
    step=trainer.compile(initial,{TEXT_KEY:np.zeros((batch,width),np.int32)})
# Host copy independent of donated state
initial=jax.device_get(initial)
def build(dtype):
    with oracle_scope(dtype):
        return AutoModelForCausalLM.from_pretrained(root,dtype=dtype,attn_implementation="eager").train()
models={dtype:build(dtype) for dtype in (torch.float32,torch.float64)}
weights={dtype:{k:v.clone() for k,v in model.state_dict().items()} for dtype,model in models.items()}
margin_threshold=float(sys.argv[8]) if len(sys.argv)>8 else 1e-6
attempts=[]
def tokens_of(seed):
    for attempt in range(128):
        key=seed if attempt==0 else 1_000_003*(attempt+1)+seed
        tokens=np.random.default_rng([45,key]).integers(0,256,(steps,batch,width)).astype(np.int32)
        if "margin" not in mode:return tokens
        with float64():
            model=models[torch.float64]
            model.load_state_dict(weights[torch.float64]);model.zero_grad(set_to_none=True)
            rows=torch.from_numpy(tokens[0].astype(np.int64))
            logits=model(input_ids=rows[:,:-1]).logits
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),rows[:,1:].reshape(-1))
            loss.backward()
            margin=min(float(p.grad.detach().abs().min()/p.grad.detach().abs().max())
                       for p in model.parameters() if p.grad is not None and p.grad.detach().abs().max()>0)
        if margin>margin_threshold:
            attempts.append(attempt+1)
            return tokens
    raise AssertionError(f"seed {seed} did not clear first-gradient margin {margin_threshold}")

def reference_inside(tokens,dtype):
    model=models[dtype]; model.load_state_dict(weights[dtype]); model.zero_grad(set_to_none=True)
    optim=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
    out=[]
    for i,rows in enumerate(tokens):
        rate=warmup_cosine(i,init=.001,peak=.01,warmup=2,decay_steps=steps,end=.001)
        for group in optim.param_groups: group['lr']=rate
        rows=torch.from_numpy(rows.astype(np.int64))
        logits=model(input_ids=rows[:,:-1]).logits
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),rows[:,1:].reshape(-1))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step();optim.zero_grad(set_to_none=True)
        out.append([float(loss.detach()),float(norm.detach())])
    return np.array(out)

def reference(tokens,dtype):
    with oracle_scope(dtype):
        return reference_inside(tokens,dtype)

def dew(tokens):
    with summed_as_run():
        state=jax.tree.map(lambda a:jnp.array(a,copy=True),initial)
        out=[]
        for rows in tokens:
            state,loss,_,_,_,commit=step(state,{TEXT_KEY:rows})
            out.append([float(loss),float(commit.norm)])
    return np.array(out)
if not mode.endswith("summary"):
    from reference_error import as_given,seeds_for,assert_as_exact
    cache={}
    def get(seed):
        seed=int(seed)
        if seed not in cache:
            tokens=tokens_of(seed)
            cache[seed]=(dew(tokens),reference(tokens,torch.float32),reference(tokens,torch.float64))
            if len(cache)%128==0:print("K EVALUATED",len(cache),flush=True)
        return cache[seed]
    split=sys.argv[7] if len(sys.argv)>7 else "pooled"
    def leaves(a):
        if split=="loss":return {"loss":a[:,0]}
        if split=="steps":return {f"{k}-{s}":a[s,i] for s in range(steps) for i,k in enumerate(["loss","norm"])}
        return {"loss":a[:,0],"norm":a[:,1]}
    candidate=as_given(lambda seed:leaves(get(seed)[0]))
    ref=as_given(lambda seed:leaves(get(seed)[1]))
    truth=as_given(lambda seed:leaves(get(seed)[2]))
    try:
        n=seeds_for(ref,lambda seed:(np.int32(seed),),"LM steps",dew_fn=candidate,truth_fn=truth,most=count)
        print("K DECLARED",n,flush=True)
        assert_as_exact(candidate,ref,lambda seed:(np.int32(seed),),"LM steps",seeds=n,truth_fn=truth)
        print("K PASS",n,flush=True)
    except Exception as error:
        print("K FAIL",type(error).__name__,str(error).replace("\n"," ")[:550],flush=True)
        raise
    finally:
        if attempts:print("K MARGIN",margin_threshold,"draws",len(attempts),"attempts",sum(attempts),"max",max(attempts),flush=True)
    raise SystemExit
start=time.time(); ours=[]; theirs=[]
for seed in range(first,first+count):
    tokens=tokens_of(seed)
    truth=reference(tokens,torch.float64); ref=reference(tokens,torch.float32); candidate=dew(tokens)
    ours.append((candidate-truth)**2);theirs.append((ref-truth)**2)
    if seed%128==0: print('K PROGRESS '+str(seed),flush=True)
ours=np.array(ours);theirs=np.array(theirs)
print('K DONE '+json.dumps(dict(steps=steps,batch=batch,width=width,count=count,seconds=time.time()-start)),flush=True)
for s in range(steps):
    for k,name in enumerate(['loss','norm']):
        a,b=ours[:,s,k],theirs[:,s,k]
        ratios=np.sqrt(a/np.maximum(b,1e-300))
        print('K STATS '+json.dumps(dict(step=s,leaf=name,ratio=float(np.sqrt(a.sum()/b.sum())),interval=list(_interval(a,b)),rmse=[float(np.sqrt(a.mean())),float(np.sqrt(b.mean()))],top1=[float(a.max()/a.sum()),float(b.max()/b.sum())],q99=[float(np.quantile(a,.99)/np.median(a)),float(np.quantile(b,.99)/np.median(b))])),flush=True)
        print('K TAIL '+json.dumps(dict(step=s,leaf=name,ratio_quantiles=np.quantile(ratios,[.1,.5,.9,.99]).tolist(),tail_seed=int(first+np.argmax(a)),tail_ratio=float(ratios[np.argmax(a)]))),flush=True)
for mode,a,b in [('pooled',ours.mean(axis=1),theirs.mean(axis=1)),('steps',ours.reshape(count,-1),theirs.reshape(count,-1))]:
    found=None
    for n in [8,16,32,64,128,256,512]:
        if count<2*n: break
        if all(_sized(a[t:t+n,k],b[t:t+n,k]) for t in [0,n] for k in range(a.shape[1])):
            found=n;break
    print('K SIZE '+json.dumps(dict(mode=mode,seeds=found,batch=batch,steps=steps)),flush=True)
    for k in range(a.shape[1]):
        if mode=="pooled":
            print('K POOL '+json.dumps(dict(mode=mode,k=k,ratio=float(np.sqrt(a[:,k].sum()/b[:,k].sum())),interval=_interval(a[:,k],b[:,k]))),flush=True)
