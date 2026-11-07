import sys; sys.path[:0]=["src","tests","."]
import functools
import jax
import jax.numpy as j
import numpy as np
from dew.algebra import scan
from dew.nn.kernels import recurrence_chunk as rc
from test_recurrence_chunk import portable_draw, portable_rule
from test_contracts_recurrence import precedent_chunked
from contract_support import normal_float32

jax.config.update("jax_enable_x64", False)
mode = sys.argv[1] if len(sys.argv)>1 else "base"
if "inverseguard" in mode:
    original_doubled=scan._doubled
    def safe_issue(role,left,right):
        l,r=left[:,:,None],right[None,:,:]
        active=(l!=0)&(r!=0)
        return j.sum(j.where(active,l,0.)*j.where(active,r,0.),axis=1)
    def doubled(overlaps,issue=None):
        return original_doubled(overlaps,safe_issue)
    scan._doubled=doubled
if "readguard" in mode:
    original_issue=scan._issued
    def issued(role,left,right):
        if role=="readout_memory" or ("stateguard" in mode and role=="correct"):
            return jax.vmap(lambda row:scan.float32_matmul(row[None,:],j.where(j.any(row!=0),right,0.))[0])(left)
        return original_issue(role,left,right)
    scan._issued=issued
if "replay" in mode:
    original_derive=scan.derive_scan
    def derive(declaration,**kwargs):
        forms=original_derive(declaration,**kwargs)
        names=tuple(n for n in declaration.operands if n!=declaration.reset)
        def replay(h,chunk):
            if "parts" in mode:
                from dew.algebra import linear_scan
                affine=linear_scan.parts(scan._traced(declaration,declaration.update,names,chunk),
                                         scan._traced(declaration,declaration.readout,names,chunk))
                tokens=scan._token_parts(declaration,names,affine,chunk)
                def token(h,xs):
                    z,B,U,s,C,Q,R=xs
                    h=j.where((z<=scan.vanishing(z.dtype,affine.base2))[:,None],0.,h)
                    d=scan._exp(affine.base2,z)[:,None]*h
                    h=d+B[:,None]*(U-s*(C@d))[None,:]
                    return h,Q@h+R
                h,out=jax.lax.scan(token,h,tokens)
                return out,h
            def token(h,xs):
                if declaration.reset:
                    h=j.where(xs[declaration.reset],0.,h)
                args=[xs[n] for n in names]
                h=declaration.update(h,*args)
                return h,declaration.readout(h,*args)
            h,out=jax.lax.scan(token,h,chunk)
            return out,h
        def body(h,chunk):
            p=forms.prepare(chunk)
            state=(j.zeros((h.shape[0],),h.dtype),h)
            return forms.mapped(p,state),forms.step(state,p)[1]
        rule=jax.custom_vjp(body)
        def fwd(h,chunk):
            return body(h,chunk),(h,chunk)
        def back(saved,c):
            return jax.vjp(replay,*saved)[1](c)
        rule.defvjp(fwd,back)
        def prepare(chunk,*args,**kwargs):return chunk
        def step(state,chunk,*args,**kwargs):return state[0],rule(state[1],chunk)[1]
        def mapped(chunk,state,*args,**kwargs):return rule(state[1],chunk)[0]
        return forms._replace(prepare=prepare,step=step,mapped=mapped)
    scan.derive_scan=derive
if "pair" in mode:
    from dew.nn import derivatives
    def paired(left,right,between,issue=None,role="scores",strict=False):
        issue = issue or scan._issued
        joined = j.where(j.tril(j.ones(between.shape[:2],bool),-1)[:,:,None],between,0.) if strict else between
        if between.shape[-1]!=1:
            return original(left,right,between,issue,role,strict=strict)
        def primal(l,r,d):
            return j.where(d!=0,issue(role,l,r.T)*d,0.)
        def tangent(p,t):
            l,r,d=p; dl,dr,dd=t
            product=issue(role,l,r.T)
            return primal(*p), j.where(d!=0, issue(role,dl,r.T)*d+issue(role,l,dr.T)*d+product*dd,0.)
        return derivatives.declared_tangent(derivatives.Protected(),primal,tangent)(left,right,joined[...,0])
    original=scan._paired
    scan._paired=paired

keys=jax.random.split(jax.random.key(1861),8)
shape=(1,37,2,8)
for packed in (False,True):
  jax.config.update("jax_enable_x64", False)
  starts=jax.random.bernoulli(keys[5],.1,(1,37)) if packed else None
  for power in (62,63):
    jax.config.update("jax_enable_x64", False)
    scale=2.**power
    operands=tuple([jax.random.normal(k,shape,dtype=j.float32)*scale for k in keys[:3]]+
                   [-jax.nn.softplus(jax.random.normal(keys[3],(1,37,2,1),dtype=j.float32)),
                    jax.nn.sigmoid(jax.random.normal(keys[4],(1,37,2),dtype=j.float32))])
    direction=tuple(jax.random.normal(keys[6],x.shape,dtype=j.float32)*(scale if i<3 else 1.) for i,x in enumerate(operands))
    cotangent=jax.random.normal(keys[7],shape,dtype=j.float32)
    def rule(*xs):
        return rc.xla_mainloop(rc.Tiles(16),*xs[:3],xs[3][...,0],xs[4],None,starts,scale=8**-.5,delta=True,dtype=xs[0].dtype)[0]
    def reference(*xs):
        return rc.recurrent_delta_rule(*xs,starts=starts)[0] if power==62 else precedent_chunked(*xs,chunk_size=16,starts=starts)
    def gradfn(fn):
        return jax.grad(lambda *xs:j.sum(fn(*xs)*cotangent),argnums=tuple(range(5)))
    def recurrence(*xs):
        return rc.recurrent_delta_rule(*xs,starts=starts)[0]
    with jax.enable_x64():
        truth=jax.jit(gradfn(recurrence))(*jax.tree.map(lambda x:x.astype(j.float64),operands))
    plain=jax.jit(gradfn(reference))(*operands)
    actual=jax.jit(gradfn(rule))(*operands)
    for at,(a,p,t) in enumerate(zip(actual,plain,truth)):
        normal=np.isfinite(t)&normal_float32(t)&np.isfinite(p)
        print("K",mode,packed,power,"vjp",at,"bad",np.sum(normal&~np.isfinite(a)),"held",np.sum(normal),flush=True)
        if packed and at==0 and "detail" in mode:
            print("K",mode,"starts",np.flatnonzero(np.asarray(starts)).tolist(),"badq",np.sum(normal&~np.isfinite(a),axis=(0,2,3)).tolist(),flush=True)
    if not packed:
        along=jax.jit(lambda *xs:jax.jvp(gradfn(rule),xs,direction)[1])(*operands)
        with jax.enable_x64():
            truealong=jax.jit(lambda *xs:jax.jvp(gradfn(recurrence),xs,jax.tree.map(lambda x:x.astype(j.float64),direction))[1])(*jax.tree.map(lambda x:x.astype(j.float64),operands))
        plainalong=jax.jit(lambda *xs:jax.jvp(gradfn(reference),xs,direction)[1])(*operands)
        for at,(a,p,t) in enumerate(zip(along,plainalong,truealong)):
            normal=np.isfinite(t)&normal_float32(t)&np.isfinite(p)
            print("K",mode,packed,power,"jvpgrad",at,"bad",np.sum(normal&~np.isfinite(a)),"held",np.sum(normal),flush=True)
        if "staged" in mode:
            for name,fn in (("remat",jax.checkpoint(rule)),("nestedjit",jax.jit(rule))):
                staged=jax.jit(lambda *xs:jax.jvp(gradfn(fn),xs,direction)[1])(*operands)
                bad=sum(int(np.sum(normal_float32(t)&np.isfinite(t)&np.isfinite(p)&~np.isfinite(a))) for a,p,t in zip(staged,plainalong,truealong))
                print("K",mode,power,name,"bad",bad,flush=True)
            def scanned(*xs):
                return jax.lax.scan(lambda carry,_:(carry,jax.jvp(gradfn(rule),xs,direction)[1]),0,None,length=1)[1]
            staged=jax.jit(scanned)(*operands)
            bad=sum(int(np.sum(normal_float32(t)&np.isfinite(t)&np.isfinite(p)&~np.isfinite(a[0]))) for a,p,t in zip(staged,plainalong,truealong))
            print("K",mode,power,"scanned","bad",bad,flush=True)
