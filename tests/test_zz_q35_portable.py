"""Scratch portable seed control for historical revisions."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from dew.interop.pretrained import Pretrained
from dew.registry import ModelConfig
from dew.objectives.lm import LMObjective
from dew.objectives.base import Step
from dew.nn.precision import FP32_SUMS


def test_portable():
    source=Path(__file__).parent/'fixtures/hf/qwen35-native-tiny'
    loaded=Pretrained.load(source,dtype='float32',attention_impl='reference')
    def compute(dtype,ids):
        mask=np.arange(32)[None,:]<np.asarray((32,28,24,20))[:,None]
        encoded=loaded.processor.from_hf({'input_ids':np.where(mask,ids,0).astype(np.int64),'attention_mask':mask.astype(np.int64)})
        model=loaded.model if dtype==jnp.float32 else ModelConfig.from_model(loaded.model).with_dtype('float64').build()
        variables=jax.tree.map(lambda a:jnp.asarray(a,dtype) if jnp.issubdtype(jnp.asarray(a).dtype,jnp.floating) else a,loaded.variables)
        objective=LMObjective(model,31,ema_decay=None,pad_id=0)
        step=Step(step=jnp.int32(0),key=jax.random.key(4),ema=None)
        def program(v):
            def loss(p):
                stat,_=objective.loss({**v,'params':p},{'text':encoded},step)
                return objective.reduce_loss(stat)[0]
            grad=jax.grad(loss)(v['params'])
            updated={**v,'params':jax.tree.map(lambda p,g:p-jnp.asarray(1e-4,dtype)*g,v['params'],grad)}
            return jnp.where(jnp.asarray(mask)[...,None],model.apply(updated,encoded.tokens,**encoded.kwargs()),0)
        return jax.jit(program)(variables)
    lines=[]
    for seed,ref in [(10,1.86229398902e-11),(700,2.87167050439e-11)]:
        ids=np.random.default_rng([495,seed]).integers(4,200,(4,32))
        with jax.default_matmul_precision(FP32_SUMS):a=np.asarray(compute(jnp.float32,ids),np.float64)
        with jax.enable_x64(True):b=np.asarray(compute(jnp.float64,ids))
        e=(a-b)**2;pos=e.sum(-1);k=np.unravel_index(pos.argmax(),pos.shape)
        lines.append('seed %d mse %.12g ratio %.8g peak %s fraction %.8g'%(seed,e.mean(),np.sqrt(e.mean()/ref),k,pos[k]/pos.sum()))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
