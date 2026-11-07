import os,sys,subprocess,tempfile,sysconfig
from pathlib import Path
if 'LLAMA_DEBUG_LAYERED' not in os.environ:
 venv=Path(tempfile.mkdtemp(prefix='llama-probe-'))/'venv'
 subprocess.run(['uv','venv','-q',str(venv),'-p',sys.executable],check=True)
 site=next((venv/'lib').glob('python3.*/site-packages'))
 (site/'dew.pth').write_text(sysconfig.get_paths()['purelib']+'\n')
 subprocess.run(['uv','pip','install','-q','--python',str(venv/'bin/python'),'cmake','ninja','sentencepiece==0.2.2'],check=True)
 os.environ['LLAMA_DEBUG_LAYERED']='1';os.environ['PATH']=str(venv/'bin')+os.pathsep+os.environ.get('PATH','')
 os.execv(str(venv/'bin/python'),[str(venv/'bin/python'),*sys.argv])
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src'),str(Path.cwd()/'tests'),str(Path.cwd()/'tools')]
import numpy as np,json,ctypes,collections
import jax,jax.numpy as jnp
from flax import linen as nn
from dew.nn.dense import Dense,DenseGeneral
from dew.nn.precision import FP32_SUMS
from tools.llama_cpp_worker import prepared
program,gguf,truth_model=prepared()
print('K prepared',program,flush=True)
source=program.parent.parent
build_directory=Path(tempfile.mkdtemp(prefix='llama-matmul-'))
(build_directory/'dot.cpp').write_text('''#include "ggml.h"
#include "ggml-cpu.h"
#include <cstring>
extern "C" void matrix(int m,int k,int n,const float*x,const float*w,float*y){
 ggml_init_params p={16*1024*1024,nullptr,false};
 auto ctx=ggml_init(p);
 auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
 auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
 memcpy(ggml_get_data(a),w,k*n*sizeof(float));
 memcpy(ggml_get_data(b),x,m*k*sizeof(float));
 auto c=ggml_mul_mat(ctx,a,b);
 auto g=ggml_new_graph(ctx);
 ggml_build_forward_expand(g,c);
 ggml_graph_compute_with_ctx(ctx,g,1);
 memcpy(y,ggml_get_data(c),m*n*sizeof(float));
 ggml_free(ctx);
}''')
libdir=program.parent/'bin'
subprocess.run(['g++','-shared','-fPIC','-O2',str(build_directory/'dot.cpp'),'-I'+str(source/'ggml/include'),'-L'+str(libdir),'-lggml','-lggml-cpu','-lggml-base','-Wl,-rpath,'+str(libdir),'-o',str(build_directory/'dot.so')],check=True)
lib=ctypes.CDLL(str(build_directory/'dot.so'))
ptr=ctypes.POINTER(ctypes.c_float)
lib.matrix.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr]
def ggml(x,k):
 x=np.ascontiguousarray(x.reshape(-1,x.shape[-1]),dtype=np.float32)
 w=np.ascontiguousarray(k.T,dtype=np.float32)
 y=np.empty((x.shape[0],w.shape[0]),np.float32)
 lib.matrix(x.shape[0],x.shape[1],w.shape[0],x.ctypes.data_as(ptr),w.ctypes.data_as(ptr),y.ctypes.data_as(ptr))
 return y
from test_llama_cpp_export import draw,exported
model,vs=exported()
from huggingface_hub import hf_hub_download
pin=json.loads(Path('tests/fixtures/llama_cpp/recording.npz.fetched.json').read_text())
st=dict(np.load(hf_hub_download(pin['repo'],pin['path'],repo_type='dataset',revision=pin['revision'])))
mode=sys.argv[1]
stats=collections.defaultdict(lambda:np.zeros(5))
def interceptor(next_fun,args,kwargs,context):
 y=next_fun(*args,**kwargs)
 if context.method_name=='__call__' and isinstance(context.module,(Dense,DenseGeneral)):
  x=np.asarray(args[0]);k=context.module.get_variable('params','kernel');k=np.asarray(k.value if hasattr(k,'value') else k)
  other=ggml(x,k).reshape(y.shape)
  t=x.astype('float64')@k.astype('float64')
  stats[context.module.scope.path_text]+=np.array([np.sum((np.asarray(y)-t)**2),np.sum((other-t)**2),np.sum(t**2),t.size,np.sum(np.asarray(y)!=other)])
  if mode=='replace':return jnp.asarray(other)
 return y
errs=[];refs=[]
os.environ['DEW_WIDE_CPU_ARITHMETIC']='1'
with jax.default_matmul_precision(FP32_SUMS),nn.intercept_methods(interceptor):
 for seed in range(40):
  truth=st['llama-cpp.truth.logits'][seed];ref=st['llama-cpp.reference.logits'][seed]
  y=np.asarray(model.apply(vs,jnp.asarray(draw(seed)['ids'])))
  errs.append(np.mean((y.astype('float64')-truth)**2));refs.append(np.mean((ref.astype('float64')-truth)**2))
print('K result',mode,'rms',np.sqrt(np.mean(errs)),'reference',np.sqrt(np.mean(refs)),'ratio',np.sqrt(np.mean(errs)/np.mean(refs)),flush=True)
for name,v in stats.items():print('K local',name,'xla',np.sqrt(v[0]/v[3]),'ggml',np.sqrt(v[1]/v[3]),'ratio',np.sqrt(v[0]/v[1]),'different',v[4]/v[3],flush=True)
