import sys; sys.path[:0]=["src","tests","."]
import jax
import jax.numpy as j
import numpy as np
from dew.algebra import scan, linear_scan

for line in str(scan._form(linear_scan.Extents(16,8,8,1),True)).splitlines():
 print("K FORM",line,flush=True)
# A vanished/packed restart's state transfer. A finite local write must survive an infinite incoming state.
incoming=j.array([[j.inf]],j.float32)
for selected in (False,True):
 def reset(h):
  return (j.where(j.array(0.)!=0,h,0.) if selected else j.array(0.)*h)+1.
 print("K restart",selected,jax.jit(reset)(incoming).item(),flush=True)
# A strictly triangular solve's transpose has only the final token's direct cotangent at its final input.
# Off-triangle zero terms cannot read an earlier token's overflowed cotangent.
lower=j.array([[0.,0.],[2.,0.]],j.float32)
cotangent=j.array([[j.inf],[1.]],j.float32)
ordinary=cotangent+lower.T@cotangent
masked=cotangent+j.sum(j.where(lower.T[:,:,None]!=0,lower.T[:,:,None]*cotangent[None,:,:],0.),axis=1)
print("K solve_transpose ordinary",np.asarray(ordinary).ravel().tolist(),"masked",np.asarray(masked).ravel().tolist(),flush=True)
# A three-token live system followed by padding: whole-tile inverse doubling forms 0*inf across blocks.
a=j.zeros((8,8),j.float32).at[1,0].set(2.e38).at[2,1].set(2.e38)
for safe in (False,True):
 def issue(role,left,right):
  if safe:
   products=left[:,:,None]*right[None,:,:]
   return j.sum(j.where((left[:,:,None]==0)|(right[None,:,:]==0),0.,products),axis=1)
  return left@right
 inv=scan._doubled(-a,issue)
 print("K inverse",safe,"nan",int(j.sum(j.isnan(inv))),"paddednan",int(j.sum(j.isnan(inv[3:]))),flush=True)
