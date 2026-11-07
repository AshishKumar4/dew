import os, sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'tests')]
from lane_environment import configure_lane
configure_lane(os.environ)
import torch
from tools.torch_precision import replaced, dtype_widened
from test_reference_precision import test_float64_truth_widens_a_tensor_softmax
try:
 if int(sys.argv[1]):
  with replaced((torch.Tensor,'softmax',dtype_widened(torch.Tensor.softmax))):
   test_float64_truth_widens_a_tensor_softmax()
 else:
  test_float64_truth_widens_a_tensor_softmax()
 print('K regression',sys.argv[1],'PASS')
except AssertionError as e:
 print('K regression',sys.argv[1],'FAIL',str(e))
