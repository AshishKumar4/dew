# A guest's reach into another context, run by benchmark_gateway.py's isolated() with `foreign`
# bound to that context's id.
for path in [f'/sessions/connections/kernel-{foreign}.json', f'/sessions/ipc/kernel-{foreign}/kernel-1']:
 try:
  Path(path).read_bytes()
 except PermissionError:
  pass
 else:
  raise AssertionError('guest read another context')
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
 sock.connect(f'/sessions/ipc/kernel-{foreign}/kernel-1')
except PermissionError:
 pass
else:
 raise AssertionError('guest connected to another context')
finally:
 sock.close()
