#!/bin/sh
# Administrative feasibility probe only, before any visitor kernels are exposed.
set -eu
id -u visitor_probe >/dev/null 2>&1 || useradd --uid 6000 --create-home --shell /bin/sh visitor_probe
id -u other_probe >/dev/null 2>&1 || useradd --uid 6001 --create-home --shell /bin/sh other_probe
mkdir -p /work /sessions/probe /sessions/other
chmod 0711 /sessions
chown visitor_probe:visitor_probe /sessions/probe
chmod 0700 /sessions/probe /sessions/other
printf 'not visible to another kernel\n' > /sessions/other/private.txt
chmod 0600 /sessions/other/private.txt
chown -R other_probe:other_probe /sessions/other
cat > /sessions/probe/check.py <<'PY'
import glob, json, os
report={'uid':os.getuid(),'gid':os.getgid(),'uid_map':open('/proc/self/uid_map').read(),
        'capabilities':[s.strip() for s in open('/proc/self/status') if s.startswith(('Cap','NoNewPrivs'))]}
try:
    os.setuid(0)
    report['setuid_zero']='allowed'
except OSError as error:
    report['setuid_zero']=f'denied: errno {error.errno}'
assert os.path.isdir('/opt/hf') and os.path.isdir('/opt/xla')
models=[p for p in glob.glob('/opt/hf/hub/**/blobs/*',recursive=True) if os.path.isfile(p)]
entries=glob.glob('/opt/xla/*-cache')
assert models and entries, 'real model and compile-cache files must exist'
for path in (models[0],entries[0],'/sessions/other/private.txt'):
    try:
        if path.endswith('private.txt'):
            open(path).read()
        else:
            fd=os.open(path,os.O_WRONLY); os.close(fd)
        report[path]='allowed'
    except (PermissionError, OSError) as error:
        report[path]=type(error).__name__
assert report['uid']==6000
assert all(line.split(':')[1].strip()=='0000000000000000' for line in report['capabilities'] if line.startswith('Cap'))
assert report['setuid_zero'].startswith('denied')
assert all(report[p] in ('PermissionError','OSError') for p in (models[0],entries[0]))
assert report['/sessions/other/private.txt']=='PermissionError'
print(json.dumps(report),flush=True)
PY
chown visitor_probe:visitor_probe /sessions/probe/check.py
# /proc is inherited read-only: Cloudflare forbids mounting another procfs.
# The trusted launcher drops the host capabilities before entering the namespace.
capsh --drop=all --no-new-privs --user=visitor_probe -- -c \
 'bwrap --unshare-user --unshare-net --ro-bind / / --bind /sessions/probe /work --tmpfs /tmp --cap-drop ALL /opt/venv/bin/python /work/check.py'

# The model-serving process is not subject to these guest limits.
rm -f /sessions/probe/resource.py
cat > /sessions/probe/limit_case.py <<'PYCODE'
import os, resource, sys, threading, time
sys.path.insert(0,'/opt/live')
import guest_limits
guest_limits.install(cpu_seconds=1)
mode=sys.argv[1]
if mode=='memory':
    try:
        bytearray(1024*1024*1024)
        raise AssertionError('guest exceeded its memory allowance')
    except MemoryError:
        print('memory allocation refused',flush=True)
    try:
        resource.setrlimit(resource.RLIMIT_AS,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
        raise AssertionError('guest raised its hard memory limit')
    except (PermissionError,ValueError):
        print('hard-limit increase refused',flush=True)
elif mode=='fork':
    try:
        os.fork()
        raise AssertionError('guest multiplied its memory allowance through fork')
    except PermissionError:
        print('fork refused',flush=True)
    thread=threading.Thread(target=lambda: None);thread.start();thread.join()
    print('ordinary thread allowed',flush=True)
elif mode=='cpu':
    while True: pass
elif mode=='wall':
    time.sleep(60)
PYCODE
chown visitor_probe:visitor_probe /sessions/probe/limit_case.py
for mode in memory fork; do
  capsh --drop=all --no-new-privs --user=visitor_probe -- -c    "bwrap --unshare-user --unshare-net --ro-bind / / --bind /sessions/probe /work --tmpfs /tmp --cap-drop ALL /opt/venv/bin/python /work/limit_case.py $mode"
done
for mode in cpu wall; do
  set +e
  timeout -s KILL 3 capsh --drop=all --no-new-privs --user=visitor_probe -- -c    "bwrap --unshare-user --unshare-net --ro-bind / / --bind /sessions/probe /work --tmpfs /tmp --cap-drop ALL /opt/venv/bin/python /work/limit_case.py $mode"
  result=$?
  set -e
  echo "limit_probe $mode exit=$result"
  case "$result" in 137|152) :;; *) echo 'limit not enforced' >&2; exit 1;; esac
done
