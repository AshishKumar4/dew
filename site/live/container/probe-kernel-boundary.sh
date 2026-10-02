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
