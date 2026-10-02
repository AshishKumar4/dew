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
import json, os
report={'uid':os.getuid(),'gid':os.getgid(),'uid_map':open('/proc/self/uid_map').read(),
        'capabilities':[s.strip() for s in open('/proc/self/status') if s.startswith(('Cap','NoNewPrivs'))]}
try:
    os.setuid(0)
    report['setuid_zero']='allowed'
except OSError as error:
    report['setuid_zero']=f'denied: errno {error.errno}'
for path in ('/opt/hf','/opt/xla','/sessions/other/private.txt'):
    try:
        if path.endswith('private.txt'):
            open(path).read()
        else:
            with open(path+'/visitor-write-probe','w') as f: f.write('must not be allowed')
        report[path]='allowed'
    except (PermissionError, OSError) as error:
        report[path]=type(error).__name__
print(json.dumps(report),flush=True)
PY
chown visitor_probe:visitor_probe /sessions/probe/check.py
# /proc is inherited read-only: Cloudflare forbids mounting another procfs.
# The trusted launcher drops the host capabilities before entering the namespace.
capsh --drop=all --no-new-privs --user=visitor_probe -- -c \
 'bwrap --unshare-user --unshare-net --ro-bind / / --bind /sessions/probe /work --tmpfs /tmp --cap-drop ALL /opt/venv/bin/python /work/check.py'
