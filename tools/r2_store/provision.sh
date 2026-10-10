#!/bin/sh
# Provision one of Dew's R2 object stores on this machine's wrangler login:
# its bucket, the lifecycle that expires its objects, its Worker (worker.js,
# the same code for every store, named and bound by NAME.jsonc) and the
# bearer token that Worker asks for.
#
#   tools/r2_store/provision.sh NAME [EXPIRE_DAYS]
#
# EXPIRE_DAYS deletes an object that many days after it is written; without
# it the bucket keeps every object until it is deleted. The token is made once
# and kept at ~/.config/dew/NAME.token (mode 600); the Worker's URL is written
# to ~/.config/dew/NAME.json. Run again, it deploys the Worker and sets the
# same token. WRANGLER names the wrangler to run (default `npx wrangler@4`).
set -eu
name=$1
days=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
config="$here/$name.jsonc"
wrangler=${WRANGLER:-npx wrangler@4}
[ -f "$config" ] || { echo "no store $name: $config is missing" >&2; exit 2; }

if ! $wrangler r2 bucket info "$name" >/dev/null 2>&1; then
  $wrangler r2 bucket create "$name"
fi
if [ -n "$days" ] && ! $wrangler r2 bucket lifecycle list "$name" 2>/dev/null | grep -q "expire-${days}d"; then
  $wrangler r2 bucket lifecycle add "$name" "expire-${days}d" --expire-days "$days" --force
fi

mkdir -p "$HOME/.config/dew"
token="$HOME/.config/dew/$name.token"
if [ ! -s "$token" ]; then
  (umask 077 && head -c 48 /dev/urandom | base64 | tr -d '/+=\n' > "$token")
fi
chmod 600 "$token"

deployed=$($wrangler deploy -c "$config" 2>&1) || { echo "$deployed" >&2; exit 1; }
url=$(echo "$deployed" | grep -o "https://$name\.[A-Za-z0-9.-]*workers\.dev" | head -1)
[ -n "$url" ] || { echo "$deployed" >&2; echo "the deploy named no URL" >&2; exit 1; }
tr -d '\n' < "$token" | $wrangler secret put CACHE_TOKEN -c "$config" >/dev/null
printf '{"url": "%s", "token_file": "%s"}\n' "$url" "$token" > "$HOME/.config/dew/$name.json"
echo "$name: $url (token in $token)"
