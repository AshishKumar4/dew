# uv at uv_least or newer (pyproject.toml's [tool.uv] required-version). bootstrap_uv [NAME=VALUE...]
# installs that release from astral's installer, NAME=VALUE in the installer's environment, where uv is
# missing or older; uv_version is the release of the uv on PATH (none where there is none), and
# uv_older VERSION says whether VERSION is older: a pre-release of uv_least is, a build of it is not.
uv_version() { (uv --version 2>/dev/null || true) | awk '{print $2}'; }
uv_older() {
	awk -v have="${1%%+*}" -v least="$uv_least" 'BEGIN {
		split(have, h, "."); split(least, l, ".")
		for (i = 1; i <= 3; i++) if (h[i] + 0 != l[i] + 0) exit !(h[i] + 0 < l[i] + 0)
		exit have !~ /[^0-9.]/
	}'
}
bootstrap_uv() {
	have=$(uv_version)
	if [ -z "$have" ] || uv_older "$have"; then
		curl -LsSf "https://astral.sh/uv/$uv_least/install.sh" | env "$@" sh
	fi
}
