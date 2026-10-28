#!/bin/sh
# Install Dew (the dewml package) with the build for this machine:
#
#     curl -LsSf https://dewml.dev/install.sh | sh
#
# It finds the hardware (an NVIDIA GPU and the CUDA build its driver runs, a TPU VM, or
# the CPU) and the Python environment, asks where to install, installs uv if it is
# missing, and runs `uv pip install "dewml[...]"` there. Options follow `sh -s --`:
#
#   -y              take every default without asking (or DEW_YES=1); so does a run
#                   with no terminal
#   --cuda N        the CUDA build N (one of cuda_builds below), whatever the driver says
#   --tpu, --cpu    the TPU or the CPU build
#   --venv PATH     install into the virtual environment at PATH, created if missing
#   --conda NAME    install into the conda environment NAME, created if missing
#   --version X     install dewml X instead of the newest release
#   --pip           use pip instead of uv
#
# The CUDA build is the newest of cuda_builds below, each build:least driver:least SM, generated from
# dew.builds.CUDA_BUILDS, that the driver and every GPU's SM run. Dew needs Python python_min or
# newer.
set -eu

# BEGIN install facts (site/scripts/gen_install.py)
python_min=3.12
package_index=https://pypi.dewml.dev/simple
pip_least='pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
cuda_builds='cuda13:580:7.5 cuda12:525:5.2'
# END install facts
# BEGIN uv bootstrap (site/scripts/gen_install.py)
uv_least=0.13.4
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
# END uv bootstrap

if [ -t 1 ]; then
	bold=$(printf '\033[1m') dim=$(printf '\033[2m') green=$(printf '\033[32m')
	red=$(printf '\033[31m') plain=$(printf '\033[0m')
else
	bold='' dim='' green='' red='' plain=''
fi
die() {
	printf '%serror:%s %s\n' "$red" "$plain" "$1" >&2
	exit 1
}
# The CUDA majors of cuda_builds, each after a space.
majors() { for entry in $cuda_builds; do printf ' %s' "${entry%%:*}"; done | sed 's/cuda//g'; }
# A word as a shell would read it back: as it is when it is plain, else in single quotes.
quote() {
	case $1 in
		'' | *[!A-Za-z0-9_./=:@%+,-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
		*) printf '%s' "$1" ;;
	esac
}

yes=${DEW_YES:-} build='' target='' conda_name='' version='' use_pip=''
while [ $# -gt 0 ]; do
	case $1 in
		-y | --yes) yes=1 ;;
		--cuda) case " $cuda_builds " in *" cuda${2:-}:"*) ;; *) die "--cuda takes one of$(majors)" ;; esac
			build=cuda$2; shift ;;
		--tpu) build=tpu ;;
		--cpu) build=cpu ;;
		--venv) [ -n "${2:-}" ] || die "--venv needs a path"; target=$2; shift ;;
		--conda) [ -n "${2:-}" ] || die "--conda needs a name"; conda_name=$2; shift ;;
		--version) [ -n "${2:-}" ] || die "--version needs a version"; version=$2; shift ;;
		--pip) use_pip=1 ;;
		*) die "unknown option $1; the options are at the top of https://dewml.dev/install.sh" ;;
	esac
	shift
done

# Questions go to the terminal, which `curl ... | sh` leaves free. With none, or with -y,
# each takes its default.
ask=
if [ -z "$yes" ] && (: </dev/tty) 2>/dev/null; then ask=1; fi
answer() { # answer PROMPT DEFAULT, into $reply
	reply=$2
	if [ -n "$ask" ]; then
		printf '%s %s[%s]%s ' "$1" "$dim" "$2" "$plain" >/dev/tty
		read -r reply </dev/tty || reply=
		[ -n "$reply" ] || reply=$2
	fi
}

# The hardware.
gpus=
if command -v nvidia-smi >/dev/null 2>&1; then
	gpus=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null) ||
		gpus=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null) || gpus=
fi
if [ -n "$gpus" ]; then
	gpu=$(printf '%s\n' "$gpus" | head -n 1 | cut -d, -f1 | sed 's/^ *//')
	count=$(printf '%s\n' "$gpus" | wc -l | tr -d ' ')
	[ "$count" -eq 1 ] || gpu="$count × $gpu"
	driver=$(printf '%s\n' "$gpus" | head -n 1 | cut -d, -f2 | tr -d ' ')
	sm=$(printf '%s\n' "$gpus" | cut -s -d, -f3 | tr -d ' ' | sort -n | head -n 1)
	at_least() { awk -v sm="$sm" -v need="$1" 'BEGIN { exit !(sm + 0 >= need) }'; }
	major=${driver%%.*}
	# The oldest build, the last entry, and its least driver.
	oldest=${cuda_builds##* } least=${cuda_builds##* } least=${least#*:} least=${least%%:*}
	runs=cpu why="SM $sm is older than JAX's CUDA builds support" name=${oldest%%:*}
	if [ "$major" -lt "$least" ]; then why="CUDA ${name#cuda} needs driver $least or newer; update the driver to use the GPU"
	elif [ -z "$sm" ]; then why="nvidia-smi did not report its compute capability; choose with --cuda, one of$(majors)"
	else
		for entry in $cuda_builds; do
			least=${entry#*:} least=${least%%:*}
			if [ "$major" -ge "$least" ] && at_least "${entry##*:}"; then runs=${entry%%:*}; break; fi
		done
	fi
	if [ "$runs" = cpu ]; then hardware="$gpu, driver $driver: no GPU build, as $why"
	else hardware="$gpu, driver $driver, runs CUDA ${runs#cuda}"
	fi
elif { [ -e /dev/accel0 ] || ls /dev/vfio/[0-9]* >/dev/null 2>&1; } &&
	tpu=$(curl -fsS -m 2 -H 'Metadata-Flavor: Google' \
		http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null); then
	runs=tpu hardware="TPU VM, $tpu"
elif [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
	runs=cpu hardware="Apple silicon, where JAX runs on the CPU"
else
	runs=cpu hardware="no GPU or TPU, so the CPU build"
fi
[ -n "$build" ] || build=$runs

# The Python environment.
conda=
command -v conda >/dev/null 2>&1 && conda=1
if [ -n "${VIRTUAL_ENV:-}" ]; then found="the virtual environment $VIRTUAL_ENV" active=$VIRTUAL_ENV
elif [ -n "${CONDA_PREFIX:-}" ]; then found="the conda environment ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}" active=$CONDA_PREFIX
else found='' active=''
fi

printf '%sDew installer%s\n' "$bold" "$plain"
printf '  hardware  %s\n' "$hardware"
printf '  python    %s\n' "${found:-no environment active}"
if [ -z "$use_pip" ]; then
	have=$(uv_version)
	if [ -z "$have" ]; then printf '  uv        not installed; uv %s will be, from astral.sh, unless you run this with --pip\n' "$uv_least"
	elif uv_older "$have"; then printf '  uv        %s, older than %s, which will be installed from astral.sh, unless you run this with --pip\n' "$have" "$uv_least"
	fi
fi
printf '\n'

# Where to install: one question. Its default is the active environment, else a new .venv.
if [ -z "$target" ] && [ -z "$conda_name" ]; then
	n=0 here='' venv='' other='' in_conda=''
	if [ -n "$ask" ]; then printf 'Where should dewml go?\n' >/dev/tty; fi
	option() { n=$((n + 1)); [ -z "$ask" ] || printf '  %s) %s\n' "$n" "$1" >/dev/tty; }
	if [ -n "$found" ]; then option "$found"; here=$n; fi
	option "a new .venv in $(pwd)"; venv=$n
	if [ -n "$conda" ]; then option "a conda environment"; in_conda=$n; fi
	option "another path"; other=$n
	answer "Choose" 1
	case $reply in
		"$here") target=$active ;;
		"$venv") target=.venv ;;
		"$in_conda") answer "Conda environment" dew; conda_name=$reply ;;
		"$other") answer "Path of the environment" .venv; target=$reply ;;
		*) die "there is no option $reply" ;;
	esac
fi

# uv, from its own installer, where it is missing or older than uv_least.
if [ -z "$use_pip" ]; then
	bootstrap_uv || die "uv's installer failed; install uv $uv_least or newer from https://docs.astral.sh/uv, or rerun with --pip"
	have=$(uv_version)
	if [ -z "$have" ] || uv_older "$have"; then
		# Where uv's installer put it (https://docs.astral.sh/uv/reference/storage/#executable-directory).
		bin=${UV_INSTALL_DIR:-${UV_UNMANAGED_INSTALL:-${XDG_BIN_HOME:-${XDG_DATA_HOME:+$XDG_DATA_HOME/../bin}}}}
		PATH="${bin:-$HOME/.local/bin}:${bin:+$bin/bin:}$PATH"
		have=$(uv_version)
	fi
	[ -n "$have" ] || die "uv is installed but not on PATH; open a new shell and rerun"
	! uv_older "$have" || die "uv $have, older than $uv_least, comes first on PATH; remove it or update it with uv self update"
fi

# The environment's Python, made if it does not exist yet.
if [ -n "$conda_name" ]; then
	[ -n "$conda" ] || die "conda is not installed"
	# shellcheck disable=SC2016 # the environment's prefix, as conda run sets it inside
	prefix() { conda run -n "$conda_name" sh -c 'printf %s "$CONDA_PREFIX"' 2>/dev/null; }
	where=$(prefix) || {
		conda create -y -q -n "$conda_name" "python>=$python_min" >/dev/null || die "conda could not create the environment $conda_name"
		where=$(prefix) || die "conda made $conda_name but cannot run in it"
	}
	# The environment's own Python: never one conda run finds on PATH outside it.
	[ -x "$where/bin/python" ] || conda install -y -q -n "$conda_name" "python>=$python_min" >/dev/null ||
		die "the conda environment $conda_name has no Python, and conda could not add one"
	python=$where/bin/python activate="conda activate $(quote "$conda_name")"
else
	if [ ! -x "$target/bin/python" ]; then
		if [ -z "$use_pip" ]; then uv venv -q --python ">=$python_min" "$target" || die "uv could not create $target"
		else python3 -m venv "$target" || die "python3 could not create $target"
		fi
	fi
	python=$target/bin/python activate=". $(quote "$target/bin/activate")"
	if [ "$target" = "${CONDA_PREFIX:-}" ]; then activate="conda activate $(quote "${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}")"; fi
fi
"$python" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '$python_min'.split('.'))))" || die "$python is Python \
$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])'); Dew needs $python_min or newer, so choose a new .venv or upgrade it"

# pip, for --pip, in an environment uv made without it, then pip 26.2.2 by its wheel's hash (tools/pip.txt):
# older pips trust the roots of eMudhra's compromised CAs in their own bundle (2026-10-20).
if [ -n "$use_pip" ]; then
	if ! "$python" -m pip --version >/dev/null 2>&1; then
		"$python" -m ensurepip -q || die "$python has no pip and ensurepip could not add it; rerun without --pip to use uv"
	fi
	"$python" -m pip install -q "$pip_least" || die "pip could not update itself to 26.2.2; rerun without --pip to use uv"
fi

# The install, shown and then run.
extra=''
[ "$build" = cpu ] || extra="[$build]"
package="dewml$extra${version:+==$version}"
# Dew runs on its own JAX build, which only Dew's package index serves (docs/installation.md).
if [ -n "$use_pip" ]; then set -- "$python" -m pip install --extra-index-url "$package_index" "$package"
else set -- uv pip install --python "$python" --extra-index-url "$package_index" "$package"
fi
shown=''
for word in "$@"; do shown="$shown${shown:+ }$(quote "$word")"; done
printf '%s$ %s%s\n' "$dim" "$shown" "$plain"
"$@" || die "the install failed; the lines above say why"

# What is installed, and how to use it. Dew refuses the CPU build beside an NVIDIA GPU unless JAX is told
# to use the CPU (docs/installation.md, Which CUDA build).
cpu=''
if [ "$build" = cpu ] && [ -n "$gpus" ]; then cpu='JAX_PLATFORMS=cpu '; export JAX_PLATFORMS=cpu; fi
report=$("$python" -c 'import importlib.metadata as m, jax, dew
print(m.version("dewml"), "with jax", jax.__version__, "on", jax.default_backend())' 2>&1) || {
	printf '%s\n' "$report" >&2
	die "dewml is installed, but Dew or JAX's backend failed to start; the error is above"
}
installed=$(printf '%s\n' "$report" | tail -n 1)
printf '\n%s✓ dewml %s%s\n' "$green" "$installed" "$plain"
printf '  check     %s%s -c "import jax; print(jax.devices())"\n' "$cpu" "$(quote "$python")"
[ -z "$cpu" ] || activate="$activate; export JAX_PLATFORMS=cpu  # Dew stops when JAX leaves a GPU unused"
printf '  activate  %s\n' "$activate"
case $build in cuda*) case $installed in *" on cpu") printf '%sJAX sees no GPU: check that nvidia-smi lists it%s\n' "$red" "$plain" ;; esac ;; esac
