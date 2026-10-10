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
#   --cuda 12|13    the CUDA build, whatever the driver says
#   --tpu, --cpu    the TPU or the CPU build
#   --venv PATH     install into the virtual environment at PATH, created if missing
#   --conda NAME    install into the conda environment NAME, created if missing
#   --version X     install dewml X instead of the newest release
#   --pip           use pip instead of uv
#
# The CUDA build follows the driver and the GPUs: driver 580 or newer with every GPU at
# SM 7.5 or newer runs CUDA 13, driver 525 or newer with SM 5.2 or newer runs CUDA 12
# (JAX's installation guide; NVIDIA's CUDA release notes, table 3). Dew needs Python 3.12
# or newer.
set -eu

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

yes=${DEW_YES:-} build='' target='' conda_name='' version='' use_pip=''
while [ $# -gt 0 ]; do
	case $1 in
		-y | --yes) yes=1 ;;
		--cuda) [ "${2:-}" = 12 ] || [ "${2:-}" = 13 ] || die "--cuda takes 12 or 13"; build=cuda$2; shift ;;
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
	at_least() { awk -v sm="$sm" -v need="$1" 'BEGIN { exit !(sm == "" || sm + 0 >= need) }'; }
	major=${driver%%.*}
	if [ "$major" -ge 580 ] && at_least 7.5; then runs=cuda13
	elif [ "$major" -ge 525 ] && at_least 5.2; then runs=cuda12
	elif [ "$major" -ge 525 ]; then runs=cpu why="SM $sm is older than JAX's CUDA builds support"
	else runs=cpu why="CUDA 12 needs driver 525 or newer; update the driver to use the GPU"
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
if [ -n "${VIRTUAL_ENV:-}" ]; then found="the virtual environment $VIRTUAL_ENV"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != base ]; then found="the conda environment $CONDA_DEFAULT_ENV"
else found=
fi

printf '%sDew installer%s\n' "$bold" "$plain"
printf '  hardware  %s\n' "$hardware"
printf '  python    %s\n' "${found:-no environment active}"
if [ -z "$use_pip" ] && ! command -v uv >/dev/null 2>&1; then
	printf '  uv        not installed; it will be, from astral.sh, unless you run this with --pip\n'
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
		"$here") if [ -n "${VIRTUAL_ENV:-}" ]; then target=$VIRTUAL_ENV; else conda_name=$CONDA_DEFAULT_ENV; fi ;;
		"$venv") target=.venv ;;
		"$in_conda") answer "Conda environment" dew; conda_name=$reply ;;
		"$other") answer "Path of the environment" .venv; target=$reply ;;
		*) die "there is no option $reply" ;;
	esac
fi

# uv, from its own installer.
if [ -z "$use_pip" ] && ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh ||
		die "uv's installer failed; install uv from https://docs.astral.sh/uv, or rerun with --pip"
	PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
	command -v uv >/dev/null 2>&1 || die "uv is installed but not on PATH; open a new shell and rerun"
fi

# The environment's Python, made if it does not exist yet.
if [ -n "$conda_name" ]; then
	[ -n "$conda" ] || die "conda is not installed"
	prefix() { conda env list | awk -v name="$conda_name" '$1 == name { print $NF }'; }
	[ -n "$(prefix)" ] || conda create -y -q -n "$conda_name" 'python>=3.12' >/dev/null ||
		die "conda could not create the environment $conda_name"
	python=$(prefix)/bin/python activate="conda activate $conda_name"
else
	if [ ! -x "$target/bin/python" ]; then
		if [ -z "$use_pip" ]; then uv venv -q --python '>=3.12' "$target" || die "uv could not create $target"
		else python3 -m venv "$target" || die "python3 could not create $target"
		fi
	fi
	python=$target/bin/python activate=". $target/bin/activate"
fi
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || die "$python is Python \
$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])'); Dew needs 3.12 or newer, so choose a new .venv or upgrade it"

# pip, for --pip, in an environment uv made without it.
if [ -n "$use_pip" ] && ! "$python" -m pip --version >/dev/null 2>&1; then
	"$python" -m ensurepip -q || die "$python has no pip and ensurepip could not add it; rerun without --pip to use uv"
fi

# The install, shown and then run.
extra=
[ "$build" = cpu ] || extra="[$build]"
package="dewml$extra${version:+==$version}"
if [ -n "$use_pip" ]; then set -- "$python" -m pip install "$package"
else set -- uv pip install --python "$python" "$package"
fi
printf '%s$ %s%s\n' "$dim" "$*" "$plain"
"$@" || die "the install failed; the lines above say why"

# What is installed, and how to use it.
installed=$("$python" -c 'import importlib.metadata as m, jax
print(m.version("dewml"), "with jax", jax.__version__, "on", jax.default_backend())' 2>/dev/null) ||
	die "dewml is installed but does not import; see why with: $python -c 'import dew'"
printf '\n%s✓ dewml %s%s\n' "$green" "$installed" "$plain"
printf '  check     %s -c "import jax; print(jax.devices())"\n' "$python"
printf '  activate  %s\n' "$activate"
case $build in cuda*) case $installed in *" on cpu") printf '%sJAX sees no GPU: check that nvidia-smi lists it%s\n' "$red" "$plain" ;; esac ;; esac
