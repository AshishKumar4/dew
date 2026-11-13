"""The native stacks of a test process a signal ends (SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE), kept beside
its whole output among the task's artifacts, so an abort inside a native library is read from what it left and
not guessed at (row 5801: train 788's pytest died by SIGABRT inside an executable's run, and faulthandler's
Python stacks named no native frame; owner, 2026-11-13: a red that does not repeat publishes only where CI
keeps the whole output and a native backtrace of any task a signal ends). A small library preloaded into the
test processes (`armed`) catches those signals after faulthandler has written the Python stacks (faulthandler
puts back the handler it found and raises the signal again): it lets a child trace the process
(PR_SET_PTRACER), runs the task environment's py-spy on it (`dump --native`, every thread, native and Python
frames), adds glibc's backtrace of the signalled thread, then dies by the same signal, so the exit pytest's
runner reads is unchanged. It is built with the container's C compiler once a process; where none builds it
the task says so (`armed`'s reason) and runs as before."""

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM = r"""
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

static const int SIGNALS[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE};

static void said(int fd, const char *text) {
    ssize_t ignored = write(fd, text, strlen(text));
    (void)ignored;
}

static void caught(int signum) {
    const char *spy = getenv("DEW_NATIVE_SPY"), *path = getenv("DEW_NATIVE_STACKS");
    if (path != NULL) {
        int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            char line[96];
            snprintf(line, sizeof line, "== process %d ended by signal %d: native stacks\n",
                     getpid(), signum);
            said(fd, line);
            if (spy != NULL && spy[0] != '\0') {
                prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
                char pid[24];
                snprintf(pid, sizeof pid, "%d", getpid());
                pid_t child = fork();
                if (child == 0) {
                    dup2(fd, 1);
                    dup2(fd, 2);
                    execl(spy, spy, "dump", "--native", "--pid", pid, (char *)NULL);
                    _exit(127);
                }
                if (child > 0) {
                    int status;
                    waitpid(child, &status, 0);
                }
            }
            said(fd, "== glibc backtrace of the signalled thread\n");
            void *frames[128];
            backtrace_symbols_fd(frames, backtrace(frames, 128), fd);
            close(fd);
        }
    }
    signal(signum, SIG_DFL);
    raise(signum);
}

__attribute__((constructor)) static void armed(void) {
    for (size_t index = 0; index < sizeof SIGNALS / sizeof SIGNALS[0]; index++) {
        struct sigaction action;
        memset(&action, 0, sizeof action);
        action.sa_handler = caught;
        action.sa_flags = SA_RESETHAND;
        sigaction(SIGNALS[index], &action, NULL);
    }
}
"""
"""The preloaded library's source: its handler, armed as each process loads it, before Python's faulthandler.
"""
MARK = "== process "
"""How the stacks file names each process a signal ended (`SHIM`)."""


@functools.cache
def built() -> tuple[Path | None, str]:
    """The library (`SHIM`) built once a process with the container's C compiler, or None and why not."""
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        return None, "no C compiler in the container"
    where = Path(tempfile.mkdtemp(prefix="dew-natives-"))
    source = where / "natives.c"
    source.write_text(SHIM)
    library = where / "natives.so"
    made = subprocess.run([compiler, "-shared", "-fPIC", "-O1", "-o", str(library), str(source)],
                          capture_output=True, text=True, check=False)
    if made.returncode != 0:
        return None, f"the library did not build: {made.stderr.strip()[-300:]}"
    return library, ""


def armed(spy: Path, stacks: Path, environment: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """The variables that arm a test process's capture (`SHIM`), writing to `stacks` with py-spy at `spy`,
    beside `environment`'s own preload; none and why where it cannot be armed (`built`)."""
    library, why = built()
    if library is None:
        return {}, why
    given = (environment or os.environ).get("LD_PRELOAD", "")
    preload = " ".join(part for part in (given, str(library)) if part)
    stacks.unlink(missing_ok=True)
    return {"LD_PRELOAD": preload, "DEW_NATIVE_SPY": str(spy) if spy.is_file() else "",
            "DEW_NATIVE_STACKS": str(stacks)}, ("" if spy.is_file()
                                                 else f"no py-spy at {spy}: glibc's backtrace alone")


def kept(root: Path, name: str, output: str, stacks: Path) -> str:
    """A signal-ended run's whole output and its native stacks kept among the task's artifacts (`root`), under
    `name`, its path there: the evidence its red rows name (armada's `artifacts`)."""
    where = root / "signalled" / name
    where.mkdir(parents=True, exist_ok=True)
    (where / "output.txt").write_text(output)
    native = stacks.read_text(errors="replace") if stacks.is_file() else "no native stacks were written\n"
    (where / "native-stacks.txt").write_text(native)
    return str(where.relative_to(root))


if __name__ == "__main__":
    # A check of the capture where it runs: a child that aborts, its stacks printed.
    target = Path(tempfile.mkdtemp(prefix="dew-natives-check-")) / "stacks.txt"
    variables, reason = armed(Path(sys.executable).with_name("py-spy"), target)
    print(reason or "armed")
    subprocess.run([sys.executable, "-c", "import os; os.abort()"], env={**os.environ, **variables},
                   check=False)
    print(target.read_text() if target.is_file() else "nothing written")
