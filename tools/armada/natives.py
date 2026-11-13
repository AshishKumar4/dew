"""The native stacks of a test process a signal ends (SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE), kept with the
run's whole output among the task's artifacts, so an abort inside a native library is read from what it left
and not guessed at (row 5801: train 788's pytest died by SIGABRT inside an executable's run, and
faulthandler's Python stacks named no native frame; owner, 2026-11-13: a red that does not repeat publishes
only where CI keeps the whole output and a native backtrace of any task a signal ends).

A collector (`collect`) starts before pytest, outside it. A small library preloaded into every test process
(`SHIM`) catches those signals after faulthandler has written the Python stacks (faulthandler puts back the
action it found and raises the signal again) and does only what is safe in a dying process: it writes its id
to the collector's pipe and waits, bounded, for the collector's word that py-spy has read every thread's
native and Python frames from outside it; then it dies by the same signal, so the exit is unchanged. Nothing
runs in the faulting process that could need a lock it holds (no fork, no allocation, no unwinding in it). The
library is built with the container's C compiler once a process; where none builds it, or no collector runs,
the task says so and runs as before."""

import functools
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHIM = r"""
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef PR_SET_PTRACER
#define PR_SET_PTRACER 0x59616d61
#endif
#define PR_ANY ((unsigned long)-1)

static const int SIGNALS[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE};
static char pipe_path[4096], done_dir[4096];
static int armed_waits;

static size_t digits(char *out, long value) {
    char reversed[24];
    size_t count = 0;
    do {
        reversed[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value > 0 && count < sizeof reversed);
    for (size_t index = 0; index < count; index++) out[index] = reversed[count - 1 - index];
    return count;
}

static void caught(int signum) {
    int fd = open(pipe_path, O_WRONLY | O_NONBLOCK);
    if (fd >= 0) {
        char line[64], done[4200];
        size_t length = digits(line, (long)getpid());
        line[length++] = ' ';
        length += digits(line + length, signum);
        line[length++] = '\n';
        ssize_t ignored = write(fd, line, length);
        (void)ignored;
        close(fd);
        size_t at = strlen(done_dir);
        memcpy(done, done_dir, at);
        memcpy(done + at, "/done.", 6);
        at += 6;
        at += digits(done + at, (long)getpid());
        done[at] = '\0';
        struct timespec pause = {0, 100000000};
        struct stat held;
        for (int wait = 0; wait < armed_waits && stat(done, &held) != 0; wait++) nanosleep(&pause, NULL);
    }
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = SIG_DFL;
    sigaction(signum, &action, NULL);
    raise(signum);
}

__attribute__((constructor)) static void armed(void) {
    const char *pipe_given = getenv("DEW_NATIVE_PIPE"), *done_given = getenv("DEW_NATIVE_DONE");
    if (pipe_given == NULL || done_given == NULL || strlen(pipe_given) >= sizeof pipe_path
            || strlen(done_given) >= sizeof done_dir - 32)
        return;
    strcpy(pipe_path, pipe_given);
    strcpy(done_dir, done_given);
    const char *seconds = getenv("DEW_NATIVE_SECONDS");
    armed_waits = 10 * (seconds != NULL ? atoi(seconds) : 60);
    prctl(PR_SET_PTRACER, PR_ANY, 0, 0, 0);
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
COLLECT_SECONDS = 60
"""How long a dying process waits for its stacks to be read (`SHIM`), and py-spy has to read them (`collect`).
"""


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


class Capture:
    """One run's capture: its collector (`collect`) started before the run and ended after it, and the
    variables that arm the run's processes (`variables`; none, and `why`, where it cannot be armed). Its
    stacks are read after the run (`stacks`)."""

    def __init__(self, spy: Path, environment: dict[str, str] | None = None) -> None:
        self.where = Path(tempfile.mkdtemp(prefix="dew-capture-")).resolve()
        self.pipe, self.done = self.where / "pipe", self.where / "done"
        self.stacks_file = self.where / "stacks.txt"
        self.variables: dict[str, str] = {}
        self.collector: subprocess.Popen | None = None
        library, self.why = built()
        spy = spy.resolve()
        if library is None:
            return
        if not spy.is_file():
            self.why = f"no py-spy at {spy}"
            return
        os.mkfifo(self.pipe)
        self.done.mkdir()
        self.collector = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "collect", str(self.pipe), str(self.done),
             str(self.stacks_file), str(spy)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, start_new_session=True)
        # Armed only once the collector reads its pipe: a dying process finds no reader before then, and dies
        # unread (a process aborting at once raced a collector still starting).
        ready = time.monotonic() + 10
        while not (self.done / "ready").exists() and time.monotonic() < ready and self.collector.poll() is None:
            time.sleep(0.02)
        if not (self.done / "ready").exists():
            self.why = "the collector did not start"
            return
        given = (environment or os.environ).get("LD_PRELOAD", "")
        self.variables = {"LD_PRELOAD": " ".join(part for part in (given, str(library)) if part),
                          "DEW_NATIVE_PIPE": str(self.pipe), "DEW_NATIVE_DONE": str(self.done),
                          "DEW_NATIVE_SECONDS": str(COLLECT_SECONDS)}

    def stacks(self) -> str:
        """The collector ended, and what it read: every signalled process's native stacks, or why none."""
        if self.collector is not None:
            self.collector.terminate()
            try:
                self.collector.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.collector.kill()
        if self.stacks_file.is_file():
            return self.stacks_file.read_text(errors="replace")
        return f"no native stacks were captured: {self.why}\n" if self.why else ""


def collect(pipe: Path, done: Path, stacks: Path, spy: Path) -> None:
    """The collector: each process that writes `<pid> <signal>` to `pipe` as a signal ends it (`SHIM`) gets
    py-spy's dump of its every thread, native and Python frames, appended to `stacks`, then `done/done.<pid>`,
    its word that it may die. Opened for reading and writing, so the pipe never reads its end between writers.
    """
    with open(os.open(pipe, os.O_RDWR), encoding="ascii", errors="replace") as said:
        (done / "ready").touch()
        for line in said:
            parts = line.split()
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                continue
            pid, signum = parts
            try:
                read = subprocess.run([str(spy), "dump", "--native", "--pid", pid], capture_output=True,
                                      text=True, timeout=COLLECT_SECONDS - 5, check=False)
                frames = read.stdout or f"py-spy read nothing: {read.stderr.strip()}"
            except subprocess.TimeoutExpired:
                frames = f"py-spy read nothing in {COLLECT_SECONDS - 5} s"
            with stacks.open("a") as out:
                at = time.strftime("%H:%M:%SZ", time.gmtime())
                out.write(f"== process {pid} ended by signal {signum} ({at}): its threads' native stacks\n"
                          f"{frames}\n")
            (done / f"done.{pid}").touch()


if __name__ == "__main__":
    if sys.argv[1:2] == ["collect"]:
        collect(*map(Path, sys.argv[2:6]))
        sys.exit(0)
    # A check of the capture where it runs: a child that aborts, its stacks printed.
    capture = Capture(Path(sys.executable).with_name("py-spy"))
    print(capture.why or "armed")
    subprocess.run([sys.executable, "-c", "import faulthandler, os; faulthandler.enable(); os.abort()"],
                   env={**os.environ, **capture.variables}, check=False)
    print(capture.stacks())
