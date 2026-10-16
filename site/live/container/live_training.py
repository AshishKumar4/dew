"""Bounds a live training cell to a couple of minutes on a shared 4-vCPU host.

A training context runs Dew itself, not the stand-ins of model_client.
`install` has IPython run each cell with its calls that name a batch, steps
or log_every, or unpack keywords, passed through `capped`: a Dew `load` or
`loader` (a dataset's or a recipe's, whose `load` reads its batch through
`dew.config.data.loader`) reads batches of at most BATCH, and a Dew `fit` runs
at most STEPS steps, logging at least four times. Each cap prints what it
changed before the run's own output, so the cell's first lines say how this
run differs from the code on the page.
"""

import ast
import inspect

BATCH, STEPS = 8, 20
CAPPED = ("batch", "steps", "log_every")
"""The keywords a call names that `Capped` sends through `capped`, beside any `**` it unpacks."""


def batch(asked):
    """The batch a cell asked for, lowered to BATCH; a token budget's batch is not counted in rows."""
    if (asked or 0) > BATCH:
        print(f"Live run: batch {asked} -> {BATCH}, so it fits a shared 4-vCPU host.")
        return BATCH
    return asked


def fitted(steps, log_every):
    """A `fit` call's steps and log_every: more than STEPS steps lowered to STEPS, logging at least four
    times."""
    changed, cut = [], steps > STEPS
    if cut:
        changed.append(f"steps {steps} -> {STEPS}")
        steps = STEPS
    if log_every > max(1, steps // 4):
        changed.append(f"log_every {log_every} -> {max(1, steps // 4)}")
        log_every = max(1, steps // 4)
    if changed:
        why = "so it finishes in about two minutes" if cut else "to log the run as it goes"
        print(f"Live run: {', '.join(changed)}, {why}.")
    return {"steps": steps, "log_every": log_every}


def capped(call, /, *args, **options):
    """`call(*args, **options)`, its keywords capped where `call` is Dew's: a `load` or `loader`'s batch
    (`batch`), a `fit`'s steps and log_every, the latter its own default where the cell names none
    (`fitted`). Whatever name the cell calls it by, and however it passes the keywords."""
    if (getattr(call, "__module__", None) or "").split(".")[0] == "dew":
        name = getattr(call, "__name__", "")
        if name in ("load", "loader") and "batch" in options:
            options["batch"] = batch(options["batch"])
        elif name == "fit" and "steps" in options:
            default = inspect.signature(call).parameters["log_every"].default
            options.update(fitted(options["steps"], options.get("log_every", default)))
    return call(*args, **options)


class Capped(ast.NodeTransformer):
    """A cell with each call that names one of `CAPPED` or unpacks keywords made through `capped`, as
    IPython runs every cell of a training context (`install`)."""

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if not any(keyword.arg is None or keyword.arg in CAPPED for keyword in node.keywords):
            return node
        # This module, as a cell reaches it without binding a name of its own.
        module = ast.Call(ast.Name("__import__", ast.Load()), [ast.Constant("live_training")], [])
        return ast.Call(ast.Attribute(module, "capped", ast.Load()), [node.func, *node.args], node.keywords)


def install():
    """Cap every later cell of this kernel: IPython transforms a cell before it runs any of it, so the
    cell that installs this is not one of them."""
    from IPython import get_ipython

    get_ipython().ast_transformers.append(Capped())
