"""Bounds a live training cell to a couple of minutes on a shared 4-vCPU host.

A training context runs Dew itself, not the stand-ins of model_client.
`install` has IPython run each cell with its own calls capped (`Capped`): a
`load` or `loader` call's batch, a dataset's or a recipe's (whose `load` reads
its batch through `dew.config.data.loader`), at BATCH, and a `fit` call's steps
at STEPS, logging at least four times. Each cap prints what it changed before
the run's own output, so the cell's first lines say how this run differs from
the code on the page.
"""

import ast

BATCH, STEPS = 8, 20


def batch(asked):
    """The batch a cell asked for, lowered to BATCH; a token budget's batch is not counted in rows."""
    if (asked or 0) > BATCH:
        print(f"Live run: batch {asked} -> {BATCH}, so it fits a shared 4-vCPU host.")
        return BATCH
    return asked


def fitted(steps, log_every=100):
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


def _here(name: str) -> ast.expr:
    """This module's `name`, as a cell reaches it without binding a name of its own."""
    module = ast.Call(ast.Name("__import__", ast.Load()), [ast.Constant("live_training")], [])
    return ast.Attribute(module, name, ast.Load())


class Capped(ast.NodeTransformer):
    """A cell with its `load` and `loader` calls' batch passed through `batch`, and its `fit` calls' steps
    and log_every through `fitted`, as IPython runs every cell of a training context (`install`)."""

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        named = {keyword.arg: keyword for keyword in node.keywords if keyword.arg is not None}
        if callee in ("load", "loader") and "batch" in named:
            named["batch"].value = ast.Call(_here("batch"), [named["batch"].value], [])
        if callee == "fit" and "steps" in named:
            fitting = ("steps", "log_every")
            capped = [named[name].value for name in fitting if name in named]
            node.keywords = [keyword for keyword in node.keywords if keyword.arg not in fitting]
            node.keywords.append(ast.keyword(None, ast.Call(_here("fitted"), capped, [])))
        return node


def install():
    from IPython import get_ipython

    get_ipython().ast_transformers.append(Capped())
