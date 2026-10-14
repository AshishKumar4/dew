"""Bounds a live training cell to a couple of minutes on a shared 4-vCPU host.

A training context runs Dew itself, not the stand-ins of model_client.
`install` caps the batch of a dataset `dew.data.load` or a recipe's loader
(`dew.config.data.loader`, which a recipe's `load` calls) reads at BATCH, and a
`Trainer.fit` at STEPS steps, logging at least four times. Each cap prints
what it changed before the run's own output, so the cell's first lines say
how this run differs from the code on the page.
"""

BATCH, STEPS = 8, 20


def capped_load(load):
    """`load`, with a batch over BATCH lowered to it; a token budget's batch is not counted in rows."""
    def capped(source, **options):
        if (options.get("batch") or 0) > BATCH:
            print(f"Live run: batch {options['batch']} -> {BATCH}, so it fits a shared 4-vCPU host.")
            options["batch"] = BATCH
        return load(source, **options)
    return capped


def capped_fit(fit):
    """`fit`, with more than STEPS steps lowered to STEPS, logging at least four times."""
    def capped(self, dataset, *, steps, log_every=100, **options):
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
        return fit(self, dataset, steps=steps, log_every=log_every, **options)
    return capped


def install():
    import warnings

    import dew.config.data
    import dew.data
    from dew.training import Trainer

    # tqdm.auto falls back to text bars in a kernel without ipywidgets, and says so on import.
    warnings.filterwarnings("ignore", message="IProgress not found")

    dew.data.load, Trainer.fit = capped_load(dew.data.load), capped_fit(Trainer.fit)
    dew.config.data.loader = capped_load(dew.config.data.loader)
