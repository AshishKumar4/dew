"""Bounds a live training cell to a couple of minutes on a shared 4-vCPU host.

A training context runs Dew itself, not the stand-ins of model_client.
`install` caps the batch of a `dew.data.load` dataset at BATCH and a
`Trainer.fit` at STEPS steps, logging at least four times. Each cap prints
what it changed before the run's own output, so the cell's first lines say
how this run differs from the code on the page.
"""

BATCH, STEPS = 8, 20


def install():
    import dew.data
    from dew.training import Trainer

    load, fit = dew.data.load, Trainer.fit

    def capped_load(source, *, batch, **options):
        if batch > BATCH:
            print(f"Live run: batch {batch} -> {BATCH}, so it fits a shared 4-vCPU host.")
            batch = BATCH
        return load(source, batch=batch, **options)

    def capped_fit(self, dataset, *, steps, log_every=100, **options):
        changed = []
        if steps > STEPS:
            changed.append(f"steps {steps} -> {STEPS}")
            steps = STEPS
        if log_every > max(1, steps // 4):
            changed.append(f"log_every {log_every} -> {max(1, steps // 4)}")
            log_every = max(1, steps // 4)
        if changed:
            print(f"Live run: {', '.join(changed)}, so it finishes in about two minutes.")
        return fit(self, dataset, steps=steps, log_every=log_every, **options)

    dew.data.load, Trainer.fit = capped_load, capped_fit
