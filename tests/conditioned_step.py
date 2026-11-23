"""The step the positions saved before the condition encoders moved hold (`saved_positions_support`'s
conditioned chains, tests/fixtures/positions/main-cdc45f8f9), alone in its module: a mix keeps each member by
a digest of the files its external steps' code is in (`dew.data.plan.StepIdentity.modules`), so an edit here
is a change of what those saved members run, and the positions must be saved again at their main
(tools/positions_saved_by_main.py); an edit beside the chains that hold it is not."""

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class Conditioned:
    """A record step whose rows depend on a condition encoder, as a diffusion run's caption step's do
    (`dew.inputs.InputSpec.dependencies`): its identity names the encoder's class by the path the saving Dew
    imports it from, dew.inputs.encoders before row 5996 and dew.interop.encoders since. Each row as it is."""

    def dependencies(self) -> Mapping[str, object]:
        from dew.inputs import CLIPText
        from dew.registry import import_path

        return {"encoder": {"class": import_path(CLIPText), "fields": {"checkpoint": "saved"}}}

    def __call__(self, row):
        return row
