"""Corpora phases share (`dew.data.pipeline.Phases`): what a phase reads (`_corpora`), a corpus by its
content however a chain names it (`_corpus_key`), and the members each phase on its own stream continues
another's by (`_shared`), a corpus a phase reads alone read as the one member of a mix (`_alone_mixed`)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator

from dew._records import JSON, json_value
from dew.data.analysis import _member_keys, _members, _name, _until, same_stage
from dew.data.pipeline import BatchStage, Combined, Mixture, Phases, Pipeline, TokenBudget
from dew.data.plan import fingerprint, identified, structured


def _corpus_key(chain: Pipeline, *, local: bool = False) -> str:
    """A corpus (`_corpora`) as phases share it, whatever a chain names it: its store's content
    (`Source.fingerprint`); a concatenation's, its members' in order, each what it reads and the steps it
    reads it through (`dew.data.plan.structured`). With `local`, a store whose content only its name states
    (a Grain dataset, an iterator), given none, is besides the one object it is in this process
    (`describe`: its place among the unnamed), which `_corpus_keys` reads its run's corpora apart by."""
    def read_of(pipeline: Pipeline) -> JSON:
        source = pipeline.source
        if not isinstance(source, Combined):
            stated = json.loads(json.dumps(source.fingerprint(), sort_keys=True, default=str))
            if local and not identified(stated):
                stated = {"unnamed": source.describe(), **stated}
            return stated
        weights = {"weights": list(source.weights)} if isinstance(source, Mixture) else {}
        members = [[read_of(child), [structured(stage) for stage in child.stages]]
                   for child in source.pipelines]
        return json_value({type(source).__name__: members, **weights}, "a corpus's key")

    return fingerprint(read_of(chain))


def _corpus_keys(phases: Phases) -> Callable[[Pipeline], str]:
    """The key each corpus `phases` read is shared and saved by: its content's (`_corpus_key`), the key a
    run's saved cursor names, for the first corpus in phase order that states that content, an unnamed
    store reused across phases among them; for each later corpus of one content (another unnamed store
    alike in every stated field), the content's with its place among them, so they share no corpus (row
    14119) and no saved cursor's key moves, a phase appended after them included (train 1484: a reused
    unnamed store's moved, its phase read again from its start). A corpus's own place in the process
    (`_corpus_key`'s `local`) tells a repeat of it from another; no key states it."""
    chains = [chain for pipeline in phases.pipelines for chain in (pipeline, *_corpora(pipeline))]
    seen: dict[str, list[str]] = {}
    for chain in chains:
        held = seen.setdefault(_corpus_key(chain), [])
        if (local := _corpus_key(chain, local=True)) not in held:
            held.append(local)

    def key(chain: Pipeline) -> str:
        content = _corpus_key(chain)
        place = seen[content].index(_corpus_key(chain, local=True))
        return content if place == 0 else fingerprint({"content": content, "place": place})

    return key


def _alone_mixed(phases: Phases) -> Phases:
    """`phases`, read each on its own stream (`StepPhases`), with each phase that reads a corpus another
    phase reads too, and reads it alone (the phase's chain a store's), reading it as the one member of a
    mix of its own: its steps before its batch the member's, the batch and the steps after it the mix's.
    A phase on its own stream continues a corpus it shares by its member's draws (`_shared`), so a corpus
    read alone continues as `mix([(corpus, 1.0)])` written in each phase does, where it was refused. A
    phase that reads it otherwise (in a mix of its own, a concatenation) stays as written. Every phase of a
    read on its own streams is staged from the phases so read (`dew.data.compile._compile_phases`)."""
    key = _corpus_keys(phases)
    keys = [{key(chain) for chain in _corpora(pipeline)} for pipeline in phases.pipelines]
    shared = {key for index, held in enumerate(keys) for key in held
              if any(key in others for at, others in enumerate(keys) if at != index)}

    def mixed(pipeline: Pipeline) -> Pipeline:
        if isinstance(pipeline.source, Combined) or key(pipeline) not in shared:
            return pipeline
        cut = _until(pipeline.stages, (BatchStage, TokenBudget))
        # The member the corpus and its steps, named as the phase names it; its read and loading the phase's.
        member = Pipeline(pipeline.source, pipeline.stages[:cut], pipeline.name)
        return dataclasses.replace(pipeline, source=Mixture((member,), (1.0,)), stages=pipeline.stages[cut:])

    return dataclasses.replace(phases, pipelines=tuple(mixed(pipeline) for pipeline in phases.pipelines))



def _shared(phases: Phases) -> tuple[dict[str, str], ...]:
    """Each phase's mix members a phase on its own stream (`StepPhases`) reads on from another phase, by
    member key, and the corpus each is (`_corpus_key`; a concatenation is one, its join): the next phase
    that reads a corpus continues it from the draws the run trained of it (`dew.data.mixing.Stamping`).
    Refuses a shared corpus a phase reads other than as a member of its own mix, whose position no key
    names, and one two phases read through other steps before their mixes, where the earlier's draws are
    no draws of the later's."""
    key = _corpus_keys(phases)
    reads: dict[str, list[tuple[int, Pipeline]]] = {}
    for index, pipeline in enumerate(phases.pipelines):
        for chain in _corpora(pipeline):
            reads.setdefault(key(chain), []).append((index, chain))
    shared = {key: read for key, read in reads.items() if len({index for index, _ in read}) > 1}
    members: list[dict[str, str]] = [{} for _ in phases.pipelines]
    keyed: dict[str, str] = {}
    for key, read in shared.items():
        (first, chain), *_ = read
        for index, held in read:
            mix = phases.pipelines[index].source
            if not isinstance(mix, Mixture) or not any(held is child for child in mix.pipelines):
                # The phase that reads it otherwise may be the first that reads it: name another beside it.
                other = next(at for at, _ in read if at != index)
                raise ValueError(f"phases {min(index, other)} and {max(index, other)} both read "
                                 f"{_name(held)}, and phase {index} reads it other than as a member of its "
                                 "own mix, where a phase continues a corpus it shares by the member's key: "
                                 f"read it as one (`mix([({_name(held)}, 1.0)])` where the phase reads it "
                                 "alone)")
            if len(held.stages) != len(chain.stages) or not all(
                    same_stage(made, kept) for made, kept in zip(held.stages, chain.stages, strict=True)):
                raise ValueError(f"phases {first} and {index} read {_name(held)} through other steps "
                                 "before their mixes, so where the one left it is no place in the other's "
                                 "read: give it the same steps in each, and the steps that differ after the "
                                 "mix")
            place = next(at for at, child in enumerate(mix.pipelines) if child is held)
            member = _member_keys(_members(mix.pipelines))[place]
            # A member's draws and its steps' randomness are keyed by its key in its mix (`_member_key`).
            if keyed.setdefault(key, member) != member:
                raise ValueError(f"phases {first} and {index} read {_name(held)} under other keys in their "
                                 f"mixes ({keyed[key]}, {member}), which key its shuffles and draws: name it "
                                 "(`.named(...)`) alike in each, or keep its place among the unnamed")
            members[index][member] = key
    return tuple(members)


def _corpora(pipeline: Pipeline) -> Iterator[Pipeline]:
    """Each corpus a phase's order places records of (`_Trained`): a chain under `pipeline`'s mixtures that
    reads a store, or a concatenation, whose join is one corpus (`_continued` reads the join on)."""
    source = pipeline.source
    if isinstance(source, Mixture | Phases):
        for child in source.pipelines:
            yield from _corpora(child)
    else:
        yield pipeline
