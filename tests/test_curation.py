"""Curating a web corpus as a chain reads it, through DataTrove's filters: heuristic quality,
n-gram decontamination against evaluation texts, a fastText classifier and language
identification, each dropping the records it rejects, counted by the reason it names, and each
keyed in the plan by the rules and the model it runs."""

import pytest

from dew.data import Loading, Memory, text
from dew.data.compile import compile, iterate
from dew.data.pipeline import Pipeline

pytest.importorskip("datatrove")

PROSE = [
    "The river runs past the old mill and through the town, where the children wait on the bridge "
    "to watch the boats go by in the evening. Their parents talk about the harvest and the weather, "
    "and about the market that opens on the square at the end of every week. When the light goes, "
    "the lamps come on along the water and the town settles down for the night.",
    "A good bread starts with flour, water, salt and time. The dough is mixed and left to rest, then "
    "folded a few times over the next hours so that it grows strong enough to hold its shape in the "
    "oven. Most of the flavour comes from the slow rise, which is why a baker plans the day around "
    "it and does not rush the loaves that are still waiting on the bench.",
    "Our team met on Tuesday to review the plans for the new library. We agreed that the reading "
    "room should face the park, that the shelves for the young readers should be low and easy to "
    "reach, and that the building should stay open late on the days when the schools are closed. "
    "The next meeting will look at the budget and at the schedule for the work.",
]
EVALUATION = ("Which of these is the capital of France and has been since the late tenth century, "
              "according to the historians who study the early kingdoms of western Europe?")


def one_pass(chain: Pipeline) -> tuple[list[int], dict[str, float]]:
    """The ids one pass of a chain keeps, in order, and what a run counts it dropping, by reason."""
    kept = [int(record["id"]) for record in iterate(chain, read="memory")]
    dropped = {key.rsplit("/", 1)[1]: count for key, count in chain.stats(records=len(PROSE) + 2).items()
               if key.startswith("data/rejected/") and count}
    return kept, dropped


def web_pages() -> Memory:
    """The prose, a fragment too short for any quality rule, and a page that quotes an evaluation
    question whole."""
    records = [{"id": index, "text": passage} for index, passage in enumerate(PROSE)]
    records.append({"id": 10, "text": "Click here."})
    records.append({"id": 11, "text": f"{PROSE[0]} Today's quiz asked: {EVALUATION}"})
    return Memory(records, shard_rows=len(records))


def test_a_curated_corpus_keeps_its_prose_and_counts_what_each_filter_drops():
    curated = Pipeline(web_pages()).filter_map(text.Quality("gopher"),
                                            text.Decontaminated(against=(EVALUATION,)))
    kept, dropped = one_pass(curated)
    assert kept == [0, 1, 2]
    assert dropped.pop("contaminated") == 1 and sum(dropped.values()) == 1, dropped


def test_a_curation_steps_rules_key_the_plan_and_the_same_rules_key_it_alike():
    def plan(*steps):
        return compile(Pipeline(web_pages()).filter_map(*steps).batch(1), read="memory",
                       loading=Loading(workers=0, threads=1, read_buffer=1)).plan.identity()

    gopher = plan(text.Quality("gopher"))
    assert gopher == plan(text.Quality("gopher"))
    assert gopher != plan(text.Quality("gopher", settings=(("min_doc_words", 20),)))
    assert gopher != plan(text.Quality("fineweb"))
    twelve, eight = (text.Decontaminated(against=(EVALUATION,), n=n) for n in (12, 8))
    assert plan(twelve) != plan(eight)


def test_a_fasttext_classifier_keeps_the_label_it_is_asked_for_and_its_model_keys_the_plan(tmp_path):
    fasttext = pytest.importorskip("fasttext")
    examples = tmp_path / "examples.txt"
    lines = [f"__label__prose {passage}" for passage in PROSE for _ in range(20)]
    lines += ["__label__spam click here buy now free offer win prize" for _ in range(60)]
    examples.write_text("\n".join(lines) + "\n")
    model = tmp_path / "model.bin"
    fasttext.train_supervised(input=str(examples), epoch=25, minCount=1, seed=0, thread=1).save_model(
        str(model))
    classified = Pipeline(web_pages()).filter_map(text.Classifier(str(model), keep=(("prose", 0.5),)))
    kept, dropped = one_pass(classified)
    assert kept == [0, 1, 2, 11] and dropped == {"classifier": 1}


@pytest.mark.network
def test_language_identification_keeps_the_language_it_is_asked_for():
    records = [{"id": index, "text": passage} for index, passage in enumerate(PROSE)]
    french = ("Le fleuve passe devant le vieux moulin et traverse la ville, où les enfants attendent sur "
              "le pont pour regarder passer les bateaux le soir.")
    records.append({"id": 10, "text": french})
    kept, dropped = one_pass(Pipeline(Memory(records, shard_rows=len(records)))
                         .filter_map(text.Language(("en",))))
    assert kept == [0, 1, 2] and dropped == {"language": 1}
