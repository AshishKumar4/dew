"""Curating a web corpus as a chain reads it, through DataTrove's filters: heuristic quality,
n-gram decontamination against evaluation texts, a fastText classifier and language
identification, each dropping the records it rejects, counted by the reason it names, and each
keyed in the plan by the rules and the model it runs; and sealing it without its near-duplicates
by DataTrove's MinHash stages, read once."""

import pytest

from dew.data import Files, Loading, Memory, text
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
FRENCH = ("Le fleuve passe devant le vieux moulin et traverse la ville, où les enfants attendent sur le pont "
          "pour regarder passer les bateaux le soir.")
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


@pytest.mark.network
def test_a_fasttext_classifier_keeps_the_label_it_is_asked_for():
    """A published fastText classifier, lid.176's compressed release, keeps the pages it scores
    English and drops the French one under the classifier's reason."""
    records = [{"id": index, "text": passage} for index, passage in enumerate(PROSE)]
    records.append({"id": 10, "text": FRENCH})
    model = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    kept, dropped = one_pass(Pipeline(Memory(records, shard_rows=len(records)))
                             .filter_map(text.Classifier(model, keep=(("en", 0.5),))))
    assert kept == [0, 1, 2] and dropped == {"classifier": 1}


@pytest.mark.network
def test_a_classifier_replaced_at_its_path_runs_as_the_identity_it_records(tmp_path):
    """A classifier step reads its model once, by content: a model replaced at the same path later
    is a new step's model, both what its filter scores with and what its plan records, while the
    first step still scores and records its own."""
    import shutil

    from datatrove.io import cached_asset_path_or_download

    released = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    path = tmp_path / "model.ftz"
    shutil.copy(cached_asset_path_or_download(released, namespace="filters", subfolder="fasttext"), path)
    records = [{"id": index, "text": passage} for index, passage in enumerate(PROSE)]
    records.append({"id": 10, "text": FRENCH})

    def chain(step):
        return Pipeline(Memory(records, shard_rows=len(records))).filter_map(step)

    def identity(step):
        return compile(chain(step).batch(1), read="memory",
                       loading=Loading(workers=0, threads=1, read_buffer=1)).plan.identity()

    english = text.Classifier(str(path), keep=(("en", 0.5),))
    first = one_pass(chain(english)), identity(english)
    path.write_bytes(path.read_bytes() + b"\0")
    assert (one_pass(chain(english)), identity(english)) == first
    replaced = text.Classifier(str(path), keep=(("en", 0.5),))
    assert identity(replaced) != first[1]
    assert replaced.pinned.read_bytes() == path.read_bytes()


@pytest.mark.network
def test_language_identification_keeps_the_language_it_is_asked_for():
    records = [{"id": index, "text": passage} for index, passage in enumerate(PROSE)]
    records.append({"id": 10, "text": FRENCH})
    kept, dropped = one_pass(Pipeline(Memory(records, shard_rows=len(records)))
                         .filter_map(text.Language(("en",))))
    assert kept == [0, 1, 2] and dropped == {"language": 1}


def test_a_corpus_sealed_without_its_near_duplicates_keeps_one_record_of_each_group(tmp_path):
    """A page copied whole, and one copied with a word changed, are near-duplicates of their source:
    the sealed corpus keeps one record of each group and every distinct page, in their order, and its
    manifest records the rule and the records removed."""
    copied = PROSE[1].replace("bread", "loaf", 1)
    pages = [PROSE[0], PROSE[1], PROSE[0], PROSE[2], copied]
    corpus = Memory([{"id": index, "text": page} for index, page in enumerate(pages)], shard_rows=len(pages))
    manifest = Pipeline(corpus).snapshot(str(tmp_path / "sealed"), deduplicate=text.NearDuplicates())
    stored = Files(str(tmp_path / "sealed")).records("indexed")
    kept = [int(stored[index]["id"]) for index in range(len(stored))]
    assert sorted({pages[index] for index in kept}) == sorted(set(PROSE))
    assert kept == sorted(kept) and len(kept) == len(PROSE)
    assert manifest["rows"] == len(PROSE)
    assert manifest["metadata"]["near_duplicates"]["removed"] == len(pages) - len(PROSE)


def test_deduplicating_snapshots_sealed_at_once_each_keep_what_one_alone_keeps(tmp_path):
    """Four corpora of pages of very different lengths sealed without their near-duplicates at once, on
    four threads: each keeps what it keeps sealed alone, as the signatures' word tokenizer, which sets
    its length bound per text, takes its turn with the other curations."""
    from concurrent.futures import ThreadPoolExecutor

    def corpus(scale):
        pages = [" ".join([page] * scale) for page in (*PROSE, PROSE[0])]
        records = [{"id": index, "text": page} for index, page in enumerate(pages)]
        return Memory(records, shard_rows=len(pages))

    def sealed(scale, root):
        Pipeline(corpus(scale)).snapshot(str(root / f"sealed-{scale}"), deduplicate=text.NearDuplicates())
        stored = Files(str(root / f"sealed-{scale}")).records("indexed")
        return [int(stored[index]["id"]) for index in range(len(stored))]

    scales = (1, 40, 3, 400)
    alone = [sealed(scale, tmp_path / "alone") for scale in scales]
    with ThreadPoolExecutor(len(scales)) as pool:
        together = list(pool.map(lambda scale: sealed(scale, tmp_path / "together"), scales))
    assert together == alone


def test_a_classifier_pins_its_model_on_a_host_that_never_pinned_one(tmp_path, monkeypatch):
    """A fresh host's temporary directory holds no pinned models yet: the first classifier step makes
    the place it pins them, and its pinned copy is its model's bytes."""
    import tempfile

    model = tmp_path / "model.bin"
    model.write_bytes(b"a model's bytes")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "fresh"))
    (tmp_path / "fresh").mkdir()
    assert text.Classifier(str(model)).pinned.read_bytes() == model.read_bytes()


def test_a_deduplicating_snapshot_of_a_pass_that_keeps_nothing_seals_nothing(tmp_path):
    """A chain whose filter rejects every page seals an empty snapshot, nothing removed."""
    rule = text.Quality("gopher", settings=(("min_doc_words", 10_000),))
    manifest = Pipeline(web_pages()).filter_map(rule).snapshot(str(tmp_path / "empty"),
                                                               deduplicate=text.NearDuplicates())
    assert manifest["rows"] == 0 and manifest["metadata"]["near_duplicates"]["removed"] == 0


def test_a_quality_rules_sequence_setting_reaches_its_filter_as_given():
    """A rule's sequence setting reaches DataTrove as the tuple it was given: FineWeb told to count only
    lines ending in "!" as punctuated keeps the page whose lines do and drops the one whose lines end
    in ".", where its own stop characters keep both."""
    pages = [{"id": index, "text": "\n".join(f"This is a reasonably long sentence, number {line} of the "
                                              f"page{end}" for line in range(5))}
             for index, end in enumerate(("!", "."))]

    def kept(rule):
        passed = iterate(Pipeline(Memory(pages)).filter_map(rule), read="memory")
        return [int(record["id"]) for record in passed]

    assert kept(text.Quality("fineweb", settings=(("stop_chars", ("!",)),))) == [0]
    assert kept(text.Quality("fineweb")) == [0, 1]


def test_a_quality_rule_that_rewrites_the_text_it_keeps_keeps_the_rewrite():
    """C4's rules drop the lines they disallow from a page they keep: the record keeps the page with
    its JavaScript notice gone, and its other fields as they were."""
    page = {"id": 0, "text": "This line has words.\nJavascript should be enabled."}
    kept = list(iterate(Pipeline(Memory([page])).filter_map(text.Quality("c4", settings=(
        ("min_num_sentences", -1),))), read="memory"))
    assert [(int(record["id"]), str(record["text"])) for record in kept] == [(0, "This line has words.")]


def test_a_quality_rule_takes_its_filters_own_settings_a_list_among_them():
    """A setting is the filter's constructor argument: Gopher's stop words, a list, given as a tuple,
    replace its own, so prose that holds none of them is rejected for that, and the step's identity
    keys the plan by them."""
    rule = text.Quality("gopher", settings=(("stop_words", ("zyzzyva", "quux")),))
    kept, dropped = one_pass(Pipeline(web_pages()).filter_map(rule))
    assert kept == [] and sum(dropped.values()) == len(PROSE) + 2


def test_curation_reads_alike_on_many_threads():
    """A process's readers share each DataTrove filter, built once: eight reader threads drop and
    keep what one does."""
    chain = Pipeline(web_pages()).filter_map(text.Quality("gopher"),
                                             text.Decontaminated(against=(EVALUATION,)))
    alone = [int(record["id"]) for record in iterate(chain.read("indexed").threads(1), read="indexed")]
    together = [int(record["id"]) for record in iterate(chain.read("indexed").threads(8), read="indexed")]
    assert together == alone == [0, 1, 2]
