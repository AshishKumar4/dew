"""Supervised fine-tuning data: conversations with a role on every token.

A `ChatMessages` source reads a parquet file of conversations and renders
each with the tokenizer's chat template. Every token gets the role of the
message that wrote it, so `LMObjective` with `loss_role=Role.ASSISTANT`
trains on assistant tokens only. Packing reuses the token pipeline's
first-fit bins with `text_roles` as one more per-token feature, and the bins
come out with `text`, `text_roles`, `text_segment_ids` and `text_positions`
aligned.
"""

from __future__ import annotations

import dataclasses
import functools
import threading
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING
import grain.python as pygrain
import jax
import numpy as np

from dew.registry import datasets

from .dataset import Batch, Dataset, DatasetSpec, Loading, local_batch
from .tokens import DocumentChunks, bounded, chunk_counts

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
ROLES_KEY = "text_roles"
"""Batch key the chat pipeline packs `[B, seq_len + 1]` int8 roles under."""


class Role(int, Enum):
    """Who wrote a token of a rendered conversation.

    The values are the `text_roles` column: 0 pads, the rest name the message
    whose span holds the token. An assistant turn's opening header belongs to
    no span; it reads as padding, so only the completion counts.
    """

    PAD = 0
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3
    TOOL = 4


def _message(message: Mapping[str, object], index: int, source: str
             ) -> tuple[Role, dict[str, str]]:
    """One `{role, content}` row validated into its role and plain-dict form."""
    role = message.get("role") if isinstance(message, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError(
            f"{source} row {index}: a message is a {{role, content}} pair of "
            f"strings, got {message!r}")
    try:
        return Role[role.upper()], {"role": role, "content": content}
    except KeyError:
        names = ", ".join(member.name.lower() for member in Role if member is not Role.PAD)
        raise ValueError(
            f"{source} row {index}: unknown role {role!r}; the roles are {names}") from None


def _token_ids(tokenizer: PreTrainedTokenizerBase, rows: list[dict[str, str]],
               generation_prompt: bool, source: str) -> list[int]:
    """`rows` through the chat template to token ids."""
    rendered = tokenizer.apply_chat_template(
        rows, tokenize=True, return_dict=False,
        add_generation_prompt=generation_prompt)
    if not isinstance(rendered, list):
        raise ValueError(
            f"tokenizer {source!r} answered the chat template with "
            f"{type(rendered).__name__}, not token ids")
    ids: list[int] = []
    for token in rendered:
        if not isinstance(token, int):
            raise ValueError(
                f"tokenizer {source!r} answered the chat template with a "
                f"non-id {token!r}")
        ids.append(token)
    return ids


def render_conversation(tokenizer: PreTrainedTokenizerBase,
                        messages: Sequence[Mapping[str, object]],
                        source: str) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and per-token roles for one conversation.

    Message k's span comes from prefix rendering: `messages[:k]` with the
    generation prompt against `messages[:k+1]` without it for an assistant
    turn, plain prefixes otherwise. Both renders must agree where they
    overlap, and the whole conversation must extend each prefix. A template
    and tokenizer pair that re-segments a boundary fails here with the
    message index. The reference is TRL's assistant mask for the same
    template and conversation (tests/fixtures/rl/chat.npz).

    `source` names the tokenizer for those refusals. The arrays are int32
    ids and int8 roles.
    """
    parsed = [_message(message, index, source) for index, message in enumerate(messages)]
    if not parsed:
        raise ValueError(f"{source} holds an empty conversation, which has no tokens to train on")
    rows = [row for _, row in parsed]
    full = _token_ids(tokenizer, rows, False, source)
    roles = np.zeros(len(full), np.int8)
    for position, (role, _) in enumerate(parsed):
        if position == 0:
            if role is Role.ASSISTANT:
                raise ValueError(
                    f"{source}: the first message is an assistant turn, so its "
                    "opening header cannot be separated from its completion "
                    "through the template; start the conversation with a system "
                    "or user message")
            prefix: list[int] = []
        else:
            prefix = _token_ids(tokenizer, rows[:position], role is Role.ASSISTANT, source)
        following = _token_ids(tokenizer, rows[:position + 1], False, source)
        if following[:len(prefix)] != prefix:
            raise ValueError(
                f"tokenizer {source!r} does not render incrementally at message "
                f"{position} ({role.name.lower()}): the prefix tokenizes to "
                f"{prefix[-8:]}, the longer render to "
                f"{following[max(0, len(prefix) - 8):len(prefix)]}")
        if full[:len(following)] != following:
            raise ValueError(
                f"tokenizer {source!r} does not render incrementally at message "
                f"{position} ({role.name.lower()}): the conversation does not "
                "extend the prefix render")
        roles[len(prefix):len(following)] = role.value
    return np.asarray(full, np.int32), roles


_tokenizer_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """The chat template's tokenizer, loaded once per process.

    The render map runs inside grain workers, which unpickle only the path,
    so each worker loads its own copy on its first record. The lock
    serializes that first import: transformers swaps in its lazy module
    object while it initializes, and two threads importing it together can
    catch it half built. The prompt source shares this cache.
    """
    with _tokenizer_lock:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)


class ConversationSource:
    """Random access over the `prompt` column of a parquet file.

    One record is one conversation, a list of `{role, content}` messages in
    the verl layout. The table is read once; rows come back as plain dicts,
    which pickle across to grain workers. Extra columns are not read.
    """

    def __init__(self, path: str):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError(
                "reading chat parquet needs pyarrow: pip install pyarrow") from exc
        table = parquet.read_table(path, columns=["prompt"])
        if table.num_rows == 0:
            raise ValueError(f"{path} holds no conversations")
        self._conversations = table.column("prompt").to_pylist()
        self.path = path

    def __repr__(self) -> str:
        # Grain writes repr(source) into a data loader iterator's checkpoint,
        # so a resumed run needs the file here, not an address in this process.
        return f"{self.__class__.__name__}(path={self.path!r})"

    def __len__(self) -> int:
        return len(self._conversations)

    def __iter__(self) -> Iterator[Batch]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Batch:
        return {"messages": self._conversations[index]}


class RenderConversation:
    """Source rows into rendered ids and per-token roles, for `map_with_index`.

    Holds only the tokenizer path, so grain workers unpickle the name and
    load their own copy. Failures name the tokenizer and the row.
    """

    def __init__(self, tokenizer: str):
        self.tokenizer = tokenizer

    def __call__(self, index: int, record: Batch) -> Batch:
        ids, roles = render_conversation(
            load_tokenizer(self.tokenizer), record["messages"],
            f"{self.tokenizer} row {index}")
        return {"text": ids, ROLES_KEY: roles}


def _lengths(source: ConversationSource, tokenizer: str) -> list[int]:
    """Every row's rendered length, validating the file up front.

    The chunk table needs the lengths before the lazy render map runs, so
    this pass renders each row once and keeps only its count. A bad row
    fails the run here with its index.
    """
    load = load_tokenizer(tokenizer)
    return [len(render_conversation(load, record["messages"], f"{source.path} row {index}")[0])
            for index, record in enumerate(source)]


@datasets("chat_messages")
@dataclasses.dataclass(frozen=True)
class ChatMessages(DatasetSpec):
    """Conversations rendered with the tokenizer's chat template, packed.

    `path` is a parquet file whose `prompt` column holds lists of `{role,
    content}` messages; `tokenizer` is the hub name or local path whose chat
    template renders them. Each conversation (in chunks, when it outgrows the
    window) is one element the packer adds to the first bin with room, and
    every emitted window carries `text_roles` beside the ids, so the loss can
    count one role's targets. `val_path` is a second parquet file scored as
    one pass; None trains without validation.
    """

    tokenizer: str
    path: str | None = None
    val_path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()
    packing_bins: int = 8

    def load(self, *, batch: int) -> Dataset:
        from grain.experimental import FirstFitPackIterDataset

        if self.path is None:
            raise ValueError("ChatMessages reads a parquet file: --data.path names it")
        per_process = local_batch(batch)
        window = self.seq_len + 1
        train_source = ConversationSource(self.path)
        train_lengths = _lengths(train_source, self.tokenizer)

        def stream(source: ConversationSource, lengths: list[int], shuffle: bool,
                   epochs: int | None):
            rendered = pygrain.MapDataset.source(source).map_with_index(
                RenderConversation(self.tokenizer))
            chunks: pygrain.MapDataset[Batch] = DocumentChunks(rendered, lengths, window)
            chunks = chunks[jax.process_index()::jax.process_count()]  # a slice is a dataset
            if shuffle:
                chunks = chunks.shuffle(self.seed)
            reads = chunks.repeat(epochs).to_iter_dataset()
            if self.loading.workers:
                # The workers render records, and the packer stays behind them
                # in this process. Grain runs a whole pipeline per worker, so
                # packing inside them would fill bins from one worker's slice
                # of the records and make the windows depend on worker_count.
                reads = reads.mp_prefetch(pygrain.MultiprocessingOptions(
                    num_workers=self.loading.workers,
                    per_worker_buffer_size=self.loading.worker_buffer))
            packed = FirstFitPackIterDataset(
                reads,
                length_struct={"text": window, ROLES_KEY: window},
                num_packing_bins=self.packing_bins,
                seed=self.seed,
                # Bins come out in packing order for val, so a validation pass
                # is the same batches every time.
                shuffle_bins=shuffle,
                padding_struct={"text": 0, ROLES_KEY: 0},
            )
            return iter(packed.batch(per_process, drop_remainder=True))

        val = None
        if self.val_path is not None:
            val_source = ConversationSource(self.val_path)
            val_lengths = _lengths(val_source, self.tokenizer)
            val = bounded(lambda: stream(val_source, val_lengths, False, 1), self.val_batches)
        return Dataset(
            train=lambda: stream(train_source, train_lengths, True, None),
            val=val,
            records=int(chunk_counts(train_lengths, window).sum()),
            batch=batch,
        )
