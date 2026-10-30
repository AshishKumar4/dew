# Decision inputs: layouts as segments, conditioning as streams

Design record, 2026-10-23.

Built so far: the joint row's user turn as segments (`JointLayout.segments`: `Text`, `Media`, `State`,
`Schema`); layouts and heads recorded by import path; and the schema's stream (`Schema.stream`): the
schema read as the row's context, in a backbone pass of its own over the same weights, which the
cross-attention reads and the head pools its questions and options from (§2, §3); and a lettered
question row's question as its context (`Instructions.stream`, `OptionLines.stream`, both "context"
together): the row holds the state and the answer cue, whose state reads the letters
(`Labels`), the context the question and its lettered options, which the cross-attention reads and a
head that pools option lines (`SlotReadout`) pools from; a label readout (`LabelReadout`) pools nothing.
A question longer than the context its backbone reads is counted unfit, and a "none of the above" a
read draws that would pass it is declined, as one past `max_len` is. Several such rows pack a window,
each reading its own context. Next: a stream for the other segments. Owner: Training (`dew.decision`); cross-attention and `dew.inputs`' encoders: Models.

## Today

A layout is one of three classes (`MarkerLayout`, `StateFirstLayout`, `JointLayout` in `src/dew/decision/layout.py`, each recorded, as a decision head is, as a registry class record by import path, `{"class", "fields"}`, an earlier run's `{"name"}` read once at the pinned run boundary). Each assembles a row's tokens in a fixed order with fixed text, and each new arrangement needs a new class and a registry entry. A context beside the state (`Example.context`) has one path of its own: the backbone's token table embeds it, and the backbone's cross-attention reads it.

## 1. A layout is data: segments

A layout is an ordered tuple of segments and a row budget (`max_len`). A segment renders one part of a request into tokens:

| Segment | Renders | Marks |
|---|---|---|
| `Text(text)` | fixed text, a system prompt or the assistant's opening | nothing |
| `Images(text)` | a request's images, as the processor expands `text` per image | the media span |
| `State(render, cut)` | the state, rendered (`render`: text, compact JSON) | nothing |
| `Schema(field, option, order)` | every question: its framing (`field` template), its instructions, each option (`option` template) in `order` (keyed, as given, noul true first) | each question's span and each option's span |
| `Markers(...)` | one question's options each with a marker token, sequential or parallel | the question's span and one marker per option |

Each segment has a budget (a token cap, or none) and a cut rule (`start`, `end`, `never`). Budgets settle in one pass: segments that never cut take their tokens first, and the rest share what remains of `max_len` in their declared priority. That is the rule each class encodes today: a schema is never cut, and the state takes what the schema leaves. A row may hold the state more than once: each copy is a `State` of its own budget (`max_tokens`, or a joint layout's `max_state_tokens`) and cut rule (`keep`), a copy kept whole takes its tokens first, and the copies that cut share what remains max-min within one token, as any row's segments share its room (`dew.data.tokens.share_tokens`: each at most what it holds and its cap, a floor's remainder to the later in ascending order of want, none left over while a copy wants more), each cut from the whole state by its own rule. A question row's copies each render as their own `render` says; a joint row's render compact.

A layout is either joint (every question in one row) or per question (one row a question, the schema segment rendering that question alone). Its record is its segment tuple, each segment a frozen dataclass, recorded and rebuilt by its import path through `dew.registry`'s typed records (`to_record`, `from_record`), with no closed registry. A composition a user writes reloads as written.

The three layouts become three configurations of these segments. The proof: each configuration lays out the existing fixtures' rows byte for byte as its class does today: tokens, spans, positions and option slots, including the parallel layout and the cut cases. That test lands before the classes go.

## 2. A segment goes to a stream

Each segment names its stream: `main`, the row the backbone reads, or a named conditioning stream. A conditioning stream is an ordinary `dew.inputs.Condition`: an encoder over a batch field. The decision's data writes that field from the segments routed to the stream. The backbone itself is one such encoder: it embeds the stream's tokens with its own table, as the context does today, or runs its own forward over them (a second pass). The model's cross-attention layers read the stream's encoded sequence as their context (`CONTEXT`, `CONTEXT_LENGTHS`).

So today's `Example.context` is one segment, `State`-like over the example's context, routed to a stream the token table encodes. "The schema on a separate path" is a `Schema` segment routed to a stream the backbone encodes. Neither needs code of its own. Packing places every stream of a row together: a window holds rows while both its tokens (`max_len`) and its contexts (the objective's `context_window`, `max_len` or one row's `context_length` where that is longer) hold them, each row's context end to end after the rows before it (`context_segments`), each question carrying the row the packer placed it in (`question_rows`), and each span moved by its row's start along the stream it indexes: its tokens, or its context where the schema is the context. A row's tokens attend to its own context alone, and the backbone's pass over the contexts reads each as a document of its own.

- **One backbone.** The backbone as a stream encoder is `dew.nn.cross_attention.encode_context`, which any model with cross-attention layers takes (`ContextEncoder`: its token table or its own pass), and holds no parameters of its own: it references the model's variables, its LoRA factors included where the model is adapted, and calls the model's public `hidden_states` with no conditioning, so a stream never reads itself. One optimizer update moves both passes; a save stores one backbone. A stream's own adapter waits for an experiment that needs it.
- **Gradients.** A stream whose encoder trains nothing (a frozen trunk with no adapter) runs under `stop_gradient`: its backward would be pure cost.
- **Work.** FLOPs count every pass: the main forward; each stream's encoder, a forward alone where it trains nothing and `frozen_training_flops` where gradients flow through it; and the cross-attention blocks over each stream's real tokens.
- **Budget.** A stream's token budget is at most its cross-attention's `context_length`; a layout whose stream budget exceeds it is refused when it is built.
- **Serving.** A task's `admit` runs a stream's encoder once per request and hands its embeddings to the existing context path (the request's digest salt, slot-resident keys and values), so the `Server` never calls the model outside its programs.

## 3. The head reads the main pass

The joint head separates what it pools from what it attends to (`JointSchemaHead.__call__`). Question and option spans give its queries (`span_means`). The row's normed states are the memory its routing and field layers read, and the row's last token is the readout.

With the schema in the main row nothing changes. With the schema on a stream, the spans index the stream's final-normed states, the same norm as the main pass's: the stream's encoder returns per-token states. The memory and the readout stay the main pass's states, which read the stream through cross-attention. Each prediction then depends on the main row's input, conditioned on the schema, and never on the schema alone. The head takes the spans' source as an input (`main` or a stream name) rather than assuming the row.

## 4. What a user writes

```python
layout = Layout(max_len=4096, joint=True, segments=(
    Text(SYSTEM), Schema(stream="main"), State(cut="end"), Text(ASSISTANT)))  # schema before state

layout = Layout(max_len=4096, joint=True, segments=(
    Text(SYSTEM), State(cut="end"), Text(ASSISTANT),
    Schema(stream="schema")))  # the schema read through cross-attention, its own pass
```

On the command line it is the same record (`--layout.segments ...`). The decision guide shows both variants on one table, trained and served with no Dew change.

## Seams, invariants, rejected alternatives

- **One owner of a row's tokens:** the segment pass. `Encoding`, `Decide._encoded` and `bounds` all read rows through it, so training and serving lay rows out alike, as they do today.
- **Spans are positions in their stream.** Each row records whether its spans index its context (`Encoded.spans_in_context`), and a packed window moves them by the row's start along that stream. Laya's head, which reads its spans in the row, refuses spans in the context (TypeError); the joint head refuses spans in a stream it is not handed (ValueError).
- **Rejected: a fourth layout class** (schema first, or a separate schema): each new arrangement would be another class and another registry entry.
- **Rejected: schema-specific conditioning code.** A stream is a `Condition`, so the schema, a context and an image tower's sequence compose the same way.
