# Working on Dew

Read `CONTRIBUTING.md` first. It is the contract; this file is the checklist an agent runs before, during and after a task. When they disagree, `CONTRIBUTING.md` wins.

## Before you write

- Find the primitive before writing code: `jax.nn.dot_product_attention`, `flax.linen`, `optax`, `orbax`, `grain`, and Google's JAX code (MaxText, tokamax, the `gemma` library). A reimplementation needs a reason a reader can check, written where the code is.
- Read the seam you are about to touch and the tests that cover it. The seams: models are pure Flax modules; objectives own parameters, loss and validation; the trainer owns the mesh, the compiled step, EMA, checkpoints and logging; data sources produce records and transforms are Grain transforms.
- Frozen at 1.0, never changed after it without a migration and a converter test: parameter tree leaf names and shapes, the checkpoint layout, wandb metric keys (`train/*`, `val/*`), the `Objective` methods, the Hugging Face parameter layout of `CausalTransformer`. Before 1.0 they change outright, with no converter.
- Optimizations carry no tradeoffs: no numerics change, no reduced-precision path, nothing observable removed. If a faster version does not match the old one to fp32 tolerance, it is not accepted.

## While you write

- Smallest correct change. No dead parameters, no helper used once, no flag nobody sets, no fallback without a demonstrated need, no change narration in comments or docstrings.
- Ports are reference-identical: same parameter layout, same operation order where numerics depend on it, same dtypes, same defaults, and a parity test against the reference at fp32 with the tolerance and the largest observed difference written in the test. The fixture generator is committed under `tools/`.
- Commit after every completed step, as `Ashish Kumar Singh <ashishkmr472@gmail.com>`, with a plain conventional message. Never push.
- Work in your own worktree when others are active. Announce shared-file edits and GPU use on the hub before, not after. The GPU is idle only when `nvidia-smi --query-compute-apps=process_name --format=csv,noheader` shows only `gnome-remote-desktop-daemon`.

## Tests

- A test is worth keeping only if it would fail on a plausible bug in the thing it names. Assert values, at the seam, through the public interface, with real inputs. No asserting that a function was called or that a shape came back.
- A bug fix ships with the test that fails before and passes after, both runs shown. A new invariant ships with a mutation that breaks it and the assertion that the mutated code fails.
- No stubs that return the value the test then checks. Mock only real external boundaries. No `importorskip` on the code under test.
- Run only the files that cover your seam: `JAX_PLATFORMS=cpu pytest tests/<file> -q -p no:cacheprovider`. The whole suite runs once, at integration.

## Before you report

- Every number ships with the command that produced it and the hardware it ran on. Every claim about behaviour ships with the file and line or the command output.
- Say what you did not do and why. A report that hides a gap costs more than the gap.
- Plain sentences, no em dashes. Read `CONTRIBUTING.md`'s Writing section before you write prose, a docstring, a comment or a commit message. It names the constructions that are banned, including colon reveals ("Measured, not adopted: where the room is"), binary contrasts, importance puffery, trailing -ing clauses that pretend to explain, negative listing, fake-profound endings, and the words that sell. Tables for comparisons.
