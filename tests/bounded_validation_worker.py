"""Workers for a pool whose held-out split holds a token past its model's table on one rank's share: each
process a rank of a fit that scores it every step.

dew is imported inside the modes, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

from multiprocess_worker import token_data, token_trainer
from process_support import serve_mode, worker_parser


def bounded_validation(args) -> dict:
    """The refusal this rank meets, by its type, every rank refusing together; None where it trained."""
    from dew.data.dataset import Loading

    trainer = token_trainer(args.name, args.out.parent, fsdp=1, seq_len=args.seq_len, checkpoints=None)
    data = token_data(args, packed=False, loading=Loading(workers=0, threads=1, read_buffer=1))
    try:
        trainer.fit(data, steps=2, eval_every=1, log_every=1)
    except Exception as error:  # the refusal itself is the report
        return {"refused": type(error).__name__}
    return {"refused": None}


MODES = {"bounded_validation": bounded_validation}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--name", default="worker")
    parser.add_argument("--tokens", help="the committed token store the run trains on")
    parser.add_argument("--held-tokens", help="the committed token store it validates on")
    parser.add_argument("--seq-len", type=int, default=4)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
