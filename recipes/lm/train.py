"""Train autoregressive, masked-diffusion or block-diffusion models on token stores.

The data is a token store `dew data export` wrote, whose manifest names its
tokenizer and vocabulary, so the run takes the vocabulary from the data, not
the command line (`dew.config.lm.LMRunConfig`). Its windows are a chain
(`dew.data.presets.token_windows`), whose settings `--set data.<call>.<setting>=`
changes: the store's as `data.source`, the windows' as `data.token_windows`.

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew data export data/shakespeare.txt data/shakespeare-byte \\
        --format array_record --tokenizer byte --eos --read memory
    python recipes/lm/train.py --set data.source.url=data/shakespeare-byte \\
        --set data.token_windows.seq_len=256 --set data.token_windows.batch=32 --trainer.epochs 10 \\
        --model.emb-features 384 --model.num-layers 6 --model.num-heads 6

Whole documents packed into the windows are a run file's chain: the store packed in its own order
(`presets.lm(..., pack="first-fit", batch=None)`), then shuffled, repeated and batched
(docs/recipes.md).
"""

from dew.config.lm import LMRunConfig
from dew.data import presets
from dew.data.sources.tokens import TokenFiles

# The windows are read by index and ordered by the shuffle and repeat written before them, so a run
# knows its epoch's length.
WINDOWS = TokenFiles("data/shakespeare-byte").shuffle(seed=0).repeat().pipe(
    presets.token_windows, seq_len=256, batch=32)

if __name__ == "__main__":
    LMRunConfig.cli(default=LMRunConfig(data=WINDOWS)).run()
