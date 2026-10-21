"""Train autoregressive, masked-diffusion or block-diffusion models on token stores.

The data is a token store `dew data export` wrote, whose manifest names its
tokenizer and vocabulary, so the run takes the vocabulary from the data, not
the command line (`dew.objectives.lm.LMRunConfig`). Its windows are a chain
(`dew.data.presets.lm`), whose settings `--set data.<call>.<setting>=` changes:
the store's as `data.source`, the windows' as `data.lm`.

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew data export data/shakespeare.txt data/shakespeare-byte \\
        --format array_record --tokenizer byte --eos --read memory
    python recipes/lm/train.py --set data.source.url=data/shakespeare-byte \\
        --set data.lm.seq_len=256 --set data.lm.batch=32 --trainer.epochs 10 \\
        --model.emb-features 384 --model.num-layers 6 --model.num-heads 6

`--set data.lm.pack=first-fit` packs whole documents into the windows instead.
"""

from dew.data import presets
from dew.data.sources.tokens import TokenFiles
from dew.objectives.lm import LMRunConfig

WINDOWS = TokenFiles("data/shakespeare-byte").pipe(presets.lm, None, seq_len=256, batch=32, pack=None)

if __name__ == "__main__":
    LMRunConfig.cli(default=LMRunConfig(data=WINDOWS)).run()
