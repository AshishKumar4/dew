"""Train autoregressive, masked-diffusion or block-diffusion models on token stores.

The data is a token store `dew data export` wrote, whose manifest names its
tokenizer and vocabulary, so the run takes the vocabulary from the data, not
the command line (`dew.objectives.lm.LMRunConfig`).

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew data export data/shakespeare.txt data/shakespeare-byte \\
        --format array_record --tokenizer byte --eos --read memory
    python recipes/lm/train.py data:text --data.source data/shakespeare-byte \\
        --data.seq-len 256 --trainer.batch-size 32 --trainer.epochs 10 \\
        --model.emb-features 384 --model.num-layers 6 --model.num-heads 6

`--data.pack` packs whole documents into the windows instead.
"""

from dew.objectives.lm import LMRunConfig

if __name__ == "__main__":
    LMRunConfig.cli().run()
