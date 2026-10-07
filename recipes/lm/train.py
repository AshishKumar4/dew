"""Train autoregressive, masked-diffusion or block-diffusion models on token files.

The data is not images but the `train.bin` / `val.bin` / `meta.json` a
tokenizer run wrote, so the run takes the vocabulary from the data, not the
command line (`dew.objectives.lm.LMRunConfig`).

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew tokenize --input data/shakespeare.txt \\
        --out data/shakespeare-byte --tokenizer byte
    python recipes/lm/train.py --data.path data/shakespeare-byte \\
        --data.seq-len 256 --trainer.batch-size 32 --trainer.epochs 10 \\
        --model.emb-features 384 --model.num-layers 6 --model.num-heads 6

`--data.pack` packs whole documents into the windows instead.
"""

from dew.objectives.lm import LMRunConfig

if __name__ == "__main__":
    LMRunConfig.cli().run()
