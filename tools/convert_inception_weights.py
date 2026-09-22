"""Convert the jax-fid FID weights into the extractor's safetensors tree.

    python tools/convert_inception_weights.py --out inception_v3_fid.safetensors

Without --pickle the pinned upstream checkpoint is downloaded and verified
first. The conversion itself is `dew.interop.inception_fid`, the one place
that opens a pickle; `fid(weights=...)` reads what this writes.
"""

import argparse
import hashlib

from dew.interop.inception_fid import (
    FID_WEIGHTS_DIGEST,
    FID_WEIGHTS_FILE,
    FID_WEIGHTS_REPO,
    FID_WEIGHTS_REVISION,
    convert,
    fetch,
    save,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", default=None,
                        help=f"the jax-fid pickle; default downloads {FID_WEIGHTS_REPO}")
    parser.add_argument("--out", required=True, help="the safetensors file to write")
    parser.add_argument("--channel-divisor", type=int, default=1,
                        help="the width the pickle holds, recorded in the header; "
                             "1 is the published network")
    args = parser.parse_args()

    source = args.pickle or fetch(FID_WEIGHTS_REPO, FID_WEIGHTS_FILE,
                                  FID_WEIGHTS_REVISION, FID_WEIGHTS_DIGEST)
    save(convert(source), args.out, args.channel_divisor)
    with open(args.out, "rb") as handle:
        print(f"{args.out} {hashlib.file_digest(handle, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
