#!/usr/bin/env python3
"""Prepare the tiny image fixture read by tests/test_tfds_read.py.

Run preparation separately from the Dew training environment:

    uv venv --python 3.13 .venv-tfds-prepare
    uv pip install --python .venv-tfds-prepare/bin/python tensorflow-datasets==4.9.10 tensorflow==2.21.0
    .venv-tfds-prepare/bin/python tools/datasets/tfds_reference.py

This writes twenty constant RGB images and their class labels as ArrayRecords.
It downloads no datasets or weights. TensorFlow encodes the generated NumPy
images during preparation; the tests read the committed records without it.
"""

from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


class DewImages(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self, disable_shuffling=True,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(8, 8, 3), encoding_format="png"),
                "label": tfds.features.ClassLabel(names=["red", "blue"]),
            }))

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples(0, 16),
                "test": self._generate_examples(16, 20)}

    def _generate_examples(self, start, stop):
        for index in range(start, stop):
            yield index, {"image": np.full((8, 8, 3), index + 10, np.uint8),
                          "label": index % 2}


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tfds"
    builder = DewImages(data_dir=str(directory), file_format="array_record")
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(
        try_download_gcs=False))
    print(builder.data_dir)
