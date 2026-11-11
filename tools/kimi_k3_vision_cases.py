"""Kimi K3's vision recording's items, shared by its worker (tools/kimi_k3_vision_worker.py, in the release's
environment) and its test (tests/test_kimi_k3_vision.py), with no framework of either."""

GRIDS = {"image": (1, 4, 6), "video": (2, 4, 4)}
"""Each drawn item's `(t, h, w)` grid in 14-pixel patches, in the order the draws and the recording take
them: an image of six merge blocks, its table resampled from the fixture's 4x4, and a two-frame video, its
frames pooled into four (`sd2_tpool`)."""
