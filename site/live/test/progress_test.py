"""Offline model revisions report stale pages without weakening the sandbox."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.errors import OfflineModeIsEnabled

spec = importlib.util.spec_from_file_location("progress", Path(__file__).parents[1] / "container/progress.py")
progress = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = progress
spec.loader.exec_module(progress)


class ReportingModelsTest(unittest.TestCase):
    def test_stale_revision_requests_reload(self):
        load = Mock(side_effect=[object(), OfflineModeIsEnabled("offline")])
        models = progress.ReportingModels(load, lambda value: value)
        with patch.object(progress, "_show"):
            models("model", revision="new")
            with self.assertRaisesRegex(ValueError, "This page was updated.*Reload"):
                models("model", revision="old")
        self.assertNotIn(("model", "old"), models.loaded)

    def test_other_models_keep_the_original_error(self):
        error = OfflineModeIsEnabled("offline")
        models = progress.ReportingModels(Mock(side_effect=error), lambda value: value)
        with patch.object(progress, "_show"), self.assertRaises(OfflineModeIsEnabled) as caught:
            models("other", revision="missing")
        self.assertIs(caught.exception, error)

    def test_cached_revision_does_not_reload(self):
        load = Mock(return_value=object())
        models = progress.ReportingModels(load, lambda value: value)
        with patch.object(progress, "_show"):
            first = models("model", revision="new")
            self.assertIs(models("model", revision="new"), first)
        load.assert_called_once_with("model", revision="new")


if __name__ == "__main__":
    unittest.main()
