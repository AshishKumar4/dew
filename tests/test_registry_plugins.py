"""Packages outside Dew register into its tables through the `dew.plugins` entry-point group."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install(site: Path, package: str, source: str, *, entry: str | None = None) -> None:
    """Lay `package` out under `site` as an installed distribution that names itself a Dew plugin."""
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text("")
    (site / package / "models.py").write_text(source)
    info = site / f"{package}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[dew.plugins]\n{package} = {entry or package}\n")


def _run(site: Path, program: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(ROOT / "src"), str(site)])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=300)


PLUGIN_MODEL = '''
import flax.linen as nn

from dew.registry import models


@models("plugin_mlp")
class PluginMLP(nn.Module):
    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
'''


def test_a_plugin_model_is_found_by_name_without_importing_the_plugin(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "import sys\n"
                          "from dew.registry import models\n"
                          "assert 'toyplugin' not in sys.modules\n"
                          "built = models.build('plugin_mlp', {'features': 7})\n"
                          "print(type(built).__module__, built.features)\n"
                          "print(models.name_of(type(built)))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["toyplugin.models 7", "plugin_mlp"]


def test_an_unknown_name_still_raises_with_plugins_installed(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "from dew.registry import models\n"
                          "try:\n"
                          "    models['no_such_model']\n"
                          "except KeyError as error:\n"
                          "    print('refused', 'plugin_mlp' in str(error))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "refused True"


def test_a_plugin_entry_naming_a_missing_package_raises(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL, entry="not_installed_anywhere")
    done = _run(tmp_path, "from dew.registry import models\n"
                          "models['plugin_mlp']\n")
    assert done.returncode != 0
    assert "not_installed_anywhere" in done.stderr and "dew.plugins" in done.stderr
