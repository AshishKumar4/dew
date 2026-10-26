"""SLOP022: a torch module is never promoted to float64 after its construction, which keeps what its
constructor computed rounded to float32 (ledger 1601); a truth's twin is built inside
tools/torch_precision.py's `float64` (ledger 1619)."""

import textwrap

import pytest
from tool_support import load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


LINEAR = "import torch\nmodel = torch.nn.Linear(4, 4)\n"

PROMOTIONS = {
    "a constructor's module": (LINEAR + "model.double()", 1),
    "a qualified import's": ("import torch\nimport torch.nn as nn\nlayer = nn.Linear(4, 4)\n"
                             "layer.to(torch.float64)", 1),
    "a reader's model, at a keyword dtype": (
        "import torch\nfrom transformers import AutoModelForCausalLM\n"
        "model = AutoModelForCausalLM.from_pretrained(path)\nmodel = model.to(dtype=torch.float64)", 1),
    "a class of the file's": ("import torch\nclass Net(torch.nn.Module):\n    pass\n"
                              "class Deeper(Net):\n    pass\n"
                              "def test():\n    net = Deeper()\n    net.double()", 1),
    "a module alias": (LINEAR + "alias = model\nalias.double()", 1),
    "a module's deep copy": ("import copy\n" + LINEAR + "twin = copy.deepcopy(model)\ntwin.double()", 1),
    "after module-preserving calls": (LINEAR + "model = model.eval().cpu()\n"
                                      "model.to('cpu').to(torch.float64)", 1),
    "a dtype alias": ("import torch\nWIDE = torch.float64\ndef test():\n    model = torch.nn.Linear(4, 4)\n"
                      "    model.to('cpu', WIDE)", 1),
    "by type, by dtype or by its tensor class": (LINEAR + "model.type(torch.double)\n"
                                                 "model.type('torch.DoubleTensor')", 2),
    "the unbound form, of anything": ("import torch\ndef widen(module):\n"
                                      "    return torch.nn.Module.double(module)", 1),
    "a factory of the file's": ("import torch\ndef built():\n    return torch.nn.Linear(4, 4)\n"
                                "def test():\n    built().double()", 1),
    "an attribute chain bound to one": ("import torch\nclass Case:\n    def setup(self):\n"
                                        "        self.model = torch.nn.Linear(4, 4)\n"
                                        "        self.model.double()", 1),
    "a module of the file's scope, in a function": (LINEAR + "def test():\n    model.double()", 1),
    "built outside float64, promoted inside it": (
        "from torch_precision import float64\n" + LINEAR + "with float64():\n    model.double()", 1),
    "a deep copy inside float64": ("import copy\nfrom tools.torch_precision import float64\n" + LINEAR
                                   + "with float64():\n    twin = copy.deepcopy(model)\n    twin.double()", 1),
    "a forward's result": (LINEAR + "logits = model(x)\nlogits.double()\nmodel(x).double()", 0),
    "a tensor's deep copy": ("import copy\nweights = copy.deepcopy(tensor)\nweights.double()", 0),
    "a module's tensor attribute and item": (LINEAR + "model.weight.double()\n"
                                             "layers = torch.nn.ModuleList([model])\nlayers[0].double()", 0),
    "a tensor": ("import torch\ntorch.zeros(3).to(torch.float64)\n"
                 "torch.nn.Parameter(torch.zeros(3)).double()\ntorch.Tensor.double(values)", 0),
    "a module built inside float64": ("from tools.torch_precision import float64\nimport torch\n"
                                      "with float64():\n    model = torch.nn.Linear(4, 4)\n"
                                      "    model.double()\nmodel.to(torch.float64)", 0),
    "a module built at float64": ("import torch\nmodel = torch.nn.Linear(4, 4, dtype=torch.float64)\n"
                                  "model.double()", 0),
    "a factory it is passed": ("def twin(build):\n    model = build()\n    model.double()", 0),
    "a name bound again": (LINEAR + "model = model(x)\nmodel.double()", 0),
    "another dtype, a device, an unknown dtype": (LINEAR + "model.to(torch.float32)\nmodel.to('cuda')\n"
                                                  "model.to(dtype)", 0),
    "a loop's target": (LINEAR + "for model in models:\n    model.double()", 0),
    "the shared twin's own build": (
        "import torch\nfrom tools.torch_precision import float64\ndef float64_twin(model, build=None):\n"
        "    with float64():\n        twin = build() if build is not None else type(model)(model.config)\n"
        "        twin.to(torch.float64)\n    return twin", 0),
}


def _codes(lint, tmp_path, relative, body):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    [module] = lint.collect([relative], tmp_path, "dew")
    return [finding.code for finding in lint.check(module)]


@pytest.mark.parametrize("case", list(PROMOTIONS))
def test_a_module_promoted_to_float64_after_its_construction_is_reported(lint, case, tmp_path):
    body, count = PROMOTIONS[case]
    assert _codes(lint, tmp_path, "tests/test_case.py", body).count("SLOP022") == count


def test_the_library_holds_no_torch_truth_so_the_rule_reads_none_of_it(lint, tmp_path):
    assert "SLOP022" not in _codes(lint, tmp_path, "src/dew/case.py", LINEAR + "model.double()")
