"""Launch credentials cross the fake CLI through private files, never receipts."""
import hashlib
import json
import sys

import pytest

from tools.colab import credentials
from tools.colab.colab_batch_vm import join_parts
from tools.colab.common import read, save
from tools.colab.ops_driver import Driver


@pytest.mark.parametrize("failed", [False, True])
def test_wandb_key_never_appears_in_logs_results_or_manifests(tmp_path, capsys, failed):
    key = "fixture-wandb-private-key"
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("machine api.wandb.ai login ashishkumar4 password " + key + "\n")
    work = tmp_path / "staging"
    work.mkdir(mode=0o700)
    plan = {"uploads": {}}
    credentials.stage_wandb(plan, work, netrc_file)
    save(work / "plan.json", plan)
    fake_cli = tmp_path / "colab"
    fake_cli.write_text(f"#!{sys.executable}\nimport pathlib,sys,shutil\n"
                        "payload=pathlib.Path(sys.argv[4]).read_bytes();print(payload.decode())\n"
                        + ("sys.exit(1)\n" if failed else "shutil.copyfile(sys.argv[4],sys.argv[5])\n"))
    fake_cli.chmod(0o700)
    driver = Driver(tmp_path / "attempt", None, str(fake_cli))
    upload = plan["uploads"]["/content/.dew-wandb.key"]
    part = work / upload["parts"][0]
    assert part.stat().st_mode & 0o777 == 0o600
    if failed:
        with pytest.raises(RuntimeError, match="output withheld") as failure:
            driver.upload("fake", part, str(tmp_path / "vm-part"), private=True)
        assert key not in str(failure.value)
    else:
        driver.upload("fake", part, str(tmp_path / "vm-part"), private=True)
        join_parts(tmp_path, ["vm-part"], tmp_path / "vm-key", hashlib.sha256(key.encode()).hexdigest())
        plan["secret_env"]["WANDB_API_KEY"] = str(tmp_path / "vm-key")
        env = credentials.environment(plan)
        assert env == {"WANDB_API_KEY": key, "WANDB_ENTITY": "ashishkumar4", "WANDB_PROJECT": "dew"}
        assert (tmp_path / "vm-key").stat().st_mode & 0o777 == 0o600
    save(tmp_path / "result.json", {"exit": int(failed), "credential_status": "private file"})
    assert key not in json.dumps(plan)
    assert key not in json.dumps(read(tmp_path / "attempt/driver.json"))
    assert key not in (tmp_path / "result.json").read_text()
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err


def test_missing_netrc_entry_fails_without_partial_secret_file(tmp_path):
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("machine unrelated login user password irrelevant\n")
    with pytest.raises(ValueError, match="api.wandb.ai"):
        credentials.stage_wandb({"uploads": {}}, tmp_path, netrc_file)
    assert not (tmp_path / "private-wandb.part").exists()


def test_vm_credential_path_cannot_be_a_symlink(tmp_path):
    real = tmp_path / "key"
    real.write_text("fixture")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="private environment"):
        credentials.environment({"secret_env": {"WANDB_API_KEY": str(link)}})
