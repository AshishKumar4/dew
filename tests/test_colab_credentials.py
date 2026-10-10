"""Launch credentials cross the fake CLI through private files, never receipts."""

import hashlib
import json
import sys

import pytest

from tools.colab import colab_batch, credentials
from tools.colab.colab_batch_vm import environment as vm_environment, join_parts
from tools.colab.common import read, save
from tools.colab.ops_driver import Driver


@pytest.mark.parametrize("failed", [False, True])
def test_wandb_key_never_appears_in_logs_results_or_manifests(tmp_path, capsys, failed):
    key = "fixture-wandb-private-key"
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("machine api.wandb.ai login fixture-user password " + key + "\n")
    config = tmp_path / "wandb.toml"
    config.write_text('entity="fixture-team"\nproject="fixture-project"\n')
    work = tmp_path / "staging"
    work.mkdir(mode=0o700)
    job = {"wandb": True}
    plan = {"uploads": {}, "jobs": [job, {"wandb": False}]}
    credentials.stage_wandb(plan, work, netrc_file, config)
    save(work / "plan.json", plan)
    fake_cli = tmp_path / "colab"
    fake_cli.write_text(
        f"#!{sys.executable}\nimport pathlib,sys,shutil\n"
        "payload=pathlib.Path(sys.argv[4]).read_bytes();print(payload.decode())\n"
        + ("sys.exit(1)\n" if failed else "shutil.copyfile(sys.argv[4],sys.argv[5])\n")
    )
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
        job["secret_env"]["WANDB_API_KEY"] = str(tmp_path / "vm-key")
        env = credentials.environment(job)
        assert env == {
            "WANDB_API_KEY": key,
            "WANDB_ENTITY": "fixture-team",
            "WANDB_PROJECT": "fixture-project",
        }
        assert credentials.environment(plan["jobs"][1]) == {}
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
    with pytest.raises(ValueError, match=r"api\.wandb\.ai"):
        credentials.stage_wandb(
            {"uploads": {}, "jobs": [{"wandb": True}]}, tmp_path, netrc_file, tmp_path / "missing-config"
        )
    assert not (tmp_path / "private-wandb.part").exists()


def test_vm_credential_path_cannot_be_a_symlink(tmp_path):
    real = tmp_path / "key"
    real.write_text("fixture")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="private environment"):
        credentials.environment({"secret_env": {"WANDB_API_KEY": str(link)}})


def test_non_opted_jobs_do_not_read_or_require_local_credentials(tmp_path):
    plan = {"uploads": {}, "jobs": [{"wandb": False}, {}]}
    credentials.stage_wandb(plan, tmp_path, tmp_path / "absent-netrc", tmp_path / "absent-config")
    assert plan["uploads"] == {} and list(tmp_path.iterdir()) == []
    assert all(credentials.environment(job) == {} for job in plan["jobs"])


def test_parent_credentials_do_not_reach_install_preamble_or_default_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "parent-key")
    monkeypatch.setenv("WANDB_ENTITY", "parent-team")
    monkeypatch.setenv("WANDB_PROJECT", "parent-project")
    env = vm_environment(tmp_path / "venv")
    assert all(key not in env for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"))


def test_opted_job_requires_entity_and_project_from_local_config(tmp_path):
    netrc_file, config = tmp_path / "netrc", tmp_path / "wandb.toml"
    netrc_file.write_text("machine api.wandb.ai login fixture password fixture-key\n")
    config.write_text('entity="fixture-team"\n')
    with pytest.raises(ValueError, match="entity and project"):
        credentials.stage_wandb({"uploads": {}, "jobs": [{"wandb": True}]}, tmp_path, netrc_file, config)
    assert not (tmp_path / "private-wandb.part").exists()


def test_opt_in_change_is_part_of_reuse_identity():
    job = {
        "commit": "sha",
        "command": "run",
        "outputs": [],
        "artifact_policy": "all",
        "ref": "sha",
        "compile_cache": "off",
        "wandb": False,
    }
    first = {"driver_sha": "driver", "base": "sha", "hardware": "gpu:T4", "jobs": [job]}
    second = {**first, "jobs": [{**job, "wandb": True}]}
    with pytest.raises(ValueError, match="identity"):
        colab_batch.validate_reuse(first, second)


def test_private_snapshots_cannot_reuse_rotated_credentials(tmp_path):
    save(tmp_path / "plan.json", {"uploads": {"private": {"private": True}}})
    with pytest.raises(ValueError, match="fresh staging"):
        colab_batch.staged({}, tmp_path)


def test_owner_only_input_files_get_private_transport(tmp_path):
    source = tmp_path / "credential"
    source.write_text("fixture-input-key")
    source.chmod(0o600)
    work = tmp_path / "work"
    work.mkdir()
    plan = {
        "uploads": {},
        "jobs": [
            {
                "index": 1,
                "artifact_policy": "measurements",
                "resume_key": "none",
                "entry": {"fields": {"inputs": f"{source} to /content/private", "command": "run"}},
            }
        ],
    }
    colab_batch.stage_inputs(plan, work)
    assert plan["uploads"]["/content/private/credential"]["private"]
    assert plan["private_parts"] == plan["uploads"]["/content/private/credential"]["parts"]
