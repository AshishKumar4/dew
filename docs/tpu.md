# Plan and run on Cloud TPUs

A Cloud TPU slice contains accelerator devices attached to one or more worker VMs. For a multi-worker training job, every worker runs the same program and joins the same JAX process pool. Dew's `dew-tpu` command wraps `gcloud`, SSH, and rsync to create a slice, install the environment on its workers, and launch a recipe on all of them.

The resource commands below include `--dry-run` and print command previews. They create no cloud resources and make no worker connections. A successful preview does not verify cloud permissions, capacity, networking, TPU execution, or distributed training.

## Prerequisites and costs

First complete a [local recipe run](recipes.md) and read [distributed training](concepts/distributed.md). Have the following ready before removing a resource command's `--dry-run` flag:

- A Google Cloud project with billing enabled, the Cloud TPU API enabled, and quota for the requested accelerator type in a supported zone. Capacity can still be unavailable when quota exists.
- An authenticated Google Cloud CLI and permissions to create/delete TPUs, use the worker service account, and access data or checkpoint buckets. Diagnose permission errors against the project's IAM policy and grant only the required permissions.
- SSH access to workers and the local `ssh` and `rsync` programs. Source setup and training run from a Git checkout. The current sync command uses the Google Compute Engine SSH key and disables SSH host-key checking; assess that policy before using it on a sensitive network.
- Worker network access to the package repositories used by setup, and passwordless sudo for its package and system-limit changes. A private-network deployment needs its own routing and access configuration; this command does not design that network.
- Dataset files, compatible model/tokenizer files if needed, and a checkpoint location available to every process. Syncing the checkout does not guarantee that ignored datasets or model weights reach the workers.

Review [Cloud TPU pricing](https://cloud.google.com/tpu/pricing), regional quota, and runtime availability for your project. Budget for accelerator time, worker/storage resources, bucket operations, and data transfer. Stopping a process or closing the log viewer does not stop billing. Spot TPUs can be preempted; the current overflow/resume defect means this guide cannot promise exact recovery after an interrupted run. See [checkpoints](guides/checkpoints.md) before choosing spot capacity.

Install Dew using the [installation guide](installation.md). `dew-tpu --help` and `dew-tpu create --help` inspect the local command interface without contacting Google Cloud.

## Configure a preview

Use a separate configuration directory while learning so you do not overwrite existing deployment defaults:

```bash
export DEW_CONFIG_DIR="$(mktemp -d)"
dew-tpu init --project dew-training --zones us-central2-b,europe-west4-a \
    --accelerator-type v5e-16 --runtime-version auto --ssh-user you \
    --gcs-bucket '' --data-disk '' --python-version 3.12 --dry-run
```

`dew-training`, `you`, and `dew-16` in this guide are example names. Replace the project and SSH user with your own values before deployment. `init --dry-run` only prints the TOML; it does not save those defaults. To use the previews below with these example values, save this file as `$DEW_CONFIG_DIR/tpu.toml`:

```toml
project = "dew-training"
zones = ["us-central2-b", "europe-west4-a"]
accelerator_type = "v5e-16"
runtime_version = "auto"
ssh_user = "you"
gcs_bucket = ""
data_disk = ""
python_version = "3.12"
```

For a real configuration, `dew-tpu init` without `--dry-run` writes the file locally; omitted flags prompt when the terminal is interactive. Without `DEW_CONFIG_DIR`, the default location is `~/.config/dew/tpu.toml` (or the XDG config directory).

`zones` defines a search order for locating an existing TPU. Creation uses the explicit `--zone`, or the first configured zone. Do not read the list as automatic capacity retry. Dew caches resolved zones in `zones.json`.

Dry runs have local side effects: creation can update the zone cache, and setup writes its rendered shell script into the configuration directory. They do not create cloud resources. The separate directory keeps these preview files away from your normal configuration.

## Preview creation and setup

```bash
dew-tpu create dew-16 --zone us-central2-b --type v5e-16 --dry-run
dew-tpu setup dew-16 --zone us-central2-b --type v5e-16 \
    --from-source --dry-run
```

Run source setup from the Dew Git checkout. Dew translates `v5e-16` to the API name `v5litepod-16` and predicts two workers for the preview. `runtime_version="auto"` selects a runtime from Dew's generation table. That table is a default, not a live availability check; verify the runtime against the current [Cloud TPU software versions](https://cloud.google.com/tpu/docs/runtimes) before creating the slice. `--version` on **create** overrides the TPU runtime. `--version` on **setup** instead selects the Dew package release.

A real create waits for the TPU's READY state. `--spot` requests interruptible capacity; `--queued` uses the queued-resources API. An optional `--disk NAME` attaches a persistent disk and mounts it at `/mnt/persist`. Confirm disk location, permissions, and data ownership before attaching it.

Setup syncs the checkout for `--from-source`, installs uv, creates `~/dew-venv`, installs `jax[tpu]` and Dew, writes `~/.dew-env`, adjusts file limits, and optionally mounts the configured bucket at `~/gcs_mount`. It also installs system packages using sudo. With no source flag, setup installs the selected package release. Repeating setup can resolve newer packages; pinning Dew does not pin JAX or the whole environment. Archive the resolved package versions with the run.

A real setup finishes by asking each worker for `jax.device_count()` and `jax.local_device_count()`. It compares the global count with the requested slice size. A two-worker v5e-16 preview predicts 16 global devices and 8 local devices per worker; those numbers are not observations of a deployed slice.

## Verify the deployed slice before training

After an authorized real setup, inspect every worker's report. READY means the resource exists, not that your training program can complete a collective operation. A global device count check also does not test checkpoint access, input throughput, or the model's sharding layout.

Preview an all-worker command with:

```bash
dew-tpu run dew-16 --zone us-central2-b --dry-run -- \
    python -c 'import jax; print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count())'
```

`run` starts one SSH command per worker and prefixes each worker's output. It sources `~/.dew-env`. Your program must still initialize distributed JAX before creating device arrays; the training recipes perform that setup. The plain `python -c` command above only inspects what JAX sees in that process and is not a distributed-training test.

For deployment acceptance, run a short recipe with the intended mesh and global batch size, confirm every process joins, observe finite updates, and verify that every process can access the data and checkpoint location. Then exercise the documented restore path with a separate short run. This page's previews provide none of that hardware evidence.

## Supply data and preview a training launch

The [recipe walkthrough](recipes.md) creates `/tmp/dew-first-recipe/tokens`. Copy that directory explicitly:

```bash
dew-tpu copy dew-16 /tmp/dew-first-recipe/tokens '~/dew-tokens' \
    --zone us-central2-b --dry-run
```

`copy` targets every worker by default. For this example SSH user, the remote data path is `/home/you/dew-tokens`; change it if your worker home differs. Return to the Dew checkout, then preview a short launch:

```bash
dew-tpu train dew-16 --zone us-central2-b --job byte-demo --dry-run -- \
    recipes/lm/train.py data:token-windows \
    --data.path /home/you/dew-tokens --data.seq-len 16 \
    --data.loading.workers 0 --data.loading.threads 1 \
    --data.loading.read-buffer 2 --data.val-batches 2 \
    --model.config '{"emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_features": 32}' \
    --trainer.batch-size 32 --trainer.steps 2 --trainer.log-every 1 \
    --trainer.eval-every 2 --trainer.checkpoint-every 2 \
    --trainer.checkpoint-dir /home/you/dew-checkpoints \
    --trainer.name byte-demo --sample-tokens 0
```

The tiny shape is for launch inspection, not a TPU performance recommendation. `train` syncs the checkout and launches the recipe detached on every worker, adding `--trainer.multi-host True`. It then follows worker 0's log. The example's worker-local checkpoint path is not a durable shared-storage design for a multi-host run. Before a real run, choose a checkpoint store accessible to all processes, configure its credentials, and test it. A local directory with the same spelling on two workers is not necessarily the same storage.

Ctrl-C stops the log follower, not the detached training job. The logs live under `~/dew-runs/<job>/worker-<index>.log`. Preview later inspection with:

```bash
dew-tpu logs dew-16 byte-demo --zone us-central2-b --worker all --dry-run
dew-tpu status dew-16 --zone us-central2-b --dry-run
dew-tpu describe dew-16 --zone us-central2-b --dry-run
```

`--follow` follows a log; `--worker N` selects a worker. Read failures from every worker. A successful launch command does not establish that its detached training process completed.

## End the allocation deliberately

List or describe resources before deleting them, and preserve outputs you need. Preview deletion with:

```bash
dew-tpu delete dew-16 --zone us-central2-b --dry-run
```

During real deletion, Dew checks whether a queued resource owns the TPU and deletes that resource when needed. A dry run cannot discover the live ownership state, so its printed deletion path may differ. Verify deletion in your project afterward, including disks and buckets that have their own lifetimes and charges.

`reset` kills processes holding accelerator devices on every worker. It is a forceful recovery operation, not normal job shutdown; it can kill another user's work on a shared slice and lose unsaved state. `stop` retains the TPU's name and disks. Neither command is a substitute for checking the remaining billable resources.

## Other commands and older scripts

| Command | Purpose |
| --- | --- |
| `list` | List TPUs in configured zones. |
| `ssh NAME --worker N -L PORT` | Open a worker shell with an optional local port forward. |
| `sync NAME` | Copy the working tree; `--delete` also removes remote files absent locally. |
| `run NAME --detach -- CMD` | Launch a general command in the background. |
| `spawn BASE N -- CMD` | Create, set up, and launch on N independent TPUs. Each adds cost. |
| `start NAME` / `stop NAME` | Change a TPU's running state without deleting its name and disks. |

Use each command's `--help` before passing destructive options. For users of the old FlaxDiff `tpu_tool.sh`, `execute` becomes `run`, the hand-copied reset script becomes `reset`, and log inspection replaces the tmux sweep dashboard. Dew does not clear `/tmp`, install the old framework pin list, or rewrite a project's DNS configuration. Use your normal secret-management process for credentials; copying a private key into every worker is not part of this setup.
