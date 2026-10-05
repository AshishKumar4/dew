# Cloud TPUs

A Cloud TPU slice has TPU chips attached to one or more worker VMs. A multi-worker job runs the same program on every worker in one JAX process pool. `dew tpu` uses `gcloud`, SSH and rsync to create the slice, install the environment and start recipes. To run another program on every worker, use `dew launch --tpu NAME` ([Multi-node training](guides/multi-node.md)).

Every resource command below uses `--dry-run`. It prints the planned commands without creating cloud resources or connecting to workers. It does not check permissions, capacity, networking, TPU execution or distributed training.

## Prerequisites

First run a [recipe locally](recipes.md) and read [distributed training](concepts/distributed.md). Before removing `--dry-run`, check these requirements:

- A Google Cloud project with billing and the Cloud TPU API enabled. It needs quota for your accelerator type in a zone that offers it. Quota does not guarantee capacity.
- An authenticated Google Cloud CLI. You need permission to create and delete TPUs, use the worker service account, and access data or checkpoint buckets. For permission errors, check the project's IAM policy and grant only what the job needs.
- SSH access to workers, with `ssh` and `rsync` installed locally. Setup from source and training use a Git checkout. Sync uses the Google Compute Engine SSH key and disables SSH host-key checking. Decide whether that is acceptable on your network before using it.
- Worker network access to the package repositories used by setup. Setup also needs passwordless sudo for package installs and system-limit changes. You must configure private-network routing and access yourself; `dew tpu` does not do this.
- Dataset, model and tokenizer files, plus a checkpoint location reachable by every process. Sync skips files excluded by `.gitignore`, including ignored datasets and model weights. Transfer those separately.

Read [Cloud TPU pricing](https://cloud.google.com/tpu/pricing) and check your project's regional quota and runtime availability. Budget for accelerator time, workers, storage, bucket operations and data transfer. Stopping a process or closing the log viewer does not stop billing.

Spot TPUs can be preempted. To resume, you need a checkpoint on storage shared by every process and a data iterator that saves its position. Read [checkpoints](guides/checkpoints.md) before choosing spot capacity.

Follow the [installation guide](installation.md) to install Dew. `dew tpu --help` and `dew tpu create --help` list options without contacting Google Cloud. Each command's help also lists the configuration defaults and their source file.

## Configuration

Use a separate configuration directory while trying these commands to protect your deployment defaults:

```bash
export DEW_CONFIG_DIR="$(mktemp -d)"
dew tpu init --project dew-training --zones us-central2-b,europe-west4-a \
    --accelerator-type v5e-16 --runtime-version auto --ssh-user you \
    --gcs-bucket '' --data-disk '' --python-version 3.12 --dry-run
```

`dew-training`, `you` and `dew-16` are example names. Before deploying, replace the project and SSH user with your own. `init --dry-run` prints TOML without saving it. For the previews below, save these example values as `$DEW_CONFIG_DIR/tpu.toml`:

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

To save a real configuration, run `dew tpu init` without `--dry-run`. In an interactive terminal, it asks for missing flags. Without `DEW_CONFIG_DIR`, the file is `~/.config/dew/tpu.toml`, or under `$XDG_CONFIG_HOME/dew/` if set.

`zones` sets the search order for existing TPUs. Creation uses `--zone` if supplied, or the first configured zone. It does not retry another zone if capacity is unavailable. Dew caches each TPU's discovered zone in `zones.json`.

Dry runs can still write locally. Creation can update the zone cache; setup writes a generated shell script in the configuration directory. The separate directory keeps these files out of your normal configuration. Dry runs create no cloud resources.

## Creation and setup

```bash
dew tpu create dew-16 --zone us-central2-b --type v5e-16 --dry-run
dew tpu setup dew-16 --zone us-central2-b --type v5e-16 \
    --from-source --dry-run
```

The first command prints:

```text
gcloud compute tpus tpu-vm create dew-16 --zone=us-central2-b --accelerator-type=v5litepod-16 --version=v2-alpha-tpuv5-lite --project=dew-training
dew-16 in us-central2-b: v5litepod-16 on 2 worker(s)
```

Run setup from source inside the Dew Git checkout. Dew translates `v5e-16` to the API name `v5litepod-16`. This preview assumes two workers.

`runtime_version="auto"` chooses a runtime from Dew's table, with one default per TPU generation. The table does not query Google's current offerings. Before creating a slice, compare the choice with [Cloud TPU software versions](https://cloud.google.com/tpu/docs/runtimes). On `create`, `--version` sets the TPU runtime. On `setup`, it chooses the Dew package release.

Without `--dry-run`, `create` waits for the TPU to report READY. `--spot` requests preemptible capacity; `--queued` uses the queued-resources API. `--disk NAME` attaches a persistent disk at `/mnt/persist`. For an existing TPU, use `attach-disk NAME DISK`. With `--read-only`, several workers can share a disk. Check its location, permissions and data ownership before attaching it.

With `--from-source`, setup syncs the checkout, installs uv and creates `~/dew-venv`. It installs the checkout with its `tpu` extra, including libtpu for Dew's pinned jax version. It writes `~/.dew-env`, raises open-file limits and, if configured, mounts a bucket at `~/gcs_mount`. System package installs use sudo.

Without a source flag, setup installs `jax[tpu]` and the chosen Dew release. Running setup again may install newer packages. Pinning Dew alone does not pin JAX or the rest of this environment, so save the resolved package versions with your run.

`--git-key FILE` copies a private key to every worker as `~/.ssh/id_ed25519` and trusts github.com for private-repository clones over ssh. Use a deploy key with access limited to the job's repositories.

After setup on a real slice, Dew queries `jax.device_count()` and `jax.local_device_count()` on each worker. It compares the global count with the requested slice size. The two-worker v5e-16 preview expects 16 global devices and 8 local devices per worker. These counts are expectations; no slice was deployed for the preview.

## Checking a deployed slice

After setup on a real slice, read every worker's report. READY confirms that the resource exists. It does not test collective operations, checkpoint access, input throughput or model sharding. The device-count check does not test these either.

To preview a command on all workers:

```bash
dew launch --tpu dew-16 --zone us-central2-b --dry-run -- \
    python -c 'import jax; print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count())'
```

`dew launch --tpu` starts one SSH command per worker. Each command sources `~/.dew-env` first. Output lines include their rank. If one worker fails, the launcher stops the others. [Multi-node training](guides/multi-node.md) covers `--cwd` and multislice runs.

Your program must initialize distributed JAX before creating device arrays, as the training recipes do. The `python -c` command above only reports devices visible to each process. It does not test distributed training.

Before a full job, run a short recipe with your planned mesh and global batch size. Check that every process joins, produces finite updates, reads the data and writes to the checkpoint location. Restore that checkpoint in a second short run. The previews on this page have not tested this on real hardware.

## Data and training jobs

The [recipe walkthrough](recipes.md) creates `/tmp/dew-first-recipe/tokens`. Copy that directory to the workers yourself:

```bash
dew tpu copy dew-16 /tmp/dew-first-recipe/tokens '~/dew-tokens' \
    --zone us-central2-b --dry-run
```

`copy` sends to every worker unless you select one with `--worker`. For the example SSH user, the destination is `/home/you/dew-tokens`. Adjust that path to your worker home directory. From the Dew checkout, preview a short launch:

```bash
dew tpu train dew-16 --zone us-central2-b --job byte-demo --dry-run -- \
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

The tiny model makes the launch easy to inspect. These settings are not for measuring TPU performance. `train` syncs the checkout and adds `--trainer.multi-host True`. It starts the recipe in the background on every worker, then follows worker 0's log.

This example writes checkpoints to each worker's local disk. The same path on two workers usually names two different disks, not durable shared storage. Before a real run, choose a checkpoint store reachable by every process, configure credentials and test access.

Ctrl-C closes the log viewer but leaves training running. Each worker writes to `~/dew-runs/<job>/worker-<index>.log`. To preview later checks:

```bash
dew tpu logs dew-16 byte-demo --zone us-central2-b --worker all --dry-run
dew tpu status dew-16 --zone us-central2-b --dry-run
dew tpu describe dew-16 --zone us-central2-b --dry-run
```

`logs` reads worker 0 by default. Select another with `--worker N` or every worker with `--worker all`. `--follow` prints new lines as they arrive. After a failure, read every worker's log. A successful launch confirms only that the job started.

## Deleting and resetting

List or describe your resources before you delete them, and save any outputs you need. To preview deletion:

```bash
dew tpu delete dew-16 --zone us-central2-b --dry-run
```

Without `--dry-run`, `delete` checks whether a queued resource owns the TPU and deletes that resource if so. A dry run cannot query ownership, so its printed command may differ from the real deletion. After deleting, confirm the result in your project. Check disks and buckets separately; they have their own lifetimes and charges.

`reset` kills every process using accelerator devices on every worker. Reserve it for stuck slices. On a shared slice it can kill someone else's job and lose unsaved work. `stop` keeps the TPU's name and disks. After either command, check which resources are still being billed.

## Command reference

| Command | Purpose |
| --- | --- |
| `init` | Write `tpu.toml` from flags, asking for what is missing in an interactive terminal. |
| `create NAME` | Create a TPU VM or pod slice and wait until it is READY. `--spot`, `--queued`, `--disk`. |
| `setup NAME` | Install uv, a virtualenv, JAX and Dew on every worker, then count the devices. `--from-source`, `--git-key`, `--gcs-bucket`. |
| `sync NAME` | Copy the git working tree to `~/<repo>` on every worker; `--delete` also removes remote files absent locally. |
| `copy NAME SRC DEST` | Copy a local file or directory to every worker, or one with `--worker N`. |
| `train NAME -- RECIPE ...` | Sync the tree and start a recipe on every worker in the background, then follow worker 0's log. |
| `logs NAME JOB` | Show a detached job's log; `--worker N` or `all`, `--follow`. |
| `status NAME` | Show what each worker is doing. |
| `run NAME -- CMD` | Run a shell command on every worker, or one with `--worker N`; `--detach` runs it in the background. |
| `list` | List TPUs in the configured zones, with worker 0's external IP. |
| `describe NAME` | Show what a TPU is and where its workers are. |
| `ssh NAME --worker N -L PORT` | Open a worker shell with an optional local port forward. |
| `ssh-config NAME` | Write `~/.ssh/config` entries: `NAME` for worker 0, `NAME-worker-N` for the others. `delete` removes them. |
| `attach-disk NAME DISK` | Attach a persistent disk to an existing TPU and mount it at `/mnt/persist` on every worker. |
| `start NAME` / `stop NAME` | Change a TPU's running state without deleting its name and disks. |
| `reset NAME` | Kill whatever holds the accelerators on every worker. |
| `delete NAME` | Delete a TPU, and the queued resource that holds it. |
| `spawn BASE N -- CMD` | Create, set up and launch on N independent TPUs. Each adds cost. |

Read each command's `--help` before passing an option that deletes or kills something.

### tpu_tool.sh equivalents

If you used `tpu_tool.sh`, the names `create`, `delete`, `start`, `stop`, `list`, `ssh`, `copy` and `setup` are unchanged. Use `run` for `execute`, `ssh-config` for `update-ssh-config`, `setup --git-key` for `copy-github-key`, and `setup --gcs-bucket` for `--mount-gcs`. `reset` has the reset script's behavior. Every command still searches the configured zones.

When workers need the credentials in `~/.netrc`, transfer it with `copy`. `spawn` labels each TPU's progress by name and ends with a table. It no longer opens a tmux dashboard; run it in tmux to detach. Dew leaves `/tmp` and project DNS configuration alone. It does not install the old framework's conda environment or pinned package list.
