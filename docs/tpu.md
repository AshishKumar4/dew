# Plan and run on Cloud TPUs

> An AI assistant maintains this document. It is presented as-is.

A Cloud TPU slice is a set of accelerator devices attached to one or more worker VMs. In a multi-worker training job, every worker runs the same program, and all of them join one JAX process pool. Dew's `dew-tpu` command wraps `gcloud`, SSH, and rsync. It creates a slice, installs the environment on each worker, and starts a recipe on all of them.

Every resource command on this page carries `--dry-run`, which prints the commands `dew-tpu` would run. A dry run creates no cloud resources and connects to no workers. A dry run that succeeds does not check your cloud permissions, capacity, networking, TPU execution, or distributed training.

## Prerequisites and costs

First do a [local recipe run](recipes.md) and read [distributed training](concepts/distributed.md). Before you remove `--dry-run` from a resource command, have the following ready:

- A Google Cloud project with billing enabled, the Cloud TPU API enabled, and quota for the accelerator type you want in a zone that offers it. Even with quota, capacity can be unavailable.
- An authenticated Google Cloud CLI, and permission to create and delete TPUs, to use the worker service account, and to reach your data or checkpoint buckets. If you get permission errors, check them against the project's IAM policy and grant only what the job needs.
- SSH access to the workers, and `ssh` and `rsync` on your machine. Setup from source and training both run from a Git checkout. The sync step uses the Google Compute Engine SSH key and turns off SSH host-key checking. Decide whether that is acceptable before you use it on a sensitive network.
- Network access from the workers to the package repositories setup installs from, and passwordless sudo for its package installs and system-limit changes. A private-network deployment needs its own routing and access setup; `dew-tpu` does not configure the network for you.
- Your dataset files, any model or tokenizer files you need, and a checkpoint location that every process can reach. Sync skips everything that `.gitignore` excludes, so ignored datasets and model weights do not reach the workers that way.

Read [Cloud TPU pricing](https://cloud.google.com/tpu/pricing), and check regional quota and runtime availability for your project. Budget for accelerator time, worker and storage resources, bucket operations, and data transfer. Billing continues after you stop a process or close the log viewer. Spot TPUs can be preempted. Resuming after a preemption needs a checkpoint on storage that every process can reach and a data iterator that saves its position; read [checkpoints](guides/checkpoints.md) before you choose spot capacity.

Install Dew by following the [installation guide](installation.md). `dew-tpu --help` and `dew-tpu create --help` show the command's options without contacting Google Cloud.

## Configure a preview

While you are learning, use a separate configuration directory so you do not overwrite your real deployment defaults:

```bash
export DEW_CONFIG_DIR="$(mktemp -d)"
dew-tpu init --project dew-training --zones us-central2-b,europe-west4-a \
    --accelerator-type v5e-16 --runtime-version auto --ssh-user you \
    --gcs-bucket '' --data-disk '' --python-version 3.12 --dry-run
```

`dew-training`, `you`, and `dew-16` are example names. Replace the project and the SSH user with your own before you deploy anything. `init --dry-run` prints the TOML and does not save it. To use the previews below with these example values, save this file as `$DEW_CONFIG_DIR/tpu.toml`:

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

For a real configuration, run `dew-tpu init` without `--dry-run` and it writes the file for you. In an interactive terminal it asks for any flag you left out. Without `DEW_CONFIG_DIR`, the file lives at `~/.config/dew/tpu.toml`, or under `$XDG_CONFIG_HOME/dew/` when that variable is set.

`zones` is the order in which `dew-tpu` searches for an existing TPU. Creation uses `--zone` if you give it, and otherwise the first zone in the list. It does not retry in the next zone when one has no capacity. Dew caches the zone it finds for each TPU in `zones.json`.

Dry runs do write local files. Creation can update the zone cache, and setup writes its generated shell script into the configuration directory. The separate directory keeps these files out of your normal configuration. No cloud resources are created.

## Preview creation and setup

```bash
dew-tpu create dew-16 --zone us-central2-b --type v5e-16 --dry-run
dew-tpu setup dew-16 --zone us-central2-b --type v5e-16 \
    --from-source --dry-run
```

Run setup from source inside the Dew Git checkout. Dew turns `v5e-16` into the API name `v5litepod-16`, and the preview assumes two workers. `runtime_version="auto"` picks a runtime from a table in Dew, one entry per TPU generation. That table is a default and does not check what Google offers today. Compare its choice with the current [Cloud TPU software versions](https://cloud.google.com/tpu/docs/runtimes) before you create the slice. `--version` means two different things: on `create` it sets the TPU runtime, and on `setup` it picks the Dew package release.

A real `create` waits until the TPU reports READY. `--spot` asks for capacity that can be preempted, and `--queued` goes through the queued-resources API. `--disk NAME` attaches a persistent disk and mounts it at `/mnt/persist`. Check the disk's location, permissions, and data ownership before you attach it.

With `--from-source`, setup first syncs the checkout. It then installs uv, creates `~/dew-venv`, installs `jax[tpu]` and Dew, writes `~/.dew-env`, raises the open-file limits, and mounts the configured bucket at `~/gcs_mount` if you set one. It also installs system packages with sudo. Without a source flag, setup installs the Dew release you chose. Running setup again can pick up newer packages, and pinning Dew does not pin JAX or anything else in the environment. Save the resolved package versions with your run.

A real setup ends by asking each worker for `jax.device_count()` and `jax.local_device_count()`, and it compares the global count with the slice size you asked for. For a two-worker v5e-16, the preview expects 16 global devices and 8 local devices per worker. These are expected values, not counts from a deployed slice.

## Verify the deployed slice before training

After a real setup, read the report from every worker. READY means the resource exists. It does not mean your training program can finish a collective operation. The device-count check does not test checkpoint access, input throughput, or your model's sharding layout either.

To preview a command on all workers:

```bash
dew-tpu run dew-16 --zone us-central2-b --dry-run -- \
    python -c 'import jax; print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count())'
```

`run` starts one SSH command per worker, sources `~/.dew-env` first, and prefixes each line of output with its worker. Your program still has to initialize distributed JAX before it creates device arrays; the training recipes do that for you. The `python -c` command above only shows what JAX sees in that one process. It does not test distributed training.

Before you trust a deployment, run a short recipe with the mesh and global batch size you plan to use. Check that every process joins, that updates are finite, and that every process can read the data and write to the checkpoint location. Then restore from that checkpoint in a second short run. None of the previews on this page do any of that on real hardware.

## Supply data and preview a training launch

The [recipe walkthrough](recipes.md) creates `/tmp/dew-first-recipe/tokens`. Copy that directory to the workers yourself:

```bash
dew-tpu copy dew-16 /tmp/dew-first-recipe/tokens '~/dew-tokens' \
    --zone us-central2-b --dry-run
```

`copy` sends to every worker unless you pass `--worker`. For the example SSH user, the data lands at `/home/you/dew-tokens`; change that path if your home directory on the workers is different. Go back to the Dew checkout and preview a short launch:

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

The tiny model is there so you can inspect the launch. It is not a TPU performance setting. `train` syncs the checkout, adds `--trainer.multi-host True`, and starts the recipe in the background on every worker. It then follows worker 0's log. In this example each worker writes checkpoints to its own local disk, which is not durable shared storage for a multi-host run. Before a real run, pick a checkpoint store that every process can reach, set up its credentials, and test it. The same local path on two workers is usually two different disks.

Ctrl-C stops following the log. The training job keeps running. The logs are in `~/dew-runs/<job>/worker-<index>.log` on each worker. To preview the commands for checking on it later:

```bash
dew-tpu logs dew-16 byte-demo --zone us-central2-b --worker all --dry-run
dew-tpu status dew-16 --zone us-central2-b --dry-run
dew-tpu describe dew-16 --zone us-central2-b --dry-run
```

`logs` reads worker 0 unless you pass `--worker N` or `--worker all`, and `--follow` keeps printing new lines. Read the logs from every worker when something fails. A launch command that succeeds only means the job started, not that it finished.

## End the allocation deliberately

List or describe your resources before you delete them, and save any outputs you need. To preview deletion:

```bash
dew-tpu delete dew-16 --zone us-central2-b --dry-run
```

A real `delete` checks whether a queued resource owns the TPU, and if so deletes the queued resource. A dry run cannot look up that ownership, so the deletion command it prints can differ from the real one. Afterward, check your project to confirm the deletion, including disks and buckets, which have their own lifetimes and charges.

`reset` kills every process that holds the accelerator devices on every worker. Use it to recover a stuck slice, not to stop a job normally. On a shared slice it can kill someone else's work, and anything unsaved is lost. `stop` keeps the TPU's name and disks. After either command, check what you are still being billed for.

## Other commands and older scripts

| Command | Purpose |
| --- | --- |
| `list` | List TPUs in configured zones. |
| `ssh NAME --worker N -L PORT` | Open a worker shell with an optional local port forward. |
| `sync NAME` | Copy the working tree; `--delete` also removes remote files absent locally. |
| `run NAME --detach -- CMD` | Launch a general command in the background. |
| `spawn BASE N -- CMD` | Create, set up, and launch on N independent TPUs. Each adds cost. |
| `start NAME` / `stop NAME` | Change a TPU's running state without deleting its name and disks. |

Read each command's `--help` before you pass an option that deletes or kills something. If you used the old FlaxDiff `tpu_tool.sh`, `run` does what `execute` did, `reset` does what the hand-copied reset script did, and `logs` takes the place of the tmux sweep dashboard. Dew does not clear `/tmp`, does not install the old framework's pinned package list, and does not rewrite the project's DNS configuration. Handle credentials with your usual secret-management process; this setup never copies a private key onto the workers.
