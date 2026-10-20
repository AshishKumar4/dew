# Live kernels

Production (`dewml-live`) and the preview (`dewml-live-preview`) each bind their own Durable Objects: the pool, the snapshot registry, the preparer and the model hosts, so a preview's warm and its visitors never reach production's (ledger 592; until then the preview called production's by `script_name`, and production's preparer, pinned to its own commit, refused the preview's).

The preview keeps no model host warm while no one asks (`MIN_HOSTS` 0; production keeps 2). A warm asks its pool for one for `WARM_SECONDS` (`ModelPool.want`): the operator's, and a visitor's request that finds no warm host. So a preview deploy warms its pool before its cells are checked:

    (cd site && node live/deploy.mjs <commit> -- --env preview --secrets-file <preview secrets>)   # POST /v1/operator/warm
    gh workflow run site.yml -f run_cells=https://preview.dewml.dev

and a host nobody uses retires ten minutes after its want ends.
