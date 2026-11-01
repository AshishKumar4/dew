"""Who runs a guest's code, and how many guests one host holds: the facts the bridge (shared_bridge.py),
the gateway's kernel manager (gateway_manager.py), the model service (model_service.py) and start-gateway.sh
share."""

import os

GUEST_UIDS = range(6100, 6200)
"""Each guest context's user id: user ctx<N> has id GUEST_UIDS.start + N (start-gateway.sh creates them)."""

MAX_CONTEXTS = 24
"""Python contexts one host holds besides its one training context: the gateway starts at most
MAX_CONTEXTS + 1 kernels."""


def user(uid: int) -> str:
    """The guest user whose id is `uid`."""
    return f"ctx{uid - GUEST_UIDS.start}"


def idle_seconds() -> int:
    """How long a session may go with no request while no cell runs: the live contract's idleSeconds
    (limits.ts), which the host passes as DEW_LIVE_IDLE_SECONDS. The bridge then ends the session, and the
    gateway culls a kernel no connection holds once it has been idle this long (gateway_manager.py)."""
    return int(os.environ.get("DEW_LIVE_IDLE_SECONDS", "300"))
