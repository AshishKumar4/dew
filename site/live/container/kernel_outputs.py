"""Bounded text and PNG outputs from a Jupyter kernel."""

MAX_MESSAGE = 900_000


def outputs_of(kind, content):
    if kind == "stream":
        text = content["text"]
        return [{"type": "stream", "name": content["name"], "text": text[i:i + MAX_MESSAGE]}
                for i in range(0, len(text), MAX_MESSAGE)]
    if kind in ("display_data", "execute_result", "update_display_data"):
        data = content.get("data", {})
        out = {"type": "display"}
        if "image/png" in data:
            if len(data["image/png"]) <= MAX_MESSAGE:
                out["png"] = data["image/png"]
            else:
                out["text"] = "[An image too large to send here was not shown.]"
        if "text/plain" in data and "text" not in out:
            out["text"] = data["text/plain"][:MAX_MESSAGE]
        return [out] if len(out) > 1 else []
    if kind == "error":
        ename, evalue = content["ename"][:200], content["evalue"][:MAX_MESSAGE // 10]
        room = MAX_MESSAGE - len(ename) - len(evalue)
        kept = []
        for line in reversed(content["traceback"][-40:]):
            if room <= 0:
                break
            kept.append(line[:room])
            room -= len(kept[-1])
        return [{"type": "error", "ename": ename, "evalue": evalue, "traceback": kept[::-1]}]
    if kind == "clear_output":
        return [{"type": "clear"}]
    return []
