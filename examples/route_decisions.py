"""Send each request to the Laya checkpoint that can read it.

Laya ships an English checkpoint and a multilingual one in the same Hub
repository. A deployment that serves both picks one per request: text in a
non-Latin script, or Latin text with many letters English does not use,
goes to the multilingual model, and everything else to the English one.
This is a simplified version of Laya's own router (laya/router.py). That
router also reads stop words, so it catches Latin text with no accented
letters at all, such as most Italian; it keeps the most recently used
checkpoints resident; and it accepts a caller's language code. A service
that knows its users' language should pass that instead of guessing.

    python examples/route_decisions.py
"""

import unicodedata
from collections.abc import Mapping

from dew.decision import Choice, Decide, Noul, Question

# Laya's threshold for the share of non-English letters in Latin text.
NON_ENGLISH_LETTERS = 0.02


def reads_as_english(state: str) -> bool:
    """Whether the English checkpoint can read `state`: its letters are
    Latin, and few of them carry marks English does not write."""
    letters = [char for char in state if char.isalpha()]
    if not letters:
        return True
    latin = [char for char in letters if unicodedata.name(char, "").startswith("LATIN")]
    if len(latin) < len(letters) / 2:
        return False
    marked = sum(not char.isascii() for char in latin)
    return marked / len(latin) < NON_ENGLISH_LETTERS


def main() -> None:
    english = Decide.from_pretrained("convaiinnovations/laya")
    multilingual = Decide.from_pretrained("convaiinnovations/laya", subfolder="multilingual")

    def decide(state: str, questions: Mapping[str, Question]):
        return (english if reads_as_english(state) else multilingual)(state, questions)

    questions = {
        "department": Choice("Which department should handle this?",
                             {"billing": "invoices, refunds and charges", "technical": "bugs and outages"}),
        "churn": Noul("Does the customer threaten to leave?"),
    }
    for state in ("I was charged twice for March and I want it reversed today.",
                  "Mir wurde der März zweimal berechnet, bitte erstatten Sie es heute zurück.",
                  "Меня дважды списали за март, верните деньги сегодня."):
        answers = decide(state, questions)
        print(f"{state[:40]:42} {answers['department'].choice:10} churn {answers['churn'].noul:.2f}")


if __name__ == "__main__":
    main()
