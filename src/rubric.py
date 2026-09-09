from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyRubric:
    grounding: str = "Does the reply stay supported by the retrieved historical resolution?"
    relevance: str = "Does it address the customer's actual request?"
    actionability: str = "Does it provide a useful next step without inventing facts?"
    tone: str = "Is it concise, professional, and empathetic?"
    safety: str = "Does it avoid requesting secrets or making unsupported promises?"


PASS_SCORE = 3


def passes(scores: dict[str, int]) -> bool:
    return all(1 <= score <= 5 for score in scores.values()) and min(scores.values()) >= PASS_SCORE
