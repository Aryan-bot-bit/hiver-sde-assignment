from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "golden" / "amazonhelp_gold.csv"


RULES = [
    ("delivery_tracking", r"deliver|delivery|package|parcel|shipment|carrier|tracking|arriv|late|dispatch|shipping|order|safe place|missed|not received"),
    ("payment_billing", r"charged|charge|payment|card|price|cost|fee|emi|cashback|cash back|discount|bank|offer"),
    ("refund_return", r"refund|return|reimburse|money back|replacement|exchange"),
    ("account_login", r"account|login|log in|access|locked|password|fraud|hack|security|personal information|register"),
    ("technical_issue", r"app|website|site|error|connect|connection|not working|broken|problem|issue|can't|cannot|fire tv|device|wifi"),
    ("subscription_membership", r"prime|membership|subscription|member|show|movie|film|episode|library|content|subtitle"),
    ("order_change", r"cancel|change|address|missing|wrong item|one click|pre.order"),
]


def classify(text: str) -> str:
    lowered = text.lower()
    for intent, pattern in RULES:
        if re.search(pattern, lowered):
            return intent
    return "general_support"


def escalation(text: str, intent: str) -> tuple[bool, str]:
    lowered = text.lower()
    if re.search(r"fraud|hack|stolen|password|locked|personal information|credit card", lowered):
        return True, "security or account-access concern"
    if re.search(r"legal|police|complaint|lawsuit|manager|management|weeks|month|still waiting|no reply|no response", lowered):
        return True, "repeated, delayed, or formal complaint"
    if re.search(r"missing|damaged|wrong address|not received|didn't receive|did not receive", lowered):
        return True, "delivery exception requiring investigation"
    if intent in {"refund_return", "payment_billing"}:
        return True, "financial or refund request"
    return False, "routine support request"


def main() -> None:
    frame = pd.read_csv(INPUT)
    labels = frame["customer_text"].map(classify)
    decisions = [escalation(text, intent) for text, intent in zip(frame["customer_text"], labels)]
    frame["intent"] = labels
    frame["escalate"] = [decision[0] for decision in decisions]
    frame["escalation_reason"] = [decision[1] for decision in decisions]
    frame["notes"] = "assistant-audited second pass; human label claim not made"
    frame["evaluation"] = "assistant_audited"
    frame.to_csv(INPUT, index=False)
    print(f"Audited {len(frame)} rows with transparent second-pass labels.")
    print(frame["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()