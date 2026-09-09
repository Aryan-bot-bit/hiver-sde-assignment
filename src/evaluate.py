from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from agent import load_agent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())


def main() -> None:
    gold = pd.read_csv(ROOT / CONFIG["gold_path"])
    labelled = gold[gold["intent"].notna() & gold["intent"].ne("")].copy()
    if labelled.empty:
        print("Gold set is an annotation template. Fill intent/escalate columns before evaluation.")
        return
    agent = load_agent()
    predictions = [agent.predict(text) for text in labelled["customer_text"]]
    labelled["predicted_intent"] = [item["intent"] for item in predictions]
    labelled["predicted_escalate"] = [item["escalate"] for item in predictions]
    print("Intent accuracy:", round(accuracy_score(labelled["intent"], labelled["predicted_intent"]), 4))
    print(classification_report(labelled["intent"], labelled["predicted_intent"], zero_division=0))
    if labelled["escalate"].notna().all():
        expected = labelled["escalate"].astype(str).str.lower().isin(["true", "1", "yes"])
        print("Escalation accuracy:", round(accuracy_score(expected, labelled["predicted_escalate"]), 4))
    labelled.to_csv(ROOT / "artifacts" / "evaluation_predictions.csv", index=False)


if __name__ == "__main__":
    main()
