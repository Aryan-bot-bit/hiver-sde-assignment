from pathlib import Path
import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
INTENT_RULES = {
    "delivery_tracking": r"track|tracking|deliver|delivery|shipment|package|parcel|arriv",
    "refund_return": r"refund|return|reimburse|money back|cancel",
    "payment_billing": r"charge|charged|payment|billing|card|price|cost|pay",
    "account_login": r"account|login|log in|password|sign in|locked",
    "technical_issue": r"app|website|site|error|not working|broken|crash|bug",
    "subscription_membership": r"prime|membership|subscription|member",
    "order_change": r"purchase|buy|change|address|wrong item|missing",
    "general_support": r".*",
}


def intent_from_rules(text: str) -> str:
    for intent, pattern in INTENT_RULES.items():
        if re.search(pattern, text.lower()):
            return intent
    return "general_support"


class SupportAgent:
    def __init__(self, pairs: pd.DataFrame):
        self.pairs = pairs.reset_index(drop=True)
        self.retriever = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
        self.matrix = self.retriever.fit_transform(self.pairs["customer_text"])
        labels = self.pairs["customer_text"].map(intent_from_rules)
        self.classifier = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("model", LogisticRegression(max_iter=300, class_weight="balanced")),
        ])
        self.classifier.fit(self.pairs["customer_text"], labels)

    def predict(self, message: str) -> dict:
        vector = self.retriever.transform([message])
        similarities = cosine_similarity(vector, self.matrix)[0]
        index = int(similarities.argmax())
        retrieval_score = float(similarities[index])
        probabilities = self.classifier.predict_proba([message])[0]
        label_index = int(probabilities.argmax())
        intent = str(self.classifier.classes_[label_index])
        confidence = float(probabilities[label_index])
        sensitive = bool(re.search(r"password|otp|one[- ]time|ssn|social security|full card|fraud|hack", message.lower()))
        escalate = sensitive or confidence < 0.55 or retrieval_score < 0.18
        reason = "sensitive account or security concern" if sensitive else ("low model confidence" if confidence < 0.55 else ("no strong historical match" if retrieval_score < 0.18 else "routine request with strong historical match"))
        reply = self.pairs.iloc[index]["reply_text"]
        return {"intent": intent, "confidence": round(confidence, 4), "retrieval_score": round(retrieval_score, 4), "draft_reply": reply, "escalate": escalate, "escalation_reason": reason}


def load_agent() -> SupportAgent:
    pairs = pd.read_csv(ROOT / CONFIG["prepared_path"])
    return SupportAgent(pairs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()
    print(json.dumps(load_agent().predict(args.message), indent=2))
