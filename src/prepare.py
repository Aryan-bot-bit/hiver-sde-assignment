from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA_PATH = ROOT / CONFIG["data_path"]
OUTPUT_PATH = ROOT / CONFIG["prepared_path"]
GOLD_PATH = ROOT / CONFIG["gold_path"]


def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    brand = CONFIG["brand"]
    rows = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200_000):
        support = chunk[(chunk["author_id"] == brand) & (~chunk["inbound"])]
        rows.append(support[["tweet_id", "text", "in_response_to_tweet_id", "created_at"]])
    replies = pd.concat(rows, ignore_index=True).dropna(subset=["in_response_to_tweet_id"])
    parent_ids = set(replies["in_response_to_tweet_id"].astype("int64"))

    parents = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200_000):
        selected = chunk[chunk["tweet_id"].isin(parent_ids) & chunk["inbound"]]
        if not selected.empty:
            parents.append(selected[["tweet_id", "text"]])
    customer = pd.concat(parents, ignore_index=True).drop_duplicates("tweet_id")
    customer = customer.rename(columns={"tweet_id": "customer_tweet_id", "text": "customer_text"})
    replies["customer_tweet_id"] = replies["in_response_to_tweet_id"].astype("int64")
    pairs = replies.merge(customer, on="customer_tweet_id", how="inner")
    pairs["customer_text"] = pairs["customer_text"].map(clean_text)
    pairs["reply_text"] = pairs["text"].map(clean_text)
    pairs = pairs[(pairs["customer_text"].str.len() >= 8) & (pairs["reply_text"].str.len() >= 8)]
    pairs = pairs.drop_duplicates("customer_tweet_id").sample(frac=1, random_state=CONFIG["random_state"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.head(CONFIG["sample_size"])[["customer_tweet_id", "customer_text", "reply_text", "created_at"]].to_csv(OUTPUT_PATH, index=False)

    gold = pairs.head(CONFIG["gold_size"])[["customer_tweet_id", "customer_text", "reply_text"]].copy()
    gold.insert(2, "intent", "")
    gold.insert(3, "escalate", "")
    gold.insert(4, "escalation_reason", "")
    gold.insert(5, "notes", "")
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD_PATH, index=False)
    print(f"Prepared {min(len(pairs), CONFIG['sample_size'])} pairs")
    print(f"Created {len(gold)}-row annotation template at {GOLD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
