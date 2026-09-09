# Hiver SDE Intern Assignment

A reproducible support-agent baseline for `AmazonHelp` using the Customer Support on Twitter dataset.

## What it does

1. Builds customer-message/support-reply pairs from the raw TWCS CSV in chunks.
2. Classifies messages into eight data-derived operational intents.
3. Retrieves a historically similar AmazonHelp reply to draft a response.
4. Escalates security-sensitive, low-confidence, or weakly matched requests.
5. Evaluates intent and escalation decisions against a reviewable gold-set file.

The system is deliberately local and deterministic. No API key is needed to run the baseline.

## Setup

```powershell
python -m pip install -r requirements.txt
python src/prepare.py
python src/agent.py "Where is my order and when will it arrive?"
python src/evaluate.py
```

The raw file is expected at `data/row/twcs.csv`. Preparation reads it in 200,000-row chunks and writes only a 12,000-pair working sample to `artifacts/amazonhelp_pairs.csv`.

## Golden set

`src/prepare.py` creates `data/golden/amazonhelp_gold.csv` with 200 sampled examples. `src/label_gold.py` fills transparent first-pass suggestions; review each row before submission and correct:

- `intent`: one of `delivery_tracking`, `refund_return`, `payment_billing`, `account_login`, `technical_issue`, `subscription_membership`, `order_change`, `general_support`
- `escalate`: `true` or `false`
- `escalation_reason`: short human explanation
- `notes`: ambiguity or evidence from the message/reply

Sampling uses a shuffled set of matched customer/support pairs from AmazonHelp. The current file is auto-suggested and marked `auto-suggested; human review required`; its evaluation numbers are a smoke test, not evidence of human agreement. For a defensible submission, audit the labels independently and report label guidelines and disagreements.

## Evaluation

The harness reports intent accuracy, per-class precision/recall/F1, and escalation accuracy. Reply quality should be judged on a 1-5 rubric:

- Grounding: factual claims are supported by the retrieved historical response.
- Relevance: addresses the customer's actual request.
- Actionability: gives a concrete next step where one exists.
- Tone: concise, professional, and empathetic.
- Safety: does not request secrets or promise unsupported outcomes.

For the final report, have a human score at least 50 replies and compare the LLM judge's pass/fail decisions with those human scores using agreement percentage and Cohen's kappa. Do not report judge agreement until those human scores exist.

## Decision log

1. Chose AmazonHelp because it has the largest outbound support volume in this local dataset.
2. Used chunked CSV reads because the source file is about 516 MB.
3. Defined intents around operational support actions visible in the messages, not generic industry categories.
4. Removed Twitter handles before retrieval to avoid matching on usernames.
5. Used historical replies as the response source to preserve brand behavior.
6. Used TF-IDF bi-grams because this is cheap, inspectable, and reproducible without an API.
7. Used a classifier confidence threshold for abstention rather than forcing every prediction.
8. Added a separate retrieval threshold because classifier confidence alone does not prove grounding.
9. Escalated security-sensitive language regardless of model confidence.
10. Kept the gold file separate from training artifacts to reduce evaluation leakage.
11. Sampled one parent customer message per reply thread for a clean evaluation unit.
12. Made the baseline API-free so reviewers can reproduce it in under 15 minutes after dependencies are installed.
13. Treat auto-suggested gold labels as unfinished until a human labels and audits them.

## Limitations and next steps

The weak labels used to train the baseline come from transparent keyword rules, so they are not a substitute for the hand-labelled gold set. The retrieved reply can contain historical context that is not appropriate for a new customer; an approved production system would add PII redaction, policy checks, and an LLM judge backed by human calibration.
