## How the Project Works

This project does **not require any API key**.

The agent runs locally using a TF-IDF-based machine-learning approach, so:

* OpenAI API key is not required.
* Gemini API key is not required.
* Hugging Face API key is not required.
* GitHub token is not required for the current HTTPS push setup.

### 1. Raw Dataset

The system reads the `twcs.csv` file from the **Customer Support on Twitter (TWCS)** dataset.

From the dataset, it selects **AmazonHelp** conversations and extracts customer messages along with their corresponding historical support replies.

### 2. Conversation Pair Creation

`prepare.py` reconstructs customer-support interactions using the tweet response relationships.

For example:

**Customer:**

> Where is my package?

**Historical AmazonHelp Reply:**

> Please check your tracking details for the latest delivery information.

These customer-message/reply pairs are stored in:

```text
artifacts/amazonhelp_pairs.csv
```

### 3. Intent Detection

`agent.py` classifies the incoming customer message into one of eight operational intents:

* `delivery_tracking`
* `refund_return`
* `payment_billing`
* `account_login`
* `technical_issue`
* `subscription_membership`
* `order_change`
* `general_support`

The classifier uses **TF-IDF features**, including word bi-grams, to identify the most likely intent.

### 4. Historical Reply Retrieval

After identifying the intent, the system searches the historical AmazonHelp conversations for a similar customer issue.

The corresponding historical support reply is then used as the basis for drafting the response.

This makes the response **grounded in actual historical AmazonHelp support behaviour** rather than generating an entirely unsupported answer.

### 5. Escalation Decision

The agent does not automatically handle every request.

It can escalate when:

* Intent confidence is too low.
* No sufficiently similar historical reply is found.
* The request contains security-sensitive language.
* The request requires human intervention or account-specific information.

For example:

**Customer:**

> Someone has accessed my account. I need help immediately.

The system can classify the request and mark it for **human escalation** rather than attempting to handle a potentially sensitive account-security issue automatically.

### 6. Evaluation

The project contains a **200-example assistant-reviewed golden-set first pass**. The `evaluation` column records this provenance as `assistant_reviewed`; a human audit is still required before claiming a hand-labelled set.

The evaluation measures:

* Intent accuracy
* Precision
* Recall
* F1-score
* Escalation accuracy
* Reply-quality rubric (manual/LLM scoring is not automated in this baseline)

Reply quality should be scored using:

1. Grounding
2. Relevance
3. Actionability
4. Tone
5. Safety

### 7. Why No API Key?

The current version intentionally uses a **local and deterministic baseline**.

This has several advantages:

* Easy for the evaluator to run.
* No API cost.
* No API-key configuration.
* No dependency on external LLM availability.
* Results are reproducible.
* Suitable for the assignment's reproducibility requirement.

An LLM-based version can be added later for improved response generation and LLM-as-judge evaluation, but the core baseline works completely locally.
