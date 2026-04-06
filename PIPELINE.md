# Hallucination Detector 2.0 — Pipeline Theory

## The Core Problem

LLMs don't retrieve facts — they generate text that *looks* like facts. When a model says "Einstein won the Nobel Prize for the theory of relativity," it isn't checking a database. It's completing a plausible-sounding sentence based on patterns in training data. The result is confident, fluent, and wrong.

No single check reliably catches all hallucinations. A claim can contradict Wikipedia but sound consistent. It can be semantically similar to true facts but factually wrong. It can fool one model while another catches it. The solution is four independent signals that compensate for each other's blind spots.

---

## Architecture Overview

```
Input Text
    ↓
Domain Classifier          ← what kind of claim is this?
    ↓
Topic Inference            ← what entity are we fact-checking?
    ↓
Domain-Appropriate Fetch   ← get ground truth from the right source
    ↓
Vector Store (ChromaDB)    ← store facts as searchable embeddings
    ↓
Per-Sentence Scoring:
    ├── NLI          (40%) ← does any fact directly contradict this?
    ├── Consistency  (35%) ← do multiple models agree on it?
    ├── Grounding    (15%) ← how close is the nearest fact?
    └── Embedding    (10%) ← how similar is the claim to that fact?
    ↓
Weighted Score + Hard Caps → grounded / hallucinated
```

---

## Stage 1 — Domain Classification

**Why it matters:** "The defendant was acquitted" and "the patient was discharged" are both factual claims, but one needs legal sources and the other needs medical ones. Routing everything to Wikipedia produces poor grounding for specialized domains.

**How it works:** BART-MNLI — a zero-shot NLI classifier — scores the input against four candidate labels: medical, legal, financial, general. Unlike a keyword classifier, it understands context. "Satya Nadella won the Nobel Prize" isn't a financial claim just because Nadella is a tech CEO. BART scores the *content* of the claim, not surface word associations.

**Sanity checks prevent misclassification:** If BART says "legal" but there are no legal terms in the text, the classification is almost certainly driven by named-entity association, not actual domain content. A gap threshold (top score must exceed second by ≥0.15) and keyword verification override weak classifications back to general.

**Fallback:** If BART is unavailable, Gemini classifies via prompt.

---

## Stage 2 — Topic Inference & Fact Fetching

**Why not just search the claim directly?** The claim "Einstein won the Nobel Prize for the theory of relativity" is what we're *checking*, not what we're searching for. We need to find the Wikipedia article about Einstein and compare the claim against established facts in that article.

**Topic inference:** Gemini extracts the primary named entity — the Wikipedia article title most likely to contain relevant ground truth. The prompt is engineered to return exact article titles ("Albert Einstein") rather than event descriptions ("Einstein Nobel Prize controversy") which often have no article.

**Three-tier fallback for fetching:**
1. Exact title match
2. MediaWiki search on the full topic string, try top 5 results
3. MediaWiki search on the first significant word(s)

**Domain-specific sources:**

| Domain | Source | Why |
|---|---|---|
| Medical | PubMed E-utilities | Wikipedia medical articles lack mechanistic depth |
| Legal | CourtListener | Case law and statute text, not encyclopedia summaries |
| Financial | Wikipedia | Sufficient for company/market facts at this scope |
| General | Wikipedia | Broad factual coverage |

---

## Stage 3 — Embeddings & Vector Store

**What an embedding is:** A sentence converted into a list of numbers (a vector) such that *similar meanings produce similar vectors*. "Einstein won the Nobel Prize" and "Einstein received the Nobel Prize" will be close together in this high-dimensional space. "The Eiffel Tower is in Paris" will be far away.

**Why cosine similarity, not Euclidean distance:** Euclidean distance measures the gap between vector endpoints. A long document and a short document saying the same thing will have very different vector magnitudes — their endpoints are far apart even though their *direction* (meaning) is identical. Cosine similarity measures the angle between vectors, ignoring magnitude. For text, direction is meaning.

**Why domain-specific embedding models:**

General-purpose embeddings treat "amoxicillin inhibits cell wall synthesis" and "amoxicillin disrupts cell membranes" as highly similar — both are about a drug and bacteria. PubMedBERT, trained on biomedical literature, has learned that cell wall synthesis and cell membrane disruption are distinct mechanisms. It produces more separated embeddings for these phrases, making the difference detectable.

| Domain | Model | Dimensions |
|---|---|---|
| Medical | NeuML/pubmedbert-base-embeddings | 768 |
| Legal | law-ai/InLegalBERT | 768 |
| Financial | ProsusAI/finbert | 768 |
| General | Gemini gemini-embedding-001 | 3072 |

**ChromaDB with cosine distance:** Facts are stored as embeddings in an in-memory vector database. At query time, the claim embedding is compared against all stored fact embeddings using cosine distance. The closest facts are retrieved in one operation — no looping, no N API calls per query.

---

## Stage 4 — The Four Scoring Signals

### Signal 1: NLI — Natural Language Inference (40%)

**The question it answers:** Does any retrieved fact directly contradict this claim?

**How it works:** Gemini receives the claim and the top 3 retrieved facts. For each fact, it classifies the logical relationship as ENTAILMENT, CONTRADICTION, or NEUTRAL. Priority: CONTRADICTION > ENTAILMENT > NEUTRAL. One contradiction is enough to reject the claim regardless of what other facts say.

**Hard cap:** If verdict is CONTRADICTION, final score is capped at 0.25 regardless of other signals. A direct factual contradiction is strong evidence — the other signals can't vote it away.

**Why NLI carries the most weight (40%):** It's the only signal that reads the *content* of the evidence. The other signals measure distances, counts, and probabilities. NLI actually understands what the fact says and whether it conflicts with the claim.

**Why it sometimes returns NEUTRAL on true hallucinations:** The retrieved fact must directly address the specific assertion. "Einstein was a German-born physicist" is NEUTRAL with respect to "Einstein won Nobel for relativity" — it doesn't confirm or deny the Nobel reason. This is correct behavior, not a bug. The other signals handle these cases.

---

### Signal 2: Self-Consistency (35%)

**The question it answers:** Does the model give consistent answers when asked about this claim repeatedly and from different angles?

**The insight:** If a model *knows* something, it says the same thing every time. If it's confabulating, its answers vary — different wrong details, different hedging, different emphasis. Variance = uncertainty = hallucination signal.

**Three independent voters:**

- **Gemini Standard** — straightforward factual evaluation: true, false, or uncertain
- **Gemini Adversarial** — same model, but prompted to actively look for reasons the claim could be wrong before rendering a verdict. Reduces confirmation bias — Gemini tends to lean TRUE on plausible-sounding claims without this prompt
- **BART-MNLI** — completely different architecture (encoder-only NLI classifier vs autoregressive decoder). Generates a factual premise via Gemini, then uses BART to measure whether the claim is supported or refuted by that premise. Different mechanism = genuinely independent signal

**Why three voters instead of one asked three times:** Asking the same model three times gives three draws from the same distribution. The same biases appear in all three. Genuine independence requires different architectures or different prompting strategies.

**Weighted voting:** Each voter produces a TRUE confidence score (1.0, 0.5, or 0.0). Final consistency score = weighted average. If BART is unavailable, weights redistribute between the two Gemini models.

**Hard cap:** If all three voters say FALSE, final score capped at 0.30.

---

### Signal 3: Grounding (15%)

**The question it answers:** How close is the nearest retrieved fact to this claim in embedding space?

**Formula:** `1 / (1 + chromadb_distance)`

This converts any non-negative distance to a (0, 1] score. Distance 0 (identical) → score 1.0. As distance increases, score approaches 0 asymptotically.

**Why it carries less weight (15%):** It only measures topical proximity, not factual accuracy. "Einstein won Nobel for relativity" is close to "Einstein made contributions to quantum theory" — both are about Einstein's scientific work. High grounding score doesn't mean the claim is correct, just that relevant facts exist in the collection.

---

### Signal 4: Embedding Similarity (10%)

**The question it answers:** How semantically similar is the claim to the closest retrieved fact?

**Formula:** `(cosine_similarity + 1) / 2`

The `+1)/2` normalization shifts cosine similarity from [-1, 1] to [0, 1]. Without this, negative cosine values corrupt the weighted average.

**Why it's the lowest weight (10%):** It partially overlaps with grounding (both use the same retrieved fact and cosine distance). It's included because it measures a slightly different thing — direct semantic alignment between the claim vector and the fact vector — but it's not independent enough to deserve more weight.

**Why embedding can be high on a hallucination:** The Amoxicillin example illustrates this perfectly. "Amoxicillin treats bacterial infections by disrupting the cell membrane" scores 0.71 embedding similarity against a PubMed fact about amoxicillin therapy. Both are about the same drug and disease. Embedding sees topical similarity. It cannot distinguish "cell membrane" from "cell wall" as a critical factual error. Consistency catches what embedding misses.

---

## The Weighted Combination

```
final_score = NLI         × 0.40
            + Consistency × 0.35
            + Grounding   × 0.15
            + Embedding   × 0.10
```

Threshold: `final_score < 0.45` → hallucinated

**Why 0.45 and not 0.50:** 0.50 is the midpoint — true uncertainty. Anything below 0.50 leans hallucinated. 0.45 adds a small buffer against false positives on borderline cases.

**Hard caps override the weighted score:**
- CONTRADICTION verdict → cap at 0.25
- All-FALSE consistency → cap at 0.30

Caps exist because some signals are so strong they shouldn't be diluted by weaker ones. A direct Wikipedia contradiction should flag the claim regardless of high embedding similarity.

---

## Why Four Signals, Not One

| Signal | Catches | Misses |
|---|---|---|
| NLI | Direct factual contradictions | Claims where retrieved fact is off-topic |
| Consistency | Confident hallucinations, model uncertainty | Well-known false claims the model consistently misidentifies |
| Grounding | Claims with no nearby evidence | Topically similar but factually wrong claims |
| Embedding | Off-topic claims | Same-topic, wrong-detail claims |

No single signal is sufficient. The system's reliability comes from their combination — each compensates for the others' specific failure modes.

---

## Known Limitations

**Intra-sentence compound claims:** "The Eiffel Tower was completed in 1889 and stands 330 meters tall" is evaluated as one unit. If one clause is true and another is false, the true clause can mask the false one. Full accuracy requires clause-level splitting, which is a separate NLP problem.

**Wikipedia coverage:** For obscure topics with no Wikipedia article, grounding and embedding signals become noise. Consistency becomes the only reliable signal in these cases — which is why it carries 35% weight despite being independent of the knowledge base.

**Domain routing errors:** If BART misclassifies the domain, the wrong embedding model and fact source are used. The keyword sanity check mitigates this but doesn't eliminate it.

**Retrieval quality:** The closest retrieved fact may not be the most relevant one for NLI purposes. NLI checks the top 3 facts to increase the chance of finding a direct contradiction, but a key contradicting fact ranked 4th or lower will be missed.
