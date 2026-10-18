# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

**Document:** FIDE *Arbiters' Manual 2026*: 374 PDF pages, 365 content pages, ≈140k words. Covers the Laws of Chess with commentary, rapid/blitz appendices, fair-play rules, Swiss pairing systems, rating, title and tie-break regulations, and online/hybrid chess rules.
**Code:** `RAG_Baseline_vs_Reranking.ipynb` (single notebook, every step commented). Raw outputs are in `results/`.

| Component | Choice |
|---|---|
| Chunking | 180-word sliding window, 40-word overlap → **1,005 chunks**, shared by both pipelines |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 384-d), L2-normalised, FAISS `IndexFlatIP` (exact cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 layers, max 512 tokens) |
| Pipeline 1 | top-10 by cosine → first 3 → LLM → *Answer 1* |
| Pipeline 2 | same top-10 → cross-encoder scores → re-sorted → top 3 → LLM (identical prompt) → *Answer 2* |
| Generator | Gemini Flash (`gemini-flash-latest`, temperature 0), with fallback to local `Qwen2.5-1.5B-Instruct` |
| Evaluation | 28 hand-written questions, each with a verbatim **gold clause** from the manual: 7 failure queries (F1–F7), 3 counter-examples (X1–X3), 18 general queries (G1–G18) |

---

## 1. Architecture comparison

**Bi-encoder (independent encoding).** The query and each chunk go through the transformer **separately**. Each becomes a single 384-d vector (mean-pooled token embeddings). Relevance is then just geometry:

$$\text{score}_{\text{bi}}(q,d)=\cos\big(E(q),E(d)\big)=\frac{E(q)\cdot E(d)}{\lVert E(q)\rVert\,\lVert E(d)\rVert}$$

Because $E(d)$ does not depend on the query, every chunk is embedded **once, offline**. A search costs one query encoding plus a nearest-neighbour lookup.

**Cross-encoder (joint encoding).** The query and the chunk are concatenated into one input, `[CLS] query [SEP] chunk [SEP]`, and passed through the transformer **together**. A linear head on the `[CLS]` vector outputs one relevance logit:

$$\text{score}_{\text{ce}}(q,d)=\mathbf{w}^{\top}\,\mathrm{BERT}\big([q;d]\big)_{\texttt{[CLS]}}+b$$

In every layer, self-attention lets each query token attend to each chunk token. So the model can check *this* query word against *that* clause: "touching a piece" against "loses the right to claim"; "must not" against "round-robins".

> **Key difference.** A bi-encoder must compress a whole chunk into one vector **before it knows the question**, so a short exception or a "not" buried in a 240-token chunk barely moves that vector. A cross-encoder reads the question and the chunk **at the same time**, so the exact clause the question is about can decide the score.

*Analogy:* a bi-encoder is a librarian who wrote a one-line summary of every page in advance and matches your question against those summaries. A cross-encoder is a librarian who reads each candidate page with your question in hand. The second is far more accurate, and far too slow to do for every page.

---

## 2. Qualitative & rank-shift analysis

### 2.1 Rank shifts on the stress-test queries

"Rank" is the position of the chunk that contains the answering clause. The bi-encoder rank is over the full corpus; the cross-encoder rank is within the re-ranked top-10. Only ranks 1–3 reach the LLM.

| ID | Failure mode | Query | Gold chunk (page) | Bi-encoder → Cross-encoder |
|---|---|---|---|---|
| F1 | Keyword trap / synonym | Can a spectator tell the arbiter that a player's time has run out? | #102 (p.38, Art. 12.7) | **#5 → #1** |
| F2 | Keyword trap / distractor | Can a player who arrives late still store their mobile phone before playing? | #96 (p.36, commentary to 11.3) | **#4 → #1** |
| F3 | Exception / corner case | Can a player claim threefold repetition after touching a piece? | #952 (p.337), #87 (p.33, Art. 9.4) | **#7 → #1**, #8 → #2 |
| F4 | Exception / penalty | What is the penalty for an incorrect draw claim? | #87 (p.33, Art. 9.5.3) | **#6 → #2** |
| F5 | Negation / constraint | Which tie-breaks must not be used in round-robin tournaments? | #722 (p.251, C.07 Art. 8) | **#5 → #1** |
| F6 | Buried at rank 4–10 | What happens if neither player keeps score in time trouble and a flag falls? | #75 (p.29, Art. 8.5.1) | **#8 → #1** |
| F7 | Buried at rank 4–10 | What should the arbiter do if both flags have fallen and it is impossible to tell which fell first? | #136 (p.52, Guideline III.3.1) | **#7 → #1** |
| X1 | Counter: CE demotes gold | When must the arbiter call a flag fall in rapid chess? | #109 (p.41, A.5.5) | #2 → #4 ✗ |
| X2 | Counter: CE demotes gold | Can a player smoke during an online game with supervision? | #875 (p.307, online 9.9 commentary) | #1 → #4 ✗ |
| X3 | Counter: recall ceiling | Can two players agree a draw before any moves have been made? | #42 (p.19, Art. 5.2.3) | #64 → not in top-10 ✗ |

![Rank shift](results/rank_shift.png)

In all seven failure queries the answering chunk was **in the top-10 but outside the top-3**, so the baseline LLM never saw it. Re-ranking promoted it into the context every time, to #1 in six of seven. I also checked that none of these clauses sits beyond the bi-encoder's 256-token window (all start at token ≤ 199), so the wins are not a truncation artefact.

### 2.2 Aggregate retrieval quality

| Query set | Pipeline | Hit@1 | Hit@3 (what the LLM sees) | MRR@10 | Recall@10 |
|---|---|---|---|---|---|
| All 28 | Baseline | 0.393 | 0.536 | 0.521 | 0.893 |
| All 28 | Re-ranked | **0.679** | **0.821** | **0.762** | 0.893 |
| General 18 | Baseline | 0.556 | 0.722 | 0.657 | 0.889 |
| General 18 | Re-ranked | **0.722** | **0.889** | **0.796** | 0.889 |

Recall@10 is identical by construction: re-ranking only permutes the same 10 candidates. **Every gain is in moving relevant chunks up into the top-3.** The "all" row is inflated on purpose, because F1–F7 were selected as failures. The general set is the fairer estimate: Hit@3 rises 0.72 → 0.89, and there it helped 3 queries (G9, G13, G17) and hurt none.

### 2.3 What each pipeline hands the LLM, and what that does to the answer

The prompt tells the LLM to answer **only** from the three passages. So when the clause is missing, the best case is "the context does not contain this", and the realistic failure is an answer stitched together from neighbouring rules. The verbatim Answer 1 / Answer 2 pairs are produced by notebook §11 and saved to `results/answer_comparison.md`.

**F1 (keyword trap).** The baseline top-3 is Appendix A (rapid scoresheets), Art. 6.8 ("a flag is considered to have fallen when the *arbiter* observes the fact or when *either player* has made a valid claim") and Art. 12.6 ("the arbiter must not intervene… shall refrain from informing a player…"). The query says "time has run out", but the manual says "flag fall". The baseline context lists only the arbiter and the players as people who can establish a flag fall, so it invites the **wrong answer "no"**. Re-ranked #1 is Art. 12.7 with its commentary: *"If a spectator sees a flag fall, or any other offence, they should tell the arbiter"*. Spectators may inform the arbiter but must not interfere.

**F2 (distractor).** All three baseline chunks share the query's vocabulary ("mobile phone", "player", "venue"): Art. 11.3.2 commentary that a phone found on a player means **immediate loss**, the zero-tolerance example, and the online-chess device rule. A baseline answer is pushed towards **"no, the player loses the game"**. The actual answer is phrased with different words ("store his/her *device*") in chunk #96: a late-arriving player *should be given the opportunity to store the device before playing their first move*. It was rank #4 and becomes #1.

**F3 (exception).** The baseline gives the general rules: touch-move (Art. 4.3) and the threefold-repetition claim procedure (Art. 9.2). Both say nothing about losing the right to claim, so Answer 1 can at best **omit the restriction** and at worst say "yes, the player having the move may claim". Re-ranked context contains Art. 9.4 (*"If the player touches a piece as in Article 4.3, he/she loses the right to claim a draw under Article 9.2 or 9.3 on that move"*) plus a worked exam answer saying the same.

**F4 (penalty).** Baseline chunk #88 contains commentary on a *different* case: *"If the player loses the right to claim in this way it is not a wrong claim and is not punished as such. The arbiter may decide to award the opponent extra time."* That is a direct route to a **misstated rule** ("incorrect claims are not punished; extra time is optional"). Art. 9.5.3, *"the arbiter shall add two minutes to the opponent's remaining thinking time"*, enters only after re-ranking.

**F5 (negation).** The baseline returns the general tie-break preamble (scope, ordering of tie-breaks), which matches "tie-break" and "tournament" but has no prohibition in it. Answer 1 can only say the context doesn't specify, or guess. Re-ranked #1 is C.07 Art. 8: *"Buchholz and other Tie-Breaks related to Buchholz — Note: These tie-breaks must not be used in round-robins"* (BH, AOB, Fore Buchholz).

**F6 (buried, contradictory neighbour).** Baseline chunk #78 is reconstruction commentary that says *"after a flag fall, the arbiter does **not** pause the clocks"*. That sentence is about the case where only one player stopped recording. Its rule, Art. 8.5.1, says the opposite for this question: *"immediately after a flag has fallen the arbiter **shall pause** the chessclock. Then both players shall update their scoresheets"*. It sat at rank #8. The baseline context therefore steers the answer towards **the opposite of the correct rule**. Re-ranking puts Art. 8.5.1 at #1.

**F7 (buried).** The baseline gives generic flag-fall commentary. The specific rule, Guideline III.3.1 (*game continues if this happens in any period except the last; drawn if it happens in the final period*), is at #7. Without it Answer 1 has nothing to go on (likely **"not covered" or a guess**). After re-ranking it is #1.

### 2.4 Where re-ranking does not help (counter-examples)

* **X1, rapid flag fall.** The bi-encoder has A.5.5 (*"The arbiter shall also call a flag fall, if he/she observes it"*) at #2. The cross-encoder prefers Guideline III (quick-play finishes) and standard-chess commentary, and pushes A.5.5 to #4. MS-MARCO training rewards topical overlap with "flag" and "arbiter", and does not know that "rapid" should select Appendix A.
* **X2, online smoking.** The bi-encoder has the online rule at #1. The cross-encoder prefers the over-the-board smoking/phone passage (#96). The context cue "online … with supervision" loses to "smoke".
* **X3, recall failure.** The answer, Art. 5.2.3 (*"provided both players have made at least one move"*), is bi-encoder rank #64. **A re-ranker cannot recover what the first stage never retrieved.** Two other general queries (G14, G15) fail the same way.

**Takeaway:** re-ranking is a precision tool. It fixes ordering inside the candidate set, but it can also mis-order when relevance hinges on a context qualifier (rapid vs. standard, online vs. over-the-board), and it does nothing for recall.

---

## 3. Latency profiling

28 queries × 5 repetitions after warm-up. The run used a 2-vCPU cloud machine **without a GPU**; on a Colab GPU every model stage is much faster, and the notebook re-measures on whatever hardware it runs on.

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
|---|---|---|---|
| Query encoding (bi-encoder) | 12.6 | 12.0 | 16.9 |
| FAISS search (1,005 × 384, exact) | 0.2 | 0.2 | 0.3 |
| **Pipeline 1 total (retrieval)** | **12.8** | **12.2** | **17.1** |
| Cross-encoder scoring of 10 pairs + sort | 393.3 | 380.8 | 486.8 |
| **Pipeline 2 total (retrieval + re-rank)** | **406.1** | **391.9** | **499.5** |

One-off cost: embedding and indexing all 1,005 chunks took 37 s.

* Re-ranking makes retrieval **≈32× slower** (median). Almost all of it is the 10 cross-encoder passes: up to 512 tokens each, and attention cost grows quadratically with length.
* In absolute terms it adds about 0.4 s on CPU (a few tens of ms on GPU). That is small next to LLM generation, which typically takes about a second or more per answer. Generation time is logged per answer in §11 (`llm_s_1`, `llm_s_2`).
* FAISS search is negligible. Bi-encoder retrieval time is dominated by encoding the query.

---

## 4. Scaling: why not run the cross-encoder over the whole collection?

**1. There is no pre-computation.** A cross-encoder score depends on the query *and* the chunk together, so nothing can be indexed ahead of time. Each query needs **N full transformer passes** over (query + chunk) inputs of up to 512 tokens. A bi-encoder pays the N document encodings once, offline. Each query then costs one encoding plus an index lookup, which is sub-linear with ANN indexes such as HNSW or IVF.

**2. Measured on this corpus** (notebook §13):

| Query | Full-corpus cross-encoder (1,005 pairs) | Retrieve-10 + re-rank | Slow-down | Gold rank, full CE | Gold rank, two-stage |
|---|---|---|---|---|---|
| F3 | 76.7 s | 0.43 s | 177× | #2 | #1 |
| F6 | 75.5 s | 0.55 s | 137× | #2 | #1 |
| X3 | 74.6 s | 0.37 s | 204× | #9 | not in top-10 |
| G7 | 75.9 s | 0.80 s | 95× | #1 | #1 |

That is about 75 ms per pair on this CPU. Cost is linear in N: at 1 million chunks it would be roughly 21 hours per query on this machine. Even a GPU at a few thousand pairs per second would take minutes per query. Two-stage retrieval keeps the expensive model's cost fixed at k = 10 pairs, whatever the corpus size.

**3. It isn't even clearly better.** For F3 and F6, full-corpus scoring ranked the gold chunk #2, whereas the two-stage pipeline ranked it #1. With 1,000 candidates there are more look-alike distractors that the cross-encoder over-scores. The bi-encoder stage acts as a cheap semantic filter. For X3 the full-corpus cross-encoder reaches the gold chunk only at #9, so brute force does not solve that recall problem either.

**Conclusion:** use the cheap, pre-computable bi-encoder for **recall** over the whole collection, and spend the expensive cross-encoder only on a short list for **precision**. This is the standard retrieve-then-rerank design.

---

## 5. Limitations and reproducibility notes

* **Query selection bias.** The failure queries were found by probing the manual for cases where the baseline stumbles, as the assignment asks. Treat the "all queries" metrics as illustrative. The 18 general queries are the fairer sample, but they are still small and hand-written.
* **Gold labels are phrase-based.** A chunk counts as relevant if it contains the verbatim answering clause. Paraphrased versions elsewhere could be missed. Two labels (X2, G14) were widened after inspection for exactly this reason.
* **One chunking configuration.** 180/40 words gives a median of 240 tokens, and 24% of chunks exceed the bi-encoder's 256-token window and lose their tail. None of the gold clauses are affected, but structure-aware chunking would likely raise baseline recall (X3, G14, G15).
* **Model weights.** The retrieval numbers above were produced on an offline machine, using copies of the same two checkpoints: the cross-encoder converted from its official ONNX export, and the bi-encoder from an fp16 copy. Both were checked against reference outputs (cosine 0.7553 on the sentence-transformers example; logits within 1e-5 of ONNX). In Colab the notebook loads the originals from Hugging Face, so near-tie ranks could differ in the last decimal.
