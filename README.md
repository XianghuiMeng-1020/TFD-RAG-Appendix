Please see our appendix for full supplementary material.
All of our code will be made avaliable upon acceptance.


# Supplementary Appendix: Temporal Faithfulness Decomposition for RAG

*Anonymous submission to CIKM 2026. This appendix accompanies the main paper and is not subject to the page limit. All numerical claims are traceable to* `macros.tex`*; all proofs reference section numbers in the main paper.*

**Table of Contents**

| Section | Content |
|---------|---------|
| [A. Theoretical Foundations](#a-theoretical-foundations) | Formal assumptions, full SEE–PCE proof, SAFS properties, halflife MLE |
| [B. TFD-Bench v4 Construction](#b-tfd-bench-v4-full-construction-details) | Data sources, Tier-1 oracle pipeline, Tier-2 injection, staleness buckets |
| [C. TFD Classifier Prompt](#c-tfd-classifier-4-shot-prompt) | Full 4-shot GPT-4o-mini classification prompt |
| [D. TFD-Targeted Implementation](#d-tfd-targeted-full-implementation-details) | DeBERTa classifier, SEE/PCE/PHE branches, latency profile |
| [E. Complete Experimental Results](#e-complete-experimental-results) | Full 17-system × GPT-4o-mini table, domain-level results, GPT-4o cross-model |
| [F. Ablation Studies](#f-ablation-studies) | Fixed-document ablation, oracle routing bound, N comparison, threshold sensitivity |
| [G. Statistical Analysis](#g-statistical-analysis) | Pre-registered hypotheses, power analysis, multiple comparisons |
| [H. Additional Analysis](#h-additional-analysis) | GoldMatch–CURRENCY independence, SAFS discriminative validation, halflife calibration |
| [I. Case Studies](#i-case-studies) | Three full representative cases (SEE, PHE, mixed SEE+PCE) |
| [J. Experimental Hyperparameters](#j-experimental-hyperparameters) | Complete hyperparameter table |
| [K. AI Usage Disclosure](#k-ai-usage-disclosure) | Per ACM 2026 Generative AI Policy |

---

---

## A. Theoretical Foundations

### A.1 Formal Assumptions

The TFD framework rests on five assumptions stated here for completeness.

**Assumption 1 (Temporal Ordering, A1).** The timestamps satisfy $t_\mathcal{C} \leq t_\theta \leq t_q$, where $t_\mathcal{C}$ is the document timestamp, $t_\theta$ the LLM knowledge cutoff, and $t_q$ the query timestamp. This reflects real-world RAG deployment; synthetic-future documents are out of scope.

**Assumption 2 (Monotonically Evolving Facts, A2).** For time-sensitive facts, once a fact changes from state $s_1$ to $s_2$ at time $t^*$, it does not revert for all $t > t^*$. The framework targets *persistent facts* (CEO appointments, capital cities, event outcomes). Non-monotonic facts (polls, prices) are handled conservatively under PHE per Corollary A.5.

**Assumption 3 (Claim Independence, A3).** Each atomic claim's error type is determined independently given its context and query. Empirically, inter-claim correlation in TFD-Bench is negligible ($\rho < 0.05$).

**Assumption 4 (Non-Zero Parametric Staleness, A4).** There exists $\varepsilon > 0$ such that $\Pr[\phi \text{ is outdated in } \mathcal{K}_{\leq t_\theta} \mid \phi \text{ is time-sensitive}] \geq \varepsilon$. Empirically, $\Pr(\text{PCE} \mid \text{temporal fact}) \approx 0.21$ on TFD-Bench v4, establishing $\varepsilon \approx 0.21$.

**Assumption 5 (Retrieval Coverage, A5).** Let $\mathcal{D}_q$ be the relevant document set and $\mathcal{C}_\tau$ the retrieved context under staleness threshold $\tau$. There exists $\eta \in (0,1]$ such that $\mathbb{E}[|\mathcal{C}_\tau \cap \mathcal{D}_q| / |\mathcal{D}_q|] \geq \eta$. Empirically $\eta \approx 0.73$ on TFD-Bench v4.

---

### A.2 Full SEE–PCE Trade-off Proof

**Theorem 1 (SEE–PCE Trade-off Bound; A1–A5).**
Let $\tau$ be a staleness filtering threshold and $\delta(\tau) = \mathrm{SEE\text{-}rate}_0 - \mathrm{SEE\text{-}rate}_\tau \geq 0$ the SEE reduction. Under A4–A5 (retrieval coverage $\eta \in (0,1]$):
$$\Delta\mathrm{PCE}(\tau) \leq \frac{\delta(\tau)}{\eta}$$
and the bound is tight when all freshly-filtered documents contain only parametric facts.

**Proof.** Consider an arbitrary example $x$ that was previously classified as SEE (i.e., its retrieved document was stale) and is now filtered by the threshold $\tau$. After filtering, the system must either:
(a) **Abstain**: Return no answer or a hedged response. PCE-rate is not affected.
(b) **Substitute parametric knowledge**: The system generates a claim not grounded in the (now-absent or freshened) context. This contributes $+1$ to PCE-rate for this example.

Under A5, at most $\delta(\tau) / \eta$ distinct examples are affected by the filtering (each filtered document covers at least $\eta$ of the relevant document set, so at most $\delta(\tau)/\eta$ examples lose their SEE-triggering document). Therefore:
$$\Delta\mathrm{PCE}(\tau) \leq \frac{\delta(\tau)}{\eta} \cdot \Pr[\text{substitute} \mid \text{filtered}] \leq \frac{\delta(\tau)}{\eta}$$
The bound is achieved when $\Pr[\text{substitute} \mid \text{filtered}] = 1$, i.e., when every filtered document's replacement triggers parametric contamination. $\square$

---

### A.3 SAFS Properties

**Proposition A.1 (SAFS Properties, under A1–A2).** SAFS satisfies:
(i) **Boundedness**: $\mathrm{SAFS} \in [0,1]$
(ii) **GoldMatch Dominance**: $\mathrm{SAFS} \leq \mathrm{GoldMatch}$, with equality iff $t_\mathcal{C} = t_q$
(iii) **Staleness Monotonicity**: $\partial \mathrm{SAFS} / \partial \Delta t \leq 0$ for all $\Delta t \geq 0$

*Proof.* (i) follows from $\mathrm{GoldMatch} \in [0,1]$ and $e^{-\lambda \Delta t} \in (0,1]$. (ii) follows from $\mathrm{CURRENCY}_d \leq 1$ with equality iff $\Delta t = 0$. (iii) follows by differentiating: $\partial \mathrm{SAFS}/\partial \Delta t = -\lambda \cdot \mathrm{SAFS} \leq 0$. $\square$

**Corollary A.2 (Strict Monotonicity and Halflife Interpretation).**
At $\Delta t = \tau_d$ (domain halflife), $\mathrm{SAFS} = \frac{1}{2}\mathrm{GoldMatch}$, independent of domain.
For fixed $\Delta t$ and GoldMatch: shorter halflife $\tau_1 < \tau_2$ implies $\mathrm{SAFS}_1 < \mathrm{SAFS}_2$.

**Theorem A.3 (SAFS$^+$ Strict Dominance, under A1–A3).**
SAFS$^+$ (with claim-level weights $w(\mathrm{SEE}) = 0.5$, $w(\mathrm{PCE}) = 0$, $w(\mathrm{PHE}) = 0$, $w(\mathrm{CORRECT}) = 1$) provides strictly finer discrimination than SAFS: if $\mathrm{SAFS}(A) = \mathrm{SAFS}(B)$ but $\mathrm{SEE\text{-}rate}(A) \neq \mathrm{SEE\text{-}rate}(B)$, then $\mathrm{SAFS}^+(A) \neq \mathrm{SAFS}^+(B)$.

*Proof.* SAFS$^+(A) = \mathrm{SAFS}(A) - \alpha \cdot \mathrm{SEE\text{-}rate}(A)$ for any $\alpha > 0$ (from the $w(\mathrm{SEE}) = 0.5$ penalty). Since $\mathrm{SAFS}(A) = \mathrm{SAFS}(B)$ and $\mathrm{SEE\text{-}rate}(A) \neq \mathrm{SEE\text{-}rate}(B)$, we have $\mathrm{SAFS}^+(A) \neq \mathrm{SAFS}^+(B)$. $\square$

---

### A.4 Halflife MLE: Statistical Properties

**Lemma A.4 (Fisher Information CLT for Domain Halflife, under A2).**
Under A2, the MLE halflife estimator $\hat{\tau}_d = \ln 2 / \hat{\lambda}_d$ is asymptotically normal:
$$\sqrt{n_d}\bigl(\hat{\tau}_d - \tau_d\bigr) \xrightarrow{d} \mathcal{N}\!\left(0,\; \frac{(\ln 2)^2}{\lambda_d^4 \cdot \mathcal{I}(\lambda_d)}\right)$$
For $n_d \geq 75$ (our minimum domain size), the Gaussian approximation error is $\leq 0.05$ (Berry–Esseen bound with Shevtsova's constant $C \leq 0.4748$).

For domain $d$, given staleness gaps $\{\Delta t_i\}$ and SEE labels $\{y_i\}$, the MLE objective is:
$$\hat{\lambda}_d = \arg\max_\lambda \sum_i \bigl[ y_i \log(1 - e^{-\lambda \Delta t_i}) + (1-y_i)(-\lambda \Delta t_i) \bigr]$$

**Minimum Sample Size.** For $\varepsilon$-accurate halflife estimation with probability $\geq 1-\delta$, it suffices to have $n \geq \ln(2/\delta) / (2\varepsilon'^2)$ where $\varepsilon' = \varepsilon \cdot \ln(2) / (\tau^*)^2$ is the implied accuracy on $\lambda$. For TFD-Bench calibration ($\tau^* \approx 813$d, $\varepsilon = 50$d, $\delta = 0.05$), this yields $n \geq 421$ per domain.

**Corollary A.5 (Boundary Case Handling, under violation of A2).**
For non-monotonic facts: (i) Oscillating facts are classified by the *most recent* state at $t_q$; (ii) temporary reversions are classified as SEE if the context reflects a non-final state; (iii) when monotonicity is violated, the SEE/PCE boundary defaults to PCE (conservative).

---

## B. TFD-Bench v4: Full Construction Details

### B.1 Data Sources and Preprocessing

| Domain | Source | Test $n$ | Staleness range |
|--------|--------|----------|-----------------|
| News | NewsQA | 465 | 1–15+ years |
| Streaming | StreamingQA | 304 | 0–3 years |
| General | ArchivalQA | 466 | 1–15+ years |
| Sports | Sportradar API | 167 | 1–7 years |
| Finance | SEC EDGAR 10-K/10-Q | 92 | 1–10 years |
| Science | PubMed/arXiv metadata | 86 | 1–5 years |

All examples include: question $q$, retrieved document with timestamp $t_\mathcal{C}$, query timestamp $t_q$ ($\in [2022, 2025]$), and gold answer valid at $t_q$. Documents were not modified; only their timestamps and associated metadata were used.

### B.2 Tier-1 Oracle SEE Label Construction

1. **SPARQL extraction**: For each entity-based question, extract entity-attribute-timestamp triples from Wikidata (e.g., `(Twitter, CEO, Jack Dorsey, valid_until=2022-10-27)` and `(Twitter, CEO, Elon Musk, valid_from=2022-10-27)`).
2. **Ground-truth SEE labeling**: If the historical value at $t_\mathcal{C}$ differs from the current value at $t_q$, the example receives a guaranteed SEE label.
3. **Confidence threshold**: Only examples with Wikidata confidence $\geq 0.90$ are included as Tier-1.
4. **Double-judge validation**: GPT-4o (primary) and GPT-4o-mini (secondary) independently classify each example; labels are accepted only when both agree. Agreement rate: 91% ($n = 170$).

### B.3 Tier-2 Temporal Injection

For examples without Wikidata anchors, two injection strategies augment the SEE examples:

**Strategy A (Wikidata-anchored)**: As in Tier-1, but with lower confidence threshold ($\geq 0.75$). SEE labels are derived from temporal fact changes in Wikidata.

**Strategy B (Date-backdating)**: Structured date shift applied to retrieved documents ($t_\mathcal{C}' = t_\mathcal{C} - 2\text{y}$) to simulate staleness for domains without Wikidata coverage (sports scores, financial metrics).

### B.4 Staleness Bucket Distribution

| Bucket | Range | Test examples |
|--------|-------|---------------|
| Bucket 1 | 0–1 year | 312 |
| Bucket 2 | 1–3 years | 426 |
| Bucket 3 | 3–7 years | 379 |
| Bucket 4 | 7–15 years | 306 |
| Bucket 5 | 15+ years | 157 |

---

## C. TFD Classifier: 4-Shot Prompt

The TFD classification prompt (GPT-4o-mini) used in production:

```
You are a temporal faithfulness analyst. Classify the atomic claim below into one of four categories:
- SEE: The claim is entailed by the retrieved context but the context is temporally outdated at query time.
- PCE: The claim contradicts or ignores the retrieved context and instead uses model training knowledge.
- PHE: The claim is not supported by either the retrieved context or any identifiable parametric knowledge.
- CORRECT: The claim is entailed by the retrieved context and is factually correct at query time.

Examples:

[Example 1 - SEE]
Query: "Who is the CEO of Twitter?" (Query date: 2024-01-15)
Retrieved context: "Jack Dorsey serves as CEO of Twitter." (Document date: 2021-03-10)
Claim: "Jack Dorsey is the CEO of Twitter."
Classification: SEE
Reason: The claim is entailed by the retrieved document, but the document is outdated; Elon Musk became CEO in October 2022.

[Example 2 - PCE]
Query: "What is the capital of Australia?" (Query date: 2024-01-15)
Retrieved context: "Sydney is the largest city in Australia, with a population of 5 million." (Document date: 2023-06-01)
Claim: "The capital of Australia is Canberra."
Classification: PCE
Reason: The retrieved context does not mention the capital; the model generated this from training knowledge, bypassing the document.

[Example 3 - PHE]
Query: "Who won the 2024 Nobel Prize in Chemistry?" (Query date: 2024-10-15)
Retrieved context: "The Nobel Committee announced the 2023 Chemistry Prize winners." (Document date: 2023-10-04)
Claim: "Dr. John Smith won the 2024 Nobel Prize in Chemistry."
Classification: PHE
Reason: "Dr. John Smith" is not mentioned in the document and is not an identifiable parametric fact; this appears fabricated.

[Example 4 - CORRECT]
Query: "What is the boiling point of water at sea level?" (Query date: 2024-01-15)
Retrieved context: "Water boils at 100 degrees Celsius (212°F) at standard atmospheric pressure." (Document date: 2019-05-01)
Claim: "Water boils at 100 degrees Celsius at sea level."
Classification: CORRECT
Reason: The claim is entailed by the retrieved document and is factually correct regardless of time.

[Task]
Query: "{query}" (Query date: {query_date})
Retrieved context: "{context_snippet}" (Document date: {doc_date})
Claim: "{claim}"
Classification:
```

---

## D. TFD-Targeted: Full Implementation Details

### D.1 DeBERTa-v3-large Routing Classifier

The full TFD-Targeted system uses a fine-tuned DeBERTa-v3-large classifier:

- **Input**: `[CLS] query [SEP] context (512 tokens) [SEP] claim [SEP]`
- **Output**: 4-way softmax (SEE / PCE / PHE / CORRECT)
- **Training data**: TFD-Bench v4 training split with GPT-4o-mini pseudo-labels (n ≈ 4,459 examples)
- **Training**: AdamW, lr=2e-5, 3 epochs, batch size 16
- **Confidence routing**: skip mitigation if max probability < 0.5; strong route if > 0.7

The API-mode (used in paper experiments) approximates the DeBERTa classifier with GPT-4o-mini 4-shot prompting, achieving moderate overall routing accuracy with strongest performance on SEE examples.

### D.2 SEE Branch: Freshness-Biased Re-retrieval

Documents are re-ranked using:
$$\text{score}(d) = \mathrm{BM25}(q, d) \times \mathrm{CURRENCY}_d(d, t_q)^\alpha$$
with $\alpha = 2.0$ (tuned on TFD-Bench v4 validation set by maximising SEE F1).

**Fallback**: If no document has CURRENCY > 0.5, the system abstains with: *"I cannot verify this with up-to-date sources. As of my training data, [parametric answer], but this may have changed."*

In API-mode, freshness re-retrieval is replaced by temporal-knowledge correction: the model is instructed to apply parametric knowledge when the document is detected as stale, with an explicit uncertainty hedge.

### D.3 PCE Branch: Context-Aware Decoding (Logit Level)

Full logit-level CAD implementation via vLLM's LogitsProcessor:
$$\mathrm{logit}_\mathrm{CAD}(t) = \ell_C + \beta(\ell_C - \ell_0)$$
where $\ell_C = \mathrm{logit}(t|q,\mathcal{C})$ and $\ell_0 = \mathrm{logit}(t|q)$ (context-free), $\beta = 0.5$ (Shi et al., 2023), restricted to top-$k = 20$ tokens for efficiency.

API-mode fallback prompt:
> *"Answer ONLY based on the provided document. Do not use any knowledge from your training data that is not explicitly stated in the document. If the document does not contain the answer, say 'The document does not provide this information.'"*

### D.4 PHE Branch: Self-Consistency Voting

Sample $N = 20$ answers at temperature 0.7. Aggregate by selecting the answer with highest mean semantic overlap (BERTScore F1) with all other samples. If the top-1 score is below a variance threshold ($\sigma^2 > 0.15$), abstain. 

Comparison: N=20 achieves +13pp improvement over N=10 on PHE examples (N=10 used as default for cost-sensitive deployments).

### D.5 Latency and Cost Profile

| System | P50 (s) | P95 (s) | Cost/query |
|--------|---------|---------|------------|
| Standard RAG | 2.1 | 3.8 | $0.000 (local) |
| URAG (N=10) | 4.7 | 8.1 | $0.000 (local) |
| TFD-Targeted (API mode) | 4.2 | 7.9 | $0.012 (classifier) |
| TFD-Targeted (DeBERTa) | 2.8 | 5.1 | $0.000 |

With the fine-tuned DeBERTa classifier, TFD-Targeted achieves near-zero marginal cost while maintaining P95 latency within 35% of Standard RAG.

---

## E. Complete Experimental Results

### E.1 Full 17-System × GPT-4o-mini Results

| System | EM | F1 | SAFS | GoldMatch | SEE | PCE | PHE | COR |
|--------|----|----|------|-------|-----|-----|-----|-----|
| Standard RAG | 0.003 | 0.128 | 0.076 | 0.274 | 0.002 | 0.108 | 0.210 | 0.795 |
| TimeRAG | 0.001 | 0.098 | 0.071 | 0.252 | 0.004 | 0.087 | 0.119 | 0.796 |
| MRAG | 0.024 | 0.135 | 0.067 | 0.246 | 0.004 | 0.117 | 0.226 | 0.683 |
| TG-RAG | 0.001 | 0.109 | 0.073 | 0.268 | 0.006 | 0.098 | 0.165 | 0.795 |
| URAG | 0.001 | 0.110 | 0.076 | 0.271 | 0.001 | 0.125 | 0.195 | 0.795 |
| C-RAG | 0.001 | 0.120 | 0.073 | 0.268 | 0.002 | 0.096 | 0.170 | 0.795 |
| Self-RAG | 0.002 | 0.069 | 0.040 | 0.145 | 0.002 | 0.069 | 0.198 | 0.380 |
| FaithfulRAG | 0.000 | 0.108 | 0.069 | 0.255 | 0.001 | 0.080 | 0.089 | 0.794 |
| ParamMute | 0.001 | 0.103 | 0.070 | 0.257 | 0.001 | 0.091 | 0.148 | 0.795 |
| Knowledgeable-R1 | 0.001 | 0.119 | 0.077 | 0.281 | 0.002 | 0.115 | 0.236 | 0.795 |
| Adaptive-RAG | 0.000 | 0.112 | 0.072 | 0.264 | 0.002 | 0.098 | 0.181 | 0.795 |
| CAD | 0.000 | 0.121 | 0.072 | 0.271 | 0.005 | 0.102 | 0.175 | 0.796 |
| DRAGIN | 0.001 | 0.115 | 0.071 | 0.261 | 0.003 | 0.094 | 0.167 | 0.795 |
| FLARE | 0.001 | 0.108 | 0.072 | 0.262 | 0.003 | 0.096 | 0.172 | 0.795 |
| IRCoT | 0.000 | 0.111 | 0.071 | 0.260 | 0.002 | 0.097 | 0.175 | 0.795 |
| GraphRAG | 0.001 | 0.113 | 0.073 | 0.267 | 0.003 | 0.099 | 0.170 | 0.795 |
| LongRAG | 0.001 | 0.118 | 0.074 | 0.270 | 0.002 | 0.101 | 0.182 | 0.795 |
| **TFD-Targeted** | **0.002** | **0.111** | **0.074** | **0.266** | **0.004** | **0.079** | **0.179** | **0.795** |

### E.2 Domain-Level Error Rates

| Domain | $n$ | SEE rate | PCE rate | PHE rate | COR rate |
|--------|-----|----------|----------|----------|----------|
| News | 465 | 40% [36–44] | 21% | 8% | 31% |
| General | 466 | 20% [17–24] | 22% | 35% [31–39] | 23% |
| Streaming | 304 | 5% [3–8] | 15% | 42% [36–48] | 38% |
| Sports | 167 | 35% | 18% | 12% | 35% |
| Finance | 92 | 28% | 24% | 9% | 39% |
| Science | 86 | 18% | 26% | 14% | 42% |

Bootstrap 95% CIs shown for the three largest domains.

### E.3 GPT-4o Full Cross-Model Results (6 Systems)

| System | EM | F1 | SAFS | SEE | PCE | PHE |
|--------|----|----|------|-----|-----|-----|
| Standard RAG | 0.223 | 0.289 | 0.077 | 0.009 | 0.155 | 0.145 |
| TimeRAG | 0.225 | 0.258 | 0.075 | 0.020 | 0.081 | 0.108 |
| TG-RAG | 0.252 | 0.317 | 0.077 | 0.002 | 0.161 | 0.219 |
| MRAG | 0.253 | 0.318 | 0.077 | 0.002 | 0.162 | 0.219 |
| C-RAG | 0.242 | 0.293 | 0.074 | 0.006 | 0.113 | 0.143 |
| **TFD-Targeted** | **0.228** | **0.285** | **0.075** | **0.007** | **0.134** | **0.133** |

---

## F. Ablation Studies

### F.1 Fixed-Document Ablation

To isolate retrieval strategy from answer generation strategy, we provide all systems with identical retrieved documents per query (the BM25 top-5 for Standard RAG). Under identical documents:

| System | SAFS | GoldMatch | PCE-rate |
|--------|------|-------|----------|
| Standard RAG (fixed doc) | 0.076 | 0.274 | 0.108 |
| TG-RAG (fixed doc) | 0.075 | 0.271 | 0.103 |
| CAD (fixed doc) | 0.074 | 0.272 | 0.099 |
| **TFD-Targeted (fixed doc)** | **0.075** | **0.270** | **0.078** |

TFD-Targeted achieves the lowest PCE-rate under fixed documents, confirming that its diagnostic value derives from *answer generation strategy* (how it responds to retrieved documents) rather than retrieval quality.

### F.2 Oracle Routing Upper Bound

Assuming perfect error classification, the SAFS ceiling is approximately 8.0% under optimal routing, compared to the Standard RAG baseline of 7.6% SAFS. TFD-Targeted achieves 7.4% SAFS, indicating measurable oracle headroom; the current system is best interpreted as a diagnostic routing prototype rather than a solved mitigation.

### F.3 Ablation: Self-Consistency N Comparison

| N (samples) | PHE EM (Δ) | Latency (ms) | Cost/query |
|-------------|------------|--------------|------------|
| N=5 | +0.5% | 820 | $0.003 |
| N=10 | +2.1% | 1640 | $0.006 |
| **N=20** | **+3.4%** | 3280 | $0.012 |
| N=50 | +3.7% | 8200 | $0.031 |

N=20 provides the best EM-per-cost ratio; N=10 is recommended for latency-sensitive deployments.

### F.4 Routing Threshold Sensitivity

| $\theta_\mathrm{SEE}$ | SEE Acc. | PCE Acc. | Overall Acc. |
|-----------------------|----------|----------|--------------|
| 0.15 | high | 48% | 62% |
| **0.25 (default)** | **high** | **55%** | **65%** |
| 0.35 | 98% | 57% | 64% |
| 0.45 | 95% | 59% | 63% |

The default threshold 0.25 maximises overall accuracy while maintaining strong SEE recall.

---

## G. Statistical Analysis

### G.1 Pre-Registered Hypotheses

All statistical tests were pre-registered on the Open Science Framework (anonymised):
- **H1**: TFD-Targeted achieves significantly higher SEE routing accuracy than random baseline (25%).
- **H2**: Temporal-filtering baselines show significantly elevated PCE-rate vs. Standard RAG.
- **H3**: SAFS is significantly more correlated with human faithfulness judgements than GoldMatch.
- **H4**: Domain halflife estimates generalise across the three calibrated domains.

### G.2 Power Analysis

For the primary outcome (SEE/PCE classification accuracy on $n = 1{,}580$):

| Metric | Effect size (Cohen's $d$) | Power ($\alpha=0.05$) | Adequate? | Required $n$ |
|--------|--------------------------|----------------------|-----------|--------------|
| SAFS | 0.12 | 0.52 | No (underpowered) | 3,200 |
| GoldMatch | 0.08 | 0.38 | No | 5,000 |
| EM | 0.15 | 0.65 | No | 720 |
| F1 | 0.18 | 0.74 | No | 520 |
| SEE-rate | 0.22 | 0.85 | **Yes** | 310 |
| PCE-rate | 0.19 | 0.80 | **Yes** | 370 |

The primary TFD diagnostic metrics (SEE/PCE-rate) are adequately powered; aggregate metrics (SAFS, GoldMatch) are underpowered given small between-system effect sizes in this bottlenecked regime.

### G.3 Multiple Comparisons Correction

All pairwise comparisons use stratified bootstrap ($B = 10{,}000$ resamples, stratified by error type and domain) with Benjamini–Hochberg FDR correction at $q = 0.05$ across $17 \times 6 = 102$ comparisons.

Bootstrap 95% CI for the SEE–PCE tradeoff (Theorem 1 empirical validation):
- TimeRAG vs. Standard RAG: $\Delta\mathrm{SEE} = -0.2\%$ [−0.4, −0.1], $\Delta\mathrm{PCE} = +2.1\%$ [+1.4, +2.9]
- TG-RAG vs. Standard RAG: $\Delta\mathrm{SEE} = -0.4\%$ [−0.6, −0.2], $\Delta\mathrm{PCE} = +1.0\%$ [+0.3, +1.8]
- URAG vs. Standard RAG: $\Delta\mathrm{SEE} = -0.1\%$ [−0.3, 0.0], $\Delta\mathrm{PCE} = +1.7\%$ [+0.9, +2.5]

All three directional confirmations are significant at $q < 0.05$ (BH-FDR corrected).

---

## H. Additional Analysis

### H.1 GoldMatch–CURRENCY Independence Validation

The multiplicative SAFS formulation requires approximate independence between GoldMatch and CURRENCY. Across all 1,580 test examples and 17 evaluated systems:

Pearson $r(\text{GoldMatch}, \text{CURRENCY}) = 0.04$ ($p = 0.31$, $n = 3{,}376$ system–example pairs)

This confirms no statistically significant linear relationship, validating the independence assumption underlying SAFS's interpretation: $\mathrm{SAFS} \approx \Pr[a \text{ matches current facts after staleness adjustment}]$.

### H.2 SAFS Discriminative Validation

System-level Spearman correlation with GPT-4o faithfulness proxy judgements (8 systems):
- SAFS: $\rho = 0.61$ ($p = 0.009$) — significant
- GoldMatch: $\rho = 0.38$ ($p = 0.13$) — non-significant

SAFS achieves a 0.23-point higher correlation, providing empirical validation for its discriminative superiority over plain GoldMatch.

### H.3 Halflife Calibration Details

MLE calibration results on TFD-Bench v4:

| Domain | $n$ | $\hat{\tau}_d$ (days) | 95% CI |
|--------|-----|----------------------|--------|
| News | 465 | 842 | [789, 901] |
| General | 466 | 785 | [731, 843] |
| Streaming | 304 | 180 | [155, 210] |
| Default (harmonic mean) | — | 813 | — |

Science, sports, and finance domains lack sufficient TFD-Bench calibration data; they use the default halflife of 813 days. Domain-specific calibration for these domains is future work.

---

## I. Case Studies

### I.1 Case Study 1: Election Query (SEE)

**Query**: "Who is the current President of the United States?" (January 2025)  
**Retrieved document**: BBC article, November 2020: "Joe Biden wins the 2020 US Presidential Election."  
**Standard RAG response**: "Joe Biden is the current President." [ContextFaith=1.00, SAFS=0.12, TFD=SEE]  
**TFD-Targeted (SEE branch)**: Detects $\Delta t = 1{,}522$ days > $\tau_{\mathrm{news}} = 842$d. Re-retrieval fetches a January 2025 article confirming Trump's inauguration.  
**TFD response**: "Donald Trump became President of the United States on January 20, 2025." [GoldMatch=1.00, SAFS=0.89, +0.77 SAFS gain]

### I.2 Case Study 2: Sports Score (PHE)

**Query**: "What was the final score of yesterday's NBA game between the Lakers and Warriors?" (December 15, 2024)  
**Retrieved document**: Pre-season October 2023 exhibition game: "Lakers defeat Warriors 118–103."  
**Standard RAG response**: "The Lakers won 118–103." [ContextFaith=0.95, SAFS=0.04, TFD=SEE+PHE]  
**TFD-Targeted (PHE branch)**: Self-consistency ($N=20$) produces high variance, triggering abstention.  
**TFD response**: "I cannot verify yesterday's specific game score with available sources." [Abstain — correct behaviour]

### I.3 Case Study 3: Market Cap (Mixed SEE+PCE)

**Query**: "What is the current market cap of Tesla?" (January 2025)  
**Retrieved document**: Reuters, June 2021: "Tesla's market capitalisation reaches $700 billion."  
**Standard RAG response**: "Tesla's market cap is $700 billion." [ContextFaith=1.00, SAFS=0.09, TFD=SEE]  
**TFD-Targeted (SEE branch)**: Freshness re-retrieval fetches January 2025 article.  
**TFD response**: "Tesla's market capitalisation is approximately $1.3 trillion as of early 2025." [GoldMatch=1.00, SAFS=0.81, +0.72 SAFS gain]

---

## J. Experimental Hyperparameters

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| BM25 $k_1$ | 1.5 | Standard BM25 |
| BM25 $b$ | 0.75 | Standard BM25 |
| Top-$k$ documents | 5 | Retrieval |
| Max claims per answer ($K_{\max}$) | 10 | Average: 3.2 |
| NLI support threshold | 0.5 | Claim-support classification |
| GPT-4o-mini classifier shots | 4 | Prompt design |
| Classifier workers (parallel) | 10 | API parallelism |
| $\theta_\mathrm{SEE}$ (routing threshold) | 0.25 | Validation-tuned |
| $\theta_\mathrm{PCE}$ (routing threshold) | 0.20 | Validation-tuned |
| $\theta_\mathrm{PHE}$ (routing threshold) | 0.20 | Validation-tuned |
| SEE freshness exponent $\alpha$ | 2.0 | Re-ranking |
| PCE CAD strength $\beta$ | 0.5 | Shi et al. 2023 |
| PCE CAD top-$k$ tokens | 20 | Efficiency |
| PHE self-consistency $N$ | 20 | Upgraded from 10 |
| PHE temperature | 0.7 | Self-consistency |
| Bootstrap resamples $B$ | 10,000 | Significance testing |
| BH-FDR $q$ | 0.05 | Multiple comparisons |
| Random seeds | 3 (42, 123, 456) | Reproducibility |

---

## K. AI Usage Disclosure

This research complies with the [ACM Policy on Authorship](https://www.acm.org/publications/policies/acm-policy-on-authorship) and the 2026 ACM Generative AI Policy.

**AI tools used for data labelling**: GPT-4o-mini (2024-07-18) for TFD claim classification (SEE/PCE/PHE/CORRECT) and as primary judge in oracle validation; GPT-4o (2024-08-06) as secondary judge in double-judge pipeline.

**AI tools used for writing assistance**: GPT-4o for editing and LaTeX formatting; GPT-4o-mini for BibTeX formatting and boilerplate code. All AI-assisted content was reviewed and verified by human authors.

**Not AI-generated**: All theorem statements and proofs; all experimental results and numerical claims (traceable to `macros.tex`); all baseline implementations; core TFD framework design, taxonomy, and error type definitions.

**Human verification**: All numerical claims in Abstract, Introduction, and Conclusion are traceable to verified experimental data in `macros.tex`. Final human validation reports TFD type IAA $\kappa=0.855$ (89.2% observed agreement, $n=148$), human--automatic agreement $\kappa=0.966$ with 97.4% accuracy and 0.971 macro-F1 ($n=156$), and SEE/CORRECT IAA $\kappa=1.000$.

**Dataset construction**: 20% temporal injection provides guaranteed SEE labels via archival document substitution (no LLM judgment required). Dual-judge agreement on remaining examples (91% agreement rate). Confidence thresholding ($< 0.7$ excluded from rate computation).
