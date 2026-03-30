# Mid-Project Checkpoint Report
**Project:** Mechanistic Interpretability of LLM Rerankers
**Model:** Qwen/Qwen2.5-1.5B-Instruct
**Datasets:** BEIR SciFact (primary), BEIR NFCorpus (OOD)
**Date:** 2026-03-30

---

## 1. BM25 Baseline Metrics — SciFact

| Metric | BM25 |
|--------|------|
| nDCG@10 | 0.5597 |
| MRR@10 | 0.5242 |
| Recall@20 | 0.7370 |

BM25 uses Okapi BM25 (k1=1.5, b=0.75, whitespace tokenization) retrieving top-20 candidates per query over 300 test queries.

---

## 2. LLM Reranker Metrics vs BM25

| Metric | BM25 | Qwen2.5-1.5B | Δ |
|--------|------|--------------|---|
| nDCG@10 | 0.5597 | 0.5817 | +0.0220 |
| MRR@10 | 0.5242 | 0.5537 | +0.0295 |
| Recall@20 | 0.7370 | 0.7370 | 0.0000 |

**Implementation note:** We use logit-based scoring (single forward pass extracting P(0)..P(3) from the lm_head at the final input token). The expected score `E[score] = 0·P(0) + 1·P(1) + 2·P(2) + 3·P(3)` is used as the continuous ranking signal. This is 8-10× faster than autoregressive generation and eliminates the regex fallback problem entirely.

Qwen2.5-1.5B achieves a modest +2.2pp nDCG@10 improvement over BM25 on SciFact. On NFCorpus (OOD), the reranker scores nDCG@10=0.2381 vs BM25=0.2666 (−2.8pp), consistent with expected domain shift degradation for a small model.

---

## 3. Score Parser Fallback Rate

**Fallback rate: 0.0%** (threshold: <5%) ✓

The logit-based approach directly computes P(0)..P(3) from the vocabulary logits — there is no text generation step and therefore no regex parsing or fallback. All 6,000 SciFact pairs and 6,460 NFCorpus pairs received valid scores.

Score distribution (SciFact):

| Score | Count | % |
|-------|-------|---|
| 0 | 256 | 4.3% |
| 1 | 4,934 | 82.2% |
| 2 | 694 | 11.6% |
| 3 | 116 | 1.9% |

Relevant pairs receive higher expected scores than non-relevant pairs, confirming the model's scoring correlates with ground-truth relevance.

---

## 4. Probe Heatmap — First Mechanistic Result

Linear probes (Ridge for regression targets, Logistic for binary targets) were trained on the residual stream activations at the decision token position across all 28 layers × 7 IR targets = 196 probes per dataset.

**SciFact peak probe scores:**

| Target | Metric | Peak Score | 95% CI | Peak Layer |
|--------|--------|-----------|--------|-----------|
| `doc_length_bucket` | AUROC | 0.9976 | [0.996, 0.999] | 3 |
| `is_relevant` | AUROC | 0.9709 | [0.954, 0.984] | 17 |
| `bm25_score` | R² | 0.7147 | [0.684, 0.740] | 17 |
| `query_term_freq` | R² | 0.4466 | [0.398, 0.491] | 20 |
| `lexical_overlap` | R² | 0.4374 | [0.391, 0.478] | 19 |
| `relevance_label` | R² | 0.3022 | [0.166, 0.409] | 19 |
| `bm25_rank` | R² | 0.0694 | [0.021, 0.119] | 17 |

**NFCorpus peak probe scores (OOD generalization):**

| Target | Metric | Peak Score | Peak Layer |
|--------|--------|-----------|-----------|
| `doc_length_bucket` | AUROC | 0.9994 | 1 |
| `is_relevant` | AUROC | 0.8861 | 17 |
| `bm25_score` | R² | 0.8141 | 17 |
| `lexical_overlap` | R² | 0.7912 | 17 |
| `query_term_freq` | R² | 0.5516 | 16 |
| `relevance_label` | R² | 0.2392 | 20 |
| `bm25_rank` | R² | 0.1459 | 20 |

---

## 5. Layer Analysis — Where Signals Emerge

**Key finding: Layer 17 is the dominant peak for IR-relevant features.**

- **Layers 0-4:** Surface features emerge. `doc_length_bucket` reaches AUROC=0.99 by layer 3 — document length is a trivial feature captured in the earliest layers.
- **Layers 5-14:** Gradual build-up. `bm25_score` rises from R²≈0.3 at layer 5 to R²≈0.6 at layer 14. `is_relevant` builds similarly.
- **Layers 15-20:** Peak performance for all IR targets. `is_relevant` AUROC=0.97, `bm25_score` R²=0.71, `lexical_overlap` R²=0.44 all peak in this window. This is the model's "relevance assessment zone."
- **Layers 21-27:** Slight decay after peak. Signals remain strong but drop marginally, consistent with later layers shifting focus toward output formatting.

The consistent peak at **layer 17** across both SciFact and NFCorpus is a strong cross-dataset generalization result, suggesting this is a stable property of Qwen2.5-1.5B's relevance processing.

---

## 6. SAE Target Layer Selection

**Original plan:** layers 7, 14, 21
**Updated decision:** layers **7, 17, 21**

Rationale:
- **Layer 7** (keep): Early representations — lexical and surface features beginning to form. Useful contrast point.
- **Layer 14** → **Layer 17** (update): Layer 14 is pre-peak; layer 17 is the confirmed peak from probe results for `is_relevant` (AUROC 0.97) and `bm25_score` (R² 0.71). SAE at layer 17 will decompose the most informative representations.
- **Layer 21** (keep): Post-peak consolidation, close to output. Captures how the model formats its final relevance decision.

---

## 7. Open Questions, Blockers, Next Steps

### Open Questions

1. **bm25_rank vs bm25_score gap:** The model encodes BM25 *score* well (R²=0.71) but BM25 *rank* weakly (R²=0.07). The model appears to represent the magnitude of lexical overlap, not the ordinal position of a document in a candidate list. This may limit rank-based interventions in Phase 8.

2. **Graded relevance:** `relevance_label` R²=0.30 with wide CI [0.17, 0.41]. It's unclear whether the model encodes genuine graded relevance or primarily learns a threshold (binary). NFCorpus (graded 0/1/2 labels) shows similar R²=0.24 — suggesting the model's representation is closer to binary than graded.

3. **OOD improvement for lexical features:** NFCorpus shows *stronger* `lexical_overlap` R²=0.79 vs SciFact's 0.44. NFCorpus has longer queries and more lexical variation, which may make the lexical overlap feature more linearly separable in activation space.

### Blockers
None. All go/no-go criteria pass. Data, activations, and probe weights are ready for Phase 7.

### Next Steps

**Phase 7 — SAE Training (layers 7, 17, 21):**
- Architecture: TopK SAE, input_dim=1536, expansion_factor=8 (hidden_dim=12288), k=32
- Training: MSE loss only (TopK enforces sparsity), Adam lr=1e-4, 50 epochs
- Training data: all sequence positions across all prompts (~60-120K vectors per layer)
- Targets: dead features <5%, reconstruction MSE monotonically decreasing
- Key output: for each SAE feature, find top-20 activating (query, doc) pairs and correlate with IR probe targets

**Phase 8 — Causal Interventions (layer 17 focus):**
- Probe-direction steering: inject ±α·w into residual stream at decision token
- Alpha values: ±1, ±3, ±5 × ‖w‖ (dose-response curve)
- SAE feature steering and ablation using features correlated with `is_relevant`
- Metric: ΔnDCG@10 and ΔMRR@10 with paired t-test across 300 SciFact queries

---

## Go/No-Go Decision

| Criterion | Result | Status |
|-----------|--------|--------|
| Reranker non-trivial nDCG@10 | 0.5817 (vs BM25 0.5597) | ✓ GO |
| Score parser fallback <5% | 0.0% (logit scoring) | ✓ GO |
| ≥3/7 targets R²>0.1 or AUROC>0.6 | 6/7 targets passing (bm25_rank R²=0.069 misses by 0.03) | ✓ GO |
| Activation cache validated | Both datasets: shape, NaN, alignment ✓ | ✓ GO |

**Overall: ✓ GO — proceed to Phase 7 (SAE Training)**
