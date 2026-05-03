# Presentation Slide Plan
**10-minute talk · 12 content slides · Qwen2.5-1.5B Mechanistic Interpretability**

---

## Slide Deck Plan

---

### Slide 1 — Title
**"Reverse-Engineering Relevance in LLM Rerankers"**
- Subtitle: *Mechanistic Interpretability with Linear Probes, SAEs, and Causal Tests*
- Team names · Course · Date
- One-line teaser: *"What does a 1.5B LLM actually look at when it ranks documents?"*

---

### Slide 2 — Motivation and Context
**Heading:** Motivation and Context

- LLM rerankers are used in RAG pipelines but are black boxes — we don't know *which signals* they use
- Prior work shows probing can find IR signals, but correlation ≠ causation
- **Gap:** no study combines probing + SAE decomposition + causal steering on the same model
- Visual: simple pipeline diagram (BM25 → LLM Reranker → ranked list) with a "?" over the LLM box

---

### Slide 3 — Research/Project Question
**Heading:** Research/Project Question *(exact title required)*

Two questions, clearly stated:
- **RQ1 (Representation):** Which IR signals are linearly encoded in which layers of the reranker?
- **RQ2 (Causality):** Are those signals causally used for ranking — or just correlated?

- Brief note: tested on BEIR SciFact (in-distribution) + NFCorpus (OOD) using Qwen2.5-1.5B

---

### Slide 4 — Methods: Pipeline
**Heading:** Methods and Approach — Pipeline

- Clean 8-phase diagram (horizontal flow):
  BM25 → LLM Reranker → Activation Cache → Feature Eng. → Linear Probes → SAE Training → Causal Interventions
- Key numbers under each box: 300 queries, 6000 pairs, 28 layers, 7 targets, 196 probes, SAE layers 7/17/21
- One sentence each on probing (Ridge/Logistic, 5-fold CV) and SAE (TopK k=64)

---

### Slide 5 — Methods: Causal Intervention Design
**Heading:** Methods and Approach — Causal Interventions

- Explain the two intervention types with a simple diagram:
  - **Probe steering:** add `α × (w/‖w‖)` to hidden state at layer 17 decision token
  - **SAE ablation:** remove feature 30's contribution from residual stream
- Note: split forward-pass design, in-pass baseline, paired t-test for significance
- Why layer 17? → confirmed by probing as the relevance peak (teaser for next slide)

---

### Slide 6 — Results: Baselines + Layer 17 Discovery
**Heading:** Results and Experiments — Baselines

Two things on one slide:
- **Left:** Small version of **Table 1** (BM25 vs Reranker) — SciFact +2.2pp, NFCorpus −2.9pp
- **Right:** **fig4_layer_story.png** (signal emergence curves) — the "aha" moment: layer 17 is the sharp peak for `is_relevant`
- Callout box: *"Layer 17: Is Relevant AUROC = 0.971"*

---

### Slide 7 — Results: What the Model Encodes
**Heading:** Results and Experiments — Layerwise Encoding

- **fig3_probe_heatmaps.png** (probe heatmap, both datasets side by side)
- Below it: a stripped-down version of **Table 2** — just 4 rows (Doc Length, Is Relevant, BM25 Score, Lexical Overlap) with peak scores
- Key annotation: blue dashed line at layer 17, circle at the is_relevant peak
- Highlight: NFCorpus lexical_overlap R²=0.79 vs SciFact 0.43 — foreshadows the OOD result

---

### Slide 8 — Results: SAE — Feature 30
**Heading:** Results and Experiments — Monosemantic Relevance Feature

- **fig8_feature30_cross_dataset.png** (Feature 30 correlation bar chart, both datasets)
- Key finding box: *"Feature 30 is the #1 is_relevant feature on BOTH SciFact (r=+0.382) and NFCorpus (r=+0.292)"*
- Small **Table 3** (trimmed to just is_relevant and doc_length rows) showing cross-dataset feature overlap
- This is the "monosemantic neuron" story — dedicated relevance direction

---

### Slide 9 — Results: Causal Test — Does Steering Change Rankings?
**Heading:** Results and Experiments — Causal Interventions

- **fig9_probe_dose_response.png** (dose-response curves, both datasets overlaid)
- **fig10_cross_dataset_causal.png** (cross-dataset causal comparison bar chart)
- Key result callouts:
  - *SciFact: is_relevant α=+3 → ΔnDCG=+0.0043, p=0.048 ✓*
  - *SciFact: Feature 30 ablation → ΔnDCG=−0.0038, p=0.046 ✓*
  - *NFCorpus: lexical_overlap α=−5 → ΔnDCG=−0.0033, p=0.028 ✓ (OOD shift)*

---

### Slide 10 — Results: Probe Faithfulness Failure
**Heading:** Results and Experiments — Probe Faithfulness

- **fig11_probe_faithfulness.png** (both datasets side by side)
- The punchline: bm25_score has R²=0.71 but **zero causal effect** on both datasets
- Small table showing the contrast:

| Target | Probe R²/AUROC | Causal? | p-value |
|---|---|---|---|
| is_relevant | 0.971 | ✅ Yes | 0.048 |
| lexical_overlap | 0.426 | ⚠️ Weak | 0.028 (NFCorpus) |
| bm25_score | 0.715 | ❌ No | >0.14 |

- One-liner: *"Linear decodability ≠ causal use"*

---

### Slide 11 — Key Takeaway
**Heading:** Key Takeaway *(exact title required)*

4 bullets, large font, clean layout:

1. **Layer 17 is the relevance computation hub** — confirmed by probing (AUROC=0.971) and causal intervention (p=0.048)
2. **Feature 30 is a monosemantic relevance neuron** — top is_relevant SAE feature on both datasets; its ablation causally degrades ranking
3. **OOD domain shift is mechanistically explained** — model switches from semantic relevance to lexical overlap as the dominant causal signal on NFCorpus
4. **Probe faithfulness failure for BM25 score** — strongly encoded (R²=0.71) but not causally used — the model stores BM25 but doesn't route decisions through it

---

### Slide 12 — Limitations and Future Work
**Heading:** Limitations and Future Work

**Limitations (2 bullets):**
- Split forward-pass uses `attention_mask=None` — effect sizes (ΔnDCG≈0.004) are relative to a degraded baseline, not the full model
- Single model (1.5B) — cross-model validation needed to confirm generalisability

**Future work (3 bullets):**
- Multi-model comparison (Mistral-7B) — does layer 17 peak generalise?
- Attention head attribution (activation patching per head) — pinpoint which heads implement the relevance circuit
- Larger alpha sweep with full attention mask — cleaner dose-response curves and absolute effect sizes

---

### Slide 13 — Thank You
Standard thank you + repository link visible

---

### Slide 14 — Team Contributions *(after Thank You)*
Table with each member's contributions

### Slide 15 — GitHub Repository *(after Thank You)*
Repository URL + QR code

---

## Additional Slides (Appendix)

These go after Slide 15 — for the instructor to review, not presented:

| Slide | Content |
|---|---|
| A1 | Full probe peak table (all 7 targets, both datasets, peak score + CI + layer) — **Table 2 full version** |
| A2 | SAE training metrics table (all 3 layers × 2 datasets, val MSE, hidden dim, k) |
| A3 | SAE IR correlation heatmap — **fig7_sae_ir_correlations.png** (top-20 features × 7 targets) |
| A4 | Cross-dataset probe generalization scatter — **fig5_cross_dataset_probe.png** |
| A5 | Score distribution by relevance — **fig2_score_distributions.png** (reranker P(0)..P(3)) |
| A6 | Full significant causal results table (all p-values, all conditions, both datasets) |
