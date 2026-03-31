#!/usr/bin/env bash
# Phase 7 — SAE Training (SciFact + NFCorpus)
# Run with: caffeinate -i bash scripts/phase7_sae.sh 2>&1 | tee logs/phase7_sae.log
#
# Checkpoints:
#   outputs/final/sae_checkpoints/scifact/layer{7,17,21}/
#   outputs/final/sae_checkpoints/nfcorpus/layer{7,17,21}/
#
# Analysis:
#   outputs/final/sae_analysis/scifact/
#   outputs/final/sae_analysis/nfcorpus/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source .venv/bin/activate
mkdir -p logs outputs/final/sae_checkpoints/scifact outputs/final/sae_checkpoints/nfcorpus

echo "============================================================"
echo " Phase 7 — SAE Training (SciFact + NFCorpus)"
echo " $(date)"
echo "============================================================"

# ── Step 1: Extract all-position activations — SciFact ─────────────────────
echo ""
echo ">>> [1/6] Extracting all-position activations — SciFact (layers 7, 17, 21)"
python3 -m src.sae.extractor \
    --dataset scifact \
    --layers 7 17 21 \
    --batch_size 4
echo ">>> SciFact extraction done."

# ── Step 2: Train SAEs — SciFact ───────────────────────────────────────────
for LAYER in 7 17 21; do
    echo ""
    echo ">>> [2/6] Training SAE — SciFact layer ${LAYER}"
    python3 -m src.sae.trainer \
        --dataset scifact \
        --layer "$LAYER" \
        --checkpoint_dir "outputs/final/sae_checkpoints/scifact"
    echo ">>> SciFact layer ${LAYER} done."
done

# ── Step 3: Extract all-position activations — NFCorpus ────────────────────
echo ""
echo ">>> [3/6] Extracting all-position activations — NFCorpus (layers 7, 17, 21)"
python3 -m src.sae.extractor \
    --dataset nfcorpus \
    --layers 7 17 21 \
    --batch_size 4
echo ">>> NFCorpus extraction done."

# ── Step 4: Train SAEs — NFCorpus ──────────────────────────────────────────
for LAYER in 7 17 21; do
    echo ""
    echo ">>> [4/6] Training SAE — NFCorpus layer ${LAYER}"
    python3 -m src.sae.trainer \
        --dataset nfcorpus \
        --layer "$LAYER" \
        --checkpoint_dir "outputs/final/sae_checkpoints/nfcorpus"
    echo ">>> NFCorpus layer ${LAYER} done."
done

# ── Step 5: Feature analysis — SciFact ─────────────────────────────────────
echo ""
echo ">>> [5/6] Feature analysis — SciFact"
python3 - <<'EOF'
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.sae.sae_store import load_sae
from src.sae.feature_analyzer import analyze_features
from src.utils.io import load_parquet

features_df = load_parquet("data/interim/scifact/features.parquet")

for layer in [7, 17, 21]:
    print(f"  Analysing SciFact layer {layer}...")
    sae, meta = load_sae(f"outputs/final/sae_checkpoints/scifact/layer{layer}")
    decision_acts = np.load(f"data/caches/activations/scifact/layer_{layer}.npy", mmap_mode="r")
    result = analyze_features(
        sae=sae,
        acts=decision_acts,
        pairs_df=features_df[["query_id", "doc_id"]],
        features_df=features_df,
        layer=layer,
        out_dir="outputs/final/sae_analysis/scifact",
    )
    print(f"    layer {layer}: active={result['n_active_features']}  "
          f"dead={result['dead_feature_pct']:.1f}%")
    for target, info in result['best_corr_per_target'].items():
        print(f"      {target:<22} best_r={info['pearson_r']:+.3f}  feat={info['feature_idx']}")

print("SciFact feature analysis done.")
EOF

# ── Step 6: Feature analysis — NFCorpus ────────────────────────────────────
echo ""
echo ">>> [6/6] Feature analysis — NFCorpus"
python3 - <<'EOF'
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.sae.sae_store import load_sae
from src.sae.feature_analyzer import analyze_features
from src.utils.io import load_parquet

features_df = load_parquet("data/interim/nfcorpus/features.parquet")

for layer in [7, 17, 21]:
    print(f"  Analysing NFCorpus layer {layer}...")
    sae, meta = load_sae(f"outputs/final/sae_checkpoints/nfcorpus/layer{layer}")
    decision_acts = np.load(f"data/caches/activations/nfcorpus/layer_{layer}.npy", mmap_mode="r")
    result = analyze_features(
        sae=sae,
        acts=decision_acts,
        pairs_df=features_df[["query_id", "doc_id"]],
        features_df=features_df,
        layer=layer,
        out_dir="outputs/final/sae_analysis/nfcorpus",
    )
    print(f"    layer {layer}: active={result['n_active_features']}  "
          f"dead={result['dead_feature_pct']:.1f}%")
    for target, info in result['best_corr_per_target'].items():
        print(f"      {target:<22} best_r={info['pearson_r']:+.3f}  feat={info['feature_idx']}")

print("NFCorpus feature analysis done.")
EOF

echo ""
echo "============================================================"
echo " Phase 7 complete — $(date)"
echo ""
echo " SciFact checkpoints:  outputs/final/sae_checkpoints/scifact/"
echo " NFCorpus checkpoints: outputs/final/sae_checkpoints/nfcorpus/"
echo " SciFact analysis:     outputs/final/sae_analysis/scifact/"
echo " NFCorpus analysis:    outputs/final/sae_analysis/nfcorpus/"
echo "============================================================"
