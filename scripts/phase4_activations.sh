#!/usr/bin/env bash
# Phase 4 — Extract activations for SciFact + NFCorpus sequentially.
# Run with caffeinate to prevent Mac sleep:
#   caffeinate -i bash scripts/phase4_activations.sh
#
# Monitor progress:
#   tail -f /tmp/extract_activations.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/extract_activations.log"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "========================================" | tee "$LOG"
echo "  Phase 4 — Activation Extraction"       | tee -a "$LOG"
echo "  Started: $(date)"                       | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

cd "$PROJECT_ROOT"

echo "" | tee -a "$LOG"
echo "--- [1/2] SciFact ($(date)) ---" | tee -a "$LOG"
"$PYTHON" -m src.activations.extractor --dataset scifact --batch_size 8 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- [2/2] NFCorpus ($(date)) ---" | tee -a "$LOG"
"$PYTHON" -m src.activations.extractor --dataset nfcorpus --batch_size 8 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  ALL DONE: $(date)"                      | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
