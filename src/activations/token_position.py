"""
Centralized decision token position logic.

Every module that needs the "decision token" position — the last input token
before the model generates its relevance score — imports from here.  This
ensures Phase 3 (scoring), Phase 4 (activation extraction), and Phase 8
(interventions) all use exactly the same position.

Strategy: ``last_input``
    Position = input_ids.shape[1] - 1
    (the final token of the prompt, right before generation begins)
"""

import torch


def get_decision_token_pos(input_ids: torch.Tensor) -> int:
    """Return the index of the decision token in the sequence dimension.

    Args:
        input_ids: Tensor of shape (batch, seq_len) or (seq_len,).

    Returns:
        Integer index of the last input token (seq_len - 1).
    """
    if input_ids.dim() == 1:
        return input_ids.shape[0] - 1
    return input_ids.shape[1] - 1


def get_decision_token_pos_batch(input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-sample decision token positions for a padded batch.

    For left-padded batches (as produced by Qwen's tokenizer with
    padding_side='left'), the decision token is always the last column,
    so all positions are identical: seq_len - 1.

    Returns:
        1-D LongTensor of shape (batch,) with all values = seq_len - 1.
    """
    seq_len = input_ids.shape[1]
    return torch.full((input_ids.shape[0],), seq_len - 1, dtype=torch.long)
