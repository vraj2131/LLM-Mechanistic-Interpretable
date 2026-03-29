"""
Build pointwise relevance assessment prompts for Qwen2.5-1.5B-Instruct.

The prompt asks the model to rate a (query, document) pair on a 0-3 scale.
The template is loaded from configs/reranker.yaml so it can be swapped for
Phase 9 robustness experiments without touching source code.

Usage:
    from src.reranking.prompt_builder import build_prompt, build_chat_messages
    messages = build_chat_messages(query, doc_title, doc_text)
    # Pass messages to tokenizer.apply_chat_template(...)
"""

from __future__ import annotations

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

_MAX_DOC_CHARS = 512 * 4  # ~512 tokens at ~4 chars/token


def _load_reranker_cfg():
    return load_config("configs/reranker.yaml")


def build_chat_messages(
    query: str,
    doc_title: str,
    doc_text: str,
    variant: str | None = None,
    cfg=None,
) -> list[dict[str, str]]:
    """Return a chat-format message list for the relevance prompt.

    Args:
        query: Query string.
        doc_title: Document title (may be empty string).
        doc_text: Document body text; truncated to ~512 tokens.
        variant: Optional prompt variant name from configs/reranker.yaml
                 (e.g. "no_rubric", "scale_0_10", "flipped_order").
                 None uses the default template.
        cfg: Pre-loaded OmegaConf config (avoids re-reading yaml in loops).

    Returns:
        List of {"role": ..., "content": ...} dicts ready for
        tokenizer.apply_chat_template().
    """
    if cfg is None:
        cfg = _load_reranker_cfg()

    # Truncate doc to avoid exceeding context window
    doc_text_trunc = doc_text[:_MAX_DOC_CHARS]

    if variant is not None:
        template = cfg.prompt_variants[variant]
    else:
        template = cfg.user_prompt_template

    user_content = template.format(
        query=query,
        doc_title=doc_title,
        doc_text=doc_text_trunc,
    )

    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user",   "content": user_content},
    ]


def build_prompt(
    query: str,
    doc_title: str,
    doc_text: str,
    tokenizer,
    variant: str | None = None,
    cfg=None,
) -> str:
    """Return the fully formatted prompt string (after chat template applied).

    Args:
        query: Query string.
        doc_title: Document title.
        doc_text: Document body text.
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        variant: Optional prompt variant name.
        cfg: Pre-loaded OmegaConf config.

    Returns:
        Prompt string with chat template applied and add_generation_prompt=True.
    """
    messages = build_chat_messages(query, doc_title, doc_text, variant=variant, cfg=cfg)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompts_for_pairs(
    pairs_df,
    tokenizer,
    variant: str | None = None,
    cfg=None,
) -> list[str]:
    """Vectorized prompt builder over a DataFrame of (query, doc) pairs.

    Args:
        pairs_df: DataFrame with columns query_text, doc_title, doc_text.
        tokenizer: HuggingFace tokenizer.
        variant: Optional prompt variant.
        cfg: Pre-loaded OmegaConf config.

    Returns:
        List of prompt strings aligned with pairs_df rows.
    """
    if cfg is None:
        cfg = _load_reranker_cfg()

    prompts = []
    for row in pairs_df.itertuples(index=False):
        prompt = build_prompt(
            query=row.query_text,
            doc_title=getattr(row, "doc_title", ""),
            doc_text=row.doc_text,
            tokenizer=tokenizer,
            variant=variant,
            cfg=cfg,
        )
        prompts.append(prompt)

    log.info(f"Built {len(prompts)} prompts (variant={variant!r})")
    return prompts
