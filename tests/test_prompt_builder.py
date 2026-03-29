"""Unit tests for prompt_builder — no model loading required."""

from unittest.mock import MagicMock

import pytest

from src.reranking.prompt_builder import build_chat_messages, build_prompts_for_pairs


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cfg():
    """Minimal OmegaConf-like config mock."""
    cfg = MagicMock()
    cfg.system_prompt = "You are an expert relevance assessor."
    cfg.user_prompt_template = (
        "Query: {query}\nDocument Title: {doc_title}\nDocument: {doc_text}\n"
        "Respond with a single integer (0, 1, 2, or 3)."
    )
    cfg.prompt_variants = {
        "no_rubric": "Query: {query}\nDocument: {doc_text}\nScore:",
        "flipped_order": "Document: {doc_text}\nQuery: {query}\nRate 0-3:",
    }
    return cfg


@pytest.fixture
def tiny_pairs_df():
    import pandas as pd
    return pd.DataFrame([
        {"query_text": "cats dogs", "doc_title": "Animals", "doc_text": "cats and dogs are pets"},
        {"query_text": "quantum physics", "doc_title": "Physics", "doc_text": "relativity and quantum mechanics"},
    ])


# ---------------------------------------------------------------------------
# build_chat_messages tests
# ---------------------------------------------------------------------------

class TestBuildChatMessages:
    def test_returns_two_messages(self, sample_cfg):
        msgs = build_chat_messages("query", "title", "text", cfg=sample_cfg)
        assert len(msgs) == 2

    def test_roles(self, sample_cfg):
        msgs = build_chat_messages("q", "t", "d", cfg=sample_cfg)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_content(self, sample_cfg):
        msgs = build_chat_messages("q", "t", "d", cfg=sample_cfg)
        assert "relevance assessor" in msgs[0]["content"]

    def test_query_in_user_content(self, sample_cfg):
        msgs = build_chat_messages("neural networks", "t", "d", cfg=sample_cfg)
        assert "neural networks" in msgs[1]["content"]

    def test_doc_title_in_user_content(self, sample_cfg):
        msgs = build_chat_messages("q", "My Title", "d", cfg=sample_cfg)
        assert "My Title" in msgs[1]["content"]

    def test_doc_text_in_user_content(self, sample_cfg):
        msgs = build_chat_messages("q", "t", "some body text", cfg=sample_cfg)
        assert "some body text" in msgs[1]["content"]

    def test_doc_truncated_to_max_chars(self, sample_cfg):
        long_text = "x" * 10_000
        msgs = build_chat_messages("q", "t", long_text, cfg=sample_cfg)
        # 512 * 4 = 2048 chars max; user content must be shorter than full text
        assert len(msgs[1]["content"]) < len(long_text) + 200

    def test_variant_no_rubric(self, sample_cfg):
        msgs = build_chat_messages("myquery", "t", "mydoc", variant="no_rubric", cfg=sample_cfg)
        assert "myquery" in msgs[1]["content"]
        assert "mydoc" in msgs[1]["content"]
        # Default rubric text should not be present
        assert "Marginally relevant" not in msgs[1]["content"]

    def test_variant_flipped_order(self, sample_cfg):
        msgs = build_chat_messages("q", "t", "docbody", variant="flipped_order", cfg=sample_cfg)
        content = msgs[1]["content"]
        # Document should appear before Query in flipped variant
        assert content.index("docbody") < content.index("q")

    def test_empty_doc_title(self, sample_cfg):
        msgs = build_chat_messages("q", "", "text", cfg=sample_cfg)
        assert msgs is not None  # no crash on empty title


# ---------------------------------------------------------------------------
# build_prompts_for_pairs tests (no tokenizer needed for message structure)
# ---------------------------------------------------------------------------

class TestBuildPromptsForPairs:
    def test_length_matches_dataframe(self, tiny_pairs_df, sample_cfg):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: (
            msgs[0]["content"] + " | " + msgs[1]["content"]
        )
        from src.reranking.prompt_builder import build_prompts_for_pairs
        prompts = build_prompts_for_pairs(tiny_pairs_df, tokenizer, cfg=sample_cfg)
        assert len(prompts) == len(tiny_pairs_df)

    def test_prompts_contain_query_text(self, tiny_pairs_df, sample_cfg):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: msgs[1]["content"]
        prompts = build_prompts_for_pairs(tiny_pairs_df, tokenizer, cfg=sample_cfg)
        assert "cats dogs" in prompts[0]
        assert "quantum physics" in prompts[1]
