"""Unit tests for model and tokenizer construction."""

import pytest
from cuking.model import (
    MODEL_ALIASES,
    SMOKE_MODEL_ID,
    build_model_and_tokenizer,
    resolve_model_id,
)


def test_alias_resolution():
    """Verify alias mapping to Hugging Face repository IDs."""
    assert resolve_model_id("indobert-base") == MODEL_ALIASES["indobert-base"]
    assert resolve_model_id("xlmr-base") == MODEL_ALIASES["xlmr-base"]
    assert resolve_model_id("custom/my-model") == "custom/my-model"


def test_smoke_allows_mismatched_sizes():
    """Verify smoke mode allows mismatched sizes for ultra-fast dummy model."""
    model, tokenizer = build_model_and_tokenizer(
        model_name_or_alias="tiny-smoke",
        num_labels=5,
        is_smoke=True,
    )
    assert model.num_labels == 5
    assert tokenizer is not None


def test_non_smoke_restricts_mismatched_sizes():
    """Verify non-smoke run fails hard when checkpoint weights mismatch without override."""
    # SMOKE_MODEL_ID has a 2-label classification head in its checkpoint.
    # Requesting 5 labels with is_smoke=False MUST fail with a RuntimeError / Exception.
    with pytest.raises(Exception):
        build_model_and_tokenizer(
            model_name_or_alias=SMOKE_MODEL_ID,
            num_labels=5,
            is_smoke=False,
        )
