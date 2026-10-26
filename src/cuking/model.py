"""Model and tokenizer builder with alias resolution for IndoNLU benchmarks."""

from typing import Dict, Optional, Tuple
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from cuking.utils import get_hf_token


# Official model alias mapping according to PRD section 7
MODEL_ALIASES: Dict[str, str] = {
    "indobert-base": "indobenchmark/indobert-base-p1",
    "indobert-large": "indobenchmark/indobert-large-p1",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "tiny-smoke": "hf-internal-testing/tiny-random-BertForSequenceClassification",
}

SMOKE_MODEL_ID = "hf-internal-testing/tiny-random-BertForSequenceClassification"


def resolve_model_id(alias_or_id: str) -> str:
    """Resolve user alias to Hugging Face model repository ID."""
    return MODEL_ALIASES.get(alias_or_id.lower(), alias_or_id)


def build_model_and_tokenizer(
    model_name_or_alias: str,
    num_labels: int,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    is_smoke: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Build tokenizer and sequence classification model.

    Args:
        model_name_or_alias: Model alias (e.g. 'indobert-base') or Hugging Face ID.
        num_labels: Number of classification classes.
        id2label: Optional mapping from label index to string.
        label2id: Optional mapping from label string to index.
        is_smoke: If True, override model with ultra-lightweight smoke test model.

    Returns:
        Tuple of (model, tokenizer).
    """
    token = get_hf_token()
    model_id = SMOKE_MODEL_ID if is_smoke else resolve_model_id(model_name_or_alias)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    # Load base config without kwargs to avoid length-mismatch warnings
    # with IndoBERT's default pre-trained config (which defaults to 5 labels).
    config = AutoConfig.from_pretrained(model_id, token=token)
    config.num_labels = num_labels

    if id2label is not None:
        config.id2label = {int(k): str(v) for k, v in id2label.items()}
    else:
        config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

    if label2id is not None:
        config.label2id = {str(k): int(v) for k, v in label2id.items()}
    else:
        config.label2id = {v: k for k, v in config.id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        token=token,
        ignore_mismatched_sizes=is_smoke,
    )

    return model, tokenizer
