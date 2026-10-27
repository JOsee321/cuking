"""Model and tokenizer builder with alias resolution for IndoNLU benchmarks."""

import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
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


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    decay: float = 0.95,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Build layer-wise learning rate decay (LLRD) parameter groups for BERT / XLM-R models.

    Embeddings receive the smallest learning rate: base_lr * (decay ** num_layers).
    The i-th encoder layer (1-indexed, i=1..num_layers) receives:
        base_lr * (decay ** (num_layers - i))
    The classification head receives:
        base_lr
    Bias and LayerNorm parameters have weight_decay = 0.0.

    Args:
        model: PyTorch model (BERT or XLM-R / RoBERTa).
        base_lr: Base learning rate for classification head.
        decay: Multiplicative decay factor per layer (e.g. 0.95).
        weight_decay: L2 regularization factor for weight parameters.

    Returns:
        List of parameter group dictionaries compatible with PyTorch optimizers.
    """
    # 1. Determine number of hidden layers in encoder
    num_layers = None
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = int(model.config.num_hidden_layers)
    else:
        max_layer = -1
        for name, _ in model.named_parameters():
            match = re.search(r"encoder\.layer\.(\d+)\.", name)
            if match:
                max_layer = max(max_layer, int(match.group(1)))
        num_layers = max_layer + 1 if max_layer >= 0 else 1

    levels: Dict[int, Dict[str, List[torch.nn.Parameter]]] = {}

    # 2. Assign parameters to levels
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "embeddings" in name:
            level = 0
        else:
            match = re.search(r"encoder\.layer\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))  # 0 to num_layers - 1
                level = layer_idx + 1             # 1 to num_layers
            else:
                level = num_layers + 1           # Classification head

        name_lower = name.lower()
        is_no_decay = ("bias" in name_lower) or ("layernorm" in name_lower) or ("layer_norm" in name_lower)

        if level not in levels:
            levels[level] = {"decay": [], "nodecay": []}

        if is_no_decay:
            levels[level]["nodecay"].append(param)
        else:
            levels[level]["decay"].append(param)

    # 3. Build ordered parameter groups
    param_groups: List[Dict[str, Any]] = []
    for lvl in sorted(levels.keys()):
        if lvl == 0:
            lr = base_lr * (decay ** num_layers)
        elif lvl <= num_layers:
            # i = lvl (1-indexed layer)
            lr = base_lr * (decay ** (num_layers - lvl))
        else:
            lr = base_lr

        if levels[lvl]["decay"]:
            param_groups.append({
                "params": levels[lvl]["decay"],
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if levels[lvl]["nodecay"]:
            param_groups.append({
                "params": levels[lvl]["nodecay"],
                "lr": lr,
                "weight_decay": 0.0,
            })

    return param_groups
