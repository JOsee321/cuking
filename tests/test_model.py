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


def test_explicit_label_maps_and_logits_shape():
    """Verify explicit label mapping and that logits shape matches class counts (SmSA 3, EmoT 5)."""
    from cuking.data import TASK_LABELS

    for task_name, expected_num in [("smsa", 3), ("emot", 5)]:
        labels = TASK_LABELS[task_name]
        assert len(labels) == expected_num
        id2label = {i: lbl for i, lbl in enumerate(labels)}
        label2id = {lbl: i for i, lbl in enumerate(labels)}

        model, tokenizer = build_model_and_tokenizer(
            model_name_or_alias="tiny-smoke",
            num_labels=expected_num,
            id2label=id2label,
            label2id=label2id,
            is_smoke=True,
        )

        assert model.config.num_labels == expected_num
        assert model.config.id2label == id2label
        assert model.config.label2id == label2id

        # Dummy batch of 2 samples
        inputs = tokenizer(["Teks contoh satu", "Teks contoh dua"], padding=True, return_tensors="pt")
        outputs = model(**inputs)
        assert outputs.logits.ndim == 2
        assert outputs.logits.shape == (2, expected_num)


def test_llrd_param_groups_monotonic_and_isolated():
    """Verify LLRD assigns every param to exactly 1 group, lr is monotonic, and no_decay is accurate."""
    from cuking.model import build_llrd_param_groups

    model, _ = build_model_and_tokenizer(
        model_name_or_alias="tiny-smoke",
        num_labels=2,
        is_smoke=True,
    )

    base_lr = 2e-5
    decay = 0.90
    weight_decay = 0.01

    param_groups = build_llrd_param_groups(
        model=model,
        base_lr=base_lr,
        decay=decay,
        weight_decay=weight_decay,
    )

    # 1. Every parameter is in exactly one group
    all_group_params = [p for g in param_groups for p in g["params"]]
    all_group_param_ids = [id(p) for p in all_group_params]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    assert len(all_group_param_ids) == len(set(all_group_param_ids)), "Duplicate parameter found across groups!"
    assert len(all_group_params) == len(trainable_params), "Not all trainable parameters were assigned to groups!"

    # 2. Learning rate increases monotonically from embeddings to head
    lrs = [g["lr"] for g in param_groups]
    for i in range(len(lrs) - 1):
        assert lrs[i] <= lrs[i + 1] + 1e-12, f"LR not monotonic non-decreasing at index {i}: {lrs[i]} > {lrs[i+1]}"

    # Embeddings (first groups) has smallest lr = base_lr * (decay ** num_layers)
    num_layers = model.config.num_hidden_layers
    expected_embed_lr = base_lr * (decay ** num_layers)
    assert abs(lrs[0] - expected_embed_lr) < 1e-9

    # Head (last groups) has base_lr
    assert abs(lrs[-1] - base_lr) < 1e-9

    # 3. Parameter weight decay grouping is strictly correct
    param_to_name = {id(p): name for name, p in model.named_parameters()}
    for g in param_groups:
        wd = g["weight_decay"]
        if wd == 0.0:
            for p in g["params"]:
                name = param_to_name[id(p)].lower()
                assert ("bias" in name) or ("layernorm" in name) or ("layer_norm" in name), (
                    f"Parameter {param_to_name[id(p)]} has weight_decay=0 but is not bias or LayerNorm!"
                )
        else:
            assert wd == weight_decay
            for p in g["params"]:
                name = param_to_name[id(p)].lower()
                assert not (("bias" in name) or ("layernorm" in name) or ("layer_norm" in name)), (
                    f"Parameter {param_to_name[id(p)]} has weight_decay={wd} but is bias or LayerNorm!"
                )


def test_cuking_trainer_llrd_activation(tmp_path):
    """Verify CukingTrainer configures LLRD optimizer when llrd: true."""
    from transformers import TrainingArguments
    from cuking.train import CukingTrainer

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_alias="tiny-smoke",
        num_labels=2,
        is_smoke=True,
    )

    out_dir = tmp_path / "trainer_out"
    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=3e-5,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = CukingTrainer(
        model=model,
        args=args,
        llrd=True,
        llrd_decay=0.85,
    )

    optimizer = trainer.create_optimizer()
    assert optimizer is not None
    assert len(optimizer.param_groups) > 1

    # Verify optimizer group lrs match LLRD calculation
    min_lr = min(g["lr"] for g in optimizer.param_groups)
    max_lr = max(g["lr"] for g in optimizer.param_groups)
    assert abs(max_lr - 3e-5) < 1e-9
    assert min_lr < max_lr
