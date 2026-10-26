"""Unit tests for dataset loaders (EmoT and SmSA)."""

from cuking.data import load_task, TASK_LABELS


def test_load_emot():
    """Verify EmoT dataset split row counts, columns, and labels."""
    dataset = load_task("emot", data_dir="data")

    assert set(dataset.keys()) == {"train", "validation", "test"}

    # Exact row counts from IndoNLU repository
    assert len(dataset["train"]) == 3521
    assert len(dataset["validation"]) == 440
    assert len(dataset["test"]) == 440

    # Column checks
    for split in ["train", "validation", "test"]:
        assert "text" in dataset[split].column_names
        assert "label" in dataset[split].column_names
        assert "label_text" in dataset[split].column_names

    # Check label set matches specification
    expected_labels = set(TASK_LABELS["emot"])
    actual_train_labels = set(dataset["train"]["label_text"])
    assert actual_train_labels == expected_labels

    # Check label IDs are non-negative integers up to 4
    train_ids = set(dataset["train"]["label"])
    assert train_ids == {0, 1, 2, 3, 4}


def test_load_smsa():
    """Verify SmSA dataset split row counts, columns, and labels."""
    dataset = load_task("smsa", data_dir="data")

    assert set(dataset.keys()) == {"train", "validation", "test"}

    # Exact row counts from IndoNLU repository
    assert len(dataset["train"]) == 11000
    assert len(dataset["validation"]) == 1260
    assert len(dataset["test"]) == 500

    # Column checks
    for split in ["train", "validation", "test"]:
        assert "text" in dataset[split].column_names
        assert "label" in dataset[split].column_names
        assert "label_text" in dataset[split].column_names

    # Check label set matches specification
    expected_labels = set(TASK_LABELS["smsa"])
    actual_train_labels = set(dataset["train"]["label_text"])
    assert actual_train_labels == expected_labels

    # Check label IDs are non-negative integers up to 2
    train_ids = set(dataset["train"]["label"])
    assert train_ids == {0, 1, 2}


def test_masked_test_splits():
    """Verify masked test split loading for CodaLab submission format."""
    emot_masked = load_task("emot", data_dir="data", use_masked_test=True)
    assert len(emot_masked["test"]) == 440

    smsa_masked = load_task("smsa", data_dir="data", use_masked_test=True)
    assert len(smsa_masked["test"]) == 500
