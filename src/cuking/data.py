"""Dataset loading and preprocessing utilities for IndoNLU tasks."""

from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import pandas as pd
from datasets import Dataset, DatasetDict


# Base raw URL for IndoNLU dataset repository
RAW_GITHUB_BASE = "https://raw.githubusercontent.com/indobenchmark/indonlu/master/dataset"

# Official task label lists aligned with paper baseline and configs
TASK_LABELS: Dict[str, List[str]] = {
    "emot": ["sadness", "anger", "love", "fear", "happy"],
    "smsa": ["positive", "neutral", "negative"],
}

# Task folder and file mappings
TASK_SPECS: Dict[str, Dict[str, any]] = {
    "emot": {
        "folder": "emot_emotion-twitter",
        "format": "csv",
        "delimiter": ",",
        "has_header": True,
        "text_col": "tweet",
        "label_col": "label",
        "files": {
            "train": "train_preprocess.csv",
            "validation": "valid_preprocess.csv",
            "test": "test_preprocess.csv",
            "test_masked": "test_preprocess_masked_label.csv",
        },
    },
    "smsa": {
        "folder": "smsa_doc-sentiment-prosa",
        "format": "tsv",
        "delimiter": "\t",
        "has_header": False,
        "text_col": 0,
        "label_col": 1,
        "files": {
            "train": "train_preprocess.tsv",
            "validation": "valid_preprocess.tsv",
            "test": "test_preprocess.tsv",
            "test_masked": "test_preprocess_masked_label.tsv",
        },
    },
}


def download_task_data(task_name: str, data_dir: str = "data") -> Path:
    """Download raw dataset files for a task idempotently from GitHub."""
    task_key = task_name.lower()
    if task_key not in TASK_SPECS:
        raise ValueError(f"Task '{task_name}' is not supported. Supported tasks: {list(TASK_SPECS.keys())}")

    spec = TASK_SPECS[task_key]
    dest_dir = Path(data_dir) / spec["folder"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file_key, filename in spec["files"].items():
        file_path = dest_dir / filename
        if not file_path.exists():
            remote_url = f"{RAW_GITHUB_BASE}/{spec['folder']}/{filename}"
            urllib.request.urlretrieve(remote_url, file_path)

    return dest_dir


def load_task(
    task_name: str,
    data_dir: str = "data",
    use_masked_test: bool = False,
) -> DatasetDict:
    """Load an IndoNLU classification task into a Hugging Face DatasetDict.

    Args:
        task_name: Name of the task ('emot' or 'smsa').
        data_dir: Local directory storing raw datasets.
        use_masked_test: If True, load test set with dummy masked labels for CodaLab.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits containing
        'text' (str), 'label' (int), and 'label_text' (str).
    """
    task_key = task_name.lower()
    if task_key not in TASK_SPECS:
        raise ValueError(f"Task '{task_name}' is not supported. Supported tasks: {list(TASK_SPECS.keys())}")

    spec = TASK_SPECS[task_key]
    task_dir = download_task_data(task_key, data_dir=data_dir)

    labels = TASK_LABELS[task_key]
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}

    split_files = {
        "train": spec["files"]["train"],
        "validation": spec["files"]["validation"],
        "test": spec["files"]["test_masked"] if use_masked_test else spec["files"]["test"],
    }

    splits = {}
    for split_name, filename in split_files.items():
        file_path = task_dir / filename
        if spec["has_header"]:
            df = pd.read_csv(file_path, delimiter=spec["delimiter"], encoding="utf-8")
        else:
            df = pd.read_csv(file_path, delimiter=spec["delimiter"], header=None, encoding="utf-8")

        texts = df[spec["text_col"]].astype(str).tolist()
        raw_labels = df[spec["label_col"]].astype(str).tolist()
        label_ids = [label2id.get(lbl, -1) for lbl in raw_labels]

        dataset_dict = {
            "text": texts,
            "label": label_ids,
            "label_text": raw_labels,
        }
        splits[split_name] = Dataset.from_dict(dataset_dict)

    return DatasetDict(splits)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int = 128,
    text_column: str = "text",
) -> DatasetDict:
    """Tokenize text column in a DatasetDict for classification."""
    def _tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize_fn, batched=True)
