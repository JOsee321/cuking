---
language:
- id
license: apache-2.0
tags:
- nlu
- indonesian
- classification
- ner
- transformers
datasets:
- indonlu
metrics:
- f1
- accuracy
---

# Model Card: Cuking

## Deskripsi Model

Cuking adalah model Natural Language Understanding (NLU) untuk Bahasa Indonesia yang di-finetune dari model dasar pretrained open-source (IndoBERT / XLM-RoBERTa). Pipeline training dioptimasi menggunakan disiplin eksperimen yang ketat (hyperparameter sweep, multi-seed, ensemble, dan layer-wise learning rate decay) dengan budget komputasi Rp0.

- Dikembangkan oleh: Rafi Wildan (JOsee321)
- Jenis model: Transformer Encoder (Sequence Classification / Token Classification)
- Bahasa: Bahasa Indonesia (`id`)
- Lisensi model dasar: Apache-2.0 / MIT
- Repositori: https://github.com/JOsee321/cuking

## Data Pelatihan dan Evaluasi

Model dievaluasi menggunakan benchmark IndoNLU:
- Task: [Diisi sesuai model rilis: EmoT / SmSA / NERGrit / NERP]
- Sumber dataset: Benchmark IndoNLU (indobenchmark/indonlu)
- Pembagian data: Train, Validation, Test sesuai standar IndoNLU

## Metode Pelatihan

- Model dasar: [Nama model Hugging Face dasar]
- Optimizer: AdamW (weight decay 0.01)
- Learning rate: [LR terpilih]
- Scheduler: Linear warmup (warmup ratio 0.1)
- Batch size: Effective batch size 16 (gradient accumulation)
- Presisi: Mixed precision FP16
- Regularisasi / Teknik khusus: Early stopping (patience 3), Layer-wise Learning Rate Decay (LLRD), Ensemble

## Hasil Evaluasi

Catatan: Skor validation resmi bersumber dari file `metrics.json` hasil run nyata.

| Task | Metrik | Baseline Paper | Cuking Validation | Cuking Test (CodaLab) |
|---|---|---|---|---|
| [Task] | Macro-F1 / Entity-F1 | [Baseline] | [Validation] | [CodaLab Test] |

## Cara Penggunaan

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "JOsee321/cuking-[model]-[task]"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Teks bahasa Indonesia yang akan dianalisis."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)

print(probs)
```

## Keterbatasan dan Bias

- Model dioptimalkan khusus untuk domain data benchmark IndoNLU dan teks informal/formal Bahasa Indonesia.
- Performa pada dialek daerah di luar Bahasa Indonesia baku atau ragam Twitter/Prosa mungkin bervariasi.
- Prediksi tidak boleh digunakan sebagai satu-satunya dasar pengambilan keputusan kritis tanpa tinjauan manusia.

## Kutipan

Jika menggunakan model ini atau pipeline Cuking, silakan merujuk:
- Repositori Cuking: https://github.com/JOsee321/cuking
- Paper IndoNLU: Wilie et al., 2020 (arXiv:2009.05387)
