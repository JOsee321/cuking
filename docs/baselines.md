# Baseline IndoNLU dan Leaderboard CodaLab

Dokumen ini diisi oleh manusia (Task H6). Agent TIDAK diperbolehkan mengarang angka baseline.
Sumber: Paper IndoNLU (arXiv 2009.05387) dan Leaderboard CodaLab IndoNLU.

## 1. Baseline Paper IndoNLU

Metrik resmi:
- Klasifikasi (EmoT, SmSA): Macro-F1 (dan Accuracy).
- NER (NERGrit, NERP): Entity-level F1 (seqeval).

### EmoT (Emotion Twitter - 5 kelas)

| Model | Valid Macro-F1 | Valid Acc | Test Macro-F1 | Test Acc | Sumber / Catatan |
|---|---|---|---|---|---|
| indobert-base-p1 | | | | | Paper IndoNLU Tabel 4 |
| indobert-large-p1 | | | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-base | | | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-large | | | | | Paper IndoNLU Tabel 4 |

### SmSA (Sentiment Analysis Prosa - 3 kelas)

| Model | Valid Macro-F1 | Valid Acc | Test Macro-F1 | Test Acc | Sumber / Catatan |
|---|---|---|---|---|---|
| indobert-base-p1 | | | | | Paper IndoNLU Tabel 4 |
| indobert-large-p1 | | | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-base | | | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-large | | | | | Paper IndoNLU Tabel 4 |

### NERGrit (NER Grit)

| Model | Valid Entity-F1 | Test Entity-F1 | Sumber / Catatan |
|---|---|---|---|
| indobert-base-p1 | | | Paper IndoNLU Tabel 4 |
| indobert-large-p1 | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-base | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-large | | | Paper IndoNLU Tabel 4 |

### NERP (NER Prosa)

| Model | Valid Entity-F1 | Test Entity-F1 | Sumber / Catatan |
|---|---|---|---|
| indobert-base-p1 | | | Paper IndoNLU Tabel 4 |
| indobert-large-p1 | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-base | | | Paper IndoNLU Tabel 4 |
| xlm-roberta-large | | | Paper IndoNLU Tabel 4 |

---

## 2. Skor Teratas Leaderboard CodaLab (Snapshot Saat Ini)

Tanggal pengecekan: [Diisi manusia]
Link leaderboard: https://competitions.codalab.org/competitions/26808

| Task | Ranking 1 Submitter | Test Metric | Nilai | Catatan Pendekatan |
|---|---|---|---|---|
| EmoT | | Test Macro-F1 | | |
| SmSA | | Test Macro-F1 | | |
| NERGrit | | Test Entity-F1 | | |
| NERP | | Test Entity-F1 | | |
