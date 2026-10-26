# Baseline IndoNLU dan Leaderboard CodaLab

Dokumen ini memuat skor baseline resmi yang diambil langsung dari paper IndoNLU (arXiv:2009.05387) dan status pengecekan leaderboard CodaLab. Seluruh angka bersumber dari dokumen yang dibuka langsung dalam sesi ini tanpa estimasi atau rekayasa data.

---

## 1. Sumber Data dan Referensi

- Paper Referensi: IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding (Bryan Wilie et al.)
- URL Paper: https://arxiv.org/abs/2009.05387 (PDF: https://arxiv.org/pdf/2009.05387.pdf)
- Nomor Tabel di Paper:
  - Tabel 5 (Halaman 7, Bagian 5.1): Results of baseline models with best performing configuration on the IndoNLU benchmark.
  - Tabel 22 (Halaman 15, Lampiran E): Results of all experiments conducted in IndoNLU benchmark.
  - Tabel 1 (Halaman 3, Bagian 3.1): Task statistics and descriptions.
- Tanggal Akses: 26 Oktober 2026

---

## 2. Pemetaan Kolom Tabel Paper ke Task

Pada Tabel 5 dan Tabel 22 paper IndoNLU, kolom hasil eksperimen dikelompokkan ke dalam dua kategori utama: Classification dan Sequence Labeling.

Urutan kolom pada header tabel paper dan pemetaannya adalah sebagai berikut:

### Kategori Classification
1. EmoT: Emotion Classification (Twitter colloquial, 5 kelas: anger, fear, happiness, love, sadness)
2. SmSA: Sentiment Analysis (Prosa, 3 kelas: positive, negative, neutral)
3. CASA: Aspect-based Sentiment Analysis (Automobile, 6 aspek, 3 label sentimen)
4. HoASA: Aspect-based Sentiment Analysis (Hotel AiryRooms, 10 aspek, 4 label sentimen)
5. WReTE: Wiki Revision Edits Textual Entailment (Wiki formal, 2 kelas: entail, not entail)
6. AVG: Rata-rata skor Macro-F1 untuk seluruh task Classification

### Kategori Sequence Labeling
7. POSP: Part-of-Speech Tagging (News formal, 26 tag)
8. BaPOS: Part-of-Speech Tagging (PAN Localization formal, 41 tag)
9. TermA: Span Extraction (Hotel colloquial, aspect dan sentiment span)
10. KEPS: Keyphrase Extraction (Twitter banking, IOB chunking)
11. NERGrit: Named Entity Recognition (Grit-ID formal: PER, PLA, ORG)
12. NERP: Named Entity Recognition (News formal: PER, LOC, IND, EVT, FNB)
13. FacQA: Question Answering Factoid Span Extraction (News formal, 6 kategori pertanyaan)
14. AVG: Rata-rata skor Sequence Labeling

### Metrik dan Split yang Digunakan Paper
- Metrik Resmi: Macro-averaged F1 (Macro-F1) untuk seluruh task klasifikasi termasuk EmoT dan SmSA (dijelaskan pada Bagian 3.4 Evaluation Metrics).
- Split Evaluasi: Split Test (data uji resmi benchmark IndoNLU).
- Status Metrik Lain di Paper:
  - Skor Valid Macro-F1: TIDAK DITEMUKAN di paper (paper hanya melaporkan skor test set pada tabel hasil).
  - Skor Valid Accuracy: TIDAK DITEMUKAN di paper (paper tidak menggunakan accuracy sebagai metrik evaluasi IndoNLU).
  - Skor Test Accuracy: TIDAK DITEMUKAN di paper (paper hanya melaporkan Macro-F1 untuk seluruh task klasifikasi).

---

## 3. Tabel Baseline Model Paper IndoNLU

### A. Tabel Baseline EmoT (Emotion Classification)

Berikut adalah skor untuk task EmoT dari seluruh model yang dilaporkan pada Tabel 5 (konfigurasi terbaik) dan variasi eksperimen Tabel 22:

#### Dari Tabel 5 (Konfigurasi Terbaik Tiap Arsitektur Model)
| Model Baseline | Test Macro-F1 | Test Acc | Valid Macro-F1 | Valid Acc | Keterangan Model di Paper |
|---|---|---|---|---|---|
| Scratch | 57.31 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | Non-pretrained Transformer (4 layers) |
| fastText-cc-id | 65.36 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | fastText Common Crawl Indonesia (4 layers) |
| fastText-indo4b | 69.23 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | fastText Indo4B corpus (4 layers) |
| mBERT | 67.30 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | bert-base-multilingual-uncased (12 layers) |
| XLM-MLM | 65.75 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-mlm-100-1280 (16 layers) |
| XLM-RBASE | 71.15 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-roberta-base (12 layers) |
| XLM-RLARGE | 78.51 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-roberta-large (24 layers) |
| IndoBERT-liteBASE (phase 1) | 73.88 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-base-p1 (seq len 128) |
| IndoBERT-liteBASE (+ phase two) | 72.27 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-base-p2 (seq len 512) |
| IndoBERTBASE (phase 1) | 75.48 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-base-p1 (seq len 128) |
| IndoBERTBASE (+ phase two) | 76.28 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-base-p2 (seq len 512) |
| IndoBERT-liteLARGE (phase 1) | 75.19 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-large-p1 (seq len 128) |
| IndoBERT-liteLARGE (+ phase two) | 70.80 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-large-p2 (seq len 512) |
| IndoBERTLARGE (phase 1) | 77.08 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-large-p1 (seq len 128) |
| IndoBERTLARGE (+ phase two) | 79.47 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-large-p2 (seq len 512) |

#### Dari Tabel 22 (Seluruh Eksperimen dan Variasi Hyperparameter)
| Nama Model di Tabel 22 | Learning Rate | Layers | Param | EmoT Test Macro-F1 |
|---|---|---|---|---|
| scratch | 1e-4 | 2 | 38.6M | 58.51 |
| scratch | 1e-4 | 4 | 52.8M | 57.31 |
| scratch | 1e-4 | 6 | 67.0M | 52.84 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 6 | 15.1M | 67.43 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 4 | 10.7M | 65.36 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 2 | 6.3M | 64.74 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 6 | 15.1M | 68.47 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 4 | 10.7M | 69.23 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 2 | 6.3M | 70.97 |
| indobert-lite-base-128-112.5k | 1e-5 | 12 | 11.7M | 73.88 |
| indobert-lite-base-128-191.5k | 1e-5 | 12 | 11.7M | 71.95 |
| indobert-lite-base-512-162.5k | 1e-5 | 12 | 11.7M | 72.27 |
| indobert-base-128 | 4e-5 | 12 | 124.5M | 75.48 |
| indobert-base-512 | 1e-5 | 12 | 124.5M | 76.61 |
| indobert-base-512 | 4e-5 | 12 | 124.5M | 76.28 |
| indobert-lite-large-128 | 1e-5 | 24 | 17.7M | 75.19 |
| indobert-lite-large-512 | 1e-5 | 24 | 17.7M | 71.67 |
| indobert-lite-large-512 | 2e-5 | 24 | 17.7M | 70.80 |
| indobert-large-128-1100k | 4e-5 | 24 | 335.2M | 77.04 |
| indobert-large-128-1000k | 4e-5 | 24 | 335.2M | 77.08 |
| indobert-large-512-1100k | 4e-5 | 24 | 335.2M | 77.39 |
| indobert-large-512-1100k | 3e-5 | 24 | 335.2M | 79.47 |
| bert-base-multilingual-uncased | 1e-5 | 12 | 167.4M | 67.30 |
| xlm-mlm-100-1280 | 1e-5 | 16 | 573.2M | 65.75 |
| xlm-roberta-base | 2e-5 | 12 | 278.7M | 71.15 |
| xlm-roberta-large | 1e-5 | 24 | 561.0M | 78.51 |

---

### B. Tabel Baseline SmSA (Sentiment Analysis Prosa)

Berikut adalah skor untuk task SmSA dari seluruh model yang dilaporkan pada Tabel 5 (konfigurasi terbaik) dan variasi eksperimen Tabel 22:

#### Dari Tabel 5 (Konfigurasi Terbaik Tiap Arsitektur Model)
| Model Baseline | Test Macro-F1 | Test Acc | Valid Macro-F1 | Valid Acc | Keterangan Model di Paper |
|---|---|---|---|---|---|
| Scratch | 67.35 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | Non-pretrained Transformer (4 layers) |
| fastText-cc-id | 76.92 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | fastText Common Crawl Indonesia (4 layers) |
| fastText-indo4b | 82.13 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | fastText Indo4B corpus (4 layers) |
| mBERT | 84.14 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | bert-base-multilingual-uncased (12 layers) |
| XLM-MLM | 86.33 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-mlm-100-1280 (16 layers) |
| XLM-RBASE | 91.39 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-roberta-base (12 layers) |
| XLM-RLARGE | 92.35 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | xlm-roberta-large (24 layers) |
| IndoBERT-liteBASE (phase 1) | 90.85 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-base-p1 (seq len 128) |
| IndoBERT-liteBASE (+ phase two) | 90.29 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-base-p2 (seq len 512) |
| IndoBERTBASE (phase 1) | 87.73 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-base-p1 (seq len 128) |
| IndoBERTBASE (+ phase two) | 87.66 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-base-p2 (seq len 512) |
| IndoBERT-liteLARGE (phase 1) | 88.66 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-large-p1 (seq len 128) |
| IndoBERT-liteLARGE (+ phase two) | 88.61 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobert-lite-large-p2 (seq len 512) |
| IndoBERTLARGE (phase 1) | 92.72 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-large-p1 (seq len 128) |
| IndoBERTLARGE (+ phase two) | 92.03 | TIDAK DITEMUKAN | TIDAK DITEMUKAN | TIDAK DITEMUKAN | indobenchmark/indobert-large-p2 (seq len 512) |

#### Dari Tabel 22 (Seluruh Eksperimen dan Variasi Hyperparameter)
| Nama Model di Tabel 22 | Learning Rate | Layers | Param | SmSA Test Macro-F1 |
|---|---|---|---|---|
| scratch | 1e-4 | 2 | 38.6M | 64.22 |
| scratch | 1e-4 | 4 | 52.8M | 67.35 |
| scratch | 1e-4 | 6 | 67.0M | 67.07 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 6 | 15.1M | 78.84 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 4 | 10.7M | 76.92 |
| fasttext-cc-id-300-no-oov-uncased | 1e-4 | 2 | 6.3M | 76.71 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 6 | 15.1M | 83.07 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 4 | 10.7M | 82.13 |
| fasttext-4B-id-300-no-oov-uncased | 1e-4 | 2 | 6.3M | 83.63 |
| indobert-lite-base-128-112.5k | 1e-5 | 12 | 11.7M | 90.85 |
| indobert-lite-base-128-191.5k | 1e-5 | 12 | 11.7M | 89.87 |
| indobert-lite-base-512-162.5k | 1e-5 | 12 | 11.7M | 90.29 |
| indobert-base-128 | 4e-5 | 12 | 124.5M | 87.73 |
| indobert-base-512 | 1e-5 | 12 | 124.5M | 90.90 |
| indobert-base-512 | 4e-5 | 12 | 124.5M | 87.66 |
| indobert-lite-large-128 | 1e-5 | 24 | 17.7M | 88.66 |
| indobert-lite-large-512 | 1e-5 | 24 | 17.7M | 90.13 |
| indobert-lite-large-512 | 2e-5 | 24 | 17.7M | 88.61 |
| indobert-large-128-1100k | 4e-5 | 24 | 335.2M | 93.71 |
| indobert-large-128-1000k | 4e-5 | 24 | 335.2M | 92.72 |
| indobert-large-512-1100k | 4e-5 | 24 | 335.2M | 92.90 |
| indobert-large-512-1100k | 3e-5 | 24 | 335.2M | 92.03 |
| bert-base-multilingual-uncased | 1e-5 | 12 | 167.4M | 84.14 |
| xlm-mlm-100-1280 | 1e-5 | 16 | 573.2M | 86.33 |
| xlm-roberta-base | 2e-5 | 12 | 278.7M | 91.39 |
| xlm-roberta-large | 1e-5 | 24 | 561.0M | 92.35 |

---

## 4. Batas Lolos Phase 1

Model acuan evaluasi Phase 1 adalah checkpoint indobenchmark/indobert-base-p1 (pada paper: IndoBERTBASE phase 1 / indobert-base-128).

Batas minimum kelulusan yang dapat diterima adalah skor paper dikurangi 2.00 poin F1:

| Task | Skor Paper indobert-base-p1 (Test Macro-F1) | Batas Minimum Lolos (-2.00 Poin) | Metrik |
|---|---|---|---|
| **EmoT** | 75.48 | 73.48 | Macro-F1 |
| **SmSA** | 87.73 | 85.73 | Macro-F1 |

Kriteria kelulusan: Jika evaluasi model Phase 1 pada split test menghasilkan skor di bawah batas minimum (EmoT < 73.48 atau SmSA < 85.73), pipeline Phase 1 belum memenuhi kriteria reproduktibilitas baseline dan harus diinvestigasi sebelum beralih ke fase berikutnya.

---

## 5. Target Phase 3

Target Phase 3 adalah melampaui skor tertinggi di antara seluruh baseline model paper IndoNLU per task:

### A. Task EmoT
- Skor Tertinggi di Paper: **79.47** (Test Macro-F1)
- Model Pencapai: IndoBERTLARGE (+ phase two) / indobert-large-512-1100k dengan learning rate 3e-5 (Tabel 5 dan Tabel 22).
- Target Phase 3 Cuking: Macro-F1 > 79.47

### B. Task SmSA
- Skor Tertinggi di Tabel 5 (Konfigurasi Utama): **92.72** (Test Macro-F1) oleh IndoBERTLARGE phase 1 (indobert-large-128-1000k LR 4e-5).
- Skor Tertinggi di Tabel 22 (Pencarian Grid Lengkap): **93.71** (Test Macro-F1) oleh indobert-large-128-1100k LR 4e-5.
- Target Phase 3 Cuking: Macro-F1 > 93.71 (atau minimal melampaui 92.72).

---

## 6. Catatan Perbandingan Split Data dan Metrik

1. **Split yang Dilaporkan Paper**: Seluruh angka hasil eksperimen pada Tabel 5 dan Tabel 22 paper IndoNLU dievaluasi pada **split test** (test set resmi benchmark).
2. **Peran Split Validation di Paper**: Split validation digunakan oleh penulis paper hanya untuk early stopping (pemberhentian awal) dan pemilihan checkpoint terbaik selama proses fine-tuning (Bagian 3.3.2 paper). Angka skor pada split validation tidak dimuat pada tabel hasil paper.
3. **Peringatan Perbandingan (Valid vs Test)**: Skor validasi Cuking (skor evaluasi pada split validasi lokal/Kaggle selama training) **TIDAK BOLEH dibandingkan langsung** dengan skor test paper tanpa label yang jelas dan transparan. Perbandingan apel-ke-apel hanya sah jika dilakukan pada split yang sama (yaitu hasil inferensi split test Cuking dibandingkan terhadap skor split test paper).

---

## 7. Leaderboard CodaLab IndoNLU

- URL Target: https://competitions.codalab.org/competitions/26537 (Tab Results)
- Status Akses: BELUM TERVERIFIKASI: leaderboard butuh login
- Keterangan Detail: Percobaan akses ke https://competitions.codalab.org/competitions/26537 pada tanggal 26 Oktober 2026 mengalami connection timeout (host server tidak merespons atau sistem CodaLab lama telah membatasi akses publik dan memerlukan autentikasi login pengguna). Sesuai aturan keras proyek, dilarang menebak kredensial, mencoba bypass autentikasi, atau mengisi angka estimasi pengganti.
