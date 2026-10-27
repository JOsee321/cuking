# Technical Decisions and Adjustments

Dokumen ini mencatat keputusan teknis, penyesuaian arsitektur, dan perbedaan antara spesifikasi PRD dengan kenyataan di lapangan.

## Format Catatan Keputusan

Setiap keputusan atau penyesuaian dicatat dengan format:
- Tanggal: YYYY-MM-DD
- Konteks: Masalah atau latar belakang
- Keputusan: Solusi atau penyesuaian yang diambil
- Dampak: Pengaruh terhadap pipeline, konfigurasi, atau hasil

---

## Log Keputusan

### DEC-001: Pinned Dependencies and Environment Setup
- Tanggal: 2026-10-26
- Konteks: Penyiapan requirements.txt dan requirements-kaggle.txt pada Phase 0.
- Keputusan: Menggunakan pustaka Transformers 4.46.3, Datasets 3.1.0, Accelerate 1.1.1, Evaluate 0.4.3, dan Seqeval 1.2.2. Pada requirements-kaggle.txt hanya disertakan pustaka tambahan karena PyTorch, NumPy, Pandas, dan Scikit-learn telah tersedia di runtime Kaggle.
- Dampak: Instalasi bersih dan konsisten di lingkungan lokal maupun Kaggle.

### DEC-002: Dataset Formats and Specifications for EmoT and SmSA
- Tanggal: 2026-10-26
- Konteks: Pengecekan langsung file asli dari repo GitHub indobenchmark/indonlu untuk task EmoT dan SmSA sebelum implementasi data loader (Task T1.1).
- Temuan Lapangan:
  1. EmoT (emot_emotion-twitter):
     - Format: CSV dengan delimiter koma (,), encoding UTF-8, memiliki header 'label,tweet'.
     - File dan jumlah baris:
       - train_preprocess.csv: 3.521 baris data (train).
       - valid_preprocess.csv: 440 baris data (validation).
       - test_preprocess.csv: 440 baris data (test).
       - test_preprocess_masked_label.csv: 440 baris data (test dengan masked dummy label).
     - Label (5 kelas): anger, happy, sadness, fear, love.
  2. SmSA (smsa_doc-sentiment-prosa):
     - Format: TSV dengan delimiter tab (\t), encoding UTF-8, tanpa header (header=None). Kolom 0 adalah teks, Kolom 1 adalah label.
     - File dan jumlah baris:
       - train_preprocess.tsv: 11.000 baris data (train).
       - valid_preprocess.tsv: 1.260 baris data (validation).
       - test_preprocess.tsv: 500 baris data (test).
       - test_preprocess_masked_label.tsv: 500 baris data (test dengan masked dummy label).
     - Label (3 kelas): positive, neutral, negative.
- Keputusan: Loader `load_task` di `src/cuking/data.py` membaca format file sesuai karakteristik nyata di atas (CSV berkoma untuk EmoT, TSV bertab tanpa header untuk SmSA), menstandarkan nama kolom menjadi `text` dan `label`, serta mengunduh secara idempoten dari raw GitHub IndoNLU ke folder lokal `data/`.
- Dampak: Dataset termuat secara presisi dan konsisten ke dalam DatasetDict Hugging Face.

### DEC-003: Lightweight Model Selection for Smoke Testing Mode
- Tanggal: 2026-10-26
- Konteks: Laptop lokal menggunakan prosesor Core i3 tanpa GPU diskrit. Mode verifikasi `--smoke` disyaratkan selesai dalam waktu di bawah 5 menit di CPU tanpa menjalankan training penuh.
- Keputusan: Menggunakan model publik ultra-ringan `hf-internal-testing/tiny-random-BertForSequenceClassification` untuk mode `--smoke`. Model ini memiliki arsitektur bert sequence classification riil dengan bobot sangat kecil (< 100 KB) dan 2 layer, sehingga proses tokenisasi, 3 step optimasi AdamW, serta inferensi logits valid dan test selesai dalam waktu sekitar 1 detik di CPU.
- Dampak: Smoke test lokal berjalan super cepat, deterministik, dan memverifikasi seluruh komponen pipeline (data, model, trainer, logits, dan penyimpanan metrik) tanpa membebani memori dan CPU laptop.

### DEC-004: Tokenizer Casing and Lowercase Preprocessing for IndoBERT
- Tanggal: 2026-10-26
- Konteks: Verifikasi perilaku casing tokenizer `indobenchmark/indobert-base-p1` pada input "Halo Dunia SEMUA" sesuai instruksi verifikasi pra-Kaggle.
- Temuan Lapangan:
  1. Tokenizer `indobenchmark/indobert-base-p1` memiliki konfigurasi bawaan `do_lower_case: True`. Hasil tokenisasi "Halo Dunia SEMUA" secara otomatis menghasilkan token berhuruf kecil: `['halo', 'dunia', 'semua']` dengan token ID `[2, 9302, 594, 366, 3]`.
  2. Sebagai perbandingan, tokenizer `xlm-roberta-base` mempertahankan casing asli: `[' Halo', ' Dunia', ' SEM', 'UA']` dengan `do_lower_case: None`.
- Keputusan:
  1. Menambahkan parameter `lowercase: bool = False` pada fungsi `tokenize_dataset` di `src/cuking/data.py` untuk mengontrol konversi teks secara eksplisit.
  2. Pada `src/cuking/train.py`, menerapkan auto-detection: bila model mengandung string "indobert", flag lowercase aktif secara default (kecuali dioverride manual via config), sedangkan model XLM-R tetap mempertahankan casing asli.
  3. Menambahkan opsi `lowercase: null` di `configs/base.yaml` sebagai titik konfigurasi eksplisit.
- Dampak: Penanganan casing menjadi deterministik dan aman dari perbedaan perilaku antar tokenizer di Hugging Face.

### DEC-005: Audit and Documentation of Test Split Usage
- Tanggal: 2026-10-26
- Konteks: Audit verifikasi pemuatan split "test" pada `data.py` dan `train.py` untuk menjamin tidak ada data leakage atau tuning berbasis test set.
- File yang Dimuat sebagai Split Test:
  1. EmoT: `data/emot_emotion-twitter/test_preprocess.csv` (atau `test_preprocess_masked_label.csv` bila opsi `use_masked_test=True`).
  2. SmSA: `data/smsa_doc-sentiment-prosa/test_preprocess.tsv` (atau `test_preprocess_masked_label.tsv` bila opsi `use_masked_test=True`).
- Konfirmasi Integritas Evaluasi:
  1. Early stopping (`EarlyStoppingCallback`) dan seleksi model terbaik (`load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`) pada Hugging Face Trainer murni hanya mengevaluasi `eval_dataset=val_data`.
  2. Split test TIDAK PERNAH digunakan dalam loop training, pemilihan epoch terbaik, penyetelan hyperparameter, maupun pemilihan arsitektur.
  3. Split test hanya dieksekusi satu kali di akhir via `trainer.predict(test_data)` untuk mengekstrak `test_logits.npy` demi keperluan pembuatan submission CodaLab dan ensemble.
- Dampak: Pipeline terverifikasi 100% bersih dari test-set leakage, sesuai Aturan Agent nomor 6.

### DEC-006: Verification of Test Split Sizes and Ground Truth vs Masked Files
- Tanggal: 2026-10-26
- Konteks: Penghitungan jumlah baris file split test EmoT dan SmSA (baik file berlabel asli maupun masked dummy), serta perbandingannya dengan angka yang dilaporkan di paper IndoNLU (Table 1: EmoT 442, SmSA 500).
- Temuan Lapangan:
  1. EmoT (`data/emot_emotion-twitter/`):
     - `test_preprocess.csv`: Total 441 baris teks (1 baris header `label,tweet` + 440 baris data). Berisi ground truth asli (distribusi label: anger 110, happy 101, sadness 100, fear 65, love 64).
     - `test_preprocess_masked_label.csv`: Total 441 baris teks (1 baris header + 440 baris data). Berisi masked dummy label (seluruh 440 baris diisi `"happy"`).
     - Perbandingan dengan paper IndoNLU: Paper mencatat ukuran test split EmoT sebesar 442 sampel, sedangkan file di repositori IndoBenchmark memuat 440 baris data (selisih 2 baris lebih sedikit dibanding angka di tabel paper).
  2. SmSA (`data/smsa_doc-sentiment-prosa/`):
     - `test_preprocess.tsv`: Total 500 baris teks (format TSV tanpa baris header = 500 baris data). Berisi ground truth asli (distribusi label: positive 208, negative 204, neutral 88).
     - `test_preprocess_masked_label.tsv`: Total 500 baris teks (format TSV tanpa baris header = 500 baris data). Berisi masked dummy label (seluruh 500 baris diisi `"neutral"`).
     - Perbandingan dengan paper IndoNLU: Paper mencatat ukuran test split SmSA sebesar 500 sampel, persis sama dengan jumlah 500 baris data pada file repositori.
- Keputusan: Mencatat temuan ini secara transparan dalam dokumentasi teknis. Pipeline tetap memuat data dari file resmi repositori IndoBenchmark apa adanya tanpa manipulasi baris.
- Dampak: Integritas data terjamin dan perbedaan 2 baris pada EmoT terdokumentasi dengan jelas.

### DEC-007: Phase 1 EmoT Run Artifact Status
- Tanggal: 2026-10-26
- Konteks: Run baseline EmoT `indobert-base_lr2e-05_s42` pada Phase 1 berasal dari eksekusi Kaggle pertama sebelum perbaikan pelaporan `best_epoch` dan explicit label maps diterapkan.
- Keputusan:
  1. File `metrics.json` untuk EmoT TIDAK diedit secara manual demi mempertahankan orisinalitas artefak dari runtime Kaggle.
  2. Nilai `best_epoch` di `metrics.json` run EmoT ini tercatat 3.0 (kebetulan sesuai log training di mana early stopping berhenti pada epoch 3, namun dihasilkan oleh logika lama).
  3. Konfigurasi label map pada run ini menggunakan mapping bawaan lama.
  4. Nilai metrik validasi (valid_macro_f1: 0.7320) dan logits (valid_logits.npy dan test_logits.npy) tetap valid untuk analisis awal.
  5. Run baseline ini akan digantikan oleh run baru yang sepenuhnya bersih pada saat sweep Phase 2 dijalankan.
- Dampak: Dokumentasi jujur dan transparan sesuai etika riset, artefak Phase 1 tetap utuh tanpa modifikasi retrospektif.

### DEC-008: Protocol for Report-Only Test Evaluation Module
- Tanggal: 2026-10-26
- Konteks: Penambahan modul evaluasi mandiri `cuking.evaluate` untuk mengukur performa prediksi `test_logits.npy` terhadap ground truth label asli pada split test IndoNLU.
- Aturan Wajib Riset:
  1. Skor split test semata-mata digunakan untuk laporan akhir (final reporting).
  2. DILARANG KERAS menggunakan metrik split test untuk memilih model terbaik, menyetel hyperparameter, early stopping, maupun memilih model anggota ensemble.
  3. Seluruh pemilihan model, hyperparameter, dan perancangan ensemble WAJIB bersandar murni pada split validation.
  4. Setiap eksekusi evaluasi split test (`python -m cuking.evaluate --split test`) mencetak peringatan tegas "[WARNING: REPORT ONLY]" pada terminal output dan logger.
  5. Integritas baris diverifikasi ketat: jumlah baris pada `test_logits.npy` wajib sama persis dengan jumlah baris ground truth test label asli.
- Dampak: Bebas dari risiko data leakage dan p-hacking, menjaga kepatuhan penuh pada Aturan Agent nomor 6.

### DEC-009: Phase 2 Large Model Sweep Grid and OOM/NaN Guards
- Tanggal: 2026-10-26
- Konteks: Penyesuaian grid sweep hyperparameter untuk model large (`xlmr-large` dan `indobert-large`), serta perlindungan stabilitas runtime (OOM fallback dan NaN guard) pada T2.2.
- Alasan Penyesuaian Grid:
  1. `xlmr-large`: Model ini sangat rentan mengalami divergen atau loss explosion pada learning rate tinggi. Paper IndoNLU menggunakan learning rate 1e-5 untuk XLM-R large. Oleh karena itu, grid learning rate disesuaikan menjadi: `5e-6, 1e-5, 2e-5`.
  2. `indobert-large`: Paper IndoNLU (Tabel 21 dan 22) menggunakan learning rate 3e-5 hingga 4e-5 untuk IndoBERT large. Grid learning rate disesuaikan menjadi: `2e-5, 3e-5, 4e-5`.
  3. Konfigurasi umum: Seed 42, `warmup_ratio=0.1`, effective batch size 16 (default per_device 8 x accumulation 2, dengan fallback otomatis ke 4 x 4 bila terjadi CUDA Out of Memory).
- Keputusan Teknis:
  1. Menambahkan `NaNGuardCallback` untuk menghentikan training segera jika loss bernilai NaN atau Inf, menulis `failed.json` di folder run, dan tidak menulis `metrics.json`.
  2. Menambahkan penanganan `OutOfMemoryError` dengan pembersihan memori CUDA dan pengulangan satu kali dengan batch 4 dan accumulation 4, mencatat `"oom_fallback": true` di `metrics.json`.
- Dampak: Eksekusi sweep model large menjadi stabil dan hemat waktu komputasi GPU Kaggle.

### DEC-010: Phase 2 EmoT Sweep Findings and Runtime Adjustments
- Tanggal: 2026-10-27
- Konteks: Analisis hasil hyperparameter sweep model large (`xlmr-large` dan `indobert-large`) pada benchmark EmoT Phase 2 yang dijalankan di Kaggle.
- Peringkat Valid Macro-F1 Sweep EmoT (Tanpa Skor Test):
  1. `xlmr-large` lr=2e-5: valid macro-F1 0.7964 (valid acc: 0.7955, best_epoch: 8.0, waktu: 700.86s)
  2. `xlmr-large` lr=5e-6: valid macro-F1 0.7923 (valid acc: 0.7864, best_epoch: 8.0, waktu: 704.94s)
  3. `xlmr-large` lr=1e-5: valid macro-F1 0.7884 (valid acc: 0.7818, best_epoch: 6.0, waktu: 711.40s)
  4. `indobert-large` lr=2e-5: valid macro-F1 0.7800 (valid acc: 0.7727, best_epoch: 5.0, waktu: 506.10s)
  5. `indobert-large` lr=4e-5: valid macro-F1 0.7711 (valid acc: 0.7705, best_epoch: 4.0, waktu: 444.94s)
  6. `indobert-large` lr=3e-5: valid macro-F1 0.7644 (valid acc: 0.7523, best_epoch: 3.0, waktu: 385.36s)
  *(Baseline Phase 1: `indobert-base` lr=2e-5: valid macro-F1 0.7320, best_epoch: 3.0)*
- Temuan Utama:
  1. Selisih performa antar learning rate pada `xlmr-large` sangat tipis (selisih ~0.4 poin persentase antara 2e-5 dan 5e-6, serta ~0.8 poin ke 1e-5). Perbedaan ini berada dalam batas noise statistik validation set yang hanya berukuran 440 sampel (di mana 1 sampel mewakili ~0.23% atau ~0.0023).
  2. `xlmr-large` mencapai `best_epoch` 8.0 pada lr 2e-5 dan 5e-6, yang mengindikasikan bahwa performa model masih menunjukkan tren kenaikan hingga epoch terakhir tanpa tanda overfitting dini.
  3. `indobert-large` mengalami overfitting lebih cepat dengan `best_epoch` berada di antara epoch 3 sampai 5, setelah itu performa validasi menurun dan memicu early stopping.
  4. Seluruh angka metrik di atas dihitung murni pada validation set tanpa menyentuh split test. Namun perlu dicatat bahwa skor validasi ini sedikit optimistis karena penentuan epoch terbaik dan pemilihan learning rate dilakukan pada split yang sama.
- Penyesuaian Perkiraan Waktu Eksekusi SmSA di Kaggle:
  Berdasarkan durasi nyata pada runtime Kaggle (EmoT xlmr-large ~11.7 menit, indobert-large ~6.5 hingga 8.5 menit), estimasi waktu untuk task SmSA yang memiliki data lebih besar dikoreksi:
  - `xlmr-large`: sekitar 45 menit per run.
  - `indobert-large`: sekitar 30 menit per run.
  - Total durasi sweep SmSA (6 run): sekitar 4 jam (masih sangat aman dalam batas sesi Kaggle 12 jam).
- Dampak: Memberikan wawasan empiris untuk perancangan model terbaik, pemilihan bobot ensemble nantinya, dan perencanaan alokasi waktu sesi GPU Kaggle untuk sweep SmSA.









