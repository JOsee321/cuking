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

