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
