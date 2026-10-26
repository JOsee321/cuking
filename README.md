# Cuking

> Model NLU Bahasa Indonesia dengan budget Rp0: Melampaui baseline IndoNLU lewat disiplin eksperimen.

Cuking adalah pipeline fine-tuning terstruktur untuk model Natural Language Understanding (NLU) Bahasa Indonesia. Proyek ini membuktikan bahwa skor tinggi dan kompetitif di benchmark publik IndoNLU dapat diraih tanpa melatih model dari nol dan tanpa biaya komputasi (budget Rp0), dengan memanfaatkan model pretrained open-source (IndoBERT, XLM-RoBERTa), GPU gratis Kaggle, serta disiplin eksperimen: sweep hyperparameter, multi-seed, layer-wise learning rate decay (LLRD), dan probability-averaging ensemble.

---

## Status Fase Proyek

| Fase | Deskripsi | Status | Target Utama |
|---|---|---|---|
| **Phase 0** | Fondasi repo dan konfigurasi | **Selesai** | Layout repo, dependencies, configs, dan dokumen fondasi |
| **Phase 1** | Reproduksi baseline | Belum dimulai | Reproduksi `indobert-base-p1` pada EmoT dan SmSA (+- 2 F1 paper) |
| **Phase 2** | Large model dan sweep | Belum dimulai | Grid sweep `xlm-roberta-large` dan `indobert-large-p1` |
| **Phase 3** | Ensemble dan optimasi | Belum dimulai | LLRD, multi-seed run, dan probability ensemble |
| **Phase 4** | Submission dan rilis | Belum dimulai | Submit CodaLab IndoNLU, rilis model dan model card di Hugging Face |
| **Phase 5** | Ekspansi NER | Belum dimulai | Pipeline token classification untuk NERGrit dan NERP |

---

## Target Benchmark

Evaluasi dilakukan murni pada benchmark resmi IndoNLU (arXiv:2009.05387):
1. **EmoT** (`emot_emotion-twitter`): Klasifikasi emosi 5 kelas (sadness, anger, love, fear, happy) - Metrik: Macro-F1.
2. **SmSA** (`smsa_doc-sentiment-prosa`): Analisis sentimen 3 kelas (positive, neutral, negative) - Metrik: Macro-F1.
3. **NERGrit** (`nergrit_ner-grit`): Named Entity Recognition - Metrik: Entity F1 (`seqeval`).
4. **NERP** (`nerp_ner-prosa`): Named Entity Recognition - Metrik: Entity F1 (`seqeval`).

### Aturan Wajib Evaluasi Split Test (REPORT ONLY)
Skor test hanya digunakan untuk laporan akhir. **DILARANG** dipakai untuk:
- Memilih model terbaik
- Menyetel hyperparameter atau arsitektur
- Menentukan early stopping
- Memilih anggota ensemble

Semua seleksi model dan penentuan ensemble WAJIB menggunakan evaluasi validation split. Setiap kali evaluasi test split dijalankan (`python -m cuking.evaluate --split test`), sistem mencetak peringatan tegas `[WARNING: REPORT ONLY]`.

---

## Struktur Repositori

```text
cuking/
  PRD.md                 # Sumber kebenaran tunggal spesifikasi proyek
  README.md              # Ringkasan proyek dan panduan reproduksi
  LICENSE                # Lisensi MIT
  requirements.txt       # Dependencies lengkap (lingkungan lokal / fresh)
  requirements-kaggle.txt# Paket tambahan untuk runtime Kaggle
  pyproject.toml         # Konfigurasi instalasi package
  configs/               # Konfigurasi training per task (YAML)
    base.yaml
    emot.yaml
    smsa.yaml
    nergrit.yaml
    nerp.yaml
  src/cuking/            # Modul utama pipeline NLU
  scripts/               # Skrip eksekusi sweep dan smoke test
  tests/                 # Unit test dan smoke test
  docs/                  # Keputusan teknis, runbook Kaggle, baselines
  results/experiments/   # Logits, config, dan metrics hasil run nyata
```

---

## Instalasi

### Lingkungan Lokal

```bash
git clone https://github.com/JOsee321/cuking.git
cd cuking
python -m venv .venv
# Di Windows:
.venv\Scripts\activate
# Di Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Lingkungan Kaggle

Notebook Kaggle sudah memiliki PyTorch, CUDA, NumPy, Pandas, dan Scikit-learn terpasang. Cukup install dependensi tambahan:

```bash
pip install -r requirements-kaggle.txt
pip install -e .
```

---

## Smoke Test Lokal

Laptop lokal hanya digunakan untuk pengembangan kode dan smoke test (subset data kecil, 1-5 step, CPU). Training sebenarnya dijalankan di Kaggle GPU.

Untuk menjalankan verifikasi smoke test:

```bash
# Menggunakan pytest:
pytest tests/test_smoke.py -v

# Atau menggunakan shell script:
bash scripts/smoke.sh
```

---

## Aturan Integritas Hasil

1. **Anti Angka Fiktif**: Tidak ada skor yang ditulis di README atau dokumen tanpa bukti file `metrics.json` hasil run nyata atau entri leaderboard CodaLab.
2. **Pemisahan Test Set**: Test set IndoNLU memiliki label tersembunyi dan tidak pernah digunakan untuk memilih model atau menyetel hyperparameter.
3. **Reproducibility**: Setiap run memiliki seed deterministik, config YAML yang tersimpan, dan submission generator yang transparan.

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
