# Kaggle Runbook: Cuking Phase 1

Panduan eksekusi training baseline `indobert-base-p1` pada benchmark EmoT dan SmSA di Kaggle Notebook dengan akselerator GPU gratis.

---

## Prasyarat Notebook Kaggle

Sebelum menjalankan sel di bawah, pastikan konfigurasi notebook Kaggle sudah diatur:
1. **Accelerator**: Pilih **GPU T4 x2** atau **GPU P100**.
2. **Internet**: Geser toggle **Internet On** pada panel kanan (Settings).
3. **Secrets**: Tambahkan secret `HF_TOKEN` lewat menu **Add-ons** > **Secrets** (label: `HF_TOKEN`, value: token Hugging Face Anda).

---

## Sel 1: Clone Repositori dan Instalasi Dependensi

```python
# Sel 1: Clone repo Cuking dan install dependensi tambahan
!git clone https://github.com/JOsee321/cuking.git
%cd cuking

# Install paket tambahan khusus runtime Kaggle
!pip install -r requirements-kaggle.txt --quiet
!pip install -e . --quiet
```

---

## Sel 2: Muat Kredensial Hugging Face dari Kaggle Secrets

```python
# Sel 2: Baca HF_TOKEN dari Kaggle Secrets ke environment variable
import os
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN berhasil dimuat dari Kaggle Secrets.")
except Exception as e:
    print(f"Catatan: HF_TOKEN tidak ditemukan di Secrets ({e}). Berjalan dalam mode unauthenticated.")
```

---

## Sel 0: Preflight Check (Verifikasi Library dan Smoke Test)

> **ATURAN WAJIB**: Run training DILARANG DILANJUTKAN bila Sel 0 ini gagal!

```python
# Sel 0: Preflight verification sebelum training dimulai
import torch
import transformers
import datasets
import accelerate
import evaluate

print("=== Preflight Library Versions ===")
print(f"PyTorch      : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU Device   : {torch.cuda.get_device_name(0)}")
print(f"Transformers : {transformers.__version__}")
print(f"Datasets     : {datasets.__version__}")
print(f"Accelerate   : {accelerate.__version__}")
print(f"Evaluate     : {evaluate.__version__}")

print("\n=== Menjalankan Smoke Test Preflight ===")
!python -m pytest tests/test_smoke.py -q
```

---

## Sel 3: Jalankan Baseline Training EmoT (indobert-base)

```python
# Sel 3: Training task EmoT dengan model indobert-base
!python -m cuking.train --task emot --model indobert-base
```

---

## Sel 4: Jalankan Baseline Training SmSA (indobert-base)

```python
# Sel 4: Training task SmSA dengan model indobert-base
!python -m cuking.train --task smsa --model indobert-base
```

---

## Sel 5: Tinjau Metrik Hasil dan Arsipkan

```python
# Sel 5: Tampilkan ringkasan file metrics.json
import glob
import json

metrics_files = glob.glob("results/experiments/**/metrics.json", recursive=True)
print(f"Ditemukan {len(metrics_files)} run selesai:")
for mf in metrics_files:
    with open(mf, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        print(f"Task: {data.get('task')}, Model: {data.get('model')}, Valid Macro-F1: {data.get('valid_macro_f1'):.4f}, Valid Acc: {data.get('valid_acc'):.4f}, Waktu: {data.get('train_time_s')}s")

# Kompres seluruh folder results/experiments ke zip
!zip -r /kaggle/working/cuking_phase1_results.zip results/experiments
print("File arsip berhasil dibuat di: /kaggle/working/cuking_phase1_results.zip")
```

---

## Sel 6: Unduh File Hasil Langsung dari Notebook

**UNDUH ZIP SEBELUM STOP SESSION, file di /kaggle/working hilang saat sesi berhenti**

```python
# Sel 6: Tampilkan tautan unduh langsung di notebook browser
from IPython.display import FileLink
FileLink('/kaggle/working/cuking_phase1_results.zip')
```

---

## Langkah Lanjutan: Sinkronisasi ke Repo Lokal

1. Klik tautan hasil Sel 6 untuk mengunduh `cuking_phase1_results.zip` langsung ke komputer lokal (atau unduh via panel output Kaggle: Data Explorer > Output).
2. Di laptop lokal, ekstrak isi zip ke folder root repositori `cuking/` sehingga folder `results/experiments/` terisi hasil run Kaggle:
   ```bash
   # Di root repo lokal:
   # Pastikan struktur folder menjadi: results/experiments/<task>/<model>_lr...
   ```
3. Jalankan agregasi metrik lokal:
   ```bash
   python -m cuking.aggregate
   ```

---

## Phase 2: Large Model Sweep Runbook

**UNDUH ZIP SEBELUM STOP SESSION. File di /kaggle/working hilang saat sesi berhenti.**

Bagian ini memuat panduan lengkap eksekusi hyperparameter sweep Phase 2 untuk model large (`xlmr-large` dan `indobert-large`) pada benchmark EmoT dan SmSA di Kaggle Notebook.

---

### Perkiraan Waktu Eksekusi

Berdasarkan perbandingan waktu eksekusi:
- **EmoT (Large model)**: sekitar 8 menit per run (total 6 run: ~48 menit). Sebagai perbandingan, model base membutuhkan sekitar 2,5 menit per run.
- **SmSA (Large model)**: sekitar 30 menit per run (total 6 run: ~180 menit / 3 jam). Sebagai perbandingan, model base membutuhkan sekitar 10 menit per run.
- **Total waktu sweep**: sekitar 3,5 hingga 4 jam untuk 12 run keseluruhan (jauh di bawah batas sesi Kaggle 12 jam).

---

### Cara Menjalankan di Latar Belakang (Background Commit)

Agar proses sweep tidak terputus saat koneksi internet terputus atau tab browser ditutup, jalankan notebook dalam mode background:
1. Pastikan konfigurasi notebook:
   - **Accelerator**: GPU T4 x2 atau GPU P100.
   - **Internet**: On (aktif).
2. Notebook harus memuat seluruh sel berurutan dari (a) sampai (g) di bawah ini tanpa error.
3. Klik tombol **Save Version** di pojok kanan atas notebook.
4. Pilih opsi **Save & Run All (Commit)**.
5. Klik **Save**. Notebook akan dijalankan secara independen di server Kaggle di latar belakang.
6. Anda dapat menutup browser. Sesi background memiliki batas waktu maksimal sekitar 12 jam.
7. Setelah selesai, buka versi notebook tersebut, masuk ke tab **Output**, dan unduh file `cuking_phase2_results.zip`.

---

### Sel Berurutan untuk Notebook Kaggle Phase 2

#### Sel (a): Atur Lingkungan GPU 0
```python
# Sel (a): Pastikan hanya GPU 0 yang dialokasikan
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES diatur ke: 0")
```

#### Sel (b): Clone Repositori dan Instalasi Dependensi
```python
# Sel (b): Clone repo Cuking dan install dependensi
!git clone https://github.com/JOsee321/cuking.git
%cd cuking

!pip install -r requirements-kaggle.txt --quiet
!pip install -e . --quiet
```

#### Sel (c): Sel 0 Preflight Check
```python
# Sel (c): Preflight verification dan smoke test
import torch
import transformers
import datasets
import accelerate
import evaluate

print("=== Preflight Environment Check ===")
print(f"PyTorch      : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU Device   : {torch.cuda.get_device_name(0)}")
print(f"Transformers : {transformers.__version__}")
print(f"Datasets     : {datasets.__version__}")
print(f"Accelerate   : {accelerate.__version__}")
print(f"Evaluate     : {evaluate.__version__}")

print("\n=== Menjalankan Smoke Test Preflight ===")
!python -m pytest tests/test_smoke.py -q
```

#### Sel (d): Eksekusi Sweep EmoT (6 Run)
```python
# Sel (d): Jalankan sweep 6 run untuk task EmoT
# Urutan: xlmr-large (1e-5), indobert-large (3e-5), xlmr-large (2e-5),
# indobert-large (4e-5), xlmr-large (5e-6), indobert-large (2e-5)
!bash scripts/run_sweep.sh --tasks emot
```

#### Sel (e): Eksekusi Sweep SmSA (6 Run)
```python
# Sel (e): Jalankan sweep 6 run untuk task SmSA
# Urutan: xlmr-large (1e-5), indobert-large (3e-5), xlmr-large (2e-5),
# indobert-large (4e-5), xlmr-large (5e-6), indobert-large (2e-5)
!bash scripts/run_sweep.sh --tasks smsa
```

#### Sel (f): Tampilkan Peringkat Metrik Terbaik
```python
# Sel (f): Tampilkan leaderboard run per task berdasarkan validation macro-F1
!python -m cuking.aggregate --top
```

#### Sel (g): Tautan Unduh Arsip Hasil Parsial / Lengkap
```python
# Sel (g): Tampilkan tautan unduh file zip hasil eksperimen
# Arsip ini diperbarui otomatis setiap kali sebuah run selesai
from IPython.display import FileLink
FileLink('/kaggle/working/cuking_phase2_results.zip')
```

