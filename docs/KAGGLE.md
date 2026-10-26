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

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
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

## Sel 6: Unduh dan Sinkronisasi ke Repo Lokal

1. Unduh file `cuking_phase1_results.zip` melalui panel output Kaggle (Data Explorer > Output).
2. Di laptop lokal, ekstrak isi zip ke folder root repositori `cuking/` sehingga folder `results/experiments/` terisi hasil run Kaggle:
   ```bash
   # Di root repo lokal:
   # Pastikan struktur folder menjadi: results/experiments/<task>/<model>_lr...
   ```
3. Jalankan agregasi metrik lokal:
   ```bash
   python -m cuking.aggregate
   ```
