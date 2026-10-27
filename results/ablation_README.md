# Ablation and Ensemble Performance Report

> **Catatan Metodologi**: CATATAN PENTING: Seleksi top-k dan evaluasi pada validation yang sama bersifat optimistis, sehingga kenaikan metrik yang kecil perlu diverifikasi lebih lanjut dengan multi-seed atau pengujian statistik independen.

## Ringkasan Performa Validasi

| task | name | valid_macro_f1 | delta_vs_best_single | members |
| --- | --- | --- | --- | --- |
| emot | best_single (xlmr-large) | 0.7963678342903201 | 0.0 | xlmr-large_lr2e-05_s42 |
| emot | best_indobert-base | 0.7319878215530816 | -0.06438001273723848 | indobert-base_lr2e-05_s42 |
| emot | best_indobert-large | 0.7800128369055802 | -0.01635499738473989 | indobert-large_lr2e-05_s42 |
| emot | best_xlmr-large | 0.7963678342903201 | 0.0 | xlmr-large_lr2e-05_s42 |
| emot | ensemble_top6 | 0.8167839855083358 | 0.020416151218015655 | emot/xlmr-large_lr2e-05_s42; emot/xlmr-large_lr5e-06_s42; emot/xlmr-large_lr1e-05_s42; emot/indobert-large_lr2e-05_s42; emot/indobert-large_lr4e-05_s42; emot/indobert-large_lr3e-05_s42 |
| emot | ensemble_per_model | 0.7991435406545058 | 0.0027757063641856705 | emot/xlmr-large_lr2e-05_s42; emot/indobert-large_lr2e-05_s42; emot/indobert-base_lr2e-05_s42 |
| emot | ensemble_top3 | 0.7955314566834286 | -0.0008363776068914808 | emot/xlmr-large_lr2e-05_s42; emot/xlmr-large_lr5e-06_s42; emot/xlmr-large_lr1e-05_s42 |
| smsa | best_single (indobert-base) | 0.9238782652168478 | 0.0 | indobert-base_lr2e-05_s42 |
| smsa | best_indobert-base | 0.9238782652168478 | 0.0 | indobert-base_lr2e-05_s42 |
