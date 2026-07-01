# Qwen3-4B — Results Manifest

Model: `Qwen/Qwen3-4B` (base, no RLHF). 36 layers, d_model=2560, 32 heads, 8 KV heads.

| File | Produced by | Contents |
|---|---|---|
| `cell1_model_summary.json` | 01 | Per-layer probe CV accuracy, probe layer, best layer |
| `cell3_probe_checkpoint.json` | 01 | Checkpoint (intermediate) |
| `best_configs.json` | 02 | Best (layer, multiplier) per emotion |
| `cell4_summary.json` | 02 | Pairwise probe accuracies + steering vector cosines (circumplex geometry) |
| `cell5_layer_sweep_raw.json` | 02 | Full layer-sweep raw data |
| `cell5_mult_sweep_full.json` | 02 | Full multiplier-sweep raw data |
| `temp_sweep_*.{json,csv}` | 03 | Temperature sweep results/checkpoint |
| `main_eval_checkpoint.json` | 03 | Checkpoint (intermediate) |
| `cell7_summary.json` | 03 | Main eval: confusion diagonal, per-emotion both/pred/coherent/mean-shift |
| `cell9_summary.json` | 04 | Head ablation summary |
| `task19_ablation_results.{json,csv}` | 04 | Raw per-head ablation results |
| `task_head_ablation_control.csv` | 04 | Head ablation control data |
| `task3_binary_vs_threepole_results.json` | 04 | Confusion-bleed analysis (mel↔horror, humor↔other) |
| `task4_surface_probe_results.json` | 04 | Surface-feature vs. activation probe accuracy gap |
| `task5_ood_results.json` | 04 | Out-of-distribution generalization |
| `cell12_summary.json` | 05 | Exp A–F: random baseline, data-size, PCA-vs-mean-diff, head specificity, drift, RoBERTa |
| `task_random_baseline.csv` | 05 (Exp A) | Raw random-vector baseline data |
| `task_data_ablation.csv` | 05 (Exp B) | Raw data-size ablation data |
| `task_repe_comparison.csv` | 05 (Exp C) | Raw PCA-vs-mean-diff comparison data |
| `task_4turn_drift.csv` | 05 (Exp E) | Raw 4-turn drift data |
| `task6_tonal_drift.csv` | 05 (Exp E, extended) | Raw tonal drift data |
| `task6_tonal_drift_summary.json` | 05 (Exp E, extended) | T1-vs-T2 drift test: mean shift, t-stat, p-value, % neutralized |
| `task_roberta_eval.csv` | 05 (Exp F) | Raw DistilRoBERTa cross-validation data |
| `cell13_summary.json` | 06 | DeepSeek judge: overall/per-emotion accuracy, mean scores by target emotion |
| `task_nim_eval*.{csv,json}` | 06 (secondary judge) | Raw secondary LLM-judge eval data |
| `simplex_results_final.csv` | 07–08 | 59-tone rollout / ternary simplex data |
| `simplex_results_T_best.csv` | 07–08 | Best-temperature simplex results |
| `emotional_axes/axis_meta.json` | 09 | PCA axes (depth/arousal/dark-bright), variance explained, tone pole coordinates |
| `emotional_axes/phase4_summary.json` | 10 | Causal PCA-axis steering validation |
| `task2_additive_vs_projinject.csv` | supplementary | Additive vs. projection-injection steering method comparison (raw, unsummarized) |
| `vector_comparison.csv` / `vector_comparison_stats.csv` | supplementary | Steering-vector construction-method comparison (raw) |

Not included (checkpoints, logs, model weights, activation tensors, raw generations): `eval.log`, `setup.log`, `sweep.log`, `*.pt`, `model/`, `outputs/`.

Note: `../scripts/` also includes `finish_cell7_qwen.py`, a helper/patch script for completing the main-eval cell (03), in addition to the 10 numbered pipeline scripts.
