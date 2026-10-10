# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying transfer learning for power load forecasting. Tetouan Region1 is the data-scarce **target domain**; Panama, Tetouan Region2, and Tetouan Region3 are data-rich **source domains**. Knowledge is transferred via pre-train → fine-tune (instance/model-based transfer). A supplementary DANN experiment (MNIST → MNIST-M) demonstrates feature-based domain adaptation.

## Environment

```
python=3.10.16, torch=2.6.0+cu126
numpy=2.2.2, pandas=2.2.3, scikit-learn=1.6.1, matplotlib=3.10.0
```

GPU (CUDA) is expected. See `实验代码/environment.txt` for the full pip freeze.

## Running Experiments

Each predict script in the subdirectories is self-contained and run directly:

```bash
cd 实验代码

# Run a single transfer experiment (pre-train on Region2 → fine-tune on Region1, LSTM):
python region2_2_region1/LSTM_predict.py

# Run without fine-tuning (pre-train only, zero-shot eval on target):
python region2_2_region1/LSTM_wo_ft_predict.py

# DANN feature-based transfer (MNIST→MNIST-M):
cd ../DANN_mnist_m
python main.py
```

Script names follow the pattern `<Model>_predict.py` (with fine-tuning) and `<Model>_wo_ft_predict.py` (without fine-tuning). Each runs 20 trials and reports mean±metrics.

## Architecture

### Data Layer
- **`实验代码/power_dataset.py`** — `Power_ds` Dataset class. Handles source/target domains, supervised framing via `series_to_supervised()`, MinMax scaling, and train/test split. Source domain uses 90% of data for training; target domain uses `running_day` days for training and `testing_day` days for testing. **⚠️ Contains hardcoded absolute paths** in `preprocessing()` — must be updated when running on a different machine.

### Model Zoo
- **`实验代码/typical_models.py`** — Classic architectures: `Linear_Model`, `CNN_model`, `LSTM_model`, `GRU_model`, `CNN_LSTM_model`, `Transformer`
- **`实验代码/promising_models/`** — Newer architectures:
  - `Informer_model/` — Informer (AAAI 2021 best paper) with ProbSparse self-attention. `my_Informer_1.py` and `my_Informer_2.py` are encoder-only variants.
  - `TCN_fc.py` — Temporal Convolutional Network
  - `GRU_seq2seq.py` — GRU Seq2Seq with attention decoder
  - `my_Transformer.py` — Custom Transformer variant

### Experiment Directories
Each subdirectory (`panama2tetouan_region1/`, `region2_2_region1/`, `region3_2_region1/`) contains predict scripts for different model types all following the same pattern:
1. Pre-train model on source domain data
2. Either fine-tune on target domain (`_predict.py`) or skip fine-tuning (`_wo_ft_predict.py`)
3. Evaluate on target test set with MAE, MAPE, RMSE, R² across 20 runs

### DANN (Domain-Adversarial Neural Network)
- **`DANN_mnist_m/functions.py`** — `ReverseLayerF`: gradient reversal layer (GRL) for adversarial domain training
- **`DANN_mnist_m/model.py`** — `CNNModel`: shared feature extractor → class classifier + domain classifier (with GRL)
- **`DANN_mnist_m/main.py`** — Training loop with progressive alpha scheduling
- Requires MNIST-M dataset downloaded separately (see `DANN_mnist_m/README.md`)

### Transfer Learning Patterns

| Pattern | Files | Description |
|---------|-------|-------------|
| Model-based with fine-tuning | `*_predict.py` | Pre-train on source → fine-tune all params on target → test |
| Model-based w/o fine-tuning | `*_wo_ft_predict.py` | Pre-train on source → directly test on target (zero-shot) |
| Feature-based (DANN) | `DANN_mnist_m/` | Adversarial domain alignment → domain-invariant features |

## Important Notes

- **Hardcoded paths**: `实验代码/power_dataset.py` lines 143 and 154 contain absolute paths to CSV data files on the original author's machine (`C:\\Users\\33103\\Desktop\\...`). These must be changed to your own paths before running.
- **Import style**: Predict scripts use absolute imports like `from 实验代码.typical_models import LSTM_model`. When running directly (not as a package), you may need to adjust `sys.path` or run from the parent directory.
- **Data files**: The `实验代码/dealt_data/` directory contains preprocessed CSV files for the four regions (not listed in glob results — verify they exist before running).
- **DANN dataset**: The MNIST-M dataset is not included in the repo and must be downloaded separately per the instructions in `DANN_mnist_m/README.md`.
