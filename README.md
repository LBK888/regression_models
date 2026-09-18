# Regression Models

A Windows desktop application (PyQt6) for **training, managing and deploying regression
models on small scientific datasets**.

It covers the whole path from a spreadsheet to a prediction: compare dozens of model
families under one evidence standard, keep only the models that earn their place, decide
which of them stay visible, and then use them on single samples or on a whole table.

**English and Traditional Chinese interface**, switchable at any time from the status bar.
English is the default. 中文說明請見 [README_TW.md](README_TW.md)。

---

## Install

```bat
install.bat
```

`install.bat` will:

1. find a Python 3.11+ interpreter (and say exactly what to install if there is none)
2. create the `.venv` virtual environment
3. install the dependencies from `requirements.txt`
4. detect the CUDA version with `nvidia-smi` and install the matching PyTorch build
   (CUDA 13.0 / 12.8 / 12.6 / 11.8, or CPU-only)
5. verify every import and run the engine self-test

## Run

```bat
start.bat
```

`start.bat` checks the virtual environment, the program files, the dependencies and the
engine self-test before opening the main window.

Command line, without a window:

```bat
.venv\Scripts\python main.py --self-test        :: check the engine
.venv\Scripts\python main.py --validate a.xlsx  :: audit a data file
.venv\Scripts\python main.py --scan             :: list what the model library sees
.venv\Scripts\python main.py --lang zh_TW       :: launch in a specific language
```

---

## The four tabs

### ① Training and hyperparameter search

Drop in a CSV or Excel file, assign a role to each sheet and column, generate the
input/output combinations you want to compare, then pick the models, scalers, losses and
augmentations to benchmark.

**Evaluation strategies**: two-stage CV Screening with optional repeated nested
Confirmation, a single Train/Test holdout, validation against an independent external
dataset, or a final fit with no evaluation at all.

**Model families**: Mean baseline, Ridge, PLS, SVR (RBF), Random Forest, Extra Trees,
Deep / Wide / Residual / Multi-branch / Bottleneck MLP, CNN1D, ResNet1D, Grouped Fusion,
Numerical-Embedding MLP, FT-Transformer, ModernNCA — plus optional TabM, RealMLP,
CatBoost, XGBoost and LightGBM.

**Augmentation**: C-Mixup, calibrated feature noise, spectral perturbation, training-time
feature masking and FOMA, applied only to the current fold's training partition and always
compared against the unmodified baseline.

Every run writes to its own `runs/run_<timestamp>/` folder — a Markdown and PDF report,
per-experiment CSVs, figures, an audit trail and the saved model bundles. Nothing is ever
overwritten. Name the run in the Output box and that name follows its models into the
library.

**Which models are kept**: only the **top 3 per Target Task**, and only if
**Reported R² is above 0 — including the first-ranked model**. An R² of 0 or below means
the model did not beat "always predict the mean", so it has no deployment value. Every
model that is dropped is logged with the reason.

### ② Model library

Scans `models/` and every `runs/` folder, groups what it finds by run, and shows each
model's **R²** and **MAPE**. You can:

- rename a run or an individual model (double-click)
- tick or untick "use", which decides what the inference tabs can see
- write a note about what a model is for
- import third-party model files
- open a model's folder or delete it from the context menu

Incompatible models are **not hidden**. They stay in the table with the reason in the
Status column — a missing architecture, absent column names, a broken pickle.

### ③ Single-sample inference

The number of input fields, their names, the number of result cards and their names **all
come from the selected model**. Switching model rebuilds the form.

### ④ Batch inference and validation

Run one worksheet through one or more models at once, and compare them.

Columns are matched automatically (ignoring case, spacing and any `Sheet::` prefix) and can
be corrected by hand. **Map the model's output columns as well and the table is treated as
test data**, producing a validation report:

- `report.md` / `report.pdf` — evidence boundary, model comparison ranked by macro NMAE,
  per-target metrics, data handling audit
- `metrics.csv` — macro and per-target MAE / RMSE / NMAE / MAPE / Reported R² /
  Diagnostic R²
- `predictions.csv`, `predictions_wide.csv` — prediction, actual and residual per row
- figures — predicted vs actual with the 1:1 line, residual distribution, residual vs
  predicted, and NMAE / R² / MAPE comparison bars

Reports are written to `reports/batch_<timestamp>/`.

---

## Model formats

The inference layer resolves three generations of artifact to one interface:

| Format | Notes |
| --- | --- |
| `.joblib` (regression_v4 bundle) | Every model the trainer produces is saved this way, so inference supports all of them |
| `.pkl` + `.md` (legacy v2.2 sensor format) | The architecture is rediscovered from the state dict; column names come from the pickle, or from the `.md` when absent |
| `.joblib` / `.pkl` (third-party estimator) | Anything with a `predict()` method; names come from scikit-learn attributes or a `.meta.json` sidecar |
| `.pt` / `.pth` | Checkpoints containing a `model_state_dict` |

To give a third-party model proper column names, put a matching `.meta.json` beside it:

```json
{
  "model_name": "External GBR",
  "run_label": "Third-party models",
  "feature_labels": ["alpha", "beta", "gamma"],
  "target_labels": ["score"],
  "metrics": { "reported_r2": 0.88, "mape_percent": 6.4 }
}
```

For a state-dict model whose architecture is not built in, put an `architectures.py` next to
it defining `class YourNet(nn.Module): def __init__(self, input_dim, output_dim)`.

---

## Language

Pick the language from the combo box in the status bar. The choice is stored in
`settings.json` next to the program and applies on the next launch.

To add a language, create `locales/<code>.py` with a `MESSAGES` dict mapping the English
source strings to the translated text, then add the code to `LANGUAGES` in `i18n.py`. Any
key you leave out falls back to English, so a partial translation is still usable.

---

## Project layout

```
main.py                 single entry point
install.bat / start.bat install and launch
i18n.py                 translation lookup and language persistence
locales/                language catalogues (zh_TW.py)
project_paths.py        single source of truth for every folder location
regression_core.py      shared core: data loading, Experiment generation, architectures
legacy_architectures.py v2.2 architecture definitions, for reloading old models
regression_v4/          training and evaluation engine
app/                    merged UI, model library, inference and batch validation
models/                 the model library (legacy_v2/ holds two sample models)
runs/                   Benchmark Run outputs
reports/                batch inference reports
docs/                   design notes and third-party notices
```

## Licence and third-party packages

See [`docs/third-party-notices.md`](docs/third-party-notices.md).
Design decisions and known trade-offs are in [`docs/MERGE_NOTES.md`](docs/MERGE_NOTES.md).
