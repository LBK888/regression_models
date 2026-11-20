# Regression Models

A Windows desktop application (PyQt6) for **training, managing and deploying regression
models on small scientific datasets**.

It covers the whole path from a spreadsheet to a prediction: compare dozens of model
families under one evidence standard, keep only the models that earn their place, decide
which of them stay visible, and then use them on single samples or on a whole table.

**English and Traditional Chinese interface**, switchable at any time from the status bar.
English is the default. 中文說明請見 [README_TW.md](README_TW.md)。

---

## Windows exe (no Python needed)

Download `RegressionModels-v…-win64-cuda12.8.7z` from the
[Releases page](https://github.com/LBK888/regression_models/releases), extract it with
[7-Zip](https://www.7-zip.org), and run `RegressionModels.exe`. If a release is split into
`.7z.001`, `.002`, …, download every part and open `.001`.

At start the program:

- **checks that CUDA really works** by running a test calculation on the GPU, and explains
  why when it does not (no NVIDIA driver, driver too old, card unsupported by the CUDA 12.8
  build) before falling back to the CPU — every model still works on the CPU;
- **creates any missing folder or setting** (`runs`, `models`, `reports`, `pretrained`, `logs`,
  `settings.json`) and installs the example models on first start;
- **offers to download the TabPFN v2 weights** (about 45 MB, no account) if they are missing.

The start-up report is written to `logs/startup.json`; `RegressionModels.exe --check` runs the
checks alone. To build the exe yourself, run `packaginguild_exe.bat` (see the comments in it).

---

## Install

```bat
install.bat
```

`install.bat` will:

1. find a Python 3.11+ interpreter (and say exactly what to install if there is none)
2. create the `.venv` virtual environment
3. read the NVIDIA driver's CUDA version and the GPU's compute capability, and install the
   newest PyTorch build that suits both (CUDA 13.0 / 12.8 / 12.6 / 11.8, or CPU-only),
   falling back to the next one when a build does not exist for this Python. A CPU-only
   build left in an existing `.venv` is replaced; a matching build is kept
4. install the dependencies from `requirements.txt`, pinned to that PyTorch build so no
   package can swap it for PyPI's CPU-only one
5. verify every import, run a test calculation on the GPU, and run the engine self-test

Running `install.bat` again on an existing `.venv` is safe: it only changes what is wrong.

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

**Evaluation strategies**: cross-validation with an optional second stage of repeated
nested cross-validation, a single train/test split (random, or the most recent rows as the
test set to measure drift over time), scoring on a separate data file, or training with no
score at all. The settings page shows only the fields the current choices use.

**Group ID**: when several rows come from the same day, tank or subject, name that column
as the Group ID so its rows are never split between training and testing. A repeated
column called Date / 日期 / Batch / Tank (and similar) is suggested automatically, and the
settings page warns when no Group ID is set.

**Data checks**: on loading, the Audit column flags gross outliers (for example a typo
far from every other value), constant columns, a column that is an exact multiple of
another, and near-duplicate columns.

**Model families**: Mean baseline, Ridge, PLS, SVR (RBF), Random Forest, Extra Trees,
CatBoost, XGBoost, LightGBM, TabPFN v2, Deep / Wide / Residual / Multi-branch / Bottleneck
MLP, Numerical-Embedding MLP, FT-Transformer, ModernNCA, TabM, RealMLP, CNN1D, ResNet1D and
Grouped Fusion. TabPFN uses its version-2 weights, which need no account; they download
once (about 45 MB) into `pretrained/tabpfn/`.

**Model size**: by default ("Auto") neural-network width scales with the square root of
the real training rows (500 rows = the preset width, clipped to ×0.5–×3), and bagged trees
grow their minimum leaf size with the data. "Fixed" keeps the preset size.

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
