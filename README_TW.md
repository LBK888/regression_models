# Regression Models

一套用於**小型科學資料集回歸模型訓練、管理與部署**的 Windows 桌面程式（PyQt6）。

它涵蓋從試算表到預測的完整流程：在同一套證據標準下比較數十種模型家族、
只保留真正值得部署的模型、決定哪些模型要出現在推論介面，
然後用它們做單筆預測或整份表格的批次預測。

**介面支援英文與繁體中文**，可隨時從狀態列切換，預設為英文。
English documentation: [README.md](README.md)

---

## 安裝

```bat
install.bat
```

`install.bat` 會：

1. 尋找 Python 3.11 以上的直譯器（找不到會明確說明該安裝什麼）
2. 建立 `.venv` 虛擬環境
3. 依 `requirements.txt` 安裝相依套件
4. 用 `nvidia-smi` 偵測 CUDA 版本，安裝對應的 PyTorch build
   （CUDA 13.0 / 12.8 / 12.6 / 11.8，或 CPU 版）
5. 驗證所有套件可匯入，並執行引擎 self-test

## 啟動

```bat
start.bat
```

`start.bat` 會檢查虛擬環境、程式檔案、相依套件與引擎 self-test，再開啟主視窗。

不開視窗的命令列用法：

```bat
.venv\Scripts\python main.py --self-test        :: 驗證引擎
.venv\Scripts\python main.py --validate a.xlsx  :: 稽核一個資料檔
.venv\Scripts\python main.py --scan             :: 列出模型庫目前看到的模型
.venv\Scripts\python main.py --lang zh_TW       :: 以指定語言啟動
```

---

## 四個分頁

### ① 訓練與超參數試探

拖入 CSV 或 Excel 檔，為每個工作表與欄位指定角色，產生要比較的輸入／輸出排列組合，
再選擇要比較的模型、scaler、loss 與 augmentation。

**評估策略**：兩階段 CV Screening 搭配可選的 repeated nested Confirmation、
單次 Train/Test holdout、以獨立外部資料集驗證，或只做 final fit 不評估。

**模型家族**：Mean 基準、Ridge、PLS、SVR (RBF)、Random Forest、Extra Trees、
Deep / Wide / Residual / Multi-branch / Bottleneck MLP、CNN1D、ResNet1D、Grouped Fusion、
Numerical-Embedding MLP、FT-Transformer、ModernNCA，
以及選配的 TabM、RealMLP、CatBoost、XGBoost、LightGBM。

**Augmentation**：C-Mixup、calibrated feature noise、spectral perturbation、
training-time feature masking 與 FOMA。所有方法只作用於當前 fold 的 training partition，
並且一定會與未處理的 baseline 比較。

每次執行都寫在自己的 `runs/run_<時間戳>/` 資料夾中 ——
Markdown 與 PDF 報告、各實驗的 CSV、圖表、稽核紀錄與保存的模型檔，不會覆蓋任何舊結果。
在「輸出」區塊填入 Run 名稱，這個名稱會跟著模型進入模型庫。

**模型保留規則**：每個 Target Task 只保留**排名前 3 名**，
且**Reported R² 必須大於 0 才會儲存 —— 第一名也不例外**。
R² ≤ 0 代表該模型沒有勝過「一律預測平均值」，沒有部署價值。
每個被淘汰的模型都會在執行紀錄中列出原因。

### ② 模型庫

掃描 `models/` 與所有 `runs/` 資料夾，依 run 分組列出，並顯示每個模型的 **R²** 與 **MAPE**。
可以：

- 雙擊重新命名 run 或個別模型
- 勾選／取消「使用」，決定推論分頁看得到哪些模型
- 為模型寫下用途備註
- 匯入第三方模型檔
- 用右鍵選單開啟模型所在資料夾或刪除模型

不相容的模型**不會被隱藏**，而是留在表格中，並在「狀態」欄說明原因 ——
找不到對應架構、缺少欄位名稱、pickle 損壞等。

### ③ 單筆推論

輸入欄位的數量與名稱、輸出卡片的數量與名稱，**全部由所選模型決定**，換模型就重建表單。

### ④ 批次推論與驗證

一次把一個工作表送進一個或多個模型，並互相比較。

欄位會自動比對（忽略大小寫、空白與 `Sheet::` 前綴），也可以手動修正。
**只要把模型的輸出欄位也對應到資料表，這份資料就會被當作 test data**，並輸出驗證報告：

- `report.md` / `report.pdf` —— 證據界線、依 macro NMAE 排名的模型比較、
  各目標指標、資料處理稽核
- `metrics.csv` —— macro 與各目標的 MAE / RMSE / NMAE / MAPE /
  Reported R² / Diagnostic R²
- `predictions.csv`、`predictions_wide.csv` —— 每列的預測值、真值與殘差
- 圖表 —— predicted vs actual（含 1:1 參考線）、殘差分佈、殘差 vs 預測值，
  以及 NMAE / R² / MAPE 模型比較長條圖

報告寫在 `reports/batch_<時間戳>/`。

---

## 支援的模型格式

推論層把三代產物收斂成同一個介面：

| 格式 | 說明 |
| --- | --- |
| `.joblib`（regression_v4 bundle） | 訓練端產生的所有模型都以這個格式儲存，因此推論端一律支援 |
| `.pkl` + `.md`（v2.2 感測器格式） | 由 state dict 重建架構；欄位名稱取自 pickle，缺漏時由 `.md` 補上 |
| `.joblib` / `.pkl`（第三方估計器） | 任何具有 `predict()` 的物件；欄位名稱取自 scikit-learn 屬性或 `.meta.json` |
| `.pt` / `.pth` | 含 `model_state_dict` 的 checkpoint |

要讓第三方模型顯示正確的欄位名稱，在模型檔旁放一個同名的 `.meta.json`：

```json
{
  "model_name": "External GBR",
  "run_label": "第三方模型",
  "feature_labels": ["alpha", "beta", "gamma"],
  "target_labels": ["score"],
  "metrics": { "reported_r2": 0.88, "mape_percent": 6.4 }
}
```

架構不在內建清單中的 state-dict 模型，可在模型檔同目錄放 `architectures.py`，
定義 `class YourNet(nn.Module): def __init__(self, input_dim, output_dim)`。

---

## 語言

從狀態列的下拉選單選擇語言。選擇會存在程式旁的 `settings.json`，下次啟動時沿用。

要新增語言，建立 `locales/<代碼>.py`，內含一個把英文原文對應到譯文的 `MESSAGES` dict，
再把代碼加進 `i18n.py` 的 `LANGUAGES`。
未翻譯的字串會自動退回英文，因此不完整的翻譯仍然可用。

---

## 專案結構

```
main.py                 單一入口
install.bat / start.bat 安裝與啟動
i18n.py                 翻譯查詢與語言設定保存
locales/                語言檔（zh_TW.py）
project_paths.py        所有資料夾位置的單一來源
regression_core.py      共用核心：資料載入、Experiment 產生、架構定義
legacy_architectures.py v2.2 架構定義（重建舊模型用）
regression_v4/          訓練與評估引擎
app/                    整合 UI、模型庫、推論與批次驗證
models/                 模型庫（legacy_v2/ 為隨附的兩個示範模型）
runs/                   Benchmark Run 輸出
reports/                批次推論報告
docs/                   設計說明與第三方套件聲明
```

## 授權與第三方套件

見 [`docs/third-party-notices.md`](docs/third-party-notices.md)。
設計決定與已知取捨見 [`docs/MERGE_NOTES.md`](docs/MERGE_NOTES.md)。
