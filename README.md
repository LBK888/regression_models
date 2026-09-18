# Regression Models

小資料回歸模型的**訓練、模型管理與推論**整合桌面程式（Windows / PyQt6）。

這個專案把兩個原本獨立的程式合併成單一入口：

| 原程式 | 功能 | 在本專案中的位置 |
| --- | --- | --- |
| `260801-泰國蝦病毒` (regression v4) | 多模型比較、超參數試探、證據導向報告 | 分頁 ①，引擎在 `regression_v4/` |
| `AI_Sensor` | 載入訓練好的模型做推論 | 分頁 ③，引擎在 `app/` |

---

## 安裝

```bat
install.bat
```

`install.bat` 會：

1. 尋找 Python 3.11 以上的直譯器（找不到會明確報錯並指示安裝來源）
2. 建立 `.venv` 虛擬環境
3. 安裝 `requirements.txt` 的相依套件
4. 用 `nvidia-smi` 偵測 CUDA 版本，安裝對應的 PyTorch build（CUDA 13.0 / 12.8 / 12.6 / 11.8 / CPU）
5. 驗證所有套件可匯入，並執行引擎 self-test

## 啟動

```bat
start.bat
```

`start.bat` 會檢查虛擬環境、程式檔案、相依套件與引擎 self-test，再開啟主視窗。

命令列用法（不開視窗）：

```bat
.venv\Scripts\python main.py --self-test        :: 驗證引擎
.venv\Scripts\python main.py --validate a.xlsx  :: 稽核一個資料檔
.venv\Scripts\python main.py --scan             :: 列出模型庫目前看到的模型
```

---

## 四個分頁

### ① 訓練與超參數試探

原 regression v4 的完整流程：拖入 CSV/Excel → 指定欄位角色 → 產生輸入／輸出排列組合 →
選擇模型、scaler、loss、augmentation → 執行 Benchmark Run → 產出 Markdown/PDF 報告與模型檔。

輸出寫在 `runs/run_<時間戳>/`，每次執行都是獨立資料夾，不會覆蓋舊結果。
在「輸出」區塊可以填 **Run 名稱**，這個名稱會寫進每個模型的 metadata，供模型庫分組。

**模型保留規則**：每個 Target Task 只保留排名前 **3** 名的模型；
**Reported R² 未大於 0 的模型一律不儲存，包含第一名** —— R² ≤ 0 代表該模型沒有勝過
「一律預測平均值」的基準，沒有部署價值。被淘汰的模型會在執行紀錄中列出原因。

### ② 模型庫

掃描 `models/` 與所有 `runs/` 底下的模型檔，依 run 分組列出，並顯示每個模型的
**R²** 與 **MAPE**。可以：

- 雙擊重新命名 run 或個別模型
- 勾選／取消「使用」，決定哪些模型會出現在推論分頁
- 寫下用途備註
- 匯入第三方模型檔到模型庫
- 右鍵開啟檔案位置或刪除模型

不相容的模型**不會被隱藏**，而是列在表格中並在「狀態」欄說明原因（例如找不到對應架構、
缺少欄位名稱資訊）。

### ③ 單筆推論

輸入欄位的**數量與名稱、輸出卡片的數量與名稱，全部由所選模型決定**，換模型就重建表單。

### ④ 批次推論與驗證

讀取整份 CSV/Excel 的一個工作表，一次對所有列做推論；可同時選多個模型互相比較。

欄位自動比對（忽略大小寫、空白、`Sheet::` 前綴），也可以手動修正。
**如果把模型的輸出欄位也對應到資料表，這份資料就會被當作 test data 驗證**，並輸出比對報告：

- `report.md` / `report.pdf`：證據界線說明、依 macro NMAE 排名的模型比較表、各目標指標、資料處理稽核
- `metrics.csv`：macro 與各目標的 MAE / RMSE / NMAE / MAPE / Reported R² / Diagnostic R²
- `predictions.csv`、`predictions_wide.csv`：每列的預測值、真值與殘差
- 圖表：predicted vs actual（含 1:1 參考線）、殘差分佈、殘差 vs 預測值、NMAE / R² / MAPE 模型比較長條圖

報告寫在 `reports/batch_<時間戳>/`。

---

## 支援的模型

**訓練引擎可產生**：Mean 基準、Ridge、PLS、SVR (RBF)、Random Forest、Extra Trees、
Deep / Wide / Residual / Multi-branch / Bottleneck MLP、CNN1D、ResNet1D、Grouped Fusion、
Numerical-Embedding MLP、FT-Transformer、ModernNCA，以及選配的 TabM、RealMLP、
CatBoost、XGBoost、LightGBM。

**推論層可載入**：

| 格式 | 說明 |
| --- | --- |
| `.joblib`（regression_v4 bundle） | 上述**所有**模型都以同一個 bundle 介面儲存，因此推論端一律支援 |
| `.pkl` + `.md`（v2.2 感測器格式） | 由 state dict 重建架構；欄位名稱取自 pickle，缺漏時由 `.md` 補上 |
| `.joblib` / `.pkl`（第三方估計器） | 任何有 `predict()` 的物件；欄位名稱取自 scikit-learn 屬性或 `.meta.json` |
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

## 專案結構

```
main.py                 單一入口
install.bat / start.bat 安裝與啟動
project_paths.py        所有資料夾位置的單一來源
regression_core.py      共用核心：資料載入、Experiment 產生、舊版架構、訓練工具
legacy_architectures.py v2.2 架構定義（舊模型重建用）
regression_v4/          訓練與評估引擎
app/                    合併 UI、模型庫、推論與批次驗證
models/                 模型庫（legacy_v2/ 為隨附的兩個示範模型）
runs/                   Benchmark Run 輸出
reports/                批次推論報告
docs/                   合併說明與第三方套件聲明
```

## 授權與第三方套件

見 [`docs/third-party-notices.md`](docs/third-party-notices.md)。
合併過程的設計決定與已知取捨見 [`docs/MERGE_NOTES.md`](docs/MERGE_NOTES.md)。
