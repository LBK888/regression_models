# 合併說明 / Merge notes

記錄 `260801-泰國蝦病毒`（regression v4）與 `AI_Sensor` 合併成本專案時的設計決定。
兩個來源資料夾**完全未被修改**，所有新工作都在本專案中進行。

---

## 1. 哪些檔案被整合、哪些沒有

### 納入

| 來源 | 檔案 | 去向 |
| --- | --- | --- |
| regression v4 | `regression_v4/*.py` | `regression_v4/`（移除 TabPFN、加入保留規則與 metadata） |
| regression v4 | `pytorch_regression_system_v3_0_pyqt.py` 的核心區段 | `regression_core.py` |
| regression v4 | `docs/third-party-notices.md` | `docs/` |
| AI_Sensor | `model_inference_ui.py` 的架構清單與推論邏輯 | `app/adapters.py`、`app/inference_page.py` |
| AI_Sensor | `architectures.py` | `legacy_architectures.py` |
| AI_Sensor | `install.bat` / `start.bat` | 擴充後放在專案根目錄 |
| AI_Sensor | 兩個訓練完成的 `.pkl` + `.md` 模型 | `models/legacy_v2/` |

### 未納入（與程式或模型無關的暫時資料）

`dataset1.xlsx`、`dataset_multiple_test.xlsx`、`泰國蝦資料qPCR.xlsx`、`紅外光總記錄表.xlsx`、
`v4_outputs/`、`tmp/`、`.venv/`、`__pycache__/`、`.vscode/`、
`pytorch_regression_system_v2.2.py`、`pytorch_regression_system_v3_0_pyqt.py` 的 v3 GUI 部分、
`v4 plan.md`、`CONTEXT.md`、`tests/`、v4 開發過程文件。

---

## 2. 從 v3 monolith 抽出 `regression_core.py`

v4 引擎原本 `from pytorch_regression_system_v3_0_pyqt import ...`，但那個檔案是 2,900 行的
單體，同時包含 v3 的完整 GUI、報告產生器與繪圖程式碼 —— 對本專案而言絕大部分是死碼。

抽出的是該檔案中**不含 Qt、不含繪圖**的核心區段：

- 資料層：`DataRole`、`ColumnRef`、`ColumnProfile`、`SheetProfile`、`DatasetBundle`、
  `FlexibleDatasetLoader`、`SelectionUnit`、`ExperimentSpec`、`generate_experiment_specs`
- 架構：`DeepFFN`、`WideNetwork`、`ResNet`、`EnsembleNet`、`AutoEncoderNet`、
  `design_architectures`、`build_model`、`create_scaler`
- 訓練工具：`DEVICE`、`seed_everything`、`EarlyStopping`、`_safe_batch_size`

程式碼逐字保留，因此兩代 pickle 出來的模型都還能載入。

---

## 3. 移除 TabPFN

**結論：移除。**

TabPFN 的 adapter 需要 `tabpfn.browser_auth`，原始碼中甚至有一段 Windows 專用的
patch（`_install_tabpfn_windows_browser_auth_compatibility`）用來處理「開瀏覽器登入並輪詢
token」的流程；capability 說明也寫著 *"model weights require separate license acceptance and
first-use download"*。也就是說，第一次使用必須註冊／登入 Prior Labs 帳號才能取得權重。

這與離線桌面工具的使用情境不符，因此依需求移除。移除範圍：

- `models.py`：`_tabpfn_capability`、`tabpfn_runtime_incompatibility_reason`、
  browser-auth patch、registry 與 configuration 條目
- `domain.py`：`tabpfn_license_acknowledged` / `tabpfn_max_rows` /
  `tabpfn_max_features` / `tabpfn_allow_cpu_over_1000` 四個設定欄位及其驗證
- `ui.py`：授權勾選框、CPU override、安全上限 spinbox、說明文字與相容性 gate

其餘選配模型（TabM、RealMLP、CatBoost、XGBoost、LightGBM）不需登入，保留為
`requirements.txt` 中註解掉的選項。

---

## 4. 模型保留規則：前三名 + R² 閘門

原本每個 Target Task 只保留第 1 名並 refit 儲存。現在：

- 每個 Target Task 保留排名前 **3** 名（`regression_v4.evaluation.MODEL_KEEP_LIMIT`）
- **Reported R² 必須大於 0 才會被儲存，第一名也不例外**
  （`MODEL_KEEP_MIN_REPORTED_R2 = 0.0`）

實作在 `evaluation.select_retained_results()`，Confirmation Stage 走同樣的規則但依
confirmation NMAE 排序。被淘汰的模型會逐一記錄原因到執行紀錄，例如：

```
Not saved (TASK_c4d0f48): mean | native | standard | original
    — rank 6: Reported R2=0.0000 is not above 0.00; no better than the mean baseline
```

**已知取捨**：`Train final only` 策略（test_size = 0）不產生任何評估證據，因此沒有 R²
可以判斷。該路徑維持原本「全部儲存」的行為，metadata 的 `evidence_stage` 標為
`final_fit_only`、`metrics` 為 `null`，模型庫的 R²／MAPE 欄會顯示 `—`。

---

## 5. 訓練時記錄欄位名稱

每個儲存的模型檔旁會寫一個同名的 `.meta.json`（`evaluation.bundle_metadata()`），記錄：

- `feature_names` / `feature_labels`、`target_names` / `target_labels`
- `rank`、`evidence_stage`、`run_label`、`experiment`、`target_task_name`
- 完整的 `metrics`（含 per-target）
- `model_id`、`loss`、`scaler`、`augmentation`、`configuration`、`training_rows`

這讓推論介面**不需要原始資料集**就能決定要顯示幾個輸入欄位、叫什麼名字，
模型庫也不必載入模型就能顯示 R² 與 MAPE。

隨附的兩個 v2.2 舊模型沒有 `.meta.json`，欄位名稱改由 pickle 內的
`feature_names` / `target_names` 提供，缺漏時再由同名 `.md` 補上（`adapters.parse_markdown_sidecar`）。
兩個模型實測都能正確還原 11 個輸入與 1 個輸出。

舊的 Benchmark Run（合併前產生、沒有 sidecar）仍可被模型庫辨識：
`model_library._metrics_from_run_results()` 會用檔名的
`experiment__model__loss__scaler__augmentation` 去 `results.csv` 反查指標。

---

## 6. 推論層對得上所有模型

`app/adapters.py` 把三種產物收斂成同一個 `Predictor` 介面：

1. **regression_v4 bundle**（`.joblib`）—— `FittedModelBundle` 自帶 estimator、x/y scaler 與
   Experiment 身分，`bundle.predict()` 對所有模型家族一致，因此訓練端新增模型不需要
   改推論端。
2. **v2.2 state-dict package**（`.pkl` + `.md`）—— 依序嘗試候選架構直到 `load_state_dict`
   成功；候選來源依序是「metadata 記錄的架構名稱」→「模型旁的 `architectures.py`」→
   內建清單（v3 架構、AI_Sensor 架構、v2.2 架構）。
3. **第三方估計器** —— 任何有 `predict()` 的物件；欄位數取自 `feature_names_in_` /
   `n_features_in_` / sidecar，輸出數以一次試算 probe 取得。

無法還原的檔案不會被靜默略過，而是帶著原因出現在模型庫。

> 原 AI_Sensor 的「動態重建」後備路徑（從 state dict 推測 Linear/BatchNorm 層堆疊）
> **未沿用**。它會產生缺少 activation function 的模型，預測值與原模型不同，卻仍以
> 正常結果呈現。改成明確回報不相容，並提示放置 `architectures.py`。

---

## 7. 單一入口與分頁

`RegressionV4MainWindow` 改寫為可嵌入的 `TrainingPage(QWidget)`，主視窗
`app/main_window.py` 以四個分頁承載。原本的獨立視窗保留為
`regression_v4.ui.RegressionV4MainWindow`，只在單獨除錯引擎時使用。

資料流：

- 訓練完成 → `TrainingPage.run_completed` → 模型庫自動重新掃描
- 模型庫勾選變更 → `LibraryPage.library_changed` → 單筆推論與批次推論重新取得可用模型

模型庫是**唯一**決定推論分頁看得到哪些模型的地方。

---

## 8. 介面多語言（i18n）

採用 gettext 式設計：**程式碼中寫英文原文，英文原文本身就是查詢鍵**。

```python
from i18n import tr

label = QLabel(tr("Model library"))
status = tr("Found {count} model files", count=12)
```

這樣做的理由：

- 英文（預設語言）完全不需要語言檔，`tr()` 直接回傳原字串
- 原始碼維持可讀的英文，不是 `ui.config.epochs_help` 這種不透明的鍵
- 缺漏的翻譯會退回英文，介面仍然可用，而不是顯示缺鍵佔位符

語言檔放在 `locales/<代碼>.py`，內含 `MESSAGES` dict。目前有 `zh_TW.py`，共 447 筆。
佔位符一律使用具名形式（`{count}` 而非 `{0}`），因此譯文可以調整語序。

**語言切換怎麼套用**：

| 頁面 | 作法 |
| --- | --- |
| 主視窗、模型庫、單筆推論、批次推論 | `retranslate()` 就地更新既有 widget 的文字 |
| 訓練頁 | **整頁重建** |

訓練頁的「評估設定」分頁有六十多個 label 與 tooltip 是在 `_build_config_tab()` 中
inline 建立的，要就地更新就得為每一個保留 widget 參照 —— 重建更簡單也更不容易遺漏。
重建會保留已載入的資料集（依路徑重新載入）、輸出資料夾與 Run 名稱。

**已知限制**：

- 訓練或批次執行進行中時無法切換語言，會跳出提示並還原下拉選單的選擇。
  正在執行的 worker 持有 UI 參照，重建會讓它指向已刪除的 widget。
- 訓練頁重建會清掉已產生的排列組合與勾選狀態，需要重新按「產生／更新排列組合」。
  資料集與角色設定不受影響。
- 報告（Benchmark Run 與批次驗證）以**產生當下的語言**寫出，之後切換語言不會改寫已存檔的報告。
- `_apply_identity_suggestions()` 中的中文欄名樣板（`編號`、`樣本id`、`個體id` 等）
  是**比對使用者試算表欄名的資料樣板，不是介面文字**，因此不翻譯也不隨語言改變。

語言選擇存在 `settings.json`（已加入 `.gitignore`，屬於每台機器的個人設定）。
