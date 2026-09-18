# -*- coding: utf-8 -*-
"""繁體中文介面翻譯。

鍵是程式碼中的英文原文，值是對應的中文。缺漏的鍵會自動退回英文，
因此即使翻譯不完整，介面仍然可用。

``{name}`` 這類佔位符必須原樣保留，順序可以依中文語序調整。
"""

MESSAGES = {
    # ------------------------------------------------------------------
    # 主視窗與分頁
    # ------------------------------------------------------------------
    "Regression Models — training, model library and inference":
        "Regression Models — 訓練、模型庫與推論",
    "① Training and hyperparameter search": "① 訓練與超參數試探",
    "② Model library": "② 模型庫",
    "③ Single-sample inference": "③ 單筆推論",
    "④ Batch inference and validation": "④ 批次推論與驗證",
    "Language": "語言",
    "Interface language: {name}": "介面語言：{name}",
    "Wait for the current run to finish before switching language.":
        "請等目前的執行結束後再切換語言。",
    "Compute device: {device}": "運算裝置：{device}",
    "GPU mode": "GPU 模式",
    "CPU mode": "CPU 模式",
    "Training finished: {path} — refreshing the model library…":
        "訓練完成：{path}　正在更新模型庫…",
    "Models available for inference: {count}": "可用於推論的模型：{count} 個",

    # ------------------------------------------------------------------
    # 訓練頁：分頁與資料
    # ------------------------------------------------------------------
    "1. Data and combinations": "1. 資料與組合",
    "2. Evaluation settings": "2. 評估設定",
    "3. Run and report": "3. 執行與報告",
    "Drag a CSV / Excel file here, or use the button below":
        "將 CSV / Excel 檔拖曳到這裡，或按下方按鈕選擇",
    "Choose a data file": "選擇資料檔",
    "No data loaded": "尚未載入資料",
    "Generate / refresh combinations": "產生／更新排列組合",
    "Observation ID and Group ID never become model features.":
        "Observation ID 與 Group ID 不會進入模型特徵。",
    "Sheet / column": "Sheet / 欄位",
    "Role": "角色",
    "Type": "型態",
    "Rows": "列數",
    "Missing": "缺失",
    "Audit": "稽核資訊",
    "numeric": "數值",
    "ID / text candidate": "ID / 文字候選",
    "No combinations generated yet": "尚未產生組合",
    "Filter by ID, Inputs, Outputs or Stable Key…":
        "篩選 ID、Inputs、Outputs、Stable Key…",
    "Select all": "全選",
    "Select none": "全不選",
    "Invert": "反選",
    "Confirm the suggested structure for the selected combinations":
        "確認已選組合的結構建議",
    "Roles changed — regenerate the combinations": "角色已變更，請重新產生組合",
    "Load a CSV or Excel file first.": "請先載入 CSV 或 Excel。",
    "No valid combination": "沒有有效組合",
    "At least one input and one output are required.": "至少需要一組輸入與一組輸出。",
    "Could not generate combinations": "組合產生失敗",
    "Could not load the data": "資料載入失敗",
    "Loaded {sheets} sheet(s) / table(s), {rows:,} rows in total. Name an "
    "Observation ID; if one subject has repeated measurements, name a Group ID "
    "as well to prevent split leakage.":
        "已載入 {sheets} 個 sheets / tables，共 {rows:,} 列。請指定 Observation ID；"
        "若同一個體有重複量測，另指定 Group ID 以避免 split leakage。",
    "Suggested as the Observation ID from the column name — please confirm. "
    "This column never becomes a model feature.":
        "依欄名自動建議為 Observation ID；請確認，這一欄不會進入模型特徵。",
    "Suggested as the Group ID from the column name — please confirm.":
        "依欄名自動建議為 Group ID；請確認。",

    # 欄位角色
    "Skip": "略過",
    "Input": "輸入",
    "Combined input": "合併輸入",
    "Output": "輸出",
    "Input or output": "可作輸入或輸出",
    "Observation ID (not a feature)": "Observation ID（不作為特徵）",
    "Group ID (not a feature)": "Group ID（不作為特徵）",

    # 組合表格
    "Use": "使用",
    "Valid rows": "有效列",
    "Input columns": "輸入欄數",
    "Output columns": "輸出欄數",
    "Structure confirmed": "結構確認",
    "Untick a combination to leave it out of training. Regenerating the "
    "combinations keeps your choice, matched by Stable Key.":
        "取消勾選即可讓這個組合不參與訓練；重新產生組合時會依 Stable Key 保留選擇。",
    "You can change the automatic suggestion. CNN1D, ResNet1D and Grouped "
    "Fusion only run on combinations that are both confirmed and compatible.":
        "可修改自動建議；CNN1D/ResNet1D/Grouped Fusion 只會執行已確認且相容的組合。",
    "Tick this only after confirming the Feature Structure matches what the "
    "data actually means.":
        "確認 Feature Structure 與實際資料語意一致後再勾選。",
    "{selected:,} of {total:,} selected; {confirmed:,} structure(s) confirmed":
        "已選 {selected:,} / {total:,} 組；結構已確認 {confirmed:,}",

    # ------------------------------------------------------------------
    # 訓練頁：Evaluation Strategy
    # ------------------------------------------------------------------
    "Decides how generalization evidence is produced. This is not a parameter "
    "preset: switching strategy only disables the fields that do not apply, and "
    "keeps the values you entered.":
        "決定如何產生泛化證據；這不是參數 preset，切換策略只會停用不適用欄位並保留原數值。",
    "Two-stage benchmark: CV Screening + optional Confirmation":
        "兩階段 Benchmark：CV Screening + optional Confirmation",
    "K-fold Screening on the development data, with optional repeated nested "
    "Confirmation. Test ratio and External mapping are ignored.":
        "以 development data 做 K-fold Screening，可選 repeated nested Confirmation；"
        "忽略 Test ratio 與 External mapping。",
    "Holdout: one Train/Test split": "Holdout：單次 Train/Test split",
    "Splits off one test set at random according to Test ratio. Screening "
    "folds, Confirmation and External mapping are ignored.":
        "依 Test ratio 隨機切出一次測試集；忽略 Screening folds、Confirmation 與 External mapping。",
    "External Validation: an independent second dataset":
        "External Validation：獨立第二資料集",
    "Selects the configuration with development CV, then evaluates once against "
    "the mapped external data. Test ratio is ignored.":
        "先用 development CV 選定設定，再只對映射後的外部資料評估一次；忽略 Test ratio。",
    "Train Final Model Only: train, do not evaluate":
        "Train Final Model Only：只訓練、不評估",
    "Builds the final model straight from every development row. Test ratio, "
    "folds, Confirmation and External mapping are ignored, and no "
    "generalization metric is produced.":
        "全部 development rows 直接建立 final model；忽略 Test ratio、folds、Confirmation "
        "與 External mapping，不產生泛化指標。",
    "Evaluation Strategy picks the evidence workflow. Unlike Run preset it never "
    "rewrites values such as epochs; settings that do not apply are disabled and "
    "ignored at run time.":
        "Evaluation Strategy 決定證據流程，不會像 Run preset 一樣改寫 epochs 等數值；"
        "不適用設定會停用並在執行時忽略。",
    "Strategy": "策略",
    "Chooses how generalization evidence is produced. It is not a preset: it "
    "never rewrites the values below, it only uses or ignores the matching "
    "fields.":
        "選擇泛化證據的產生方式；它不是 preset，不會改寫下方數值，只會使用或忽略對應欄位。",
    "Test ratio": "Test ratio",
    "Holdout only: splits off one test set by percentage. At 0% every row goes "
    "into the final fit, and no test evidence is produced.":
        "只供 Holdout 使用：依百分比分出一次測試資料；設為 0% 時全部資料用於 final fit，"
        "但不產生測試證據。",
    "Screening folds": "Screening folds",
    "Cuts the development data into K parts, evaluating on one and training on "
    "the rest in turn. A larger K means more evaluations and more time.":
        "把 development data 輪流切成 K 份，每次用一份評估、其餘訓練；K 越大評估次數與時間越多。",
    "Random seed": "Random seed",
    "Controls the random sequence for splitting, initialization and "
    "augmentation. The same data, settings and seed reproduce the result.":
        "控制資料切分、初始化與 augmentation 的隨機序列；相同資料與設定使用相同 seed 可重現結果。",
    "After evaluation, refit the final model on all available data":
        "評估完成後，以全部資料重新 fit final model",
    "Once evaluation has chosen a configuration, retrain a deployable final "
    "model on every development row. That fit is not generalization evidence.":
        "評估結束並選定設定後，以全部 development rows 重新訓練可保存的 final model；"
        "這個 fit 不會產生泛化證據。",
    "Run repeated nested Confirmation after Screening":
        "Screening 後執行 repeated nested Confirmation",
    "Re-runs the candidates promoted from Screening through repeated nested CV. "
    "More reliable, substantially more compute, CV strategies only.":
        "將 Screening 晉級的候選模型再做 repeated nested CV；結果較可靠但運算量顯著增加，"
        "只適用 CV 型策略。",
    "Finalists / Target Task": "Finalists / Target Task",
    "How many candidate configurations per identical output target are promoted "
    "to Confirmation. More finalists means a longer Confirmation.":
        "每個相同輸出目標最多晉級多少候選設定到 Confirmation；數量越多，確認時間越長。",
    "Confirmation outer folds": "Confirmation outer folds",
    "How many splits the outer Confirmation CV uses. Every outer test fold stays "
    "entirely out of that round's tuning and training.":
        "Confirmation 外層 CV 的切分數；每個 outer test fold 都完全不參與該次調參與訓練。",
    "Confirmation outer repeats": "Confirmation outer repeats",
    "Repeats the whole outer CV with different seeds, which estimates the "
    "variance of the result and multiplies the compute accordingly.":
        "用不同 seed 重複整套 outer CV；可估計結果變異，但運算量近似按次數倍增。",
    "Confirmation inner folds": "Confirmation inner folds",
    "Inner CV runs only inside each outer training partition, so parameters are "
    "never tuned against outer test data.":
        "只在每個 outer training partition 內做 inner CV 選參數；可避免用 outer test data 調參。",
    "Tuning budget / finalist": "Tuning budget / finalist",
    "How many parameter sets each finalist may try per outer fold. A larger "
    "budget searches wider and takes longer.":
        "每個 finalist 在每個 outer fold 最多嘗試的參數組數；數值越大搜尋更廣但耗時增加。",
    "Compare Original against the selected augmentation on identical outer folds":
        "以相同 outer folds 比較 Original 與已選 augmentation",
    "Compares Original against the enabled augmentation on exactly the same "
    "outer folds. Ignored under Original only.":
        "在完全相同的 outer folds 上比較 Original 與啟用的 augmentation；"
        "Original only 時此項會被忽略。",
    "Note: a greyed-out field is one the current strategy does not use. "
    "Switching strategy keeps whatever you typed.":
        "提示：灰色欄位代表目前策略不使用；切換策略後原輸入值仍會保留。",
    "This is an evaluation workflow, not a preset. It uses Screening folds, with "
    "optional Confirmation. Test ratio and External mapping are ignored; every "
    "other training, model, loss and augmentation setting applies as usual.":
        "這是評估工作流程，不是 preset。使用 Screening folds；可選 Confirmation。"
        "Test ratio 與 External mapping 會忽略，其他 training／model／loss／augmentation "
        "設定照常使用。",
    "This is an evaluation workflow, not a preset. It uses Test ratio for one "
    "Train/Test split. Screening folds, every Confirmation setting and External "
    "mapping are ignored. At Test ratio = 0% it performs the final fit only.":
        "這是評估工作流程，不是 preset。只使用 Test ratio 做一次 Train/Test split；"
        "Screening folds、所有 Confirmation 設定與 External mapping 會忽略。"
        "Test ratio=0% 時只做 final fit。",
    "This is an evaluation workflow, not a preset. It selects a configuration on "
    "the development data with Screening folds and optional Confirmation, then "
    "evaluates once through External mapping. Test ratio is ignored.":
        "這是評估工作流程，不是 preset。先使用 Screening folds 在 development data 選定設定，"
        "可選 Confirmation，最後才使用 External mapping 評估一次；Test ratio 會忽略。",
    "This is an evaluation workflow, not a preset. It trains the final models "
    "straight from every development row. Test ratio, Screening folds, "
    "Confirmation, External mapping and the refit option are all ignored, and no "
    "generalization metric is produced.":
        "這是評估工作流程，不是 preset。使用全部 development rows 直接訓練 final models；"
        "Test ratio、Screening folds、Confirmation、External mapping 與 refit 選項都會忽略，"
        "且不產生泛化指標。",

    # ------------------------------------------------------------------
    # 訓練頁：模型
    # ------------------------------------------------------------------
    "Models (greyed-out entries are kept on purpose, so none is forgotten)":
        "模型（灰色項目刻意保留，以免後續遺漏）",
    "Tick the models to compare. Each extra model expands into more training "
    "runs across the selected scalers, losses, augmentations and repetitions.":
        "勾選要比較的模型；每增加一個模型都會依 scaler、loss、augmentation 與 repetitions "
        "展開更多訓練。",
    "Ticking this adds the model to every compatible Experiment. More models "
    "means more total run time.":
        "勾選後會在每個相容 Experiment 中加入此模型比較；模型越多，總運算時間越長。",
    "Currently unavailable: {reason}": "目前不可執行：{reason}",

    "Mean baseline: predicts the training targets' mean and nothing else. Tick "
    "it to see whether a complex model really beats the simplest possible "
    "prediction.":
        "平均值基準模型：只用訓練資料的目標平均值預測；勾選後可判斷複雜模型是否真的優於最簡單基準。",
    "Ridge linear regression: L2 regularization shrinks unstable coefficients, "
    "which makes the linear fit more robust to collinear features.":
        "Ridge 線性迴歸：以 L2 正則化縮小不穩定係數；勾選後會評估較抗共線性的線性關係。",
    "PLS: compresses highly correlated features into a few latent components "
    "related to the target. Suited to small data with many collinear features.":
        "PLS：把高度相關特徵壓縮成少數與目標相關的潛在成分；適合特徵多且共線的小資料。",
    "RBF-SVR: learns a nonlinear relationship through a kernel while bounding "
    "the error band. Adds one nonlinear small-sample comparison.":
        "RBF-SVR：以核函數學習非線性關係並控制誤差帶；勾選後會增加一組非線性小樣本比較。",
    "Random Forest: averages many randomized decision trees. Captures "
    "nonlinearity and interactions, but takes longer to train.":
        "Random Forest：對多棵隨機決策樹取平均；可描述非線性與交互作用，但訓練時間較長。",
    "Extra Trees: builds a tree ensemble with more random splits. Usually "
    "faster, and reduces the variance of any single tree.":
        "Extra Trees：使用更隨機的切分建立樹集成；通常較快並可降低單棵樹的變異。",
    "Deep MLP: a multi-layer fully connected network. Learns general "
    "nonlinearity and honours the neural loss you selected.":
        "Deep MLP：多層全連接神經網路；會學習一般非線性關係並使用所選 neural loss。",
    "Wide MLP: a wider fully connected network. More representational capacity "
    "per layer, at the cost of more parameters and compute.":
        "Wide MLP：較寬的全連接神經網路；增加同層表示能力，也會增加參數與運算量。",
    "Residual MLP: residual connections let a deeper network train, improving "
    "gradient flow and adding nonlinear capacity.":
        "Residual MLP：以殘差連接訓練較深網路；可改善梯度傳遞並增加非線性容量。",
    "Multi-branch MLP: several parallel paths extract different representations "
    "before merging. More capacity, and more compute.":
        "Multi-branch MLP：以多條並行網路路徑抽取不同表示後合併；會增加模型容量與運算量。",
    "Bottleneck MLP: compresses then rebuilds a higher-level representation, "
    "pushing the model towards a more compact feature combination.":
        "Bottleneck MLP：先壓縮再重建高階表示；可促使模型學到較精簡的特徵組合。",
    "CNN1D: slides a convolution kernel along the ordered 1D features you "
    "confirmed. Only appropriate when neighbouring columns really are locally "
    "related.":
        "CNN1D：沿使用者確認的有序一維特徵滑動卷積核；只適合相鄰欄位確實具有局部關係的資料。",
    "ResNet1D: residual convolution blocks over ordered 1D features. Suited to "
    "sequence-like data; spectral column naming is not required.":
        "ResNet1D：在有序一維特徵上使用殘差卷積區塊；適合序列型資料，不要求光譜命名。",
    "Grouped Fusion: a separate encoder per Feature Group, then a learned "
    "fusion. Runs only when there are at least two meaningful groups.":
        "Grouped Fusion：每個 Feature Group 使用獨立編碼器再融合；只有至少兩個有意義群組時才執行。",
    "Numerical-Embedding MLP: turns each numeric feature into a learnable "
    "representation before the MLP. Captures general tabular nonlinearity.":
        "Numerical-Embedding MLP：先把每個數值特徵轉成可學習表示再交給 MLP；可捕捉一般表格非線性。",
    "FT-Transformer: treats each feature as a token and models feature "
    "interaction with attention. Good for general tabular data, but slower.":
        "FT-Transformer：把每個特徵視為 token 並用 attention 建模特徵互動；"
        "適合一般表格資料但較耗時。",
    "ModernNCA: learns an embedding where similar targets sit close together, "
    "then predicts from neighbours. Useful for exploring local sample structure.":
        "ModernNCA：學習讓相似目標彼此接近的嵌入空間，再以鄰居預測；適合探索局部樣本結構。",
    "TabM: a parameter-shared multi-member tabular network ensemble. Runs only "
    "when the optional package is installed.":
        "TabM：以參數共享的多成員表格神經網路集成預測；套件可用時才會執行。",
    "RealMLP: pytabkit's tuned tabular MLP configuration. Joins the comparison "
    "when the optional package is installed.":
        "RealMLP：使用 pytabkit 的表格 MLP 設定；套件可用時加入比較。",
    "CatBoost: gradient-boosted decision trees, strong on general tabular "
    "nonlinearity. Needs the optional package installed.":
        "CatBoost：梯度提升決策樹；擅長一般表格非線性，安裝 optional package 後才可使用。",
    "XGBoost: regularized gradient-boosted trees. Adds the common tabular "
    "boosting comparison, and more compute.":
        "XGBoost：正則化梯度提升樹；勾選後加入常用表格 boosting 比較，運算量會增加。",
    "LightGBM: leaf-wise gradient-boosted trees, usually fast. Needs the "
    "optional package installed.":
        "LightGBM：以 leaf-wise 策略建立梯度提升樹；通常速度快，安裝 optional package 後才可使用。",

    # ------------------------------------------------------------------
    # 訓練頁：進階設定
    # ------------------------------------------------------------------
    "Advanced training settings (a preset only fills in defaults; every field "
    "stays editable)":
        "進階訓練設定（Preset 只會填入預設值，所有欄位仍可修改）",
    "Controls neural training, repetitions and feature scaling. Run preset is "
    "the only preset that immediately rewrites some of these values.":
        "控制 neural training、重複次數與特徵縮放；Run preset 是唯一會立即改寫部分數值的 preset。",
    "Quick look: fewer epochs, less patience and fewer repetitions. Saves time, "
    "with less stable results.":
        "快速預覽：降低 epochs、patience 與 repetitions，較省時間但結果穩定性較低。",
    "Balanced: moderate epochs and patience with 3 repetitions, trading time "
    "against stability.":
        "平衡模式：使用中等 epochs、patience 與 3 次 repetitions，兼顧時間與穩定性。",
    "Rigorous: more epochs, more patience, wider models and 5 repetitions. "
    "Substantially more compute.":
        "嚴謹模式：提高 epochs、patience、模型寬度與 5 次 repetitions，運算量大幅增加。",
    "Custom: rewrites nothing and keeps your current settings.":
        "自訂模式：不改寫任何數值，保留目前手動設定。",
    "Run preset": "Run preset",
    "The one real parameter preset: choosing Economy, Balanced or Rigorous "
    "immediately rewrites epochs, patience, repetitions, validation ratio and "
    "model multiplier. Nothing else changes.":
        "真正的參數 preset：切換 Economy／Balanced／Rigorous 會立即改寫 epochs、patience、"
        "repetitions、validation ratio 與 model multiplier；其他欄位不變。",
    "Epochs": "Epochs",
    "How many complete passes over the training data a neural model may make. "
    "More can learn more, at the cost of time and overfitting risk.":
        "Neural model 最多完整掃過 training data 的次數；較大可能學得更充分，"
        "但會增加時間並可能過擬合。",
    "Early-stopping patience": "Early-stopping patience",
    "How many epochs without validation-loss improvement to allow before "
    "stopping. Larger waits longer; smaller may stop too early.":
        "Validation loss 連續多少 epochs 沒改善才停止；較大會等待更久，較小可能過早停止。",
    "Batch size": "Batch size",
    "How many training rows each gradient update uses. Larger is usually faster "
    "but needs more memory; smaller makes updates noisier.":
        "每次梯度更新使用的 training rows 數；較大通常較快但耗記憶體，較小更新較有隨機性。",
    "Learning rate": "Learning rate",
    "The step size the neural optimizer takes per weight update. Too large is "
    "unstable; too small converges slowly.":
        "Neural optimizer 每次更新權重的步幅；太大可能不穩定，太小會收斂緩慢。",
    "Model width multiplier": "Model width multiplier",
    "Scales the neural hidden width up or down. Larger raises capacity, memory "
    "use and overfitting risk together.":
        "按倍率放大或縮小 neural hidden width；較大增加模型容量、記憶體與過擬合風險。",
    "Training repetitions": "Training repetitions",
    "Repeats every configuration with a different random initialization and "
    "pools the predictions. Reduces luck, multiplies the time.":
        "用不同隨機初始化重複每個 configuration 並彙整預測；可降低偶然性，但時間近似按倍數增加。",
    "Internal validation ratio": "Internal validation ratio",
    "Carves an early-stopping watch set out of the neural model's fold-training "
    "partition only. Fold test data is never touched.":
        "只從 neural model 的 fold-training partition 再切一部分監控 early stopping；"
        "不會取用 fold test data。",
    "Scalers": "Scalers",
    "Tick one or more fold-local scalings. Each extra scaler adds a whole set of "
    "model configurations.":
        "勾選一種或多種 fold-local scaling；每多一種 scaler 都會增加一整組 model configurations。",
    "Controls how X and y are scaled inside each training fold. Every ticked "
    "scaler is trained and evaluated separately.":
        "控制每個 training fold 如何縮放 X 與 y；每個勾選 scaler 都會分開訓練與評估。",
    "Standard scaler: uses the training fold's mean and standard deviation to "
    "reach roughly zero mean and unit scale. Suits most models.":
        "Standard scaler：以 training fold 的平均值與標準差轉成約為零均值、單位尺度；適合多數模型。",
    "Robust scaler: scales by the median and interquartile range, so outliers "
    "matter less.":
        "Robust scaler：以中位數與四分位距縮放；較不受離群值影響。",
    "MinMax scaler: scales into a fixed range using the training fold's minimum "
    "and maximum. More sensitive to extreme values.":
        "MinMax scaler：依 training fold 最小與最大值縮放到固定範圍；對極端值較敏感。",

    # ------------------------------------------------------------------
    # 訓練頁：Loss
    # ------------------------------------------------------------------
    "Loss functions (only for neural adapters that accept a custom loss)":
        "Loss functions（只套用於支援自訂 loss 的 neural adapters）",
    "The loss decides how a neural model measures prediction error while "
    "training. Each extra tick adds one neural configuration.":
        "Loss 決定 neural model 訓練時如何衡量預測誤差；每多勾一種就增加一組 neural configuration。",
    "Huber / SmoothL1 (suggested for noisy measurements)":
        "Huber / SmoothL1（建議用於 noisy measurements）",
    "Classical models use their own native objective, so ticking several losses "
    "does not run them more than once.":
        "Classical models 使用其原生 objective，不會因勾選多個 loss 而重複執行。",
    "MSE: squares the error, so large outlying deviations dominate. Ticking it "
    "trains one more MSE configuration for every neural model.":
        "MSE：平方較大的誤差，因此會更重視離群的大偏差；勾選後 neural models 會多訓練一組 "
        "MSE configuration。",
    "Huber / SmoothL1: squared for small errors, roughly linear for large ones, "
    "which limits how much outliers steer neural training.":
        "Huber／SmoothL1：小誤差用平方、大誤差改用近似線性；可降低離群值對 neural training 的影響。",
    "MAE / L1: every absolute error counts in proportion. Less sensitive to "
    "outliers, but the gradient is less smooth.":
        "MAE／L1：所有絕對誤差按比例計算；較不受離群值影響，但梯度較不平滑。",
    "Log-cosh: close to MSE for small errors and to MAE for large ones — a "
    "smooth, outlier-tolerant compromise.":
        "Log-cosh：小誤差近似 MSE、大誤差近似 MAE；提供平滑且較耐離群值的折衷。",

    # ------------------------------------------------------------------
    # 訓練頁：Augmentation
    # ------------------------------------------------------------------
    "Decides how synthetic training rows enter the comparison. Original only "
    "ignores every ticked method.":
        "決定 synthetic training rows 如何加入比較；Original only 會忽略所有已勾選 methods。",
    "Original only (no augmentation)": "Original only（不做 augmentation）",
    "Compare separately against the original data (suggested)":
        "與原始資料分開比較（建議）",
    "Combine the ticked methods": "合併已勾選方法",
    "Builds the Original configuration only. Your method ticks are kept but "
    "completely ignored at run time.":
        "只建立 Original configuration；methods 的勾選狀態會保留但執行時完全忽略。",
    "Builds Original plus one separate configuration per ticked method, so you "
    "can see directly whether each method improves the result.":
        "建立 Original 與每個已勾選 method 的獨立 configuration，可直接判斷各方法是否改善結果。",
    "Builds Original plus one configuration combining every ticked method. "
    "Tests the combination, but cannot attribute the effect to one method.":
        "建立 Original 與一個合併所有已勾選 methods 的 configuration；"
        "可測組合效果但無法分辨單一方法貢獻。",
    "Policy": "Policy",
    "Controls how augmentation configurations expand: Original only skips "
    "augmentation, Separate compares each method on its own, Combined applies "
    "them together.":
        "控制 augmentation configurations 的展開方式；Original only 不做 augmentation，"
        "Separate 分開比較，Combined 合併方法。",
    "Methods": "Methods",
    "C-Mixup (target-aware)": "C-Mixup（target-aware）",
    "Calibrated feature noise": "Calibrated feature noise",
    "Spectral perturbation (needs a valid wavelength axis)":
        "Spectral perturbation（需有效波長軸）",
    "Training-time feature masking": "Training-time feature masking",
    "Choose which training-only augmentations to generate. Policy decides "
    "whether they are compared separately or combined.":
        "選擇要產生的 training-only augmentation；是否分開或合併由 Policy 決定。",
    "Tick augmentation algorithms. Under Original only they are all ignored; "
    "under Separate each is compared on its own; under Combined they are applied "
    "together.":
        "勾選 augmentation 演算法；Original only 時全部忽略，Separate 時各自比較，"
        "Combined 時一起套用。",
    "Include the original rows in the augmented training set":
        "將原始資料加入 augmentation 訓練集",
    "Synthetic rows / train rows": "Synthetic rows / train rows",
    "How many synthetic rows each augmentation produces relative to the "
    "fold-training rows. 0.5 means about half as many.":
        "每次 augmentation 產生的 synthetic rows 相對於 fold-training rows 的比例；"
        "0.5 表示產生約一半數量。",
    "Augmentation repeats": "Augmentation repeats",
    "Multiplies the synthetic row count again. More data and more training time "
    "— this is not the same as model training repetitions.":
        "把 synthetic row 數量再乘上此次數；增加資料量與訓練時間，但不是模型 training repetitions。",
    "When ticked, an augmented variant trains on the original rows plus the "
    "synthetic ones. When cleared, that variant uses synthetic rows only; the "
    "Original baseline is kept either way.":
        "勾選時 augmented variant 使用 Original rows 加 synthetic rows；"
        "取消時該 variant 只使用 synthetic rows，Original baseline 仍保留。",
    "C-Mixup alpha": "C-Mixup alpha",
    "The Beta distribution shape parameter controlling the interpolation weight "
    "between two rows. Larger mixes closer to the middle; smaller stays closer "
    "to one of them.":
        "Beta 分布的形狀參數，控制兩筆資料內插權重；較大更接近中間混合，較小更靠近其中一筆。",
    "Noise fraction": "Noise fraction",
    "Calibrated noise as a fraction of the training feature's standard "
    "deviation. Larger perturbs more, and is more likely to leave the plausible "
    "measurement range.":
        "Calibrated noise 相對於 training feature 標準差的比例；"
        "越大擾動越強，也越可能偏離合理量測範圍。",
    "Mask probability": "Mask probability",
    "The chance that each training feature is masked to its scaled mean. Larger "
    "regularizes harder, and can discard too much information.":
        "每個 training feature 被遮蔽為縮放後平均值的機率；越大正則化越強，也可能損失過多資訊。",
    "FOMA alpha": "FOMA alpha",
    "The Beta scaling coefficient FOMA applies to non-leading singular "
    "components. Larger favours moderate scaling; smaller more often either "
    "preserves or strongly compresses them.":
        "控制 FOMA 非主要奇異成分的 Beta 縮放係數；較大偏向中等縮放，"
        "較小較常接近保留或大幅壓縮。",
    "FOMA retained components (k)": "FOMA retained components (k)",
    "How many leading joint X/y SVD components FOMA keeps intact. A larger k "
    "preserves more of the data's main structure and perturbs more "
    "conservatively.":
        "FOMA 完整保留的前 k 個聯合 X/y SVD 成分；k 越大資料主結構保留越多、擾動越保守。",
    "Every method acts only on the current fold's training partition, and the "
    "Original baseline is always kept. Bootstrap is not counted as "
    "augmentation; SMOGN needs a target relevance definition first, so this "
    "version offers no blind switch for it.":
        "所有方法只作用於當前 fold 的 training partition，並保留 Original baseline。"
        "Bootstrap 不列為 augmentation；SMOGN 必須先定義 target relevance，"
        "因此本版不提供任意開關。",
    "C-Mixup: picks two training rows with similar target values and "
    "interpolates both X and y, producing target-aware synthetic rows.":
        "C-Mixup：依目標值相近程度挑選兩筆訓練資料並內插 X 與 y；"
        "勾選後會建立 target-aware synthetic rows。",
    "Calibrated noise: adds small random noise to X, scaled by how spread the "
    "training features are, and keeps y at its source value. Models measurement "
    "variation.":
        "Calibrated noise：依訓練特徵離散程度對 X 加入小幅隨機雜訊，y 保持來源值；"
        "用來模擬量測變動。",
    "Spectral perturbation: adds smooth baseline, multiplicative and noise "
    "variation along a valid wavelength axis. Non-spectral Experiments are "
    "skipped automatically.":
        "Spectral perturbation：沿有效波長軸加入平滑基線、倍率與雜訊變化；"
        "非光譜 Experiment 會自動略過。",
    "Feature masking: randomly masks some scaled features during training, so "
    "the model cannot lean too hard on one column.":
        "Feature masking：訓練時隨機遮蔽部分已縮放特徵；迫使模型不要過度依賴單一欄位。",
    "FOMA: takes an SVD of the fold-training joint X/y and shrinks the "
    "non-leading manifold components, producing synthetic rows near the "
    "manifold.":
        "FOMA：對 fold-training 的聯合 X/y 做 SVD 並縮放非主要流形成分；"
        "用來產生流形附近的 synthetic rows。",
    "Right now only Original runs: every ticked method and its parameters are "
    "ignored. Your ticks are kept, so switching policy resumes them.":
        "目前只執行 Original：所有已勾選 methods 與其參數都會忽略；"
        "勾選狀態會保留，切回其他 policy 即可繼續使用。",
    "Right now this builds the Original baseline plus one configuration per "
    "ticked method. The methods are never mixed.":
        "目前會建立 Original baseline，再將每個已勾選 method 各自建立一個 configuration；"
        "方法之間不混合。",
    "Right now this builds the Original baseline plus one configuration "
    "combining every ticked method. The report can judge the combination as a "
    "whole, but cannot attribute the effect to one method.":
        "目前會建立 Original baseline，再建立一個合併所有已勾選 methods 的 configuration；"
        "報告只能判斷整體組合，不能拆解單一方法貢獻。",

    # ------------------------------------------------------------------
    # 訓練頁：External Validation
    # ------------------------------------------------------------------
    "For the External Validation strategy only: map each development column to "
    "the independent external data. The external target is read only after the "
    "model has been selected.":
        "只供 External Validation 策略使用：將 development 欄位逐一對應到獨立外部資料，"
        "模型選定後才讀取外部 target。",
    "Load independent external data": "載入獨立外部資料",
    "Loads a fully independent file that took no part in development selection. "
    "Used for exactly one final evaluation under the External Validation "
    "strategy.":
        "載入完全獨立、未參與 development selection 的資料檔；"
        "只在 External Validation 策略執行一次最終評估。",
    "Not loaded; used by the External Validation strategy only":
        "尚未載入；只在 External Validation 策略使用",
    "Use the external row position": "使用外部資料列位置",
    "External Observation ID (optional)": "External Observation ID（選填）",
    "The column that uniquely identifies each external observation. Without one, "
    "rows are aligned by position and the report says so.":
        "選擇外部資料中可唯一識別每筆觀測的欄位；未指定時用資料列位置對齊並在報告中警告。",
    "Names the external observation identifier. It is used for tracking and "
    "alignment only, never as a model input feature.":
        "指定外部資料的觀測識別欄；只用於追蹤與對齊，不會當成模型輸入特徵。",
    "Development column": "Development 欄位",
    "External column": "External 欄位",
    "The left column lists what the model was built on; pick the matching "
    "external column on the right. Every input and target must be mapped "
    "explicitly.":
        "左欄是模型建立時使用的欄位，右欄選擇外部資料的對應欄位；"
        "所有 inputs 與 target 都必須明確映射。",
    "Every model feature and target must be mapped explicitly. The external "
    "target is not read until development CV and Confirmation have finished and "
    "the winning configuration is frozen.":
        "必須明確映射所有入模特徵與目標。外部 target 在 development CV／Confirmation "
        "完成並凍結勝出設定前不會被讀取。",
    "Choose the independent External Validation data file":
        "選擇獨立 External Validation 資料檔",
    "Could not load the external data": "外部資料載入失敗",
    "— select —": "— 請選擇 —",
    "External mapping is incomplete": "External mapping 不完整",
    "External mapping is incomplete: {names}": "External mapping 尚未完成：{names}",
    "External data required": "尚需外部資料",
    "Load the independent External Validation data and finish the column "
    "mapping.":
        "請載入獨立 External Validation 資料並完成欄位 mapping。",

    # ------------------------------------------------------------------
    # 訓練頁：輸出與執行
    # ------------------------------------------------------------------
    "The parent folder for every result, figure, Markdown file, PDF and final "
    "model bundle.":
        "指定所有實驗結果、圖表、Markdown、PDF 與 final model bundles 的主輸出資料夾。",
    "The parent report folder. Each run creates its own timestamped subfolder "
    "inside it, so earlier results are never overwritten.":
        "這是報告主資料夾；每次開始執行都會在其中自動建立帶時間的獨立 run 子資料夾，"
        "避免覆蓋舊結果。",
    "Browse": "選擇",
    "Opens a folder picker and fills in the Master report folder.":
        "開啟資料夾選擇器並把選定路徑填入 Master report 資料夾。",
    "Master report folder": "Master report 資料夾",
    "The parent directory for every Benchmark Run. Each run creates a "
    "run_<date>_<time>_<microseconds> subfolder, and never moves or overwrites "
    "earlier results.":
        "作為所有 Benchmark Runs 的上層目錄；每次執行會建立 run_年月日_時間_微秒子資料夾，"
        "不會搬移或覆蓋先前結果。",
    "Run name": "Run 名稱",
    "e.g. shrimp qPCR first pass (blank uses the folder name)":
        "例如：泰國蝦 qPCR 初篩（留空則用資料夾名稱）",
    "This name is written into every retained model's metadata, and is what "
    "tells training batches with different purposes apart on the Model library "
    "tab.":
        "這個名稱會寫進每個保留模型的 metadata，之後在「模型庫」頁面用來分辨"
        "不同實驗目的的訓練批次。",
    "Only the top {limit} models per Target Task are kept. A model whose "
    "Reported R² is not above 0 is never saved, including the first-ranked one.":
        "每個 Target Task 只保留排名前 {limit} 的模型；Reported R² 未大於 0 的模型"
        "（包含第一名）不會被儲存。",
    "Choose the output folder": "選擇輸出資料夾",
    "Start": "開始執行",
    "Cancel": "取消",
    "Training log — not started": "Training log — 尚未開始",
    "Training log — completed — {path}": "Training log — 已完成 — {path}",
    "Training log — completed, but the PDF could not be rendered":
        "Training log — 已完成，但 PDF 產生失敗",
    "Cancellation requested; the run stops after the current fold.":
        "已送出取消請求；目前 fold 完成後停止。",
    "Run cancelled.": "已取消訓練。",
    "Run failed": "執行失敗",
    "Finished": "完成",
    "Master report written to:\n{path}": "Master report 已輸出：\n{path}",
    "Training finished, but the PDF failed": "訓練完成，但 PDF 產生失敗",
    "The CSV, Markdown, PNG, model and audit files are all intact.\n"
    "Error log: {path}":
        "CSV、Markdown、PNG、模型與稽核檔均已保留。\n錯誤紀錄：{path}",
    "Nothing to run": "執行計畫為空",
    "Tick at least one data combination and one available model.":
        "請至少勾選一個資料組合與一個可用模型。",
    "Incomplete settings": "設定不完整",
    "Load the data and assign column roles first.": "請先載入資料並設定角色。",
    "A run is in progress": "訓練進行中",
    "Cancel it first and wait for the current fold to finish.":
        "請先取消並等待目前 fold 結束。",
    "  Warning: no Observation ID was named; row position is used as the "
    "alignment fallback.":
        "  警告：未指定 Observation ID，目前使用列位置作為對齊備援。",

    # ------------------------------------------------------------------
    # 模型庫
    # ------------------------------------------------------------------
    "Model library": "模型庫",
    "Model library / {folder}": "模型庫／{folder}",
    "External path": "外部路徑",
    "Not scanned yet": "尚未掃描",
    "🔄 Rescan": "🔄 重新掃描",
    "➕ Import model files": "➕ 匯入模型檔",
    "Name your runs and models here, and tick the ones you want to see on the "
    "inference tabs. Third-party and legacy models are listed too; anything "
    "that cannot run says why in the Status column.":
        "在這裡為 run 與模型命名、勾選要在推論頁面看到的模型。"
        "第三方或舊格式模型也會列出；無法執行的會在「狀態」欄說明原因。",
    "Filter by run, model name, target or status…":
        "篩選 run、模型名稱、目標或狀態…",
    "Show runnable models only": "只顯示可執行的模型",
    "Tick all": "全部勾選",
    "Untick all": "全部取消",
    "Name / Run": "名稱 / Run",
    "Model": "模型",
    "Inputs": "輸入",
    "Outputs": "輸出",
    "Kind": "類型",
    "Status": "狀態",
    "Details and notes for the selected item (notes are saved automatically)":
        "選取項目詳細資料 / 備註（備註會自動儲存）",
    "Describe what this model or run is for…": "為這個模型或 run 寫下用途說明…",
    "Scanning…": "掃描中…",
    "Scanning {current}/{total}: {name}": "掃描中 {current}/{total}：{name}",
    "{total} model file(s); {usable} runnable; {enabled} ticked for inference":
        "共 {total} 個模型檔；可執行 {usable} 個；已勾選供推論使用 {enabled} 個",
    "Cannot be ticked": "無法勾選",
    "“{name}” cannot run, so it cannot be added to the inference tabs.\n\n"
    "{reason}":
        "「{name}」目前不可執行，因此不能加入推論頁面。\n\n{reason}",
    "Run folder: {path}": "Run 資料夾：{path}",
    "File: {path}": "檔案：{path}",
    "Kind: {kind}": "類型：{kind}",
    "Architecture / model: {name}": "架構／模型：{name}",
    "Run: {name}": "Run：{name}",
    "Experiment: {name}": "Experiment：{name}",
    "Target Task: {name}": "Target Task：{name}",
    "Rank inside this Target Task: {rank}": "該 Target Task 內排名：第 {rank} 名",
    "Evidence stage: {stage}": "證據階段：{stage}",
    "Saved at: {timestamp}": "儲存時間：{timestamp}",
    "Metrics: {metrics}": "指標：{metrics}",
    "Inputs ({count}): {names}": "輸入（{count}）：{names}",
    "Outputs ({count}): {names}": "輸出（{count}）：{names}",
    "Status: {status}": "狀態：{status}",
    "Open the containing folder": "在檔案總管開啟位置",
    "Remove from the library (deletes the file)": "從模型庫移除（刪除檔案）",
    "Delete model file": "刪除模型檔",
    "Permanently delete this file?\n\n{path}\n\nAny matching .meta.json and .md "
    "sidecar is deleted with it.":
        "要永久刪除這個檔案嗎？\n\n{path}\n\n（同名的 .meta.json / .md 說明檔也會一併刪除）",
    "Delete failed": "刪除失敗",
    "Choose model files to import": "選擇要匯入模型庫的檔案",
    "Model files ({patterns})": "模型檔（{patterns}）",
    "File already exists": "檔案已存在",
    "{name} is already in the library. Overwrite it?":
        "{name} 已經在模型庫中，要覆蓋嗎？",
    "Import failed": "匯入失敗",

    # 模型相容性
    "Runnable": "可執行",
    "Runnable ({notes})": "可執行（{notes}）",
    "Incompatible: {reason}": "不相容：{reason}",
    "Incompatible: loading raised {error_type}: {error}":
        "不相容：載入時發生 {error_type}: {error}",
    "regression_v4 bundle": "regression_v4 bundle",
    "legacy state-dict package": "舊版 state-dict 模型包",
    "wrapped estimator": "包裝過的估計器",
    "third-party estimator": "第三方估計器",
    "unknown": "未知",
    "architecture={name}": "架構={name}",
    "scaler={name}": "scaler={name}",
    "no scaler stored; raw feature values are fed to the network":
        "沒有儲存 scaler；會直接把原始特徵值送進網路",
    "feature names were not recorded; placeholder names are shown in order":
        "沒有記錄欄位名稱；依順序顯示替代名稱",
    "the file no longer exists": "檔案已不存在",
    "the pickle has no 'model_state_dict' entry": "pickle 中沒有 'model_state_dict'",
    "the pickled dict has neither 'model_state_dict' nor a 'model'/'estimator' "
    "entry":
        "pickle 的 dict 既沒有 'model_state_dict'，也沒有 'model'／'estimator'",
    "no feature names or input_dim were recorded in the package or its .md":
        "模型包與其 .md 都沒有記錄欄位名稱或 input_dim",
    "no known architecture accepts this state dict (recorded name {name}; tried "
    "{count} candidate class(es)). Put a matching class in an architectures.py "
    "file next to the model.":
        "沒有任何已知架構能載入這個 state dict（記錄的名稱為 {name}；已嘗試 {count} 個候選類別）。"
        "請在模型旁建立 architectures.py 並定義對應的 class。",
    "a bare nn.Module was saved without feature/target names; save a "
    "regression_v4 bundle or add a sidecar .meta.json":
        "這是沒有欄位名稱的裸 nn.Module；請改存 regression_v4 bundle，或補上 .meta.json 說明檔",
    "loaded a {type_name} which exposes no predict() method":
        "載入的是 {type_name}，沒有 predict() 方法",
    "the number of input features is unknown: the estimator exposes neither "
    "feature_names_in_ nor n_features_in_, and no sidecar .meta.json records "
    "them":
        "無法得知輸入特徵數：估計器沒有 feature_names_in_ 或 n_features_in_，"
        "也沒有 .meta.json 記錄",
    "a trial prediction failed: {error}": "試算預測失敗：{error}",
    "torch.load failed: {error}": "torch.load 失敗：{error}",
    "joblib.load failed: {error}": "joblib.load 失敗：{error}",
    "pickle.load failed: {error}": "pickle.load 失敗：{error}",
    "unsupported .pt payload of type {type_name}": "不支援的 .pt 內容型別 {type_name}",
    "unsupported file type {suffix}": "不支援的檔案類型 {suffix}",
    "Expected {expected} features, received {received}":
        "預期 {expected} 個特徵，實際收到 {received} 個",
    "The model returned a different number of rows than it was given":
        "模型回傳的列數與輸入不同",
    "The model returned {returned} output column(s) but {expected} target "
    "name(s) are recorded":
        "模型回傳 {returned} 個輸出欄位，但記錄了 {expected} 個目標名稱",

    # ------------------------------------------------------------------
    # 單筆推論
    # ------------------------------------------------------------------
    "Model selection": "模型選擇",
    "Model:": "使用模型：",
    "Tick the models you want to use on the Model library tab.":
        "請先到「模型庫」勾選要使用的模型。",
    "Input feature values": "輸入特徵值",
    "Enter a value…": "輸入數值…",
    "Predicted target": "預測目標",
    "Prediction": "推論結果",
    "Waiting for a prediction…": "等待推論…",
    "🔮  Predict": "🔮  預測",
    "🗑 Clear": "🗑 清除",
    "— select a model —": "— 請選擇模型 —",
    "No model is both ticked and runnable. Tick one on the Model library tab, or "
    "run a training pass first.":
        "目前沒有已勾選且可執行的模型。請到「模型庫」分頁勾選，或先完成一次訓練。",
    "{count} model(s) available — pick one to start.":
        "可用模型 {count} 個，請選擇一個開始推論。",
    "⏳ Loading the model…": "⏳ 載入模型中…",
    "✅ Loaded {architecture} — {inputs} input(s), {outputs} output(s), on "
    "{device}":
        "✅ 已載入：{architecture}　輸入 {inputs}　輸出 {outputs}　裝置 {device}",
    "❌ Loading failed: {error}": "❌ 載入失敗：{error}",
    "Metrics recorded when this model was evaluated: {metrics} (from its "
    "evaluation stage, not the accuracy of this input)":
        "訓練時記錄的評估指標：{metrics}（來自該模型的評估階段，不是本次輸入的準確度）",
    "This model has no recorded evaluation metrics.": "這個模型沒有記錄評估指標。",
    "“{name}” has not been filled in": "「{name}」尚未填寫",
    "“{name}” is not a valid number: {value}": "「{name}」格式不正確：{value}",
    "Invalid input": "輸入錯誤",
    "Inference error": "推論錯誤",

    # ------------------------------------------------------------------
    # 批次推論
    # ------------------------------------------------------------------
    "1. Data source": "1. 資料來源",
    "Choose a CSV / Excel file": "選擇 CSV / Excel",
    "No table loaded yet": "尚未載入資料表",
    "Worksheet:": "工作表：",
    "Observation ID column:": "觀測 ID 欄：",
    "Row and column counts appear once a table is loaded.":
        "載入後會顯示列數與欄位數。",
    "Worksheet “{sheet}”: {rows:,} rows × {columns} columns":
        "工作表「{sheet}」：{rows:,} 列 × {columns} 欄",
    "Choose the table to run inference on": "選擇批次推論資料表",
    "Could not load the table": "資料載入失敗",
    "Data ({patterns})": "資料檔（{patterns}）",
    "2. Models (tick several to compare them)": "2. 選擇模型（可多選比較）",
    "3. Column mapping (matched automatically, editable)":
        "3. 欄位對應（自動比對，可手動修正）",
    "Model column name": "模型欄位名稱",
    "Table column": "資料表欄位",
    "(not mapped)": "（不對應）",
    "Map the model's output columns as well and this table is treated as test "
    "data: the run then produces a validation comparison report.":
        "把模型的輸出欄位也對應到資料表，就會把這份資料當作 test data 驗證並輸出比對報告。",
    "No model is both ticked and runnable. Tick the ones you want on the Model "
    "library tab first.":
        "目前沒有已勾選且可執行的模型。請先到「模型庫」分頁勾選要使用的模型。",
    "Inputs: {names}": "輸入：{names}",
    "Outputs: {names}": "輸出：{names}",
    "⚠️ {count} input column(s) are still unmapped: {names}":
        "⚠️ 還有 {count} 個輸入欄位未對應：{names}",
    "✅ Inputs and every ground-truth column are mapped: this run will produce a "
    "validation comparison report.":
        "✅ 輸入與全部 ground truth 欄位都已對應：這次會輸出驗證比對報告。",
    "Inputs are fully mapped. Map the output columns too and this table is "
    "treated as test data.":
        "輸入已對應完成。把輸出欄位也對應到資料表，就會把這份資料當作 test data 驗證。",
    "4. Run and results": "4. 執行與結果",
    "▶ Run batch inference": "▶ 執行批次推論",
    "Open the report folder": "開啟報告資料夾",
    "Validation metrics": "驗證指標",
    "Prediction preview": "預測結果預覽",
    "Run log": "執行紀錄",
    "Message": "訊息",
    "No ground-truth column was mapped, so there are no validation metrics.":
        "沒有 ground truth 欄位，因此沒有驗證指標。",
    "No table loaded": "尚未載入資料",
    "Choose the CSV or Excel file to run inference on first.":
        "請先選擇要批次推論的 CSV 或 Excel 檔。",
    "No model selected": "尚未選擇模型",
    "Tick at least one model.": "請至少勾選一個模型。",
    "Incomplete column mapping": "欄位對應不完整",
    "These models still have unmapped input columns and will be reported as "
    "failed:\n\n{names}\n\nRun anyway?":
        "這些模型還有未對應的輸入欄位，執行後會被標記為失敗：\n\n{names}\n\n仍要繼續嗎？",
    "Cancellation requested; the run stops after the current model.":
        "已送出取消請求；目前模型完成後停止。",
    "Report written to: {path}": "報告已輸出：{path}",
    "The PDF could not be produced (every other file is intact): {error}":
        "PDF 產生失敗（其他檔案不受影響）：{error}",
    "Batch inference finished": "批次推論完成",
    "{total} model(s) finished, {validated} of them with ground truth to "
    "validate against.\n\nReport folder:\n{path}":
        "{total} 個模型完成，其中 {validated} 個有 ground truth 可驗證。\n\n報告資料夾：\n{path}",
    "Batch inference failed": "批次推論失敗",
    "Could not open the folder": "無法開啟資料夾",
    "Loading failed: {error_type}: {error}": "載入失敗：{error_type}: {error}",
    "loading failed: {error}": "載入失敗：{error}",
    "{rows} row(s)": "{rows} 列",
    "predicted {rows} row(s); no ground truth to validate against":
        "{rows} 列預測完成（無 ground truth）",

    # 批次推論引擎
    "{count} input column(s) are still unmapped: {names}":
        "尚未對應 {count} 個輸入欄位：{names}",
    "The table has no such column(s): {names}": "資料表沒有這些欄位：{names}",
    "Only some ground-truth columns are mapped, so this run produces "
    "predictions without validation.":
        "只對應到部分 ground truth 欄位，因此這次只輸出預測值，不進行驗證。",
    "No row has a complete set of input values (and ground truth, where it was "
    "mapped).":
        "沒有任何一列同時具備完整的輸入值（以及 ground truth，若有選取）。",
    "Inference failed: {error_type}: {error}": "推論失敗：{error_type}: {error}",
    "Validation needs at least two ground-truth rows; R² cannot be computed.":
        "驗證需要至少兩列 ground truth，R² 無法計算。",
    "Metric calculation failed: {error}": "指標計算失敗：{error}",
    "{count} row(s) were skipped because an input value was missing or not "
    "numeric.":
        "{count} 列因輸入值缺失或非數值而略過。",
    "{count} row(s) were left out of the validation because ground truth was "
    "missing.":
        "{count} 列因 ground truth 缺失而未納入驗證。",

    # ------------------------------------------------------------------
    # 批次驗證報告
    # ------------------------------------------------------------------
    "Batch inference and validation report": "批次推論與驗證報告",
    "Generated: {timestamp}": "產生時間：{timestamp}",
    "Source: `{path}`": "資料來源：`{path}`",
    "Worksheet: `{sheet}` ({rows:,} rows)": "工作表：`{sheet}`（共 {rows:,} 列）",
    "Models in this run: {count}": "參與模型：{count} 個",
    "Evidence boundary": "證據界線",
    "The metrics below compare each model's predictions against the ground truth "
    "in this table. They are independent generalization evidence only if these "
    "observations **never took part in that model's training or tuning**. If "
    "this table overlaps the training data, read it as a reproducibility check "
    "rather than as new evidence.":
        "以下指標是模型對本表格的預測與表格內 ground truth 的比較。"
        "只有當這些觀測值**從未參與該模型的訓練或調參**時，這些數字才構成獨立的泛化證據；"
        "若本表格與訓練資料重疊，請把它視為重現性檢查而非新的證據。",
    "No usable ground-truth column was mapped, so this run reports predictions "
    "only and computes no validation metrics.":
        "本次沒有可用的 ground truth 欄位，因此只輸出預測值，不計算任何驗證指標。",
    "Reported R² is clamped at 0: a 0 means the model did not beat the "
    "always-predict-the-mean baseline. Diagnostic R² keeps negative values for "
    "diagnosis. MAPE is descriptive only — it distorts as the true value "
    "approaches 0 — so it is never used for ranking.":
        "Reported R² 已在 0 截斷：0 代表模型沒有勝過「一律預測平均值」的基準。"
        "Diagnostic R² 保留負值供診斷。MAPE 只作描述用，當真值接近 0 時會失真，"
        "因此不作為排名依據。",
    "Model comparison (ranked by macro NMAE)": "模型比較（依 macro NMAE 排名）",
    "Rank": "排名",
    "Reported R²": "Reported R²",
    "Diagnostic R²": "Diagnostic R²",
    "Per-target metrics": "各目標指標",
    "Target": "目標",
    "Data handling audit": "資料處理稽核",
    "Table rows": "表格列數",
    "Rows used": "實際使用",
    "Skipped: input missing": "輸入缺失略過",
    "Ground truth missing": "Ground truth 缺失",
    "validated": "已驗證",
    "predictions only": "僅預測",
    "Notes": "備註",
    "Figures": "圖表",
    "Machine-readable appendix": "機器可讀附錄",
    "predicted value, actual value and residual per row and target":
        "每列每個目標的預測值、真值與殘差",
    "one row per observation, ready to paste back into the sheet":
        "每個觀測一列，可直接貼回原表",
    "macro and per-target metrics": "macro 與各目標指標",
    "runtime, model list and column mapping": "執行環境、模型清單與欄位對應",
    "Predicted vs actual — {model}": "預測值 vs 真值 — {model}",
    "Actual": "真值",
    "Predicted": "預測值",
    "Residual": "殘差",
    "Residual distribution": "殘差分佈",
    "Residual vs predicted": "殘差 vs 預測值",
    "Actual − Predicted": "真值 − 預測值",
    "Count": "次數",
    "Macro NMAE (lower is better)": "Macro NMAE（越低越好）",
    "Reported R² (higher is better)": "Reported R²（越高越好）",
    "MAPE % (descriptive only)": "MAPE %（僅供描述）",
}
