Regression Models — Windows 版（CUDA 12.8）
=============================================

【安裝】
1. 下載 RegressionModels-v…-win64-cuda12.8.7z。若該版本分成多個分割檔
   （.7z.001、.7z.002 …），請全部下載到同一個資料夾。
2. 用 7-Zip（https://www.7-zip.org）按右鍵 →「解壓縮至…」；分割檔請對 .7z.001 操作，
   會自動合併。
3. 執行資料夾中的 RegressionModels.exe。
   第一次執行若出現「Windows 已保護您的電腦」，請按「其他資訊」→「仍要執行」
   （本程式沒有數位簽章）。

【啟動時的自動檢查】
- GPU：實際在顯示卡上做一次測試運算。不能用時會說明原因並改用 CPU 執行，
  其他功能不受影響。NVIDIA 顯示卡請安裝最新驅動程式（需支援 CUDA 12.8）。
- 資料夾與設定：自動建立 runs、reports、models、pretrained、logs 與 settings.json，
  並在第一次啟動時把範例模型放到 models\legacy_v2。
- TabPFN v2 權重（pretrained\tabpfn\tabpfn-v2-regressor.ckpt）：已隨附；
  若被刪除，啟動時會詢問是否重新下載（約 45 MB，不需註冊帳號）。

【資料夾說明】
  RegressionModels.exe   主程式
  _internal\             程式本體，請勿修改
  runs\                  每次訓練的報告與模型
  models\                模型庫（可放入自己的模型）
  reports\               批次推論與驗證報告
  pretrained\tabpfn\     TabPFN v2 權重
  logs\                  啟動檢查（startup.json）、錯誤紀錄（crash.log）

整個資料夾可以直接複製到另一台電腦使用，訓練結果與模型會一起帶過去。

【命令列】
  RegressionModels.exe --check       只執行啟動檢查，結果寫入 logs\console.log
  RegressionModels.exe --self-test   引擎自我測試
  RegressionModels.exe --model-check 每種模型實際訓練一次，確認這台電腦都能執行
  RegressionModels.exe --lang zh_TW  以繁體中文啟動


Regression Models — Windows build (CUDA 12.8)
=============================================

Install: download RegressionModels-v...-win64-cuda12.8.7z (if a release is split
into .7z.001, .7z.002, ..., download every part into one folder and open .001),
extract it with 7-Zip and run RegressionModels.exe. If SmartScreen appears, choose
"More info" -> "Run anyway" (the program is not code-signed).

At start the program checks that CUDA really works (a test calculation on the
GPU) and explains why when it does not, then runs on the CPU. It also creates
any missing folders and settings.json, installs the example models on first
start, and offers to download the TabPFN v2 weights if they are missing.
Logs are in the logs folder. `RegressionModels.exe --model-check` trains every
model family once and writes pass/fail to logs\console.log.
