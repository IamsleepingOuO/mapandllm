# MapAndLLM Vision v2

以 Safari 全螢幕相機介面整合招牌辨識、室內地圖、共享房間與 Ollama 導航。
版本分支：`feat/camera-navigation-v2`。原始 `mapandllm` 工作目錄保持原樣。

## 啟動

目前可先沿用已建立的 `sign-ocr-live` 環境啟動介面與房間 API：

```bash
conda activate sign-ocr-live
PORT=40012 bash ~/mapandllm-v2/start_http.sh
```

腳本自動切換至專案目錄並設定目前 Python 環境的 `lib` 路徑，可從家目錄執行。
連接埠可由 `PORT` 或額外的 `--port` 參數指定。HTTP 可測桌面地圖與聊天介面；手機相機與感測器需 HTTPS（或 localhost）。

建立獨立完整環境時：

```bash
cd ~/mapandllm-v2
conda env create -f environment.yml
conda activate mapandllm-v2
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install paddlepaddle==3.2.2 --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
cp .env.example .env
bash start_http.sh
```

所有 CUDA/cuDNN/GL 等執行庫僅裝在 conda 環境，不需要安裝系統 CUDA toolkit 或修改 NVIDIA 驅動。
PyTorch cu124 安裝組合參照 [官方版本清單](https://docs.pytorch.org/get-started/previous-versions/)。
若沿用 `sign-ocr-live` 執行地圖解析，仍需安裝本專案 `requirements.txt` 中新增的 EasyOCR 與 Ultralytics 依賴；此版本開發時未修改原環境的套件。

## 模型設定（稍後補上）

`.env` 中提供以下設定入口。沒有 `.env` 時使用預設值。

```dotenv
YOLO_MODEL_PATH=train6/weights/best.pt
OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=gemma4
DINO_DEVICE=cuda
OCR_DEVICE=cpu
OCR_ENABLE_MKLDNN=0
```

依使用者要求，本次未下載地圖權重、建立 Ollama 服務或下載 LLM。地圖 YOLO 權重缺少時，上傳端點會回傳 503 與設定提示。首次相機辨識才載入 Grounding DINO / PaddleOCR，因此啟動介面不會觸發下載。

## 使用流程

1. 啟動相機，或選「先使用地圖與導航」。
2. 打開地圖面板，建立房間或以邀請碼／連結加入房間。
3. 上傳平面圖，等待解析完成。共享房間會輪詢地圖狀態與其他使用者位置。
4. 在聊天面板輸入起點與目的地。最近 60 秒辨識到的店名可作為起點線索，勾選框可關閉；不會把模糊線索直接當作精確位置。
5. LLM 比對地圖中的區域後，由拓樸圖「近似最短距離＋少轉彎」演算法規劃路徑，並由確定性規則產生導航文字；地圖面板同步顯示路線。
6. 可沿用兩點步行校正：點起點、實際走一段並數步數、點終點輸入步數，再允許感測器開始定位。定位仍是原專案的估算方式，未經真實手機實測。

相機辨識保留一次一張、不排隊、自動／手動辨識與辨識框；照片、結果 JSON、標註圖和可選招牌裁切保存在新版本的 `saved/`。
房間、位置只在記憶體保存，重啟會清空；請使用單一 Uvicorn worker。閒置房間 30 分鐘後可回收，每房間最多兩個共享定位使用者。

## 主要檔案

- `app/server.py`：統一 FastAPI 入口、相機辨識與保存。
- `navigation.py`：拓樸圖導航、少轉彎路徑選擇、房間 API 與 Ollama 地點語意判斷，掛於 `/api`。
- `map_processor.py`：原平面圖分割流程。
- `app/static/`：實際使用的 Safari 風格介面；`navigation.js` 連接房間／地圖／聊天，`pdr_engine.js` 保留步行定位。
- `main.py`：相容入口，亦可 `python -m uvicorn main:app`。
- 原 `index.html` 與 `static/` 保留作舊介面參考，服務不再使用它們。

## 測試

```bash
conda activate sign-ocr-live
cd ~/mapandllm-v2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m unittest discover -s tests -v
```

API 測試以替身取代模型／Ollama，覆蓋房間邀請、上傳限制、OCR 成功與失敗保存、OCR 線索傳遞、LLM 離線備援、定位上限及拓樸少轉彎路徑。這些測試不代表模型推論品質或手機感測器已完成實測。

瀏覽器串接測試：`tests/browser_smoke.py` 使用 Playwright、模擬相機與模型資料，實際呼叫 FastAPI，驗證 390×844 手機尺寸下的建立／加入房間、上傳、路線繪製、相機辨識與 OCR 線索開關。執行方式（需另備 Playwright 與 Chromium）：

```bash
PYTHONPATH=. python tests/browser_smoke.py
```

本次 12 項 API／路徑測試與 Chromium 手機尺寸串接測試已通過。測試工具、瀏覽器及其函式庫均置於 `/tmp`，未安裝系統套件或修改原 conda 環境。
