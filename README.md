# MapAndLLM Vision v2

以 Safari 全螢幕相機介面整合招牌辨識、室內地圖、共享房間與 Ollama 導航。
版本分支：`v2`。以下指令若無另外說明，皆從專案根目錄執行。

## 啟動

使用目前的 `mapandllm-v2` conda 環境啟動介面與房間 API：

```bash
conda activate mapandllm-v2
PORT=40012 bash start_http.sh
```

腳本會設定目前 Python 環境所需的 `lib` 路徑。
連接埠可由 `PORT` 或額外的 `--port` 參數指定。HTTP 可測桌面地圖與聊天介面；手機相機與感測器需 HTTPS（或 localhost）。

建立獨立完整環境時：

```bash
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

## 模型設定

第一次安裝先建立設定檔：

```bash
cp .env.example .env
```

`start_http.sh` 會透過 Uvicorn 讀取 `.env`。修改模型或裝置設定後必須重新啟動服務；已經匯入 Python 的模型設定不會在執行中自動更新。

### 建議設定

```dotenv
# 地圖解析模型
YOLO_MODEL_PATH=train6/weights/best.pt

# 相機招牌偵測：Grounding DINO 使用 GPU
DINO_MODEL_ID=IDEA-Research/grounding-dino-tiny
DINO_DEVICE=cuda
DINO_SHORT_EDGE=640
DINO_LONG_EDGE=1067
DINO_BOX_THRESHOLD=0.25
DINO_TEXT_THRESHOLD=0.22
MAX_DETECTIONS=3

# 招牌文字辨識：PaddleOCR 使用 CPU，避免與 DINO／YOLO／Ollama 搶 VRAM
OCR_DEVICE=cpu
OCR_DET_MODEL=PP-OCRv5_server_det
OCR_REC_MODEL=PP-OCRv5_server_rec
OCR_LANG=chinese_cht
OCR_VERSION=PP-OCRv5
OCR_SCORE_THRESHOLD=0.45
OCR_ENABLE_MKLDNN=0

# 地點語意比對
OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=gemma4:latest
LLM_NUM_CTX=4096
LLM_KEEP_ALIVE=30m
# 留空代表由 Ollama 自動決定；設為 0 代表只用 CPU
LLM_NUM_GPU=
```

### 各模型用途

| 模型 | 用途 | 預設執行裝置 | 載入時機 |
|---|---|---|---|
| YOLO `train6/weights/best.pt` | 從上傳的平面圖辨識地圖物件 | CUDA GPU 0 | 上傳地圖後，由獨立 `map_worker.py` 載入 |
| Grounding DINO tiny | 從相機照片找出店面招牌區域 | CUDA | 建立第一個房間後在背景預熱，之後常駐 FastAPI 程序 |
| PaddleOCR v5 server | 讀取招牌上的繁體中文／英文 | CPU | 與相機 DINO 一起預熱並常駐 |
| Ollama `gemma4:latest` | OCR 與地圖地名比對、解析導航語句 | Ollama 自動配置 | 地圖處理完成後預熱；由 `LLM_KEEP_ALIVE` 控制常駐時間 |

Grounding DINO 只負責找招牌，PaddleOCR 只負責讀字；LLM 不負責計算路線。起終點確定後，路線由 `navigation.py` 的拓樸圖演算法產生。

### Ollama 模型準備

確認 Ollama 服務與模型：

```bash
ollama serve
ollama list
ollama pull gemma4:latest
```

若 `gemma4` 因 VRAM 不足回傳 HTTP 500，可先使用已安裝的較小模型並強制走 CPU：

```dotenv
LLM_MODEL=TwinkleAI/gemma-3-4B-T1-it:latest
LLM_NUM_GPU=0
LLM_NUM_CTX=4096
```

CPU 模式可避免與相機 DINO、地圖 YOLO 爭用 VRAM，但第一次載入與每次生成會比較慢。使用 GPU 時可用以下指令檢查占用：

```bash
nvidia-smi
curl http://127.0.0.1:11434/api/ps
```

### 相機辨識調校

完整可調參數請看 `.env.example`。常用參數如下：

| 設定 | 預設值 | 說明 |
|---|---:|---|
| `DINO_SHORT_EDGE` | `640` | DINO 輸入短邊；提高可能看清小招牌，但更慢、更耗 VRAM |
| `DINO_LONG_EDGE` | `1067` | DINO 輸入長邊上限 |
| `MAX_DETECTIONS` | `3` | 每張照片最多送入 OCR 的候選招牌數 |
| `DINO_BOX_THRESHOLD` | `0.25` | 招牌框最低信心值 |
| `OCR_SCORE_THRESHOLD` | `0.45` | OCR 文字最低信心值 |
| `OCR_ENABLE_MKLDNN` | `0` | 預設關閉，避免部分 Paddle 3.x oneDNN 問題 |

速度優先時可以改用 PaddleOCR mobile 模型：

```dotenv
OCR_DET_MODEL=PP-OCRv5_mobile_det
OCR_REC_MODEL=PP-OCRv5_mobile_rec
```

mobile 模型雖然較快，但先前樣本的店名準確率低於 server 模型，正式使用前應以人工標記資料驗證。

### 地圖與 LLM 工作程序

```dotenv
MAP_WORKER_TIMEOUT=7200
MAP_KMEANS_K=6
LLM_POST_WORKER_DELAY=5
LLM_WARMUP_RETRIES=4
LLM_WARMUP_BASE_DELAY=5
LLM_REQUEST_RETRIES=3
LLM_REQUEST_BASE_DELAY=5
```

- 地圖處理會先要求 Ollama 卸載模型，再啟動獨立的 YOLO／EasyOCR 工作程序。
- 工作程序退出後會等待 `LLM_POST_WORKER_DELAY`，再預熱 Ollama。
- HTTP 500、CUDA 冷啟動或暫時連線失敗會依重試設定處理。
- 程序內的 GPU 鎖只能協調同一份 FastAPI；其他容器、其他服務與另一份專案仍可能占用 GPU。

地圖 YOLO 權重缺少時，上傳端點會回傳 HTTP 503 並提示 `YOLO_MODEL_PATH`。模型名稱不存在或 Ollama 未啟動，而且 OCR 文字無法唯一精確匹配地圖別名時，相機定位會回傳 `llm_unavailable`；若能唯一匹配，系統仍可使用 `exact_ocr_alias_fallback` 更新位置。

## 使用流程

1. 啟動相機，或選「先使用地圖與導航」。
2. 打開地圖面板，建立房間或以邀請碼／連結加入房間。
3. 上傳平面圖，等待解析完成。共享房間會輪詢地圖狀態與其他使用者位置。
4. 在聊天面板輸入起點與目的地。相機拍照間隔以毫秒設定；相機啟動後每 5 秒將最新保存的 OCR JSON 與地圖 JSON 的地點索引交給 LLM 比對。只有比對到合法且唯一的地點 ID 才更新地圖上的使用者位置；勾選框可關閉自動定位。
5. LLM 比對地圖中的區域後，由拓樸圖「近似最短距離＋少轉彎」演算法規劃路徑，並由確定性規則產生導航文字；地圖面板同步顯示路線。
6. 可沿用兩點步行校正：點起點、實際走一段並數步數、點終點輸入步數，再允許感測器開始定位。定位仍是原專案的估算方式，未經真實手機實測。

相機辨識保留一次一張、不排隊與自動／手動辨識；用戶端不顯示辨識框，工作站仍可將照片、結果 JSON、標註圖和可選招牌裁切保存在 `saved/`。
房間、位置只在記憶體保存，重啟會清空；請使用單一 Uvicorn worker。閒置房間 30 分鐘後可回收，每房間最多兩個共享定位使用者。

## 主要檔案

- `app/server.py`：統一 FastAPI 入口、相機辨識與保存。
- `navigation.py`：拓樸圖導航、少轉彎路徑選擇、房間 API 與 Ollama 地點語意判斷，掛於 `/api`。
- `map_processor.py`：原平面圖分割流程。
- `app/static/`：實際使用的 Safari 風格介面；`navigation.js` 連接房間／地圖／聊天，`pdr_engine.js` 保留步行定位。
- `main.py`：相容入口，亦可 `python -m uvicorn main:app`。
- 原 `index.html` 與 `static/` 保留作舊介面參考，服務不再使用它們。

## 測試

單獨以保存的相片 OCR 結果測試定位（預設驗證 PUMA、ID 12、座標 2977×433）：

```bash
conda run -n mapandllm-v2 python scripts/test_photo_location.py
```

若要強制要求結果必須由 LLM 回傳，而不能使用精確店名備援：

```bash
conda run -n mapandllm-v2 python scripts/test_photo_location.py --require-llm
```

GPU 記憶體不足時可用已安裝的小模型走 CPU：

```bash
conda run -n mapandllm-v2 python scripts/test_photo_location.py \
  --model TwinkleAI/gemma-3-4B-T1-it:latest --cpu --require-llm
```

```bash
conda activate mapandllm-v2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m unittest discover -s tests -v
```

API 測試以替身取代模型／Ollama，覆蓋房間邀請、上傳限制、OCR 成功與失敗保存、OCR 線索傳遞、LLM 離線備援、定位上限及拓樸少轉彎路徑。這些測試不代表模型推論品質或手機感測器已完成實測。

瀏覽器串接測試：`tests/browser_smoke.py` 使用 Playwright、模擬相機與模型資料，實際呼叫 FastAPI，驗證 390×844 手機尺寸下的建立／加入房間、上傳、路線繪製、相機辨識與 OCR 線索開關。執行方式（需另備 Playwright 與 Chromium）：

```bash
PYTHONPATH=. python tests/browser_smoke.py
```

本次 12 項 API／路徑測試與 Chromium 手機尺寸串接測試已通過。

## 相機辨識效能測試

相機設定可選 768 或 960 px 輸入圖片；靜止畫面會略過重複自動辨識，但最長 10 秒會重新辨識一次，手動拍照不受影響。預設 `MAX_DETECTIONS=3`、`DINO_SHORT_EDGE=640`、`DINO_LONG_EDGE=1067`；若小字或遠處招牌漏辨識，可先把圖片改回 960 px，並依序把 DINO 尺寸改回 800／1333、候選數改回 8。環境變數變更後需重新啟動服務。OCR 模型可用 `OCR_DET_MODEL` 與 `OCR_REC_MODEL` 切換；預設保留 PP-OCRv5 server 版本。若要試驗速度優先版本，可設為 `PP-OCRv5_mobile_det`／`PP-OCRv5_mobile_rec`，但必須先用有人工標記正確店名的影像驗證準確率，不能只看與舊 OCR 文字是否相同。

每次辨識的 `saved/results/<日期>/*.json` 會包含 `processing_ms` 與 `timings`：模型載入、DINO 前處理／推論／後處理、每個 OCR 裁切、標註圖保存等耗時。比較設定時請使用同一批招牌照片，分別記錄中位數耗時與正確辨識率；首次模型載入應單獨計算。`processing_ms` 不包含瀏覽器上傳與每 5 秒一次的 LLM 定位比對。
