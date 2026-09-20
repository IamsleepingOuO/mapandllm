# 室內導航系統 — 0721_4 整合版

此版本保留原本的網頁操作流程與 API：建立/加入房間、上傳地圖、輪詢分析狀態、對話查詢、SVG 路線繪製與 PDR 碰撞矩陣皆維持原介面。

## 啟動

1. 建立虛擬環境並安裝 `requirements.txt`。
2. 將 YOLO 權重放到 `train6/weights/best.pt`，或設定環境變數 `YOLO_MODEL_PATH`。
3. 確認 Ollama 已安裝並已下載 `LLM_MODEL` 指定的模型。
4. 執行 `run_server.bat`，或：

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

瀏覽器開啟 `http://127.0.0.1:8000`。

## 主要資料流

`index.html` → `/upload` → `process_map_pipeline()` → `navigation_data.json` → `/chat` → 拓樸圖「近似最短距離＋最少轉彎」路徑 → `path_coords` → 前端 `drawPathOnMap()`。

### LLM 讀取內容

LLM 定位只讀 `navigation_data.json` 的 `llm_context`：地點 ID、名稱、別名、物件與形狀。它不讀 `W_*` 航點或整份 edge 清單。路徑由後端確定性 Dijkstra 計算，LLM 只負責解析需求與潤飾導航文字。

### 檔名相容層

- 新主資料：`navigation_data.json`
- 新路網：`llm_navigation_graph.json`
- 0721_4 原始空間資料：`room_data_0721_4.json`、`_0721_4.csv`
- 舊前端相容：`map_matrix.csv`
- 舊後端資料名相容：`room_data.json`
- 完整輸出索引：`map_manifest.json`

## 未包含的外部檔案

YOLO `best.pt` 與 Ollama 模型體積大，未隨此壓縮檔提供。`reference/` 保存使用者上傳的原始後端與 `0721_4.py`，不參與執行。


## 2.1 路徑規劃與除錯圖修正

- `debug_selected_route.jpg` 會以 `debug_navigation_graph.jpg` 為底圖，再用黃線疊加實際選中的路徑；紅點為起點、藍點為終點。
- 路徑演算法先算出嚴格最短距離，再使用包含進入方向的狀態式 Dijkstra 計算少轉彎候選路徑。
- 少轉彎候選只有在距離不超過「最短距離 × 1.03 + 5 px」時才會採用，因此不會為了少轉彎而明顯繞路。
- 小於 25° 的方向變化視為同一直線，可避免路網微小抖動被誤算成轉彎。

可選環境變數：

```powershell
$env:NAV_TURN_PENALTY_PX="80"
$env:NAV_U_TURN_PENALTY_PX="240"
$env:NAV_MAX_DETOUR_RATIO="1.03"
$env:NAV_MAX_DETOUR_SLACK_PX="5"
$env:NAV_STRAIGHT_TOLERANCE_DEG="25"
```

一般情況不需要設定，直接啟動伺服器即可。修改環境變數後必須重新啟動 Uvicorn。
