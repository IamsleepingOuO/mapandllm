# 整合報告

## 已修改

### `main.py`

- 保留原有 FastAPI 路由與前端 request/response contract。
- `process_map_background()` 改呼叫 `process_map_pipeline()`。
- 導航由舊版逐像素 A* 改為讀取 `llm_navigation_graph.json`，使用拓樸圖 Dijkstra。
- 起點與目的地使用 `attachment_node`；路線終點停在走道抵達點，不進入房間中心。
- LLM 定位只接收 `llm_context`，不接收 `W_*` 節點與 edges。
- Ollama 無法回應時，保留本地名稱比對與確定性導航文字 fallback。
- 原本 `/create_room`、`/join_room`、`/upload`、`/room_status`、`/chat`、`/update_position` 與 `/` 均保留。

### `map_processor.py`

- 以 `0721_4.py` 為主體整合 OCR 快取、K-Means、牆體兩階段修補、RoomSegmenter 與 WaypointGraphGenerator。
- 新增 `process_map_pipeline()` 作為網頁後端單一入口。
- 新增 schema v2 `navigation_data.json`，將語意地點資料與拓樸路網分層。
- 保留 0721_4 原始輸出，另產生舊版相容檔名。
- YOLO 權重不存在時不再讓整個背景任務直接崩潰，而是略過 icon 推論並繼續其他分析。

## 未修改，已原樣複製

- `index.html`
- `static/js/pdr_engine.js`
- `static/js/socket_log.js`
- `static/css/style.css`

SHA-256 驗證結果記錄於 `integration_manifest.json`。

## 執行期間輸出

每個房間會建立在 `uploads/<room_id>/`：

- `navigation_data.json`：後端主資料，schema v2。
- `llm_navigation_graph.json`：0721_4 拓樸圖。
- `room_data_0721_4.json`：0721_4 房間語意原始輸出。
- `_0721_4.csv`：0721_4 空間矩陣原始輸出。
- `map_matrix.csv`：供未修改前端 PDR 碰撞載入。
- `room_data.json`：舊檔名相容資料。
- `map_manifest.json`：輸出索引、模型狀態與計時。
- `runtime_profile_fast.json`：各階段耗時。
- 其餘 `debug_*.jpg/json`：0721_4 偵錯輸出。

## 已完成檢查

1. `main.py`、`map_processor.py` 均通過 `py_compile`。
2. FastAPI 路由建立、加入房間、狀態、聊天前置判斷與位置同步通過合成測試。
3. 拓樸圖 Dijkstra、轉折座標輸出與 LLM guidance adapter 通過合成路網測試。
4. `process_map_pipeline()` 的五個核心輸出檔名通過合成管線 contract 測試。
5. 四個未修改前端檔案與上傳原檔 SHA-256 完全相同。

## 尚未在此環境執行的項目

完整地圖推論需要使用者本機的 EasyOCR、Ultralytics、Ollama 與 YOLO `best.pt`。目前封裝已做語法、API、檔名、schema 與合成路網測試，但未在此環境以實際模型權重跑完整地圖。


## 2.1 修正

1. DEBUG 路徑圖不再建立黑色空白畫布，而是讀取完整 `debug_navigation_graph.jpg` 後疊加選中路徑。
2. 導航器加入方向狀態、轉彎成本、U-turn 成本與最大繞路限制。
3. 距離仍是硬性限制，只有近似最短候選才可因較少轉彎而被採用。
