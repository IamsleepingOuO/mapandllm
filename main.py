import os
import uuid
import json
import shutil
import secrets
import time
import re
import cv2
import numpy as np
import pandas as pd
import heapq
import torch
import gc
from pathlib import Path
from fastapi import FastAPI, Response, Request, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
from map_processor import get_ocr_data, analyze_colors_and_corridor, extract_walls_with_repair, RoomSegmenter

# ==========================================
# ⚙️ 系統設定與全域變數
# ==========================================
# 強制優化 PyTorch 記憶體碎片管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ⚠️ 請確保此 YOLO 模型路徑正確
YOLO_MODEL_PATH = 'train6/weights/best.pt'
# 使用的 LLM 模型 (依照你程式碼中的設定，若無此模型可改回 'llama3')
LLM_MODEL = 'TwinkleAI/gemma-3-4B-T1-it'

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ROOMS = {}
CODE_TO_UUID = {}
IDLE_TIMEOUT = 1800

def cleanup_expired_rooms():
    now = time.time()
    expired_uuids = [uid for uid, data in ROOMS.items() if now - data["last_active"] > IDLE_TIMEOUT]
    for uid in expired_uuids:
        code = ROOMS[uid]["invite_code"]
        del ROOMS[uid]
        if code in CODE_TO_UUID:
            del CODE_TO_UUID[code]
        print(f"🧹 房間 {code} 已回收")

# ==========================================
# 🧠 AI 視覺處理模組 (來自 0526.py 升級版)
# ==========================================
def safe_imread(image_path, flags=cv2.IMREAD_COLOR):
    return cv2.imread(str(image_path), flags)

def process_map_background(room_id: str, image_path: Path):
    try:
        # 更新狀態為處理中，讓前端顯示「AI 視覺解析中...」
        ROOMS[room_id]["status"] = "processing"
        ROOMS[room_id]["image_url"] = None
        ROOMS[room_id]["room_data"] = None
        
        # 為這個房間建立專屬的輸出資料夾
        output_folder = UPLOAD_DIR / room_id
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 開始處理房間 {room_id} 的地圖...")
        
        # 1. 執行 OCR
        ocr_results = get_ocr_data(str(image_path))

        # 2. 獲取走道遮罩 (這裡對應你 __main__ 裡設定的 k=6)
        corridor_mask_k = analyze_colors_and_corridor(str(image_path), ocr_results, k=6)

        # 3. 提取精密牆體與幾何修補
        repaired_wall_matrix = extract_walls_with_repair(str(image_path), output_folder, ocr_results)
        
        # 4. 進行空間分配與屬性標記
        if repaired_wall_matrix is not None:
            segmenter = RoomSegmenter(output_folder, YOLO_MODEL_PATH)
            segmenter.process(
                str(image_path), 
                wall_matrix=repaired_wall_matrix,
                corridor_mask=corridor_mask_k, 
                ocr_data=ocr_results
            )
            
            # 讀取剛剛產生的 JSON 資料
            json_file_path = output_folder / "room_data_0526_6.json"
            csv_file_path = output_folder / "_0526_6.csv"
            if json_file_path.exists():
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    room_data = json.load(f)
            else:
                room_data = {}

            # 🌟 更新房間狀態：處理完成，通知前端可以拿圖了！
            ROOMS[room_id]["status"] = "ready"
            # 指向你模組產生的那張 debug_0526_6.jpg
            ROOMS[room_id]["image_url"] = f"/uploads/{image_path.name}" 
            ROOMS[room_id]["room_data"] = room_data
            ROOMS[room_id]["json_path"] = str(json_file_path)
            ROOMS[room_id]["csv_path"] = str(csv_file_path)

    except Exception as e:
        print(f"❌ 地圖處理失敗: {e}")
        ROOMS[room_id]["status"] = "error"

@app.post("/upload")
async def upload_map(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    room_id: str = Form(...)
):
    if room_id not in ROOMS:
        return {"error": "房間不存在"}

    # 儲存使用者上傳的原始圖片
    file_ext = file.filename.split('.')[-1]
    save_path = UPLOAD_DIR / f"{room_id}_raw.{file_ext}"
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 將耗時的影像處理丟入背景執行，讓前端立刻收到回應
    background_tasks.add_task(process_map_background, room_id, save_path)
    
    return {"message": "地圖上傳成功，開始背景解析"}

# ==========================================
# 📍 A* 導航模組與 LLM 生成 (來自 0427_llm1.py)
# ==========================================
class IndoorNavigator:
    def __init__(self, map_csv_path, room_json_path, room_id):
        self.map_matrix = pd.read_csv(map_csv_path, header=None).values
        self.H, self.W = self.map_matrix.shape
        self.room_id = room_id
        with open(room_json_path, 'r', encoding='utf-8') as f:
            self.room_data = json.load(f)
        self.portal_ids = {int(k) for k, v in self.room_data.items() if v.get("portal") is True}

    def get_room_centroid(self, room_id):
        ys, xs = np.where(self.map_matrix == int(room_id))
        if len(ys) == 0: return None
        return int(np.mean(xs)), int(np.mean(ys))

    def find_portal_entry(self, room_id):
        centroid = self.get_room_centroid(room_id)
        if not centroid: return None, None
        cx, cy = centroid
        directions = {'上': (0, -1), '下': (0, 1), '左': (-1, 0), '右': (1, 0)}
        best_pt, min_dist, room_to_portal_dir = None, float('inf'), None
        for d_name, (dx, dy) in directions.items():
            x, y = cx, cy
            first_portal, last_portal, dist_to_first = None, None, 0
            while 0 <= x < self.W and 0 <= y < self.H:
                if self.map_matrix[y, x] in self.portal_ids:
                    if first_portal is None: first_portal = (x, y); dist_to_first = abs(x - cx) + abs(y - cy)
                    last_portal = (x, y)
                else:
                    if first_portal is not None: break
                x += dx; y += dy
            if first_portal is not None and dist_to_first < min_dist:
                min_dist = dist_to_first
                best_pt = ((first_portal[0] + last_portal[0]) // 2, (first_portal[1] + last_portal[1]) // 2)
                room_to_portal_dir = (dx, dy)
        return best_pt, room_to_portal_dir

    def a_star_path(self, start_pt, end_pt):
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        # 狀態紀錄擴充：座標 (x, y) 加上 抵達該點的方向 (dx, dy)
        start_state = (start_pt[0], start_pt[1], 0, 0)
        open_set = []
        heapq.heappush(open_set, (0, start_state))
        came_from = {}
        g_score = {start_state: 0}
        TURN_PENALTY = 0.5  # 轉彎懲罰值，保證距離最短的前提下轉彎最少
        
        while open_set:
            _, current_state = heapq.heappop(open_set)
            cx, cy, cdx, cdy = current_state
            
            if heuristic((cx, cy), end_pt) < 5: 
                path = [end_pt]
                curr = current_state
                while curr in came_from:
                    curr = came_from[curr]
                    path.append((curr[0], curr[1]))
                path.reverse()
                return path
                
            for ndx, ndy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + ndx, cy + ndy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    if self.map_matrix[ny, nx] in self.portal_ids or heuristic((nx, ny), end_pt) < 5:
                        move_cost = 1
                        if (cdx, cdy) != (0, 0) and (ndx, ndy) != (cdx, cdy):
                            move_cost += TURN_PENALTY # 判斷轉彎並加上懲罰
                            
                        tentative_g = g_score[current_state] + move_cost
                        neighbor_state = (nx, ny, ndx, ndy)
                        
                        if tentative_g < g_score.get(neighbor_state, float('inf')):
                            came_from[neighbor_state] = current_state
                            g_score[neighbor_state] = tentative_g
                            f_score = tentative_g + heuristic((nx, ny), end_pt)
                            heapq.heappush(open_set, (f_score, neighbor_state))
        return []

    def extract_path_events(self, path, room_to_portal_dir):
        if len(path) < 5: return []
        facing_vec = (-room_to_portal_dir[0], -room_to_portal_dir[1])
        simplified_segments = []; current_dir = None; segment_points = []
        sampled_path = path[::10] 
        if path[-1] not in sampled_path: sampled_path.append(path[-1])
            
        for i in range(len(sampled_path) - 1):
            p1, p2 = sampled_path[i], sampled_path[i+1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            step_dir = (1, 0) if dx > 0 else (-1, 0) if abs(dx) > abs(dy) else (0, 1) if dy > 0 else (0, -1)
            if current_dir is None: current_dir = step_dir
            if step_dir == current_dir: segment_points.append(p2)
            else: simplified_segments.append({'dir': current_dir, 'points': segment_points}); current_dir = step_dir; segment_points = [p2]
        if segment_points: simplified_segments.append({'dir': current_dir, 'points': segment_points})

        events = []; current_facing = facing_vec
        def get_turn_direction(v_face, v_move):
            cross = v_face[0] * v_move[1] - v_face[1] * v_move[0]
            dot = v_face[0] * v_move[0] + v_face[1] * v_move[1]
            return "右手邊" if cross > 0 else "左手邊" if cross < 0 else "前方" if dot > 0 else "後方"

        for idx, seg in enumerate(simplified_segments):
            move_vec = seg['dir']
            turn_str = get_turn_direction(current_facing, move_vec)
            passed_rooms = set()
            for px, py in seg['points']:
                for rx in range(px-15, px+16, 5):
                    for ry in range(py-15, py+16, 5):
                        if 0 <= rx < self.W and 0 <= ry < self.H:
                            cid = self.map_matrix[ry, rx]
                            if cid != 1 and cid not in self.portal_ids:
                                str_cid = str(cid)
                                if str_cid in self.room_data:
                                    names = self.room_data[str_cid].get("names", [])
                                    valid_names = []
                                    for n in names:
                                        n_str = str(n).strip()
                                        if n_str.isdigit() or (any(c.isdigit() for c in n_str) and len(n_str) <= 5): continue
                                        valid_names.append(n_str)
                                    if valid_names: passed_rooms.add(valid_names[0])
            
            if idx == 0: events.append(f"[第1步-起步] 請向 {turn_str} 走。")
            else: events.append(f"[轉角] 向 {turn_str} 轉。")
            if passed_rooms: events.append(f"  -> 直行經過：{', '.join(list(passed_rooms))}")
            current_facing = move_vec 
        return events

    def draw_debug_path(self, path, start_pt, end_pt):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[self.map_matrix == 1] = (255, 255, 255)
        if path and len(path) > 1:
            pts = np.array(path, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=3)
        if start_pt:
            cv2.circle(img, start_pt, radius=8, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Start", (start_pt[0]+10, start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if end_pt:
            cv2.circle(img, end_pt, radius=8, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "End", (end_pt[0]+10, end_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        debug_path = UPLOAD_DIR / f"{self.room_id}_debug.jpg"
        cv2.imwrite(str(debug_path), img)
        return f"/uploads/{self.room_id}_debug.jpg"

    def generate_llm_guidance(self, start_id, end_id, user_start_name=None, user_end_name=None):
        start_pt, start_vec = self.find_portal_entry(start_id)
        end_pt, _ = self.find_portal_entry(end_id)
        if not start_pt or not end_pt: return "無法找到合適的出入口座標進行路徑規劃。", None, None
            
        path = self.a_star_path(start_pt, end_pt)
        debug_url = self.draw_debug_path(path, start_pt, end_pt)
        if not path: return "抱歉，無法規劃出連通的路徑。", debug_url, None
        
        events = self.extract_path_events(path, start_vec)
        
        def get_best_name(names, default_name):
            valid_names = []
            for n in names:
                n_str = str(n).strip()
                if n_str.isdigit() or (any(c.isdigit() for c in n_str) and len(n_str) <= 5): continue
                valid_names.append(n_str)
            return valid_names[0] if valid_names else default_name
            
        start_names = self.room_data[str(start_id)].get("names", [f"{start_id}號房間"])
        end_names = self.room_data[str(end_id)].get("names", [f"{end_id}號房間"])
        
        start_name = user_start_name if user_start_name and not user_start_name.isdigit() else get_best_name(start_names, start_names[0])
        end_name = user_end_name if user_end_name and not user_end_name.isdigit() else get_best_name(end_names, end_names[0])
        events_text = "\n".join(events)
        
        prompt = f"""
你是一個親切的室內導航語音助手。我已經透過路徑規劃演算法計算出了一條路線，以下是路線的關鍵轉折點與經過的房間紀錄：
【導航基本資訊】
- 起點房間：{start_name}
- 終點房間：{end_name}
【路線指令紀錄】
{events_text}

請將上述紀錄潤飾成一段自然、友善的語音導航。請嚴格遵守以下對話結構：
1. 絕對不要說任何「好的」、「沒問題」、「為您規劃如下」等開場白廢話！
2. 開頭第一句「必須」精確讀取紀錄中 [第1步-起步] 的方向，告訴使用者：「請以面向『{start_name}』為正前方，接著向你的（填入紀錄中的方向）開始走」。
3. 沿途導航時，若紀錄提到直行經過很多房間，請最多挑選 2 個名字說出，並加上「等區域」，不要報流水帳。
4. 依照紀錄順序明確指示轉向（向左或向右）。
5. 結尾恭喜使用者抵達目的地 {end_name}。
6. 寫成一段流暢連續的對話段落，不要輸出任何標題或條列符號。
"""
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        
        # 保留強制的物理截斷機制，斬掉廢話與 <think>
        import re
        clean_reply = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL).strip()
        if "請以面向" in clean_reply:
            clean_reply = "請以面向" + clean_reply.split("請以面向", 1)[1]
            
        path_coords = [[int(p[0]), int(p[1])] for p in path]
        return clean_reply, debug_url, path_coords

def get_user_location(user_input, room_data):
    # 【保留修正】開放所有 ID，允許使用者從「第一噴水池廣場」出發
    valid_room_ids = list(room_data.keys()) 

    prompt = f"""
你是一個專業的室內導航定位系統。你的任務是分析使用者的自然語言描述，並比對給定的室內地圖資料庫，精準判斷使用者「現在的位置」與「目的地」。

【地圖資料庫說明】
{json.dumps(room_data, ensure_ascii=False)}

【核心限制與推理規則】
1. **起點與終點限制 (極重要)**：合法區域 ID 清單：{valid_room_ids}。使用者所在的起點與終點可以是實體房間，也可以是廣場或走道（portal）。
2. **分析線索**：透過對比所有房間的names與objects找到關聯。同時可透過shape屬性確認形容詞描述的大小寬窄。
3. **語意比對**：使用者可能透過情境或別稱來描述地點，請推理出最可能的房間。
4. **文字修正**: 若遇到疑似錯別字，請自行判斷正確詞義。

使用者現在說："{user_input}"

【輸出格式要求】
請「務必」僅輸出合法的 JSON 格式，絕對不要包含任何額外的解說文字。格式如下：
{{
    "current_room_id": "推斷出的起點合法房間ID",
    "current_room_name": "請『完全複製』使用者在句子中實際使用的字詞（例如：VALENTINO），絕對不要輸出純數字門牌！",
    "destination_id": "推斷出的目的地合法房間ID",
    "destination_name": "請『完全複製』使用者在句子中實際使用的字詞（例如：starbucks），絕對不要輸出純數字門牌！",
    "reason": "簡短說明你的推論邏輯"
}}
"""
    response = ollama.generate(model=LLM_MODEL, prompt=prompt, format='json')
    clean_out = re.sub(r'^```json\s*', '', response['response']).replace('```', '')
    try:
        result = json.loads(clean_out)
        curr_id, dest_id = result.get("current_room_id"), result.get("destination_id")
        if curr_id and str(curr_id) not in valid_room_ids: result["current_room_id"] = None
        if dest_id and str(dest_id) not in valid_room_ids: result["destination_id"] = None
        return result
    except Exception as e:
        print(f"[錯誤] 解析 JSON 失敗: {e}")
        return {"current_room_id": None, "destination_id": None, "reason": "Error"}

# ==========================================
# 🌐 API 路由與任務控制
# ==========================================

@app.post("/create_room")
async def create_room():
    cleanup_expired_rooms()
    room_uuid = str(uuid.uuid4())
    invite_code = secrets.token_urlsafe(4)[:6].upper()
    while invite_code in CODE_TO_UUID: invite_code = secrets.token_urlsafe(4)[:6].upper()

    ROOMS[room_uuid] = {
        "invite_code": invite_code, "last_active": time.time(), 
        "image_url": None, "status": "idle", "csv_path": None, "json_path": None,
        "users": {}
    }
    CODE_TO_UUID[invite_code] = room_uuid
    return {"room_id": room_uuid, "invite_code": invite_code}
class JoinRequest(BaseModel):
    code_or_id: str

@app.post("/join_room")
async def join_room(req: JoinRequest):
    cleanup_expired_rooms()
    room_uuid = req.code_or_id.upper() if req.code_or_id.upper() in ROOMS else CODE_TO_UUID.get(req.code_or_id.upper())
    if not room_uuid: raise HTTPException(status_code=404, detail="房間不存在")
    ROOMS[room_uuid]["last_active"] = time.time()
    return {"room_id": room_uuid, "invite_code": ROOMS[room_uuid]["invite_code"]}

@app.get("/room_status/{room_id}")
async def get_room_status(room_id: str):
    if room_id not in ROOMS: raise HTTPException(status_code=404, detail="房間不存在")
    return {"image_url": ROOMS[room_id].get("image_url"), "status": ROOMS[room_id].get("status"), "users": ROOMS[room_id].get("users", {})}

class ChatRequest(BaseModel):
    message: str
    room_id: str | None = None

@app.post("/chat")
async def chat_with_llama(req_data: ChatRequest):
    if not req_data.room_id or req_data.room_id not in ROOMS: return {"reply": "⚠️ 房間已失效。"}
    room = ROOMS[req_data.room_id]; room["last_active"] = time.time()
    current_status = room.get("status")

    if current_status == "processing": return {"reply": "⏳ 地圖分析中，請稍候。"}
    if current_status == "error": return {"reply": "❌ 地圖解析發生錯誤。"}
    if current_status == "idle" or not room.get("json_path"):
        return {"reply": "嗨！請先點擊上方上傳地圖，我才能幫你導航喔！"}

    with open(room["json_path"], 'r', encoding='utf-8') as f:
        room_data = json.load(f)

    # 呼叫我們剛剛整合的 LLM 位置分析函數
    loc = get_user_location(req_data.message, room_data)
    start_id = loc.get("current_room_id")
    end_id = loc.get("destination_id")
    start_name = loc.get("current_room_name") # 🌟 新增抓取名稱
    end_name = loc.get("destination_name")    # 🌟 新增抓取名稱
    
    if not start_id: return {"reply": "🤔 抱歉，我不太確定你的「現在位置」在哪裡，可以再描述得更明確一點嗎？"}
    if not end_id: return {"reply": "🤔 抱歉，我不太確定你要去的「目的地」是哪裡，可以換個說法嗎？"}

    nav = IndoorNavigator(room["csv_path"], room["json_path"], req_data.room_id)
    # 🌟 傳入 start_name 與 end_name 給導航器
    final_text, debug_url, path_coords = nav.generate_llm_guidance(
        int(start_id), int(end_id),
        user_start_name=start_name,
        user_end_name=end_name
    )
    
    reply = final_text
    if debug_url: reply += f"\n\n🗺️ [系統] DEBUG 路徑圖：{debug_url}"
        
    return {"reply": reply, "path_coords": path_coords}

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

class PositionUpdate(BaseModel):
    user_id: str
    x: float
    y: float
    color: str

@app.post("/update_position/{room_id}")
async def update_position(room_id: str, pos: PositionUpdate):
    if room_id in ROOMS:
        # 限制最多兩人：如果字典裡沒有這個人，且人數已達2人，就拒絕更新
        if pos.user_id not in ROOMS[room_id]["users"] and len(ROOMS[room_id]["users"]) >= 2:
            return {"status": "full"}
            
        ROOMS[room_id]["users"][pos.user_id] = {
            "x": pos.x,
            "y": pos.y,
            "color": pos.color,
            "last_update": time.time()
        }
        ROOMS[room_id]["last_active"] = time.time()
    return {"status": "ok"}