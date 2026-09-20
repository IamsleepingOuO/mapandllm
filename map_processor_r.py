# V9: joint public classification, architectural envelopes and complete public road coverage.
# Preserve enclosed rooms, background voids, wall safety, and primary topology checks.
import cv2
import numpy as np
from pathlib import Path
import random
import json
import gc
import os
import math
import time
import hashlib
import heapq
from contextlib import contextmanager
from collections import Counter
import networkx as nx  # 🌟 新增：處理圖論與航點連線的套件

# Pipeline diagnostics do not depend on parent-process stdout consumption.
import builtins
import sys
import threading
import traceback
from contextvars import ContextVar
from functools import wraps

_PIPELINE_DIAGNOSTICS = ContextVar('map_pipeline_diagnostics', default=None)

def print(*args, **kwargs):
    diagnostic = _PIPELINE_DIAGNOSTICS.get()
    if diagnostic is None or kwargs.get('file') is not None:
        kwargs.setdefault('flush', True)
        return builtins.print(*args, **kwargs)
    text = kwargs.get('sep', ' ').join(str(a) for a in args)
    diagnostic['message'] = text
    with diagnostic['lock']:
        diagnostic['log'].write(text + kwargs.get('end', '\n'))
        diagnostic['log'].flush()
    # A worker launched with stdout=PIPE can deadlock when its parent waits
    # without draining the pipe. File progress remains available in either case.
    if getattr(sys.stdout, 'isatty', lambda: False)():
        kwargs.setdefault('flush', True)
        builtins.print(*args, **kwargs)


def _pipeline_diagnostics(func):
    @wraps(func)
    def wrapped(image_path, output_dir, *args, **kwargs):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        worker_id = threading.get_ident()
        stop = threading.Event()
        with open(directory / 'pipeline_progress.log', 'w', encoding='utf-8', buffering=1) as log:
            state = {'log': log, 'lock': threading.Lock(), 'stage': 'pipeline_start',
                     'message': '', 'status': 'running', 'error': None}
            token = _PIPELINE_DIAGNOSTICS.set(state)
            def snapshot(include_stack=False):
                data = {k: state[k] for k in ('stage','message','status','error')}
                data.update(pid=os.getpid(), elapsed_seconds=round(time.perf_counter()-started,2))
                frame = sys._current_frames().get(worker_id)
                if frame is not None and include_stack:
                    data['stack'] = traceback.format_stack(frame)[-16:]
                _write_json_cache(directory / 'pipeline_progress.json', data)
            def heartbeat():
                while not stop.wait(5.0):
                    try:
                        snapshot(include_stack=True)
                    except OSError:
                        pass
            thread = threading.Thread(target=heartbeat, daemon=True, name='map-progress')
            snapshot()
            thread.start()
            try:
                print('[Pipeline] worker entered; progress is written to pipeline_progress.json')
                result = func(image_path, output_dir, *args, **kwargs)
                state['status'] = 'completed'
                return result
            except BaseException as exc:
                state['status'] = 'failed'
                state['error'] = f'{type(exc).__name__}: {exc}'
                print(traceback.format_exc())
                raise
            finally:
                stop.set()
                thread.join()
                try:
                    snapshot()
                finally:
                    _PIPELINE_DIAGNOSTICS.reset(token)
    return wrapped

# 強制優化 PyTorch 記憶體碎片管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================
# FAST V1：共用快取、單次模型推論與階段計時
# =========================================
_EASYOCR_READER_CACHE = {}
_IMAGE_READ_CACHE = {}
CACHE_VERSION = "0915_v9_joint_public_partition"
RECOGNITION_CACHE_VERSION = "0914_wall_v7_width_partition_guard"
SEGMENTATION_RELEASE = "V8_PHYSICAL_PUBLIC_WALL_GUARDS"


def _path_signature(path):
    p = Path(path)
    try:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return str(p)


def _cache_key(*parts):
    payload = "||".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _read_json_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def _write_json_cache(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


@contextmanager
def stage_timer(name, timings=None):
    start = time.perf_counter()
    diagnostic = _PIPELINE_DIAGNOSTICS.get()
    if diagnostic is not None:
        diagnostic['stage'] = name
    print(f"[開始] {name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[name] = elapsed
        print(f"[計時] {name}: {elapsed:.2f} 秒")


def get_yolo_data(model_path, image_path, cache_dir=None, conf=0.15, imgsz=896):
    """YOLO 只推論一次；同一張圖與同一權重再次執行時直接讀 JSON 快取。"""
    cache_file = None
    if cache_dir is not None:
        key = _cache_key(
            RECOGNITION_CACHE_VERSION,
            "yolo",
            _path_signature(image_path),
            _path_signature(model_path),
            conf,
            imgsz,
        )
        cache_file = Path(cache_dir) / f"yolo_{key}.json"
        cached = _read_json_cache(cache_file)
        if cached is None:
            previous_key = _cache_key("0915_physical_public_v8_narrow_repair", "yolo",
                _path_signature(image_path), _path_signature(model_path), conf, imgsz)
            cached = _read_json_cache(Path(cache_dir) / f"yolo_{previous_key}.json")
        if cached is not None and isinstance(cached.get("detections"), list):
            detections = cached["detections"]
            boxes = [tuple(map(int, d["box"])) for d in detections]
            print(f"[快取] YOLO：載入 {len(detections)} 筆偵測。")
            return detections, boxes

    print("[模型] 載入 PyTorch / YOLO 與權重...")
    import torch
    from ultralytics import YOLO
    model = YOLO(model_path)
    predict_kwargs = {
        "source": safe_imread(image_path),
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if torch.cuda.is_available():
        predict_kwargs.update({"device": 0, "half": True})
    result = model.predict(**predict_kwargs)[0]
    detections = []
    for obj in result.boxes:
        coords = obj.xyxy[0].detach().cpu().tolist()
        x1, y1, x2, y2 = [int(round(v)) for v in coords]
        cls_id = int(obj.cls[0].detach().cpu().item())
        score = float(obj.conf[0].detach().cpu().item())
        label = model.names[cls_id] if isinstance(model.names, (list, tuple)) else model.names.get(cls_id, str(cls_id))
        detections.append({
            "box": [x1, y1, x2, y2],
            "center": [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))],
            "conf": score,
            "label": str(label),
            "class_id": cls_id,
        })

    if cache_file is not None:
        _write_json_cache(cache_file, {"version": CACHE_VERSION, "detections": detections})
    boxes = [tuple(map(int, d["box"])) for d in detections]
    return detections, boxes

# =========================================
# 工具函數：支援多格式的影像讀取
# =========================================
def safe_imread(image_path, flags=cv2.IMREAD_COLOR):
    key = (_path_signature(image_path), int(flags))
    cached = _IMAGE_READ_CACHE.get(key)
    if cached is None:
        # Python opens Unicode paths; OpenCV only receives encoded image bytes.
        try:
            data = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
            cached = cv2.imdecode(data, flags) if data.size else None
        except (OSError, cv2.error):
            return None
        if cached is None:
            return None
        # 單一 worker 只處理一張圖；仍設小上限避免長駐服務直接呼叫時累積。
        if len(_IMAGE_READ_CACHE) >= 4:
            _IMAGE_READ_CACHE.clear()
        _IMAGE_READ_CACHE[key] = cached
    return cached.copy()

def safe_imwrite(image_path, image, params=None):
    """Encode in memory, then write through Python for Windows Unicode paths."""
    path = Path(image_path)
    try:
        ok, encoded = cv2.imencode(path.suffix, image, params or [])
        if not ok:
            raise MapProcessingError(f"影像編碼失敗：{path}")
        path.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as exc:
        raise MapProcessingError(f"影像輸出失敗：{path}；{exc}") from exc
    return True


# =========================================
# 全域 OCR 提取模組
# =========================================
def get_ocr_data(image_path, cache_dir=None, canvas_size=None):
    """
    FAST V1：保留 EasyOCR/CRAFT polygon 與文字辨識功能，但避免每次重建模型。

    - 同一個 Python 行程內共用 Reader。
    - 同一張地圖再次執行時使用 JSON 快取。
    - GPU 使用 2048 canvas；CPU 使用 1792 canvas，避免 2560 canvas 的平方級成本。
    """
    print("[模型] 載入 PyTorch / 檢查 OCR 快取...")
    import torch
    use_gpu = torch.cuda.is_available()
    if canvas_size is None:
        canvas_size = 2048 if use_gpu else 1792

    cache_file = None
    if cache_dir is not None:
        key = _cache_key(RECOGNITION_CACHE_VERSION, "ocr", _path_signature(image_path), use_gpu, canvas_size, "ch_tra,en")
        cache_file = Path(cache_dir) / f"ocr_{key}.json"
        cached = _read_json_cache(cache_file)
        if cached is None:
            previous_key = _cache_key("0915_physical_public_v8_narrow_repair", "ocr",
                _path_signature(image_path), use_gpu, canvas_size, "ch_tra,en")
            cached = _read_json_cache(Path(cache_dir) / f"ocr_{previous_key}.json")
        if cached is not None and isinstance(cached.get("items"), list):
            print(f"[快取] OCR：載入 {len(cached['items'])} 筆文字。")
            return cached["items"]

    print("[系統] 正在執行 OCR 文字辨識 + CRAFT polygon 偵測...")
    reader_key = (("ch_tra", "en"), bool(use_gpu))
    reader = _EASYOCR_READER_CACHE.get(reader_key)
    if reader is None:
        print('[模型] 初始化 EasyOCR Reader（首次使用可能下載模型）...')
        import easyocr
        reader = easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu)
        _EASYOCR_READER_CACHE[reader_key] = reader

    temp_img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
    if temp_img is None:
        return []
    H, W = temp_img.shape[:2]

    read_kwargs = {
        "canvas_size": int(canvas_size),
        "detail": 1,
        "paragraph": False,
        "decoder": "greedy",
        "beamWidth": 1,
        "workers": 0,
    }
    # GPU 才提高 batch；CPU 過大 batch 反而可能增加記憶體交換。
    if use_gpu:
        read_kwargs["batch_size"] = 8

    results = reader.readtext(temp_img, **read_kwargs)
    ocr_data = []
    for bbox, text, prob in results:
        poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in bbox], dtype=np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
        xs = poly[:, 0]
        ys = poly[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))
        ocr_data.append({
            "text": str(text),
            "center": [cx, cy],
            "box": [x_min, y_min, x_max, y_max],
            "poly": poly.tolist(),
            "prob": float(prob),
        })

    if cache_file is not None:
        _write_json_cache(cache_file, {"version": CACHE_VERSION, "items": ocr_data})
    return ocr_data

# =========================================
# 模組 A：前置光影校正與 K-Means 色彩萃取 (修正版)
# =========================================
def _build_corridor_cluster_mask(labels_2d, color_id):
    """將單一 K-Means 色群轉成平滑但不過度擴張的走道候選遮罩。"""
    mask = (labels_2d == int(color_id)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    )
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)


def _nearest_mask_point(point, mask, radius):
    """Snap an eroded boundary contact locally, never to a distant component."""
    h, w = mask.shape
    x, y = map(int, point)
    x1, y1, x2, y2 = _clip_box(x-radius, y-radius, x+radius+1, y+radius+1, w, h)
    ys, xs = np.where(mask[y1:y2, x1:x2] > 0)
    if not len(xs):
        return None
    xs, ys = xs+x1, ys+y1
    i = int(np.argmin((xs-x)**2 + (ys-y)**2))
    return int(xs[i]), int(ys[i])


def _rectilinear_chain_path(chain, mask):
    """Use a safe skeleton as a witness; greedily remove turns with H/V visibility.

    No full-map search, no diagonal physical edges, and no percentage-based
    permission to cross a wall. Work is bounded by 64 probes per retained bend.
    """
    points = [tuple(map(int, p)) for p in chain]
    if len(points) < 2:
        return points
    h, w = mask.shape

    def safe(a, b):
        x, y = a
        u, v = b
        if not (0 <= x < w and 0 <= u < w and 0 <= y < h and 0 <= v < h):
            return False
        if y == v:
            return bool(np.all(mask[y, min(x,u):max(x,u)+1] > 0))
        if x == u:
            return bool(np.all(mask[min(y,v):max(y,v)+1, x] > 0))
        return False

    def join(a, b):
        if safe(a, b):
            return [a, b]
        for elbow in ((a[0], b[1]), (b[0], a[1])):
            if safe(a, elbow) and safe(elbow, b):
                return [a, elbow, b]
        return None

    result = [points[0]]
    i = 0
    while i < len(points)-1:
        indices = np.linspace(i+1, len(points)-1, min(64, len(points)-1-i), dtype=int)
        selected = None
        for j in reversed(indices.tolist()):
            route = join(points[i], points[j])
            if route is not None:
                selected = (j, route)
                break
        if selected is None:
            return None
        i, route = selected
        for pt in route[1:]:
            if pt == result[-1]:
                continue
            if len(result) >= 2 and (
                result[-2][0] == result[-1][0] == pt[0]
                or result[-2][1] == result[-1][1] == pt[1]
            ):
                result[-1] = pt
            else:
                result.append(pt)
    return result


def _compiled_axis_path(start, goal, mask, turn_penalty=20.0, deadline=None):
    """Target-directed four-neighbour A*: no full sparse pixel graph allocation.

    Two orientation states preserve turn costs. Every returned path is exact
    at image resolution; timeout/budget exhaustion is never called no_path.
    """
    started = time.perf_counter()
    call_budget = float(np.clip(float(os.environ.get('MAP_AXIS_SEARCH_SECONDS','1.5')), .05, 10.0))
    deadline = min(deadline, started+call_budget) if deadline is not None else started+call_budget
    if started >= deadline:
        return {'status':'time_budget','expansions':0}
    start, goal = tuple(map(int,start)), tuple(map(int,goal))
    h,w = mask.shape
    if any(not (0 <= x < w and 0 <= y < h and mask[y,x] > 0) for x,y in (start,goal)):
        return {'status':'endpoint_outside_mask'}
    simple = _rectilinear_chain_path([start,goal],mask)
    if simple is not None:
        return {'status':'ok','points':simple,'scale':1,'expansions':0,
                'path_len':float(sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(simple,simple[1:]))),
                'turn_count':max(0,len(simple)-2),'engine':'axis_visibility'}
    ys,xs = np.where(mask > 0)
    x0,y0,x1,y1 = int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    local = mask[y0:y1,x0:x1]>0
    lh,lw = local.shape
    cells = lh*lw
    max_cells = int(os.environ.get('MAP_AXIS_MAX_ROI_PIXELS','6000000'))
    if cells > max_cells:
        return {'status':'memory_budget','roi_pixels':cells,'expansions':0}
    if time.perf_counter() >= deadline:
        return {'status':'time_budget','expansions':0}
    # No per-pixel Python objects and no 8N COO edge arrays / CSR conversion.
    dist = np.full(cells*2,np.inf,np.float64)
    pred = np.full(cells*2,-1,np.int32)
    sx,sy = start[0]-x0,start[1]-y0
    gx,gy = goal[0]-x0,goal[1]-y0
    queue = []
    for axis in (0,1):
        sid = (sy*lw+sx)*2+axis
        dist[sid] = 0
        heapq.heappush(queue,(abs(sx-gx)+abs(sy-gy),0.0,sid))
    expansions = 0
    max_expansions = int(os.environ.get('MAP_AXIS_MAX_EXPANSIONS','200000'))
    max_queue = int(os.environ.get('MAP_AXIS_MAX_QUEUE','300000'))
    end = None
    while queue:
        if expansions % 128 == 0:
            if time.perf_counter() >= deadline:
                return {'status':'time_budget','expansions':expansions}
            if expansions >= max_expansions or len(queue) > max_queue:
                return {'status':'search_budget','expansions':expansions}
        _, negcost, sid = heapq.heappop(queue)
        cost = -negcost
        if cost != dist[sid]:
            continue
        cell,old_axis = divmod(sid,2)
        y,x = divmod(cell,lw)
        if (x,y)==(gx,gy):
            end = sid
            break
        expansions += 1
        for nx,ny,axis in ((x-1,y,0),(x+1,y,0),(x,y-1,1),(x,y+1,1)):
            if not (0<=nx<lw and 0<=ny<lh and local[ny,nx]):
                continue
            nid = (ny*lw+nx)*2+axis
            candidate = cost+1.0+float(turn_penalty)*(axis!=old_axis)
            if candidate < dist[nid]:
                dist[nid] = candidate
                pred[nid] = sid
                heuristic = abs(nx-gx)+abs(ny-gy)
                heapq.heappush(queue,(candidate+heuristic,-candidate,nid))
    if end is None:
        return {'status':'no_path','expansions':expansions}
    chain = []
    while end >= 0:
        y,x = divmod(end//2,lw)
        chain.append((x+x0,y+y0))
        end = int(pred[end])
    chain.reverse()
    points = _rectilinear_chain_path(chain,mask)
    if points is None:
        return {'status':'validation_failed'}
    return {'status':'ok','points':points,'scale':1,'expansions':expansions,
            'path_len':float(len(chain)-1),'turn_count':max(0,len(points)-2),
            'engine':'bounded_target_axis_astar'}


def _raw_free_gateway_pairs(region, public, walls, other_rooms, max_gap, separation, limit=3):
    """Search distributed boundary contacts and verify a local, possibly bent entry.

    The raw wall mask is authoritative here. Collision dilation is a navigation
    margin, not evidence that a physical doorway is closed.
    """
    from scipy.spatial import cKDTree
    radius=max(3,int(np.ceil(max_gap)))
    nearby=cv2.dilate(((region|public).astype(np.uint8)),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))>0
    allowed=nearby & (walls==0) & ~other_rooms
    a=region & allowed;b=public & allowed
    boundary=a & ~(cv2.erode(a.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    ay,ax=np.where(boundary);by,bx=np.where(b & ~(cv2.erode(b.astype(np.uint8),np.ones((3,3),np.uint8))>0))
    if not len(ax) or not len(bx):return []
    # Uniform boundary sampling avoids starving wider entrances behind the
    # hundreds of minimum-distance points along a shared closed wall.
    sample=np.linspace(0,len(ax)-1,min(1200,len(ax)),dtype=int)
    pts=np.column_stack((ax[sample],ay[sample]));targets=np.column_stack((bx,by))
    distance,index=cKDTree(targets).query(pts)
    order=np.argsort(distance,kind='stable');selected=[];tested=[]
    h,w=walls.shape
    for i in order:
        if distance[i]>max_gap:break
        p=tuple(map(int,pts[i]));q=tuple(map(int,targets[index[i]]))
        if any(math.hypot(p[0]-v[0],p[1]-v[1])<max(3,separation*.25) for v in tested):continue
        tested.append(p)
        if len(tested)>100:break
        if any(math.hypot(p[0]-v[0][0],p[1]-v[0][1])<separation for v in selected):continue
        pad=max(12,int(max_gap*2))
        x0,y0,x1,y1=_clip_box(min(p[0],q[0])-pad,min(p[1],q[1])-pad,
                             max(p[0],q[0])+pad+1,max(p[1],q[1])+pad+1,w,h)
        result=_compiled_axis_path((p[0]-x0,p[1]-y0),(q[0]-x0,q[1]-y0),
                                  allowed[y0:y1,x0:x1].astype(np.uint8)*255,8.0)
        if result.get('status')!='ok':continue
        selected.append((p,q,float(distance[i])))
        if len(selected)>=limit:break
    return selected


def _build_blocked_background(bg_mask, image, ocr_data):
    """Keep exterior background and conservative, unlabeled interior voids.

    Interior candidates must match the known canvas background, have substantial
    area, and be surrounded by a contrasting coloured band on all four sides.
    White rooms with OCR labels and monochrome plans are explicitly preserved.
    Only candidate pixels are blocked; never fill its bounding box or hull.
    """
    h, w = bg_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats((bg_mask > 0).astype(np.uint8), 8)
    keep = np.zeros(count, dtype=bool)
    report = {"version": "background_void_v4", "components": []}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image is not None else None
    text_mask = np.zeros((h, w), np.uint8)
    for item in ocr_data or []:
        poly = _ocr_poly_from_item(item, w, h)
        if poly is not None:
            cv2.fillPoly(text_mask, [np.asarray(poly, np.int32)], 255)
    for cid in range(1, count):
        x, y, bw, bh, area = map(int, stats[cid])
        ratio = area / float(h*w)
        exterior = x <= 1 or y <= 1 or x+bw >= w-1 or y+bh >= h-1
        reason = "preserved"
        if exterior and .004 <= ratio <= .90:
            keep[cid] = True
            reason = "canvas_background"
        elif not exterior and ratio >= .008 and min(bw,bh) >= min(h,w)*.10 and hsv is not None:
            r = max(4, int(round(min(h,w)*.009)))
            x1,y1,x2,y2 = _clip_box(x-r*2,y-r*2,x+bw+r*2,y+bh+r*2,w,h)
            comp = (labels[y1:y2,x1:x2] == cid).astype(np.uint8)
            # Closing only supplies a text-overlap test; its pixels are NOT blocked.
            envelope = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((r*2+1,r*2+1), np.uint8))
            text_present = np.any((text_mask[y1:y2,x1:x2] > 0) & (envelope > 0))
            ring = (cv2.dilate(comp, np.ones((r*2+1,r*2+1),np.uint8)) > 0) & (comp == 0)
            sat = hsv[y1:y2,x1:x2,1]
            inside_sat = float(np.median(sat[comp > 0]))
            contrast = (sat.astype(np.float32) >= max(24.0, inside_sat + 16.0)) & ring
            yy,xx = np.indices(comp.shape)
            cx,cy = x+bw*.5-x1, y+bh*.5-y1
            sectors = [ring & (yy < cy) & (abs(xx-cx) <= abs(yy-cy)),
                       ring & (yy >= cy) & (abs(xx-cx) <= abs(yy-cy)),
                       ring & (xx < cx) & (abs(xx-cx) > abs(yy-cy)),
                       ring & (xx >= cx) & (abs(xx-cx) > abs(yy-cy))]
            votes = sum(np.count_nonzero(contrast & side) / float(max(1,np.count_nonzero(side))) >= .50 for side in sectors)
            contrast_ratio = np.count_nonzero(contrast) / float(max(1,np.count_nonzero(ring)))
            if not text_present and inside_sat < 45 and votes == 4 and contrast_ratio >= .65:
                keep[cid] = True
                reason = "unlabeled_background_void_inside_coloured_ring"
            report["components"].append({"id":cid,"area":area,"blocked":bool(keep[cid]),
                "reason":reason,"text_present":bool(text_present),"contrast_ratio":round(contrast_ratio,4),
                "surround_votes":int(votes),"inside_saturation":inside_sat})
        if keep[cid] and reason == "canvas_background":
            report["components"].append({"id":cid,"area":area,"blocked":True,"reason":reason})
    return keep[labels].astype(np.uint8)*255, report


def _infer_background_color_ids(labels_2d, centers, k):
    """以四邊與角落的一致性辨識真正的畫布背景。

    舊版只看一條邊界線的總占比；當房間色塊貼到右側或下側時，房間色也會被
    當成背景，之後整片寫入 wall matrix。新版要求候選色同時出現在多個角落或
    至少三個方向的邊帶，並只允許與最佳背景顏色相近的次要背景色加入。

    若地圖本身裁切到完全沒有背景，寧可回傳空集合，也不刪除貼邊房間。
    """
    H, W = labels_2d.shape[:2]
    k = int(k)
    band = int(np.clip(round(min(H, W) * 0.018), 6, 36))
    corner = int(np.clip(round(min(H, W) * 0.055), 18, 96))

    side_slices = [
        labels_2d[:band, :],
        labels_2d[max(0, H - band):H, :],
        labels_2d[:, :band],
        labels_2d[:, max(0, W - band):W],
    ]
    corner_slices = [
        labels_2d[:corner, :corner],
        labels_2d[:corner, max(0, W - corner):W],
        labels_2d[max(0, H - corner):H, :corner],
        labels_2d[max(0, H - corner):H, max(0, W - corner):W],
    ]

    records = []
    for color_id in range(k):
        side_ratios = [float(np.mean(side == color_id)) if side.size else 0.0 for side in side_slices]
        corner_ratios = [float(np.mean(block == color_id)) if block.size else 0.0 for block in corner_slices]
        side_votes = sum(ratio >= 0.08 for ratio in side_ratios)
        corner_votes = sum(ratio >= 0.18 for ratio in corner_ratios)
        border_mean = float(np.mean(side_ratios))

        # 多方向一致才像畫布背景；單純貼著一至兩側的房間不可通過。
        eligible = bool(
            corner_votes >= 2
            or side_votes >= 3
            or (corner_votes >= 1 and side_votes >= 3)
        )
        score = (
            0.52 * (corner_votes / 4.0)
            + 0.33 * (side_votes / 4.0)
            + 0.15 * min(1.0, border_mean / 0.40)
        )
        records.append({
            "color_id": int(color_id),
            "eligible": eligible,
            "score": float(score),
            "side_votes": int(side_votes),
            "corner_votes": int(corner_votes),
            "side_ratios": side_ratios,
            "corner_ratios": corner_ratios,
        })

    eligible = [rec for rec in records if rec["eligible"]]
    if not eligible:
        return [], records

    eligible.sort(key=lambda rec: rec["score"], reverse=True)
    best = eligible[0]
    bg_ids = [int(best["color_id"])]

    # 漸層或陰影背景可能被 K-Means 切成兩色；只有色距接近且邊界證據也夠強
    # 才合併，避免把淺色房間順手視為第二背景。
    centers_arr = np.asarray(centers, dtype=np.float32)
    best_center = centers_arr[int(best["color_id"])] if centers_arr.ndim == 2 else None
    for rec in eligible[1:]:
        if rec["score"] < max(0.46, best["score"] * 0.72):
            continue
        if best_center is None:
            continue
        color_id = int(rec["color_id"])
        bgr_distance = float(np.linalg.norm(centers_arr[color_id] - best_center))
        if bgr_distance <= 24.0 and (rec["corner_votes"] >= 2 or rec["side_votes"] == 4):
            bg_ids.append(color_id)

    return sorted(set(bg_ids)), records


def analyze_colors_and_corridor(image_path, ocr_data, k=8, max_dim=1200, return_details=False):
    """
    FAST V4：保留原版 K-Means 品質，但不再把 OCR 中心點當成走道色的最終依據。

    OCR bbox 中心經常剛好落在白色字、黑色字或字元空隙，而不是店面/走道底色；
    因此舊版「哪個色群上的 OCR 中心較少，就選哪個」在不同配色地圖上可能於店面色
    與走道色之間翻轉。此函式仍產生 legacy 暫定遮罩以維持相容性，但同時回傳
    所有非背景色群，交由 RoomSegmenter 使用牆體拓樸選出真正走道色。
    """
    img = safe_imread(image_path)
    if img is None:
        return None

    H, W = img.shape[:2]
    print("[系統] 正在執行品質保護式 K-Means 走道分析...")

    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    blurred = cv2.morphologyEx(
        filtered_img,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    scale = min(1.0, float(max_dim) / float(max(H, W)))
    if scale < 1.0:
        proc_img = cv2.resize(
            blurred,
            (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
            interpolation=cv2.INTER_AREA
        )
    else:
        proc_img = blurred

    pixels = proc_img.reshape((-1, 3)).astype(np.float32)
    if pixels.size == 0:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(
        pixels,
        int(k),
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS
    )

    labels_small = labels.reshape(proc_img.shape[:2]).astype(np.uint8)
    if labels_small.shape != (H, W):
        labels_2d = cv2.resize(labels_small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        labels_2d = labels_small

    bg_ids, background_records = _infer_background_color_ids(labels_2d, centers, int(k))
    if bg_ids:
        print(f"[色彩] 四邊/角落一致性背景色群={bg_ids}")
    else:
        print("[色彩] 未找到具多邊一致性的畫布背景；保守地不刪除任何貼邊色群。")

    counts = np.bincount(labels_2d.ravel(), minlength=int(k)).astype(np.int64)
    non_bg_counts = counts.copy()
    for bg_id in bg_ids:
        non_bg_counts[bg_id] = 0

    top2_ids = non_bg_counts.argsort()[-2:][::-1]
    id_1, id_2 = int(top2_ids[0]), int(top2_ids[1])
    text_count_1 = 0
    text_count_2 = 0
    for item in ocr_data:
        cx, cy = map(int, item['center'])
        if 0 <= cx < W and 0 <= cy < H:
            label_at_text = int(labels_2d[cy, cx])
            text_count_1 += int(label_at_text == id_1)
            text_count_2 += int(label_at_text == id_2)

    legacy_corridor_id = id_1 if text_count_1 <= text_count_2 else id_2
    corridor_mask = _build_corridor_cluster_mask(labels_2d, legacy_corridor_id)

    bg_mask = np.zeros((H, W), dtype=np.uint8)
    if bg_ids:
        bg_mask[np.isin(labels_2d, np.asarray(bg_ids, dtype=labels_2d.dtype))] = 255

    min_candidate_pixels = max(64, int(H * W * 0.002))
    candidate_ids = [
        int(i) for i in range(int(k))
        if i not in bg_ids and int(counts[i]) >= min_candidate_pixels
    ]

    details = {
        "labels_2d": labels_2d,
        "centers": centers,
        "counts": counts,
        "bg_ids": [int(v) for v in bg_ids],
        "candidate_ids": candidate_ids,
        "background_records": background_records,
        "legacy_corridor_id": int(legacy_corridor_id),
        "legacy_top2_ids": [id_1, id_2],
        "legacy_text_counts": [int(text_count_1), int(text_count_2)],
    }

    if return_details:
        return corridor_mask, bg_mask, details
    return corridor_mask, bg_mask

# =========================================
# 模組 B：牆體極致提取與幾何修補 V4 — evidence-gated two-stage repair
# =========================================
def _skeletonize_uint8(mask):
    """回傳 0/255 單像素骨架；優先 ximgproc，其次 skimage，最後有界形態學。"""
    src = ((mask > 0).astype(np.uint8) * 255)
    try:
        return cv2.ximgproc.thinning(src)
    except AttributeError:
        try:
            from skimage.morphology import skeletonize
            return skeletonize(src > 0).astype(np.uint8) * 255
        except ImportError:
            pass
        # Zero padding guarantees termination even for an all-foreground mask.
        src = np.pad(src, 1)
        skeleton = np.zeros(src.shape, np.uint8)
        eroded = src.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(eroded, opened)
            eroded = cv2.erode(eroded, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            if cv2.countNonZero(eroded) == 0:
                break
        return skeleton[1:-1, 1:-1]


def _estimate_wall_width_map(wall_mask):
    """以 distance transform 估每個牆像素的局部牆寬。牆中心約為 2 * distance。"""
    wall_u8 = ((wall_mask > 0).astype(np.uint8) * 255)
    dist = cv2.distanceTransform(wall_u8, cv2.DIST_L2, 3)
    return dist * 2.0


def _clip_box(x1, y1, x2, y2, W, H):
    return max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))


def _ocr_poly_from_item(item, W, H):
    """讀取 OCR polygon；若舊資料沒有 poly，就退回 box 的四點 polygon。"""
    if item.get('poly') is not None:
        poly = np.array(item['poly'], dtype=np.int32)
    else:
        x1, y1, x2, y2 = item['box']
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return poly


def _build_adaptive_text_masks(ocr_data, shape, reference_wall_mask):
    """
    V2 文字遮罩：
    1) 使用 OCR/text detector polygon，而不是外接矩形。
    2) 膨脹量依局部牆寬估計，不再使用固定 pad。
    3) tight_mask 用來偵錯；adaptive_mask 才用於真正抹除與 repair region。
    """
    H, W = shape
    tight_mask = np.zeros((H, W), dtype=np.uint8)
    adaptive_mask = np.zeros((H, W), dtype=np.uint8)
    width_map = _estimate_wall_width_map(reference_wall_mask)

    for item in ocr_data:
        poly = _ocr_poly_from_item(item, W, H)
        one = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(one, [poly], 255)
        tight_mask = cv2.bitwise_or(tight_mask, one)

        x, y, w, h = cv2.boundingRect(poly)
        context_pad = int(np.clip(max(10, int(max(w, h) * 0.35)), 10, 60))
        x1, y1, x2, y2 = _clip_box(x - context_pad, y - context_pad, x + w + context_pad, y + h + context_pad, W, H)
        context = np.zeros((H, W), dtype=np.uint8)
        context[y1:y2, x1:x2] = 255
        around_text = cv2.dilate(one, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, context_pad // 2 * 2 + 1), max(5, context_pad // 2 * 2 + 1))))
        around_text = cv2.bitwise_and(around_text, context)

        local_widths = width_map[(around_text > 0) & (reference_wall_mask > 0)]
        if local_widths.size > 0:
            local_wall_width = float(np.median(local_widths))
        else:
            # fallback：文字高度越小，pad 越小；避免長文字造成巨型框。
            local_wall_width = max(2.0, min(8.0, min(max(w, 1), max(h, 1)) * 0.18))

        pad = int(np.clip(round(local_wall_width * 0.65 + 1), 1, 10))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        adaptive_mask = cv2.bitwise_or(adaptive_mask, cv2.dilate(one, kernel, iterations=1))

    # 只做非常小的 close，連接同一段文字內的裂縫，但避免跨牆跨區合併。
    adaptive_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return tight_mask, adaptive_mask, width_map


def _component_touches_outside_foreground(comp_full, raw_binary, region_mask, dilate_r=3):
    """檢查某 foreground component 是否和 region 外的 foreground 相接。
    真正穿過文字框/YOLO 框的牆線通常會延伸到框外；文字筆畫通常不會。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r * 2 + 1, dilate_r * 2 + 1))
    comp_d = cv2.dilate(comp_full, k)
    outside = cv2.bitwise_and(raw_binary, cv2.bitwise_not(region_mask))
    return int(cv2.countNonZero(cv2.bitwise_and(comp_d, outside)))


def _is_wall_like_foreground_component(area, cw, ch, outside_contact, local_region_w, local_region_h, mode='text'):
    """判斷 region 內的 foreground component 比較像牆線還是文字/icon 筆畫。

    核心原則：
    - 文字刪除只刪 OCR region 內實際的白色筆畫，不再刪整個 polygon。
    - 若 component 是長、細、並且能和 region 外的 foreground 連續，優先視為牆線保留。
    - YOLO 框常誤框到房間，所以 YOLO 模式更保守：只刪 compact/短小物件，保留長線。
    """
    length = max(cw, ch)
    thickness_est = area / float(max(length, 1))
    aspect = length / float(max(min(cw, ch), 1))

    # 長、薄、而且接到框外 foreground，最像「牆穿過文字/圖示框」。
    strong_wall_line = (length >= 22 and aspect >= 5.0 and thickness_est <= 5.5 and outside_contact >= 3)

    # 非常長的線，就算接觸框外較少，也多半不是文字本身。
    very_long_line = (length >= 0.65 * max(local_region_w, local_region_h) and aspect >= 4.0 and thickness_est <= 6.5)

    if mode == 'yolo':
        # YOLO 若誤抓整個房間，裡面會有很多長牆線；要更偏向保留。
        return strong_wall_line or very_long_line or (length >= 70 and aspect >= 3.5 and thickness_est <= 7.0)

    return strong_wall_line or very_long_line


def _accumulate_precise_region(local_region, origin, raw_u8, mode, outputs):
    """只在單一 OCR/YOLO ROI 內做 connected-components，不再為每個筆畫配置全圖 mask。"""
    x1, y1 = map(int, origin)
    h, w = local_region.shape[:2]
    if h <= 0 or w <= 0:
        return
    x2, y2 = x1 + w, y1 + h
    union_mask, remove_mask, repair_seed, preserve_mask = outputs
    union_roi = union_mask[y1:y2, x1:x2]
    remove_roi = remove_mask[y1:y2, x1:x2]
    repair_roi = repair_seed[y1:y2, x1:x2]
    preserve_roi = preserve_mask[y1:y2, x1:x2]
    raw_roi = raw_u8[y1:y2, x1:x2]

    region_u8 = ((local_region > 0).astype(np.uint8) * 255)
    union_roi[region_u8 > 0] = 255
    local_fg = cv2.bitwise_and(raw_roi, region_u8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((local_fg > 0).astype(np.uint8), connectivity=8)
    outside = cv2.bitwise_and(raw_roi, cv2.bitwise_not(region_u8))
    _, _, region_w, region_h = cv2.boundingRect(region_u8)
    contact_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i, :5])
        if area < 2:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        outside_contact = int(cv2.countNonZero(cv2.bitwise_and(cv2.dilate(comp, contact_kernel), outside)))
        wall_like = _is_wall_like_foreground_component(area, cw, ch, outside_contact, region_w, region_h, mode=mode)
        length = max(cw, ch)
        aspect = length / float(max(min(cw, ch), 1))
        extent = area / float(max(cw * ch, 1))

        if wall_like:
            preserve_roi[comp > 0] = 255
            continue

        if mode == 'text':
            remove = not (area > 900 and outside_contact >= 10)
        else:
            compact_icon = area <= 900 and length <= 90 and aspect < 8.0
            tiny_noise = area < 40
            closed_icon = extent > 0.28 and area <= 700 and length <= 80
            remove = tiny_noise or compact_icon or closed_icon

        if remove:
            remove_roi[comp > 0] = 255
            if outside_contact >= (12 if mode == 'text' else 8):
                repair_roi[cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0] = 255
        else:
            preserve_roi[comp > 0] = 255


def _build_precise_foreground_removal_from_masks(region_masks, shape, raw_binary, mode='text'):
    """相容舊 API；實際 connected-components 改成逐 ROI 執行。"""
    H, W = shape
    outputs = tuple(np.zeros((H, W), dtype=np.uint8) for _ in range(4))
    raw_u8 = ((raw_binary > 0).astype(np.uint8) * 255)
    for full_mask in region_masks:
        ys, xs = np.where(full_mask > 0)
        if len(xs) == 0:
            continue
        pad = 4
        x1, y1, x2, y2 = _clip_box(xs.min() - pad, ys.min() - pad, xs.max() + pad + 1, ys.max() + pad + 1, W, H)
        _accumulate_precise_region(full_mask[y1:y2, x1:x2], (x1, y1), raw_u8, mode, outputs)
    union_mask, remove_mask, repair_seed, preserve_mask = outputs
    remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return union_mask, remove_mask, repair_seed, preserve_mask


def _build_precise_text_stroke_masks(ocr_data, shape, raw_binary):
    """直接由 polygon 的局部 bbox 建立筆畫遮罩，避免建立 OCR 數量份全圖陣列。"""
    H, W = shape
    outputs = tuple(np.zeros((H, W), dtype=np.uint8) for _ in range(4))
    raw_u8 = ((raw_binary > 0).astype(np.uint8) * 255)
    for item in ocr_data:
        poly = _ocr_poly_from_item(item, W, H)
        x, y, w, h = cv2.boundingRect(poly)
        pad = 4
        x1, y1, x2, y2 = _clip_box(x - pad, y - pad, x + w + pad, y + h + pad, W, H)
        local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted = poly - np.array([x1, y1], dtype=np.int32)
        cv2.fillPoly(local, [shifted], 255)
        _accumulate_precise_region(local, (x1, y1), raw_u8, 'text', outputs)
    union_mask, remove_mask, repair_seed, preserve_mask = outputs
    remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return union_mask, remove_mask, repair_seed, preserve_mask


def _build_precise_box_stroke_masks(boxes, shape, raw_binary, pad=2):
    """直接在 YOLO box 的局部 ROI 分析，不再建立 box 數量份全圖遮罩。"""
    H, W = shape
    outputs = tuple(np.zeros((H, W), dtype=np.uint8) for _ in range(4))
    raw_u8 = ((raw_binary > 0).astype(np.uint8) * 255)
    for x1, y1, x2, y2 in boxes:
        bx1, by1, bx2, by2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)
        context = 4
        rx1, ry1, rx2, ry2 = _clip_box(bx1 - context, by1 - context, bx2 + context, by2 + context, W, H)
        local = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
        cv2.rectangle(local, (bx1 - rx1, by1 - ry1), (max(bx1 - rx1, bx2 - rx1 - 1), max(by1 - ry1, by2 - ry1 - 1)), 255, -1)
        _accumulate_precise_region(local, (rx1, ry1), raw_u8, 'yolo', outputs)
    union_mask, remove_mask, repair_seed, preserve_mask = outputs
    remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return union_mask, remove_mask, repair_seed, preserve_mask


def _make_box_mask(boxes, shape, pad=4):
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        ax1, ay1, ax2, ay2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)
        cv2.rectangle(mask, (ax1, ay1), (ax2, ay2), 255, -1)
    return mask


def _extract_structural_room_separators(
    image_path,
    base_walls,
    ocr_data,
    yolo_boxes=None,
    bg_mask=None,
):
    """從原圖補回細白線、淡色邊界與被文字截斷的長直分隔線。

    這是房間分割的第二條證據，不取代既有牆體：
      1) OCR/圖示區先暫時 inpaint，避免字的水平筆畫被當成隔間；
      2) 同時使用 LAB 色差邊緣與亮線 top-hat，支援黑牆、白線與彩色底圖；
      3) Hough 候選必須接近既有牆、具有兩側色差或亮線證據；
      4) 全程在縮小影像運算，2K/4K 地圖不會多出長時間等待。
    """
    report = {
        "version": "appearance_partition_v3",
        "enabled": bool(_env_flag("MAP_ENABLE_APPEARANCE_ROOM_PARTITION", True)),
        "candidate_line_count": 0,
        "accepted_line_count": 0,
        "added_wall_pixels": 0,
        "added_wall_ratio": 0.0,
    }
    if not report["enabled"]:
        return np.zeros_like(base_walls, dtype=np.uint8), report

    img = safe_imread(image_path)
    if img is None or base_walls is None:
        report["reason"] = "missing_image_or_wall_mask"
        return np.zeros_like(base_walls, dtype=np.uint8), report

    H, W = base_walls.shape[:2]
    max_dim = int(np.clip(int(os.environ.get("MAP_ROOM_PARTITION_MAX_DIM", "1500")), 900, 2200))
    scale = min(1.0, max_dim / float(max(H, W)))
    work_w = max(1, int(round(W * scale)))
    work_h = max(1, int(round(H * scale)))
    work_short = max(1, min(work_h, work_w))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    work = cv2.resize(img, (work_w, work_h), interpolation=interpolation)
    base = cv2.resize(
        ((base_walls > 0).astype(np.uint8) * 255),
        (work_w, work_h),
        interpolation=cv2.INTER_NEAREST,
    )

    erase_full = np.zeros((H, W), dtype=np.uint8)
    for item in ocr_data or []:
        try:
            poly = _ocr_poly_from_item(item, W, H)
        except (KeyError, TypeError, ValueError):
            continue
        cv2.fillPoly(erase_full, [poly], 255)
    if yolo_boxes:
        erase_full = cv2.bitwise_or(erase_full, _make_box_mask(yolo_boxes, (H, W), pad=2))

    erase = cv2.resize(erase_full, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    erase_pad = int(np.clip(round(work_short * 0.0025), 2, 6))
    erase = cv2.dilate(
        erase,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erase_pad * 2 + 1, erase_pad * 2 + 1)),
    )
    if np.any(erase):
        # 此影像只供分隔線偵測，不會覆寫原圖，也不會改變 OCR 結果。
        work = cv2.inpaint(work, erase, max(2, erase_pad), cv2.INPAINT_TELEA)

    smooth = cv2.bilateralFilter(work, 7, 42, 42)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)

    # 低對比淡藍/米色邊界在灰階 Canny 中很容易消失；LAB 三通道聯集較穩定。
    edge_evidence = np.zeros((work_h, work_w), dtype=np.uint8)
    for idx, channel in enumerate(channels):
        low, high = ((18, 58) if idx == 0 else (10, 36))
        edge_evidence = cv2.bitwise_or(
            edge_evidence,
            cv2.Canny(channel, low, high, L2gradient=True),
        )

    # 亮色細分隔線：L - opening(L)。黑字不會成為正的 top-hat，能降低誤抓文字。
    ridge_kernel_size = int(np.clip(round(work_short * 0.0045), 5, 11))
    if ridge_kernel_size % 2 == 0:
        ridge_kernel_size += 1
    lightness = channels[0]
    top_hat = cv2.morphologyEx(
        lightness,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ridge_kernel_size, ridge_kernel_size)),
    )
    positive = top_hat[top_hat > 0]
    auto_ridge = float(np.percentile(positive, 68)) if positive.size else 8.0
    ridge_threshold = int(np.clip(auto_ridge, 6, 18))
    ridge_evidence = (top_hat >= ridge_threshold).astype(np.uint8) * 255
    ridge_evidence = cv2.morphologyEx(
        ridge_evidence,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    evidence = cv2.bitwise_or(edge_evidence, ridge_evidence)
    # inpaint 邊界本身也可能產生人工 edge；保留 Hough 的跨缺口能力，不保留該區 edge。
    evidence[erase > 0] = 0

    base_near_radius = int(np.clip(round(work_short * 0.0040), 3, 8))
    base_near = cv2.dilate(
        base,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (base_near_radius * 2 + 1, base_near_radius * 2 + 1),
        ),
    )
    endpoint_radius = int(np.clip(round(work_short * 0.0080), 7, 16))
    endpoint_near = cv2.dilate(
        base,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (endpoint_radius * 2 + 1, endpoint_radius * 2 + 1),
        ),
    )
    ridge_near = cv2.dilate(ridge_evidence, np.ones((3, 3), dtype=np.uint8))

    foreground_near = None
    if bg_mask is not None:
        bg_work = cv2.resize(bg_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
        foreground_near = cv2.dilate(
            (bg_work == 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )

    min_line_length = int(np.clip(round(work_short * 0.022), 30, 70))
    max_line_gap = int(np.clip(round(work_short * 0.010), 8, 22))
    hough_threshold = int(np.clip(round(work_short * 0.016), 22, 55))
    lines = cv2.HoughLinesP(
        evidence,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        report["reason"] = "no_hough_candidates"
        return np.zeros_like(base_walls, dtype=np.uint8), report

    # OpenCV 4 usually returns (N, 1, 4), while OpenCV 5 may return (N, 4).
    # Normalizing here keeps the partition stage portable across deployments.
    lines = np.asarray(lines).reshape(-1, 4)
    report["candidate_line_count"] = int(len(lines))
    candidates = []
    side_offset = int(np.clip(round(work_short * 0.0030), 3, 7))
    for raw_line in lines:
        x1, y1, x2, y2 = map(int, raw_line)
        dx, dy = x2 - x1, y2 - y1
        length = float(math.hypot(dx, dy))
        if length < min_line_length:
            continue

        sample_count = int(np.clip(round(length), 18, 700))
        ts = np.linspace(0.04, 0.96, sample_count, dtype=np.float32)
        xs = np.clip(np.rint(x1 + dx * ts).astype(np.int32), 0, work_w - 1)
        ys = np.clip(np.rint(y1 + dy * ts).astype(np.int32), 0, work_h - 1)
        evidence_support = float(np.mean(evidence[ys, xs] > 0))
        ridge_support = float(np.mean(ridge_near[ys, xs] > 0))
        base_support = float(np.mean(base_near[ys, xs] > 0))
        foreground_support = (
            float(np.mean(foreground_near[ys, xs] > 0))
            if foreground_near is not None else 1.0
        )
        endpoint_hits = int(endpoint_near[y1, x1] > 0) + int(endpoint_near[y2, x2] > 0)

        nx, ny = -dy / max(length, 1.0), dx / max(length, 1.0)
        ax = np.clip(np.rint(xs + nx * side_offset).astype(np.int32), 0, work_w - 1)
        ay = np.clip(np.rint(ys + ny * side_offset).astype(np.int32), 0, work_h - 1)
        bx = np.clip(np.rint(xs - nx * side_offset).astype(np.int32), 0, work_w - 1)
        by = np.clip(np.rint(ys - ny * side_offset).astype(np.int32), 0, work_h - 1)
        side_differences = np.linalg.norm(
            lab[ay, ax].astype(np.float32) - lab[by, bx].astype(np.float32),
            axis=1,
        )
        side_delta = float(np.median(side_differences)) if side_differences.size else 0.0

        if evidence_support < 0.14:
            continue
        if foreground_support < 0.28 and base_support < 0.36:
            continue
        # A real missing partition normally joins existing structure at both
        # ends.  A line with weaker attachment is accepted only when much of it
        # already follows the pre-clean wall prior.  This rejects text baselines
        # and logo edges without assuming horizontal/vertical architecture.
        strong_attachment = bool(
            endpoint_hits == 2
            or base_support >= 0.42
            or (
                endpoint_hits >= 1
                and base_support >= 0.24
                and evidence_support >= 0.20
            )
        )
        if not strong_attachment:
            continue
        if ridge_support < 0.08 and side_delta < 5.5 and base_support < 0.45:
            continue

        score = (
            1.20 * evidence_support
            + 0.95 * ridge_support
            + 0.85 * base_support
            + 0.22 * endpoint_hits
            + 0.018 * min(side_delta, 24.0)
            + 0.0008 * min(length, 500.0)
        )
        angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
        mid_x, mid_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        normal_x = -math.sin(math.radians(angle))
        normal_y = math.cos(math.radians(angle))
        rho = mid_x * normal_x + mid_y * normal_y
        candidates.append({
            "line": (x1, y1, x2, y2),
            "length": length,
            "score": score,
            "angle": angle,
            "rho": rho,
            "evidence_support": evidence_support,
            "ridge_support": ridge_support,
            "base_support": base_support,
            "side_delta": side_delta,
            "endpoint_hits": endpoint_hits,
        })

    # Hough 常對同一條白線回傳多條幾乎重合的線；以 angle/rho 做輕量 NMS。
    candidates.sort(key=lambda rec: (rec["score"], rec["length"]), reverse=True)
    configured_max_lines = int(os.environ.get("MAP_ROOM_PARTITION_MAX_LINES", "0") or 0)
    auto_max_lines = int(np.clip(round((work_h * work_w) / 8000.0), 60, 600))
    max_lines = int(np.clip(configured_max_lines, 40, 700)) if configured_max_lines > 0 else auto_max_lines
    rho_step = float(np.clip(round(work_short * 0.004), 3, 10))
    selected = []
    occupied_bins = set()
    for rec in candidates:
        angle_bin = int(round(rec["angle"] / 2.5))
        rho_bin = int(round(rec["rho"] / rho_step))
        key = (angle_bin, rho_bin)
        if key in occupied_bins:
            continue
        occupied_bins.add(key)
        selected.append(rec)
        if len(selected) >= max_lines:
            break

    separator_work = np.zeros((work_h, work_w), dtype=np.uint8)
    line_thickness = int(np.clip(round(2.0 * scale), 1, 3))
    for rec in selected:
        x1, y1, x2, y2 = rec["line"]
        cv2.line(separator_work, (x1, y1), (x2, y2), 255, line_thickness, cv2.LINE_8)

    separator_full = cv2.resize(separator_work, (W, H), interpolation=cv2.INTER_NEAREST)
    if scale < 0.82:
        separator_full = cv2.dilate(separator_full, np.ones((2, 2), dtype=np.uint8))
    # 只補短缺口；不使用大 closing，避免兩條相鄰隔間線互相黏住。
    separator_full = cv2.morphologyEx(
        separator_full,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    added = (separator_full > 0) & (base_walls == 0)
    report.update({
        "accepted_line_count": int(len(selected)),
        "added_wall_pixels": int(np.count_nonzero(added)),
        "added_wall_ratio": round(float(np.mean(added)), 7),
        "processing_scale": round(float(scale), 5),
        "ridge_threshold": int(ridge_threshold),
        "min_line_length_px_work": int(min_line_length),
        "max_line_gap_px_work": int(max_line_gap),
        "max_lines": int(max_lines),
        "max_lines_mode": "configured" if configured_max_lines > 0 else "scale_adaptive",
        "rho_nms_step_px_work": round(float(rho_step), 3),
    })
    return separator_full, report


def _partition_topology_signature(wall_mask):
    """Summarise enclosure quality at several scale-relative closing sizes."""
    walls = ((wall_mask > 0).astype(np.uint8) * 255)
    h, w = walls.shape[:2]
    map_area = float(max(1, h * w))
    nominal = max(3, int(round(math.hypot(h, w) * 0.010)))
    kernel_sizes = []
    for multiplier in (0.72, 1.0, 1.28):
        size = max(3, int(round(nominal * multiplier)))
        if size % 2 == 0:
            size += 1
        kernel_sizes.append(size)

    records = []
    for size in sorted(set(kernel_sizes)):
        closed = cv2.morphologyEx(
            walls,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (closed == 0).astype(np.uint8), connectivity=8
        )
        accepted_areas = []
        tiny_area = 0
        min_area = max(32, int(map_area / 8000.0))
        for cid in range(1, count):
            x, y, bw, bh, area = map(int, stats[cid, :5])
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            if area < min_area:
                tiny_area += area
                continue
            accepted_areas.append(area)
        total = max(1, int(sum(accepted_areas)))
        records.append({
            'closing_size_px': int(size),
            'region_count': int(len(accepted_areas)),
            'median_area_px': round(float(np.median(accepted_areas)), 2) if accepted_areas else 0.0,
            'largest_region_fraction': round(
                float(max(accepted_areas) / total), 6
            ) if accepted_areas else 1.0,
            'tiny_area_fraction': round(float(tiny_area / map_area), 7),
        })

    counts = [rec['region_count'] for rec in records]
    return {
        'scales': records,
        'median_region_count': float(np.median(counts)) if counts else 0.0,
        'minimum_region_count': int(min(counts)) if counts else 0,
        'maximum_region_count': int(max(counts)) if counts else 0,
        'count_stability': round(
            float(min(counts) / max(1, max(counts))), 6
        ) if counts else 0.0,
        'median_largest_region_fraction': round(float(np.median([
            rec['largest_region_fraction'] for rec in records
        ])), 6) if records else 1.0,
        'maximum_tiny_area_fraction': round(float(max([
            rec['tiny_area_fraction'] for rec in records
        ])), 7) if records else 0.0,
    }


def _accept_partition_candidate(base_walls, separator_mask, report):
    """Global rollback guard for appearance-derived separator lines.

    The candidate must add stable enclosed regions without causing a fragment
    explosion.  When evidence is ambiguous the untouched legacy wall result is
    returned, preserving maps that already segment well.
    """
    candidate = cv2.bitwise_or(
        ((base_walls > 0).astype(np.uint8) * 255),
        ((separator_mask > 0).astype(np.uint8) * 255),
    )
    before = _partition_topology_signature(base_walls)
    after = _partition_topology_signature(candidate)
    before_count = float(before['median_region_count'])
    after_count = float(after['median_region_count'])
    gain = after_count - before_count
    maximum_safe_count = before_count + max(18.0, before_count * 0.65)

    added_ratio = float(report.get('added_wall_ratio', 0.0) or 0.0)
    stable_enough = bool(
        after['count_stability'] >= max(0.58, before['count_stability'] - 0.18)
    )
    no_fragment_explosion = bool(
        after_count <= maximum_safe_count
        and after['maximum_tiny_area_fraction']
        <= before['maximum_tiny_area_fraction'] + 0.012
    )
    structural_gain = bool(
        gain >= 1.0
        or after['median_largest_region_fraction']
        <= before['median_largest_region_fraction'] - 0.025
    )
    accepted = bool(
        added_ratio <= 0.025
        and stable_enough
        and no_fragment_explosion
        and structural_gain
    )
    if added_ratio <= 0.00002:
        accepted = False

    reasons = []
    if added_ratio > 0.025:
        reasons.append('added_wall_ratio_too_high')
    if not stable_enough:
        reasons.append('partition_unstable_across_scales')
    if not no_fragment_explosion:
        reasons.append('fragmentation_guard')
    if not structural_gain:
        reasons.append('no_stable_enclosure_gain')
    if added_ratio <= 0.00002:
        reasons.append('negligible_candidate')
    return accepted, candidate, {
        'accepted': bool(accepted),
        'reasons': reasons or ['stable_enclosure_gain'],
        'median_region_gain': round(float(gain), 3),
        'maximum_safe_region_count': round(float(maximum_safe_count), 3),
        'before': before,
        'after': after,
    }


def _skeleton_degree(skeleton):
    sk = (skeleton > 0).astype(np.uint8)
    neigh = cv2.filter2D(sk, cv2.CV_16S, np.ones((3, 3), np.int16), borderType=cv2.BORDER_CONSTANT) - sk.astype(np.int16)
    return neigh


def _cc_centroids(mask, min_area=1):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    pts = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            pts.append((int(round(centroids[i][0])), int(round(centroids[i][1]))))
    return pts


def _sample_wall_width(pt, wall_width_map, fallback=5.0, radius=4):
    """讀取節點附近的局部牆寬；沒有有效值時回傳 fallback。"""
    H, W = wall_width_map.shape[:2]
    x, y = pt
    x1, y1, x2, y2 = _clip_box(x - radius, y - radius, x + radius + 1, y + radius + 1, W, H)
    vals = wall_width_map[y1:y2, x1:x2]
    vals = vals[vals > 0]
    return float(np.median(vals)) if vals.size else float(fallback)


def _dedupe_points_by_radius(points, radius=10):
    """把太接近的節點合併，避免同一個 junction 被畫成一串點。"""
    if not points:
        return []
    points = [tuple(map(int, p)) for p in points]
    used = [False] * len(points)
    merged = []
    for i, p in enumerate(points):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        changed = True
        # 用簡單 region-growing，處理同一段抖動上連續靠近的候選點。
        while changed:
            changed = False
            cx = sum(q[0] for q in cluster) / len(cluster)
            cy = sum(q[1] for q in cluster) / len(cluster)
            for j, q in enumerate(points):
                if used[j]:
                    continue
                if math.hypot(q[0] - cx, q[1] - cy) <= radius:
                    cluster.append(q)
                    used[j] = True
                    changed = True
        merged.append((int(round(sum(q[0] for q in cluster) / len(cluster))),
                       int(round(sum(q[1] for q in cluster) / len(cluster)))))
    return merged


def _angular_separation_deg(a, b):
    """回傳兩個有向向量的夾角，範圍 0~180 度。"""
    ax, ay = _normalized(a[0], a[1])
    bx, by = _normalized(b[0], b[1])
    dot = float(np.clip(ax * bx + ay * by, -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _axis_residual_deg(vec, axis):
    """把 axis 視為無方向軸線，計算 vec 偏離此軸或反向軸的最小角度。"""
    d = _angular_separation_deg(vec, axis)
    return min(d, 180.0 - d)


def _validate_true_junction(pt, skeleton, wall_width_map, local_width=5.0):
    """
    過濾 skeleton degree>=3 造成的假 junction。

    核心想法：
    真正 T/X junction 在節點周圍會分裂出至少 3 條「足夠長」且方向有明顯差異的骨架分支。
    沿著同一牆體方向的小抖動通常只有 2 條長分支，或第 3 條只是很短的毛刺，因此會被剔除。
    """
    H, W = skeleton.shape[:2]
    x, y = pt
    lw = _sample_wall_width(pt, wall_width_map, fallback=local_width, radius=5)

    inner_r = int(np.clip(round(max(3.0, lw * 0.75)), 3, 8))
    outer_r = int(np.clip(round(max(20.0, lw * 5.0)), 20, 55))
    min_branch_len = float(np.clip(max(12.0, lw * 2.6), 10.0, 35.0))

    x1, y1, x2, y2 = _clip_box(x - outer_r, y - outer_r, x + outer_r + 1, y + outer_r + 1, W, H)
    if x2 <= x1 or y2 <= y1:
        return False, {'reason': 'empty_patch'}

    local_sk = (skeleton[y1:y2, x1:x2] > 0).astype(np.uint8)
    if cv2.countNonZero(local_sk) < 6:
        return False, {'reason': 'too_few_skeleton_pixels'}

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist_map = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inner = (dist_map <= inner_r).astype(np.uint8)
    outer = (dist_map <= outer_r).astype(np.uint8)
    annulus = ((outer > 0) & (inner == 0)).astype(np.uint8)

    branch_zone = cv2.bitwise_and(local_sk, annulus)
    if cv2.countNonZero(branch_zone) == 0:
        return False, {'reason': 'no_branch_zone'}

    touch_zone = cv2.dilate(inner, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(branch_zone, connectivity=8)

    branches = []
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(comp, touch_zone)) == 0:
            # 不是從中心 junction 接出去的分支，忽略。
            continue
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            continue
        gx = xs + x1
        gy = ys + y1
        dists = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)
        max_idx = int(np.argmax(dists))
        length = float(dists[max_idx])
        if length < min_branch_len:
            # 典型的牆線鋸齒/毛刺，不應視為 junction 分支。
            continue
        far_pt = (float(gx[max_idx]), float(gy[max_idx]))
        vx, vy = _normalized(far_pt[0] - x, far_pt[1] - y)
        branches.append({'dir': (vx, vy), 'length': length, 'area': int(stats[i, cv2.CC_STAT_AREA])})

    if len(branches) < 3:
        return False, {'reason': 'less_than_three_long_branches', 'branches': len(branches)}

    # 找最強主軸，若所有分支幾乎都落在同一條軸線上，代表只是共線方向上的抖動。
    dirs = [b['dir'] for b in branches]
    best_axis = None
    best_axis_score = -1.0
    for axis in dirs:
        score = sum(abs(axis[0] * d[0] + axis[1] * d[1]) for d in dirs)
        if score > best_axis_score:
            best_axis_score = score
            best_axis = axis

    residuals = [_axis_residual_deg(d, best_axis) for d in dirs]
    off_axis_count = sum(1 for r in residuals if r >= 32.0)
    if off_axis_count == 0:
        return False, {'reason': 'collinear_jitter', 'branches': len(branches), 'max_residual': round(max(residuals), 2)}

    # 方向多樣性：至少要有一個分支和主軸形成明顯非共線，且分支之間最大夾角不能全都集中在同線反向。
    max_pair_sep_from_collinear = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            sep = _axis_residual_deg(dirs[i], dirs[j])
            max_pair_sep_from_collinear = max(max_pair_sep_from_collinear, sep)
    if max_pair_sep_from_collinear < 30.0:
        return False, {'reason': 'low_angular_diversity', 'branches': len(branches)}

    return True, {
        'reason': 'true_junction',
        'branches': len(branches),
        'off_axis_count': off_axis_count,
        'max_residual': round(max(residuals), 2),
        'width': round(lw, 2)
    }


def _normalized(vx, vy, fallback=(1.0, 0.0)):
    n = math.hypot(vx, vy)
    if n <= 1e-6:
        return fallback
    return vx / n, vy / n


def _region_centroid(region_mask):
    m = cv2.moments((region_mask > 0).astype(np.uint8))
    if m['m00'] == 0:
        return None
    return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])


def _estimate_node_direction(pt, skeleton, region_mask, radius=18):
    """以 skeleton 局部重心 + repair region 方向決定端點朝向，不再由 repair box 邊框決定。"""
    H, W = skeleton.shape[:2]
    x, y = pt
    x1, y1, x2, y2 = _clip_box(x - radius, y - radius, x + radius + 1, y + radius + 1, W, H)
    local = (skeleton[y1:y2, x1:x2] > 0)
    ys, xs = np.where(local)

    rc = _region_centroid(region_mask)
    if rc is not None:
        to_region = _normalized(rc[0] - x, rc[1] - y)
    else:
        to_region = (1.0, 0.0)

    if len(xs) > 1:
        cx = float(xs.mean() + x1)
        cy = float(ys.mean() + y1)
        vx, vy = _normalized(x - cx, y - cy, fallback=to_region)
    else:
        vx, vy = to_region

    # 方向必須指向修補區；若相反就翻轉。
    if vx * to_region[0] + vy * to_region[1] < 0:
        vx, vy = -vx, -vy
    return vx, vy


def _build_wall_prior(binary):
    """
    從「抹掉文字/icon 之前」的二值圖建立 wall prior。
    這不是最終牆體，而是用來判斷 repair region 內是否真的有牆線證據。
    目的：避免品牌文字、店名文字在房間中央被誤當成需要修牆的區域。
    """
    prior = np.zeros_like(binary, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 16:
            continue
        aspect = max(w, h) / float(max(min(w, h), 1))
        # 牆線通常是較長、較薄、或和大結構連在一起；文字小碎片會被排除一部分。
        if (w > 35 or h > 35 or area > 160) and (aspect > 1.8 or area > 260):
            prior[labels == i] = 255
    prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return prior


def _component_orientation_count(mask, center):
    """把 mask 內的接觸像素依相對中心的方向分桶，用來判斷是否只是單側文字/雜訊。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0
    cx, cy = center
    bins = set()
    for x, y in zip(xs[::max(1, len(xs)//400)], ys[::max(1, len(xs)//400)]):
        ang = (math.degrees(math.atan2(y - cy, x - cx)) + 360) % 360
        bins.add(int(ang // 45))
    return len(bins)


def _region_wall_damage_evidence(region_mask, pre_wall_prior, current_walls, wall_width_map, stage='outer', region_bbox=None):
    H, W = region_mask.shape[:2]
    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        if pts is None:
            return False, {'reason': 'no_region_contour'}
        x, y, w, h = cv2.boundingRect(pts)
    else:
        x, y, w, h = map(int, region_bbox)

    width_roi = wall_width_map[max(0, y - 20):min(H, y + h + 20), max(0, x - 20):min(W, x + w + 20)]
    local_widths = width_roi[width_roi > 0]
    lw = float(np.median(local_widths)) if local_widths.size else 5.0
    contact_r = int(np.clip(lw * (2.0 if stage == 'outer' else 2.6) + 5, 8, 22))

    rx1, ry1, rx2, ry2 = _clip_box(x - contact_r - 2, y - contact_r - 2,
                                    x + w + contact_r + 2, y + h + contact_r + 2, W, H)
    local_region = region_mask[ry1:ry2, rx1:rx2]
    close_region = cv2.dilate(local_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contact_r * 2 + 1, contact_r * 2 + 1)))
    contact_band = cv2.bitwise_and(close_region, cv2.bitwise_not(local_region))
    prior_inside = cv2.countNonZero(cv2.bitwise_and(pre_wall_prior[ry1:ry2, rx1:rx2], close_region))
    wall_contact = cv2.bitwise_and(current_walls[ry1:ry2, rx1:rx2], contact_band)
    contact_pixels = cv2.countNonZero(wall_contact)

    m = cv2.moments((local_region > 0).astype(np.uint8))
    if m['m00'] > 0:
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    else:
        center = (x - rx1 + w // 2, y - ry1 + h // 2)
    sector_count = _component_orientation_count(wall_contact, center)

    min_prior = max(10, int(lw * 2.0))
    min_contact = max(8, int(lw * 1.8))
    if stage == 'inner':
        min_prior = max(6, int(lw * 1.2))
        min_contact = max(5, int(lw * 1.0))
    if prior_inside < min_prior:
        return False, {'reason': 'no_pre_wall_evidence', 'prior_inside': int(prior_inside), 'contact_pixels': int(contact_pixels), 'local_width': round(lw, 2)}
    if contact_pixels < min_contact:
        return False, {'reason': 'no_nearby_wall_contact', 'prior_inside': int(prior_inside), 'contact_pixels': int(contact_pixels), 'local_width': round(lw, 2)}
    if stage == 'outer' and sector_count < 1:
        return False, {'reason': 'contact_too_isolated', 'prior_inside': int(prior_inside), 'contact_pixels': int(contact_pixels), 'sectors': int(sector_count), 'local_width': round(lw, 2)}
    return True, {'reason': 'wall_damage_evidence_ok', 'prior_inside': int(prior_inside), 'contact_pixels': int(contact_pixels), 'sectors': int(sector_count), 'local_width': round(lw, 2), 'contact_r': int(contact_r)}


def _endpoint_points_into_region(pt, direction, region_mask, max_probe=70):
    """端點方向前方必須真的打進 repair region，否則多半只是鄰近正常牆端。"""
    H, W = region_mask.shape[:2]
    x0, y0 = pt
    vx, vy = direction
    dil = cv2.dilate(region_mask, np.ones((5, 5), np.uint8), iterations=1)
    hits = 0
    first_hit = None
    for r in range(2, int(max_probe), 2):
        x = int(round(x0 + vx * r))
        y = int(round(y0 + vy * r))
        if x < 0 or x >= W or y < 0 or y >= H:
            break
        if dil[y, x] > 0:
            hits += 1
            if first_hit is None:
                first_hit = r
            if hits >= 2:
                return True, {'first_hit': first_hit, 'hits': hits}
    return False, {'first_hit': first_hit, 'hits': hits}


def _endpoint_has_backbone(pt, direction, skeleton, min_len=10, tolerance=2):
    """
    檢查端點背後是否有穩定牆線。
    毛刺末端也會是 degree==1，但它背後通常只有很短的 skeleton 支撐。
    """
    H, W = skeleton.shape[:2]
    x0, y0 = pt
    vx, vy = direction
    # 背後方向：與修補方向相反。
    bx, by = -vx, -vy
    px, py = -by, bx
    farthest = 0
    hit_count = 0
    for r in range(2, int(max(min_len * 2.5, min_len + 16)), 2):
        found = False
        for off in range(-tolerance, tolerance + 1):
            x = int(round(x0 + bx * r + px * off))
            y = int(round(y0 + by * r + py * off))
            if 0 <= x < W and 0 <= y < H and skeleton[y, x] > 0:
                found = True
                break
        if found:
            farthest = r
            hit_count += 1
    return farthest >= min_len and hit_count >= max(2, min_len // 5), {'farthest': int(farthest), 'hit_count': int(hit_count)}


def _restore_prior_line_segments(candidate_walls, region_mask, pre_wall_prior, wall_width_map, max_lines=80, region_bbox=None):
    H, W = candidate_walls.shape[:2]
    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        if pts is None:
            return []
        x, y, w, h = cv2.boundingRect(pts)
    else:
        x, y, w, h = map(int, region_bbox)
    pad = 16
    rx1, ry1, rx2, ry2 = _clip_box(x - pad, y - pad, x + w + pad, y + h + pad, W, H)
    local_region = region_mask[ry1:ry2, rx1:rx2]
    context = cv2.dilate(local_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    local_prior = cv2.bitwise_and(pre_wall_prior[ry1:ry2, rx1:rx2], context)
    local_prior = cv2.morphologyEx(local_prior, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    lines = cv2.HoughLinesP(local_prior, 1, np.pi / 180, threshold=12, minLineLength=10, maxLineGap=5)
    if lines is None:
        return []
    wall_near = cv2.dilate(candidate_walls[ry1:ry2, rx1:rx2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    accepted = []
    for line in np.asarray(lines).reshape(-1, 4)[:max_lines]:
        lx1, ly1, lx2, ly2 = map(int, line)
        if math.hypot(lx2 - lx1, ly2 - ly1) < 10:
            continue
        e1_touch = wall_near[ly1, lx1] > 0
        e2_touch = wall_near[ly2, lx2] > 0
        p1, p2 = (lx1 + rx1, ly1 + ry1), (lx2 + rx1, ly2 + ry1)
        ratio = _line_region_ratio(region_mask, p1, p2, thickness=3)
        if ratio < 0.18 and not (e1_touch and e2_touch):
            continue
        if not (e1_touch or e2_touch):
            continue
        midx, midy = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
        width = _sample_wall_width((midx, midy), wall_width_map, fallback=4.0, radius=6)
        thick = int(np.clip(round(width * 0.42), 2, 5))
        cv2.line(candidate_walls, p1, p2, 255, thick)
        accepted.append((p1, p2))
    return accepted


def _find_skeleton_nodes_near_region(walls, region_mask, wall_width_map, stage='outer', skeleton=None, degree_map=None, region_bbox=None):
    """FAST V1：每個 stage 共用 skeleton；每個 repair region 只在局部 ROI 建 contact/search band。"""
    H, W = walls.shape[:2]
    if skeleton is None:
        skeleton = _skeletonize_uint8(walls)
    if degree_map is None:
        degree_map = _skeleton_degree(skeleton)

    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        if pts is None:
            return [], [], skeleton
        x, y, w, h = cv2.boundingRect(pts)
    else:
        x, y, w, h = map(int, region_bbox)

    local_widths = wall_width_map[max(0, y - 20):min(H, y + h + 20), max(0, x - 20):min(W, x + w + 20)]
    local_widths = local_widths[local_widths > 0]
    local_width = float(np.median(local_widths)) if local_widths.size else 5.0
    contact_r = int(np.clip(local_width * (1.7 if stage == 'outer' else 2.2) + 5, 8, 20))
    search_r = int(np.clip(local_width * 4.5 + 16, 22, 58))

    rx1, ry1, rx2, ry2 = _clip_box(x - search_r - 3, y - search_r - 3,
                                    x + w + search_r + 3, y + h + search_r + 3, W, H)
    local_region = region_mask[ry1:ry2, rx1:rx2]
    contact_outer = cv2.dilate(local_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contact_r * 2 + 1, contact_r * 2 + 1)))
    contact_band = cv2.bitwise_and(contact_outer, cv2.bitwise_not(local_region))
    search_outer = cv2.dilate(local_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_r * 2 + 1, search_r * 2 + 1)))
    search_band = cv2.bitwise_and(search_outer, cv2.bitwise_not(local_region))

    sk_local = skeleton[ry1:ry2, rx1:rx2]
    deg_local = degree_map[ry1:ry2, rx1:rx2]
    endpoint_mask = ((sk_local > 0) & (contact_band > 0) & (deg_local == 1)).astype(np.uint8) * 255
    raw_junction_mask = ((sk_local > 0) & (search_band > 0) & (deg_local >= 3)).astype(np.uint8) * 255

    endpoint_pts = _cc_centroids(cv2.dilate(endpoint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), min_area=1)
    endpoint_pts = [(px + rx1, py + ry1) for px, py in endpoint_pts]
    endpoint_pts = _dedupe_points_by_radius(endpoint_pts, radius=int(np.clip(local_width * 1.5 + 4, 7, 16)))

    raw_junction_pts = _cc_centroids(cv2.dilate(raw_junction_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), min_area=1)
    raw_junction_pts = [(px + rx1, py + ry1) for px, py in raw_junction_pts]
    junction_pts = []
    for pt in raw_junction_pts:
        ok, _ = _validate_true_junction(pt, skeleton, wall_width_map, local_width=local_width)
        if ok:
            junction_pts.append(pt)
    junction_pts = _dedupe_points_by_radius(junction_pts, radius=int(np.clip(local_width * 2.2 + 6, 10, 24)))

    endpoints = []
    max_probe = int(np.clip(max(w, h) + local_width * 5, 24, 120))
    min_backbone = int(np.clip(local_width * 2.2 + 6, 9, 24))
    for pt in endpoint_pts:
        vx, vy = _estimate_node_direction(pt, skeleton, region_mask)
        ok_region, region_info = _endpoint_points_into_region(pt, (vx, vy), region_mask, max_probe=max_probe)
        if not ok_region:
            continue
        ok_backbone, backbone_info = _endpoint_has_backbone(pt, (vx, vy), skeleton, min_len=min_backbone, tolerance=2)
        if not ok_backbone:
            continue
        width = _sample_wall_width(pt, wall_width_map, fallback=local_width, radius=3)
        endpoints.append({'pt': pt, 'dir': (vx, vy), 'width': width, 'paired': False, 'region_hit': region_info, 'backbone': backbone_info})

    junctions = [{'pt': pt, 'width': _sample_wall_width(pt, wall_width_map, fallback=local_width, radius=5)} for pt in junction_pts]
    return endpoints, junctions, skeleton

def _line_mask(shape, p1, p2, thickness=1):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, int(max(1, thickness)))
    return mask


def _line_region_ratio(region_mask, p1, p2, thickness=1):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    pad = max(2, int(thickness) + 1)
    H, W = region_mask.shape[:2]
    x1, y1, x2, y2 = _clip_box(min(p1[0], p2[0]) - pad, min(p1[1], p2[1]) - pad,
                                max(p1[0], p2[0]) + pad + 1, max(p1[1], p2[1]) + pad + 1, W, H)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    lm = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.line(lm, (p1[0] - x1, p1[1] - y1), (p2[0] - x1, p2[1] - y1), 255, max(1, int(thickness)))
    total = cv2.countNonZero(lm)
    if total == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(lm, region_mask[y1:y2, x1:x2])) / float(total)


def _pair_cost(ep1, ep2, region_mask, relaxed=False, region_bbox=None):
    p1, p2 = ep1['pt'], ep2['pt']
    d1, d2 = ep1['dir'], ep2['dir']
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(vx, vy)
    if dist < 3:
        return None

    # 避免跨過過大的 repair region；大型修補會交給局部方向延伸而不是硬配。
    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        bbox = cv2.boundingRect(pts) if pts is not None else None
    else:
        bbox = tuple(map(int, region_bbox))
    if bbox is not None:
        _, _, w, h = bbox
        max_reasonable = max(80.0, math.hypot(w, h) * 1.5)
        if dist > max_reasonable:
            return None

    ux, uy = vx / dist, vy / dist
    dot1 = d1[0] * ux + d1[1] * uy
    dot2 = d2[0] * (-ux) + d2[1] * (-uy)
    angle_gate = 75 if relaxed else 65
    if dot1 < math.cos(math.radians(angle_gate)) or dot2 < math.cos(math.radians(angle_gate)):
        return None

    # 線段必須主要穿過修補區附近，避免直接把兩條無關牆線接起來。
    thick = int(np.clip(round((ep1.get('width', 4) + ep2.get('width', 4)) * 0.35), 1, 5))
    ratio = _line_region_ratio(cv2.dilate(region_mask, np.ones((5, 5), np.uint8)), p1, p2, thickness=max(1, thick))
    ratio_gate = 0.08 if (relaxed and dist < 90) else (0.14 if relaxed else 0.20)
    if ratio < ratio_gate:
        return None

    # 角度共線、牆寬一致、距離與橫向偏移。
    col_cost = (1.0 - dot1) + (1.0 - dot2)
    w1, w2 = max(ep1.get('width', 4.0), 1.0), max(ep2.get('width', 4.0), 1.0)
    width_cost = abs(w1 - w2) / max((w1 + w2) * 0.5, 1.0)
    lateral1 = abs(d1[0] * vy - d1[1] * vx)
    lateral2 = abs(d2[0] * (-vy) - d2[1] * (-vx))
    lateral_cost = max(lateral1, lateral2) / max(dist, 1.0)

    # 成本越低越好；後續轉為 profit 做全域 matching。
    return dist * 0.45 + col_cost * 90.0 + width_cost * 45.0 + lateral_cost * 35.0


def _global_match_endpoints(endpoints, region_mask, relaxed=False, region_bbox=None):
    """候選端點圖做全域最大收益匹配，取代局部 greedy。"""
    G = nx.Graph()
    for i in range(len(endpoints)):
        G.add_node(i)
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            cost = _pair_cost(endpoints[i], endpoints[j], region_mask, relaxed=relaxed, region_bbox=region_bbox)
            if cost is None:
                continue
            profit = 300.0 - cost
            if profit > 0:
                G.add_edge(i, j, weight=profit, cost=cost)
    if G.number_of_edges() == 0:
        return []
    pairs = list(nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight='weight'))
    pairs = [(int(a), int(b), float(G.edges[a, b]['cost'])) for a, b in pairs]
    pairs.sort(key=lambda x: x[2])
    return pairs


def _estimate_dominant_orientations(walls, region_mask, region_bbox=None):
    H, W = walls.shape[:2]
    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        if pts is None:
            return []
        x, y, w, h = cv2.boundingRect(pts)
    else:
        x, y, w, h = map(int, region_bbox)
    pad = 50
    rx1, ry1, rx2, ry2 = _clip_box(x - pad, y - pad, x + w + pad, y + h + pad, W, H)
    local_region = region_mask[ry1:ry2, rx1:rx2]
    context = cv2.dilate(local_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
    local_walls = cv2.bitwise_and(walls[ry1:ry2, rx1:rx2], context)
    edges = cv2.Canny(local_walls, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=18, maxLineGap=6)
    bins = {}
    if lines is not None:
        # OpenCV builds return either (N, 1, 4) or (N, 4).  Normalising here
        # keeps the wall-repair stage portable without changing its geometry.
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 8:
                continue
            key = int(round(math.degrees(math.atan2(dy, dx) % math.pi) / 10.0) * 10) % 180
            bins[key] = bins.get(key, 0.0) + length
    return [(math.cos(math.radians(deg)), math.sin(math.radians(deg))) for deg, _ in sorted(bins.items(), key=lambda kv: kv[1], reverse=True)[:6]]


def _trace_endpoint_to_wall(walls, region_mask, endpoint, orientations, max_len=160):
    """
    未配對端點用局部主方向場延伸，而不是單一射線 + 90 度吸附。
    回傳最佳 hit point 或 None。
    """
    H, W = walls.shape[:2]
    pt = endpoint['pt']
    ex, ey = endpoint['dir']

    candidate_dirs = []
    # endpoint 方向永遠保留。
    candidate_dirs.append((ex, ey, 0.0))
    for ox, oy in orientations:
        # orientation 沒有方向性，選擇與 endpoint 朝向同側的符號。
        if ox * ex + oy * ey < 0:
            ox, oy = -ox, -oy
        dot = ox * ex + oy * ey
        if dot >= math.cos(math.radians(55)):
            angle_penalty = 1.0 - dot
            candidate_dirs.append((ox, oy, angle_penalty))

    best = None
    for vx, vy, angle_penalty in candidate_dirs:
        for r in range(6, max_len, 2):
            x = int(round(pt[0] + vx * r))
            y = int(round(pt[1] + vy * r))
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            if walls[y, x] > 0 and region_mask[y, x] == 0:
                # 5px 線寬提供與舊版「先將 region 膨脹 2px」相同的容忍度，
                # 但 _line_region_ratio 只處理線段 ROI，不必為每個 endpoint 掃全圖。
                ratio = _line_region_ratio(region_mask, pt, (x, y), thickness=5)
                if ratio < 0.12:
                    continue
                cost = r + angle_penalty * 80.0
                if best is None or cost < best[0]:
                    best = (cost, (x, y))
                break
    return None if best is None else best[1]


def _endpoint_count(mask):
    sk = _skeletonize_uint8(mask)
    deg = _skeleton_degree(sk)
    return int(np.sum((sk > 0) & (deg == 1)))


def _free_component_count(mask):
    free = (mask == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    # 忽略極小雜訊洞。
    return sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 20)


def _topology_accepts(before, after, region_mask, relaxed=False, region_bbox=None):
    H, W = before.shape[:2]
    if region_bbox is None:
        pts = cv2.findNonZero((region_mask > 0).astype(np.uint8))
        if pts is None:
            return True, {'reason': 'no_contour'}
        x, y, w, h = cv2.boundingRect(pts)
    else:
        x, y, w, h = map(int, region_bbox)
    pad = 60
    x1, y1, x2, y2 = _clip_box(x - pad, y - pad, x + w + pad, y + h + pad, W, H)
    b = before[y1:y2, x1:x2]
    a = after[y1:y2, x1:x2]
    ep_b, ep_a = _endpoint_count(b), _endpoint_count(a)
    fc_b, fc_a = _free_component_count(b), _free_component_count(a)
    ok = (ep_a <= ep_b + (1 if relaxed else 0)) and (fc_a <= fc_b + (6 if relaxed else 4))
    return ok, {'endpoint_before': ep_b, 'endpoint_after': ep_a, 'free_cc_before': fc_b, 'free_cc_after': fc_a}


def extract_walls_with_repair_legacy(image_path, output_dir, ocr_data, bg_mask=None, yolo_boxes=None):
    img = safe_imread(image_path)
    if img is None:
        return None
    H, W = img.shape[:2]

    print("[系統] 正在提取牆體並進行牆線保護式精確文字/圖示刪除 V5...")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_mean = np.mean(hsv[:, :, 1])

    if s_mean < 15:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 6)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        enhanced_color = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        filtered = cv2.bilateralFilter(enhanced_color, 9, 75, 75)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 6)
        edges = cv2.Canny(filtered, 50, 150)
        binary = cv2.bitwise_or(binary, edges)

    raw_wall_binary = binary.copy()

    # V4 修正：two-stage repair 的 evidence gate 需要一份「清掉文字/icon 之前」的牆線先驗。
    # 這份 prior 用來判斷 repair region 裡面是否真的曾經有牆線證據；
    # 若沒有定義，後面的 _region_wall_damage_evidence() 會發生 NameError。
    pre_wall_prior = _build_wall_prior(raw_wall_binary)
    safe_imwrite(str(output_dir / "debug_pre_wall_prior_0721_4.jpg"), pre_wall_prior)

    print("[系統] 建立 V5 精確文字筆畫遮罩：polygon 只作 ROI，不再整塊挖掉...")
    text_poly_mask, text_stroke_mask, text_repair_seed, text_preserve_mask = _build_precise_text_stroke_masks(ocr_data, (H, W), raw_wall_binary)
    safe_imwrite(str(output_dir / "debug_text_polygon_mask_0721_4.jpg"), text_poly_mask)
    safe_imwrite(str(output_dir / "debug_text_precise_strokes_0721_4.jpg"), text_stroke_mask)
    safe_imwrite(str(output_dir / "debug_text_preserved_wall_like_0721_4.jpg"), text_preserve_mask)
    safe_imwrite(str(output_dir / "debug_text_repair_seed_0721_4.jpg"), text_repair_seed)

    print("[系統] 執行雜訊抹除 (precise OCR strokes / precise YOLO strokes / 幾何複雜度分析)...")
    noise_boxes = []
    repair_seed_mask = text_repair_seed.copy()

    # V5：只刪 OCR polygon 內實際的文字前景筆畫；牆線-like component 會保留。
    binary[text_stroke_mask > 0] = 0

    if yolo_boxes:
        yolo_roi_mask, yolo_stroke_mask, yolo_repair_seed, yolo_preserve_mask = _build_precise_box_stroke_masks(yolo_boxes, (H, W), raw_wall_binary, pad=2)
        safe_imwrite(str(output_dir / "debug_yolo_roi_mask_0721_4.jpg"), yolo_roi_mask)
        safe_imwrite(str(output_dir / "debug_yolo_precise_strokes_0721_4.jpg"), yolo_stroke_mask)
        safe_imwrite(str(output_dir / "debug_yolo_preserved_wall_like_0721_4.jpg"), yolo_preserve_mask)
        binary[yolo_stroke_mask > 0] = 0
        repair_seed_mask = cv2.bitwise_or(repair_seed_mask, yolo_repair_seed)
        noise_boxes.extend(yolo_boxes)

    # 小型複雜元件：只在元件 bbox 內分析與清除，避免每個元件都掃描整張圖。
    num_labels_icon, labels_icon, stats_icon, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels_icon):
        x, y, w, h, area = map(int, stats_icon[i])
        if not (w < 150 and h < 150 and area > 10):
            continue
        aspect_ratio = max(w, h) / float(max(min(w, h), 1))
        if aspect_ratio >= 5.0:
            continue
        local_labels = labels_icon[y:y+h, x:x+w]
        comp_local = (local_labels == i).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(comp_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        internal_holes = sum(1 for h_info in hierarchy[0] if h_info[3] != -1) if hierarchy is not None else 0
        extent = area / float(max(w * h, 1))
        complexity = 0.0
        if contours:
            perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
            complexity = (perimeter * perimeter) / float(max(area, 1))
        delete_flag = False
        needs_repair = False
        if area < 40 or w < 10 or h < 10:
            delete_flag = True
        elif internal_holes >= 2 and area > 60:
            delete_flag = needs_repair = True
        elif complexity > 60 and area > 60:
            delete_flag = needs_repair = True
        elif aspect_ratio < 1.5 and extent < 0.4 and internal_holes == 1:
            delete_flag = needs_repair = True
        elif 15 <= w <= 50 and 15 <= h <= 50 and aspect_ratio < 1.5 and extent > 0.5:
            delete_flag = needs_repair = True
        if delete_flag:
            binary_roi = binary[y:y+h, x:x+w]
            binary_roi[comp_local > 0] = 0
            if needs_repair:
                dil = cv2.dilate(comp_local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                seed_roi = repair_seed_mask[y:y+h, x:x+w]
                seed_roi[dil > 0] = 255
                noise_boxes.append((x, y, x + w, y + h))

    if bg_mask is not None:
        if bg_mask.shape != binary.shape:
            bg_mask = cv2.resize(bg_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        fg_mask = cv2.bitwise_not(bg_mask)
        num_labels_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        if num_labels_fg > 1:
            largest_label = 1 + np.argmax(fg_stats[1:, cv2.CC_STAT_AREA])
            clean_fg = (fg_labels == largest_label).astype(np.uint8) * 255
        else:
            clean_fg = fg_mask

        clean_fg = cv2.morphologyEx(clean_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours_bg, _ = cv2.findContours(clean_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary, contours_bg, -1, 255, 4)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = (((stats[:, cv2.CC_STAT_WIDTH] > 35) | (stats[:, cv2.CC_STAT_HEIGHT] > 35)) &
            (stats[:, cv2.CC_STAT_AREA] > 60))
    keep[0] = False
    final_walls = (keep[labels].astype(np.uint8) * 255)

    print("[系統] 執行牆體加粗與微型縫合...")
    final_walls = cv2.dilate(final_walls, np.ones((3, 3), np.uint8), iterations=1)
    final_walls = cv2.morphologyEx(final_walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    safe_imwrite(str(output_dir / "debug_walls_before_repair_0721_4.jpg"), final_walls)

    wall_width_map = _estimate_wall_width_map(final_walls)
    debug_repair_img = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)
    debug_nodes_img = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)

    # V5 repair region：只來自「精確刪除筆畫且靠近牆線」的小 seed，不再由整個 OCR/YOLO 框觸發。
    repair_region_mask = cv2.morphologyEx(repair_seed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    safe_imwrite(str(output_dir / "debug_repair_regions_0721_4.jpg"), repair_region_mask)

    num_regions, region_labels, region_stats, _ = cv2.connectedComponentsWithStats((repair_region_mask > 0).astype(np.uint8), connectivity=8)
    repair_report = []

    print("[系統] 以 evidence-gated two-stage repair 修補牆體...")

    def run_repair_stage(stage_name, walls_in, relaxed=False, allow_prior_restore=False):
        walls_out = walls_in.copy()
        stage_report = []
        stage_width_map = _estimate_wall_width_map(walls_out)
        stage_skeleton = _skeletonize_uint8(walls_out)
        stage_degree = _skeleton_degree(stage_skeleton)
        skeleton_refresh_batch = max(
            1, int(os.environ.get("MAP_WALL_REPAIR_SKELETON_BATCH", "8"))
        )
        accepted_since_refresh = 0
        dirty_bboxes = []
        # 重用同一塊 buffer，避免每個 OCR/icon repair region 都配置一張全圖 mask。
        region_mask = np.zeros_like(repair_region_mask)
        previous_bbox = None

        def bbox_gap(a, b):
            ax, ay, aw, ah = map(int, a)
            bx, by, bw, bh = map(int, b)
            dx = max(0, bx - (ax + aw), ax - (bx + bw))
            dy = max(0, by - (ay + ah), ay - (by + bh))
            return math.hypot(dx, dy)

        def refresh_stage_skeleton():
            nonlocal stage_skeleton, stage_degree, accepted_since_refresh, dirty_bboxes
            stage_skeleton = _skeletonize_uint8(walls_out)
            stage_degree = _skeleton_degree(stage_skeleton)
            accepted_since_refresh = 0
            dirty_bboxes = []

        for ridx in range(1, num_regions):
            area = int(region_stats[ridx, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            if area > H * W * 0.08:
                stage_report.append({'stage': stage_name, 'region': ridx, 'accepted': False, 'reason': 'region_too_large', 'area': area})
                continue

            x, y, rw, rh = map(int, region_stats[ridx, :4])
            region_bbox = (x, y, rw, rh)
            if previous_bbox is not None:
                px, py, pw, ph = previous_bbox
                region_mask[py:py+ph, px:px+pw] = 0
            local_labels = region_labels[y:y+rh, x:x+rw]
            region_mask[y:y+rh, x:x+rw] = (local_labels == ridx).astype(np.uint8) * 255
            previous_bbox = region_bbox
            eligible, evidence = _region_wall_damage_evidence(
                region_mask, pre_wall_prior, walls_out, stage_width_map,
                stage=stage_name, region_bbox=region_bbox
            )
            if not eligible:
                stage_report.append({'stage': stage_name, 'region': ridx, 'accepted': False, 'area': area, 'reason': evidence.get('reason'), 'evidence': evidence})
                continue

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_repair_img, contours, -1, (0, 255, 255) if stage_name == 'outer' else (255, 255, 0), 1)

            # 已接受的遠端 repair 不會改變目前 ROI 的 skeleton。累積到一批，或
            # 新 region 與 dirty repair 幾乎相鄰時才重算全圖 skeleton。
            if (
                accepted_since_refresh >= skeleton_refresh_batch
                or any(bbox_gap(region_bbox, dirty) <= 12.0 for dirty in dirty_bboxes)
            ):
                refresh_stage_skeleton()

            before = walls_out
            candidate_walls = walls_out.copy()
            endpoints, junctions, skeleton = _find_skeleton_nodes_near_region(
                candidate_walls, region_mask, stage_width_map, stage=stage_name,
                skeleton=stage_skeleton, degree_map=stage_degree, region_bbox=region_bbox
            )

            for ep in endpoints:
                cv2.circle(debug_nodes_img, ep['pt'], 3, (0, 0, 255), -1)
                dx, dy = ep['dir']
                cv2.line(debug_nodes_img, ep['pt'], (int(ep['pt'][0] + dx * 16), int(ep['pt'][1] + dy * 16)), (0, 0, 255), 1)
            for jn in junctions:
                cv2.circle(debug_nodes_img, jn['pt'], 4, (255, 0, 255), -1)

            pairs = _global_match_endpoints(endpoints, region_mask, relaxed=relaxed, region_bbox=region_bbox)
            accepted_pairs = []
            for i, j, cost in pairs:
                p1, p2 = endpoints[i]['pt'], endpoints[j]['pt']
                thick = int(np.clip(round((endpoints[i]['width'] + endpoints[j]['width']) * 0.35), 2, 6))
                cv2.line(candidate_walls, p1, p2, 255, thick)
                endpoints[i]['paired'] = True
                endpoints[j]['paired'] = True
                accepted_pairs.append((i, j, round(cost, 2)))

            orientations = _estimate_dominant_orientations(before, region_mask, region_bbox=region_bbox)
            extensions = []
            for ep_idx, ep in enumerate(endpoints):
                if ep.get('paired'):
                    continue

                best_junction = None
                best_j_cost = float('inf')
                for jn in junctions:
                    vx, vy = jn['pt'][0] - ep['pt'][0], jn['pt'][1] - ep['pt'][1]
                    dist = math.hypot(vx, vy)
                    if dist < 5 or dist > (150 if relaxed else 120):
                        continue
                    ux, uy = vx / dist, vy / dist
                    align = ep['dir'][0] * ux + ep['dir'][1] * uy
                    if align < math.cos(math.radians(68 if relaxed else 58)):
                        continue
                    if _line_region_ratio(region_mask, ep['pt'], jn['pt'], thickness=5) < (0.08 if relaxed else 0.14):
                        continue
                    cost = dist + (1 - align) * 60
                    if cost < best_j_cost:
                        best_j_cost = cost
                        best_junction = jn['pt']

                hit_pt = best_junction
                if hit_pt is None:
                    hit_pt = _trace_endpoint_to_wall(candidate_walls, region_mask, ep, orientations, max_len=(190 if relaxed else 150))

                if hit_pt is not None:
                    thick = int(np.clip(round(ep.get('width', 4.0) * 0.45), 2, 6))
                    cv2.line(candidate_walls, ep['pt'], hit_pt, 255, thick)
                    ep['paired'] = True
                    extensions.append((ep_idx, hit_pt))

            prior_restores = []
            if allow_prior_restore:
                prior_restores = _restore_prior_line_segments(candidate_walls, region_mask, pre_wall_prior, stage_width_map, region_bbox=region_bbox)

            ok, topo = _topology_accepts(before, candidate_walls, region_mask, relaxed=relaxed, region_bbox=region_bbox)
            has_work = bool(accepted_pairs or extensions or prior_restores)
            if ok and has_work:
                for i, j, cost in accepted_pairs:
                    p1, p2 = endpoints[i]['pt'], endpoints[j]['pt']
                    cv2.line(debug_repair_img, p1, p2, (0, 0, 255), 2)
                for ep_idx, hit_pt in extensions:
                    cv2.line(debug_repair_img, endpoints[ep_idx]['pt'], hit_pt, (100, 255, 100), 2)
                for p1, p2 in prior_restores:
                    cv2.line(debug_repair_img, p1, p2, (255, 180, 0), 1)
                walls_out = candidate_walls
                accepted_since_refresh += 1
                dirty_bboxes.append(region_bbox)
                stage_report.append({
                    'stage': stage_name,
                    'region': ridx,
                    'accepted': True,
                    'area': area,
                    'evidence': evidence,
                    'endpoints': len(endpoints),
                    'junctions': len(junctions),
                    'pairs': accepted_pairs,
                    'extensions': len(extensions),
                    'prior_restores': len(prior_restores),
                    'topology': topo
                })
            else:
                cv2.drawContours(debug_repair_img, contours, -1, (0, 165, 255), 2)
                stage_report.append({
                    'stage': stage_name,
                    'region': ridx,
                    'accepted': False,
                    'area': area,
                    'evidence': evidence,
                    'endpoints': len(endpoints),
                    'junctions': len(junctions),
                    'pairs': accepted_pairs,
                    'extensions': len(extensions),
                    'prior_restores': len(prior_restores),
                    'topology': topo,
                    'reason': 'topology_rejected_or_no_work' if not ok else 'no_candidate_work'
                })
        return walls_out, stage_report

    # 第一階段：外層/主要缺口修補。門檻較寬鬆，但 endpoint 必須通過 damage evidence 與 contact-band 過濾。
    final_walls, outer_report = run_repair_stage('outer', final_walls, relaxed=True, allow_prior_restore=False)
    final_walls = cv2.morphologyEx(final_walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 第二階段：在外層修好後重新估牆寬與 skeleton，修內部短牆段；允許使用 pre-clean wall prior 作為導引。
    final_walls, inner_report = run_repair_stage('inner', final_walls, relaxed=False, allow_prior_restore=True)
    repair_report.extend(outer_report)
    repair_report.extend(inner_report)

    final_walls = cv2.morphologyEx(final_walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # V3 房間分隔補強：舊牆體仍是主體，只把經過文字排除、亮線/色差與牆體接續
    # 三重 gate 的長直線加入。這能處理「淡色房間 + 細白分隔線」的商場地圖。
    safe_imwrite(
        str(output_dir / "debug_cleaned_walls_before_room_partition_v2.jpg"),
        final_walls,
    )
    separator_mask, partition_report = _extract_structural_room_separators(
        image_path=image_path,
        base_walls=final_walls,
        ocr_data=ocr_data,
        yolo_boxes=yolo_boxes,
        bg_mask=bg_mask,
    )
    partition_accepted, partition_candidate, topology_guard = _accept_partition_candidate(
        final_walls, separator_mask, partition_report
    )
    partition_report["topology_guard"] = topology_guard
    if partition_accepted:
        final_walls = partition_candidate
        final_walls = cv2.morphologyEx(
            final_walls,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        partition_report["accepted_by_global_guard"] = True
    else:
        partition_report["accepted_by_global_guard"] = False
        partition_report["guard_reason"] = ",".join(topology_guard.get("reasons", []))

    partition_overlay = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)
    if partition_accepted:
        partition_overlay[separator_mask > 0] = (0, 0, 255)
    else:
        # Orange means the appearance candidate was inspected but rolled back.
        partition_overlay[separator_mask > 0] = (0, 165, 255)
    safe_imwrite(str(output_dir / "debug_room_partition_evidence_v2.jpg"), partition_overlay)
    safe_imwrite(str(output_dir / "debug_room_separator_mask_v2.jpg"), separator_mask)
    with open(output_dir / "room_partition_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(partition_report, f, ensure_ascii=False, indent=2)

    safe_imwrite(str(output_dir / "debug_cleaned_walls_0721_4.jpg"), final_walls)
    safe_imwrite(str(output_dir / "debug_repair_boxes_0721_4.jpg"), debug_repair_img)
    safe_imwrite(str(output_dir / "debug_wall_skeleton_nodes_0721_4.jpg"), debug_nodes_img)
    with open(output_dir / "wall_repair_report_0721_4.json", 'w', encoding='utf-8') as f:
        json.dump(repair_report, f, ensure_ascii=False, indent=4)

    return (final_walls / 255).astype(np.uint8)

# =========================================
# FAST V5：多可通行公共空間恢復（免訓練 / geometry + topology）
# =========================================
# ---- Wall evidence V6 (self-contained, no extra local module required) ----
"""Evidence-led wall extraction. OpenCV + NumPy; optional scikit-image skeleton.
Coordinates are original-image pixels. OCR accepts poly or box; icons xyxy.
No model inference is performed here. Output wall mask uses 0/1, not 0/255.
"""
import cv2
import numpy as np
import time
import json
import os
from pathlib import Path


def _v6_skeleton(mask):
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(mask)
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0).astype(np.uint8) * 255
    except ImportError:
        # Portable morphological skeleton, also used by the original project.
        result = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        work = mask.copy()
        while cv2.countNonZero(work):
            eroded = cv2.erode(work, element)
            result |= cv2.subtract(work, cv2.dilate(eroded, element))
            work = eroded
        return result


def _v6_clean_skeleton(walls, unit):
    # Suppress narrow enclosed edge-pair loops only in the skeleton source.
    # Physical wall mask and open doorway connectivity are not thickened.
    source=walls.copy()
    contours,hierarchy=cv2.findContours(source,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i,c in enumerate(contours):
            if hierarchy[0,i,3]<0: continue
            x,y,w,h=cv2.boundingRect(c)
            if min(w,h)<=max(4,round(8*unit)) and cv2.contourArea(c)<=max(32,round(256*unit*unit)):
                cv2.drawContours(source,[c],-1,255,-1)
    skel=_v6_skeleton(source)
    n,labels,stats,_=cv2.connectedComponentsWithStats(skel,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=max(5,round(12*unit));keep[0]=False
    skel=(keep[labels].astype(np.uint8)*255)
    degree=cv2.filter2D((skel>0).astype(np.uint8),cv2.CV_16S,np.ones((3,3),np.int16))-(skel>0)
    endpoints=np.column_stack(np.where((skel>0)&(degree==1)))
    remove=[];limit=max(3,round(5*unit));height,width=skel.shape
    for y,x in endpoints:
        path=[(int(y),int(x))];prev=None;current=path[0]
        for _ in range(limit):
            cy,cx=current
            neighbors=[(ny,nx) for ny in range(max(0,cy-1),min(height,cy+2))
                       for nx in range(max(0,cx-1),min(width,cx+2))
                       if (ny,nx)!=current and (ny,nx)!=prev and skel[ny,nx]>0]
            if len(neighbors)!=1:break
            nxt=neighbors[0]
            if degree[nxt]>=3:
                remove.extend(path);break
            if degree[nxt]!=2:break
            path.append(nxt);prev,current=current,nxt
    for y,x in remove:skel[y,x]=0
    return skel


def _v6_roi_mask(shape, ocr_data, yolo_boxes, sx, sy):
    mask = np.zeros(shape, np.uint8)
    for item in ocr_data or []:
        if item.get('poly') is not None:
            poly = np.asarray(item['poly'], np.float64)
        elif item.get('box') is not None:
            x1, y1, x2, y2 = item['box']
            poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float64)
        else:
            continue
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            continue
        poly = np.rint(poly * [sx, sy]).astype(np.int32)
        poly[:,0] = np.clip(poly[:,0], 0, shape[1]-1)
        poly[:,1] = np.clip(poly[:,1], 0, shape[0]-1)
        cv2.fillPoly(mask, [poly], 255)
    for box in yolo_boxes or []:
        x1,y1,x2,y2 = np.rint(np.asarray(box) * [sx,sy,sx,sy]).astype(int)
        x1,x2 = np.clip([x1,x2], 0, shape[1]-1)
        y1,y2 = np.clip([y1,y2], 0, shape[0]-1)
        if x2>x1 and y2>y1:
            cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
    return cv2.dilate(mask, np.ones((3,3),np.uint8))


def _v6_runs(values):
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def extract_wall_evidence_v6(image, ocr_data=None, yolo_boxes=None, max_dim=2048, budget_seconds=18.0):
    """Return (uint8 0/1 walls, debug masks, report).

    Budget bounds optional repair, not a hard real-time guarantee. Image I/O,
    OCR/YOLO inference and caller debug encoding are outside this core timer.
    No filename, floor-specific coordinates or map-specific thresholds.
    """
    started = time.perf_counter()
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a nonempty BGR image')
    H,W = image.shape[:2]
    if min(H,W) < 8:
        raise ValueError('Image too small for structural extraction')
    scale = min(1.0, float(max_dim) / max(H,W))
    w,h = max(8,round(W*scale)),max(8,round(H*scale))
    work = cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()
    roi = _v6_roi_mask((h,w),ocr_data,yolo_boxes,w/W,h/H)
    unit = max(h,w)/2048.0
    min_line = max(18, int(round(30*unit)))
    # Smooth JPEG/camera noise without erasing single-pixel boundaries.
    lab = cv2.cvtColor(cv2.GaussianBlur(work,(3,3),0.55),cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    luminosity=lab[:,:,0]
    local_mean=cv2.GaussianBlur(luminosity,(0,0),3.0)
    local_contrast=np.clip(128+4*(luminosity.astype(np.float32)-local_mean.astype(np.float32)),0,255).astype(np.uint8)
    channels = [luminosity,local_contrast]
    for channel in [lab[:,:,1],lab[:,:,2]]:
        lo,hi = np.percentile(channel,[2,98])
        if hi-lo >= 6:
            channels.append(np.clip((channel.astype(np.float32)-lo)*min(3.0,180/(hi-lo)),0,255).astype(np.uint8))
    # Both dark ink and bright separators contribute, without a global saturation switch.
    evidence = np.zeros((h,w),np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segments = []
    for ci,channel in enumerate(channels):
        gradient = cv2.morphologyEx(channel,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        if ci != 1:  # Local contrast is a proposal channel, never pixel evidence.
            evidence |= ((gradient >= (3 if ci == 0 else 7)).astype(np.uint8)*255)
        lines = detector.detect(channel)[0]
        if lines is None:
            continue
        for x1,y1,x2,y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2-x1,y2-y1))
            if length >= min_line:
                segments.append((length,float(x1),float(y1),float(x2),float(y2)))
    report = {'version':'wall_v6','original_size':[W,H],'working_size':[w,h],
              'ocr_regions':len(ocr_data or []),'icon_regions':len(yolo_boxes or []),
              'candidate_segments':len(segments),'budget_seconds':budget_seconds,
              'repair_budget_exhausted':False}
    # Long geometry seeds select nearby evidence; unconnected text strokes never grow globally.
    structural = np.zeros_like(evidence)
    anchors = []
    through_roi = []
    supported_pixels=cv2.dilate(evidence,np.ones((3,3),np.uint8))
    for length,x1,y1,x2,y2 in sorted(segments,reverse=True)[:12000]:
        n = max(2,int(length)+1)
        xs = np.clip(np.rint(np.linspace(x1,x2,n)).astype(int),0,w-1)
        ys = np.clip(np.rint(np.linspace(y1,y2,n)).astype(int),0,h-1)
        outside = roi[ys,xs] == 0
        if np.mean(supported_pixels[ys,xs]>0)<0.8:
            continue
        # OCR-contained text baselines/icons cannot seed walls. Crossing lines
        # need real visible continuation on both sides of each masked run.
        if outside.mean() < 0.55 or np.count_nonzero(outside) < min_line:
            continue
        line = np.column_stack([xs,ys])
        margin = max(8,min_line//3)
        for a,b in _v6_runs(~outside):
            if a>=margin and b+margin<=n and outside[a-margin:a].all() and outside[b:b+margin].all():
                through_roi.append((line[a-1].copy(),line[b].copy()))
        for a,b in _v6_runs(outside):
            if b-a >= max(7,min_line//3):
                cv2.line(structural,tuple(line[a]),tuple(line[b-1]),255,2)
                p0=line[a].astype(float);p1=line[b-1].astype(float)
                visible_length=float(np.linalg.norm(p1-p0))
                if visible_length>=min_line:
                    anchors.append((visible_length,p0,p1))
    # Pixel evidence maintains actual shape; dilation is only a search band,
    # not global thickening of the returned walls.
    band = cv2.dilate(structural,np.ones((5,5),np.uint8))
    observed = cv2.bitwise_and(evidence,band)
    observed[roi>0] = 0
    observed |= structural
    # Fill one-pixel seams, but do not close genuine multi-pixel door openings.
    # No blanket close: even narrow real openings stay open.
    repair = np.zeros_like(observed)
    near_evidence = cv2.dilate(evidence,np.ones((3,3),np.uint8))
    # Spatial endpoint buckets avoid quadratic all-pairs matching.
    max_gap = max(32,round(64*unit))
    if cv2.countNonZero(roi):
        _,_,roi_stats,_=cv2.connectedComponentsWithStats(roi)
        sizes=np.max(roi_stats[1:,:2+2][:,2:4],axis=1)
        if len(sizes): max_gap=max(max_gap,min(160,int(np.percentile(sizes,90))+8))
    bucket = {}
    ends = []
    for length,p,q in anchors:
        direction = (q-p)/length
        for point,outward in [(p,-direction),(q,direction)]:
            idx=len(ends); ends.append((point,outward,length))
            key=tuple(np.floor(point/max_gap).astype(int))
            bucket.setdefault(key,[]).append(idx)
    accepted=[]; seen=set()
    for p,q in through_roi:
        cv2.line(repair,tuple(p),tuple(q),255,2)
        accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                         'end':(q/[w/W,h/H]).round(2).tolist(),'kind':'crossing_roi'})
    for i,(p,dp,lp) in enumerate(ends):
        if time.perf_counter()-started > budget_seconds:
            report['repair_budget_exhausted']=True; break
        cell=np.floor(p/max_gap).astype(int)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in bucket.get((cell[0]+dx,cell[1]+dy),[]):
                    if j<=i: continue
                    q,dq,lq=ends[j]; v=q-p; gap=float(np.linalg.norm(v))
                    if gap<2 or gap>max_gap: continue
                    direction=v/gap
                    if np.dot(dp,direction)<0.985 or np.dot(dq,-direction)<0.985: continue
                    if gap>min(lp,lq)*0.9: continue
                    n=max(3,int(gap)+1)
                    xs=np.clip(np.rint(np.linspace(p[0],q[0],n)).astype(int),0,w-1)
                    ys=np.clip(np.rint(np.linspace(p[1],q[1],n)).astype(int),0,h-1)
                    masked=roi[ys,xs]>0
                    support=near_evidence[ys,xs]>0
                    missing=observed[ys,xs]==0
                    if missing.mean()<0.15: continue
                    # Evidence outside OCR must be continuous enough; a blank
                    # unmasked doorway is never treated as an occlusion.
                    supported=(support|masked)[1:-1]
                    uncovered=max((b-a for a,b in _v6_runs(~supported)),default=0)
                    kind='occlusion' if masked.mean()>=0.15 else 'faint_line'
                    if supported.mean()<0.88 or uncovered>max(2,round(3*unit)): continue
                    if kind=='faint_line' and support[1:-1].mean()<0.92: continue
                    key=tuple(np.rint(np.r_[p,q]/3).astype(int))
                    if key in seen: continue
                    seen.add(key)
                    cv2.line(repair,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),255,2)
                    accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                                     'end':(q/[w/W,h/H]).round(2).tolist(),'kind':kind})
    merged = observed | repair
    # Small attached corners/curves are recovered only in a narrow structural
    # neighborhood. This cannot flood an entire connected text component.
    curve_band=cv2.dilate(merged,np.ones((5,5),np.uint8))
    extra=evidence & curve_band
    extra[roi>0]=0
    merged |= extra
    skeleton = _v6_clean_skeleton(merged,unit)
    report.update({'retained_segments':len(anchors),'repairs':accepted,
                   'repair_count':len(accepted),'core_seconds':round(time.perf_counter()-started,4),
                   'wall_pixels_working':int(np.count_nonzero(merged))})
    masks={'observed':observed,'repairs':repair,'occlusion_mask':roi,'skeleton':skeleton,'walls':merged}
    if (w,h)!=(W,H):
        masks={k:cv2.resize(v,(W,H),interpolation=cv2.INTER_NEAREST) for k,v in masks.items()}
        masks['skeleton']=_v6_clean_skeleton(masks['walls'],max(H,W)/2048.0)
        report['core_seconds']=round(time.perf_counter()-started,4)
    return (masks['walls']>0).astype(np.uint8),masks,report


def _v6_write_image(path, image):
    # OpenCV imwrite is unreliable with Chinese paths on some Windows builds.
    path=Path(path)
    ok,encoded=cv2.imencode(path.suffix,image)
    if not ok: raise IOError('Could not encode image: '+str(path))
    encoded.tofile(str(path))


def extract_walls_with_repair_v6(image_path, output_dir, ocr_data, bg_mask=None, yolo_boxes=None):
    started=time.perf_counter()
    image=cv2.imdecode(np.fromfile(str(image_path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: return None
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    walls,masks,report=extract_wall_evidence_v6(image,ocr_data,yolo_boxes,
        max_dim=int(os.environ.get('MAP_WALL_MAX_DIM','2048')),
        budget_seconds=float(os.environ.get('MAP_WALL_BUDGET_SECONDS','18')))
    # Background masks remain navigation obstacles in the existing caller.
    # Their color-cluster contours are not assumed to be physical walls.
    report['background_policy']='handled_by_existing_pipeline_after_wall_extraction'
    _v6_write_image(output_dir/'debug_cleaned_walls_0721_4.jpg',masks['walls'])
    if os.environ.get('MAP_WALL_DEBUG','1')!='0':
        for name,mask in masks.items():
            _v6_write_image(output_dir/('wall_v6_'+name+'.png'),mask)
        overlay=image.copy()
        overlay[masks['observed']>0]=(0,90,255)
        overlay[masks['repairs']>0]=(255,0,255)
        _v6_write_image(output_dir/'wall_v6_overlay.jpg',overlay)
    report['stage_seconds']=round(time.perf_counter()-started,4)
    report['within_20_seconds']=report['stage_seconds']<=20
    (output_dir/'wall_v6_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[Wall V6] {:.3f}s, {} evidence-gated repairs'.format(report['stage_seconds'],report['repair_count']))
    return walls


"""Evidence-led wall extraction. OpenCV + NumPy; optional scikit-image skeleton.
Coordinates are original-image pixels. OCR accepts poly or box; icons xyxy.
No model inference is performed here. Output wall mask uses 0/1, not 0/255.
"""
import cv2
import numpy as np
import time
import json
import os
from pathlib import Path


def _v7_skeleton(mask):
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(mask)
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0).astype(np.uint8) * 255
    except ImportError:
        # Portable morphological skeleton, also used by the original project.
        result = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        work = mask.copy()
        while cv2.countNonZero(work):
            eroded = cv2.erode(work, element)
            result |= cv2.subtract(work, cv2.dilate(eroded, element))
            work = eroded
        return result


def _v7_barrier_raster(skeleton):
    # 8-connected free-space requires a 4-connected digital barrier. Add only
    # one corner pixel at diagonal steps; do not dilate the whole mask.
    out=skeleton.copy()
    a=skeleton[:-1,:-1]>0;b=skeleton[:-1,1:]>0
    c=skeleton[1:,:-1]>0;d=skeleton[1:,1:]>0
    down=a&d&~b&~c
    up=b&c&~a&~d
    out[:-1,1:][down]=255
    out[:-1,:-1][up]=255
    return out


def _v7_roi_mask(shape, ocr_data, yolo_boxes, sx, sy):
    mask = np.zeros(shape, np.uint8)
    for item in ocr_data or []:
        if item.get('poly') is not None:
            poly = np.asarray(item['poly'], np.float64)
        elif item.get('box') is not None:
            x1, y1, x2, y2 = item['box']
            poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float64)
        else:
            continue
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            continue
        poly = np.rint(poly * [sx, sy]).astype(np.int32)
        poly[:,0] = np.clip(poly[:,0], 0, shape[1]-1)
        poly[:,1] = np.clip(poly[:,1], 0, shape[0]-1)
        cv2.fillPoly(mask, [poly], 255)
    for box in yolo_boxes or []:
        x1,y1,x2,y2 = np.rint(np.asarray(box) * [sx,sy,sx,sy]).astype(int)
        x1,x2 = np.clip([x1,x2], 0, shape[1]-1)
        y1,y2 = np.clip([y1,y2], 0, shape[0]-1)
        if x2>x1 and y2>y1:
            cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
    return cv2.dilate(mask, np.ones((3,3),np.uint8))


def _v7_runs(values):
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def extract_wall_evidence_v7(image, ocr_data=None, yolo_boxes=None, max_dim=2048, budget_seconds=18.0):
    """Return (uint8 0/1 walls, debug masks, report).

    Budget bounds optional repair, not a hard real-time guarantee. Image I/O,
    OCR/YOLO inference and caller debug encoding are outside this core timer.
    No filename, floor-specific coordinates or map-specific thresholds.
    """
    started = time.perf_counter()
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a nonempty BGR image')
    H,W = image.shape[:2]
    if min(H,W) < 8:
        raise ValueError('Image too small for structural extraction')
    scale = min(1.0, float(max_dim) / max(H,W))
    w,h = max(8,round(W*scale)),max(8,round(H*scale))
    work = cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()
    roi = _v7_roi_mask((h,w),ocr_data,yolo_boxes,w/W,h/H)
    unit = max(h,w)/2048.0
    min_line = max(6, int(round(8*unit)))
    long_line = max(18,int(round(30*unit)))
    # Smooth JPEG/camera noise without erasing single-pixel boundaries.
    lab = cv2.cvtColor(cv2.GaussianBlur(work,(3,3),0.55),cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    luminosity=lab[:,:,0]
    local_mean=cv2.GaussianBlur(luminosity,(0,0),3.0)
    local_contrast=np.clip(128+4*(luminosity.astype(np.float32)-local_mean.astype(np.float32)),0,255).astype(np.uint8)
    channels = [luminosity,local_contrast]
    for channel in [lab[:,:,1],lab[:,:,2]]:
        lo,hi = np.percentile(channel,[2,98])
        if hi-lo >= 6:
            channels.append(np.clip((channel.astype(np.float32)-lo)*min(3.0,180/(hi-lo)),0,255).astype(np.uint8))
    proposal_only={1}
    # Both dark ink and bright separators contribute, without a global saturation switch.
    evidence = np.zeros((h,w),np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segments = []
    for ci,channel in enumerate(channels):
        gradient = cv2.morphologyEx(channel,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        if ci not in proposal_only:  # Enhanced channels propose geometry only.
            evidence |= ((gradient >= (3 if ci == 0 else 7)).astype(np.uint8)*255)
        lines = detector.detect(channel)[0]
        if lines is None:
            continue
        for x1,y1,x2,y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2-x1,y2-y1))
            if length >= min_line:
                segments.append((length,float(x1),float(y1),float(x2),float(y2)))
    report = {'version':'wall_v7','original_size':[W,H],'working_size':[w,h],
              'ocr_regions':len(ocr_data or []),'icon_regions':len(yolo_boxes or []),
              'candidate_segments':len(segments),'budget_seconds':budget_seconds,
              'repair_budget_exhausted':False}
    # Long geometry seeds select nearby evidence; unconnected text strokes never grow globally.
    structural = np.zeros_like(evidence)
    gx=cv2.Sobel(luminosity,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(luminosity,cv2.CV_32F,0,1,ksize=3)
    anchors = []
    through_roi = []
    # Short separators must meet a structural line at both ends. A short glyph
    # touching one wall is insufficient; no connected-component area deletion.
    long_guide=np.zeros_like(evidence)
    for length,x1,y1,x2,y2 in segments:
        if length>=long_line:
            cv2.line(long_guide,(round(x1),round(y1)),(round(x2),round(y2)),255,1)
    guide_near=cv2.dilate(long_guide,np.ones((7,7),np.uint8))
    roi_near=cv2.dilate(roi,np.ones((9,9),np.uint8))
    supported_pixels=cv2.dilate(evidence,np.ones((3,3),np.uint8))
    for length,x1,y1,x2,y2 in sorted(segments,reverse=True)[:12000]:
        # Collapse channel-dependent edge offsets onto the raw-image edge.
        # The search band estimates a coordinate; its pixels are not walls.
        normal=np.array([-(y2-y1),x2-x1])/length
        sample_n=max(3,min(48,int(length)))
        tx=np.linspace(x1,x2,sample_n);ty=np.linspace(y1,y2,sample_n)
        offsets=np.arange(-3,4)
        xx=np.clip(np.rint(tx[:,None]+normal[0]*offsets).astype(int),0,w-1)
        yy=np.clip(np.rint(ty[:,None]+normal[1]*offsets).astype(int),0,h-1)
        weights=np.abs(gx[yy,xx]*normal[0]+gy[yy,xx]*normal[1])
        good=(weights.sum(axis=1)>8)&(roi[np.clip(np.rint(ty).astype(int),0,h-1),np.clip(np.rint(tx).astype(int),0,w-1)]==0)
        if np.count_nonzero(good)>=3:
            shifts=(weights[good]*offsets).sum(axis=1)/weights[good].sum(axis=1)
            offset=float(np.median(shifts));x1+=offset*normal[0];x2+=offset*normal[0];y1+=offset*normal[1];y2+=offset*normal[1]
        n = max(2,int(length)+1)
        xs = np.clip(np.rint(np.linspace(x1,x2,n)).astype(int),0,w-1)
        ys = np.clip(np.rint(np.linspace(y1,y2,n)).astype(int),0,h-1)
        outside = roi[ys,xs] == 0
        if np.mean(supported_pixels[ys,xs]>0)<0.8:
            continue
        # OCR-contained text baselines/icons cannot seed walls. Crossing lines
        # need real visible continuation on both sides of each masked run.
        if np.count_nonzero(outside) < max(5,min_line//2):
            continue
        if length<long_line:
            ends_on_guide=bool(guide_near[ys[0],xs[0]]) and bool(guide_near[ys[-1],xs[-1]])
            occlusion_stub=(bool(guide_near[ys[0],xs[0]]) and bool(roi_near[ys[-1],xs[-1]])) or (bool(guide_near[ys[-1],xs[-1]]) and bool(roi_near[ys[0],xs[0]]))
            if not (ends_on_guide or occlusion_stub):continue
        line = np.column_stack([xs,ys])
        margin = max(3,min_line//3)
        for a,b in _v7_runs(~outside):
            if a>=margin and b+margin<=n and outside[a-margin:a].all() and outside[b:b+margin].all():
                through_roi.append((line[a-1].copy(),line[b].copy()))
        visible=outside & (supported_pixels[ys,xs]>0)
        for a,b in _v7_runs(visible):
            if b-a >= max(4,min_line//3):
                cv2.line(structural,tuple(line[a]),tuple(line[b-1]),255,1)
                p0=line[a].astype(float);p1=line[b-1].astype(float)
                visible_length=float(np.linalg.norm(p1-p0))
                if visible_length>=4:
                    anchors.append((visible_length,p0,p1))
    # Pixel evidence maintains actual shape; dilation is only a search band,
    # not global thickening of the returned walls.
    # Only centerline proposals enter the wall mask. Gradient search bands
    # are evidence buffers and must never be OR'ed into output geometry.
    observed=structural.copy()
    # Recover short curved joins by following actual Canny pixels between
    # existing line supports. Never union a dilated gradient band into walls.
    curve_edges=cv2.Canny(luminosity,12,36,L2gradient=True)
    # A false OCR box may cover a real curved corner. Keep its raw edge
    # eligible only for the bounded two-support contour test below.
    curves,_=cv2.findContours(curve_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    seed_near=cv2.dilate(structural,np.ones((5,5),np.uint8))
    curve_count=0
    for contour in curves:
        pts=contour[:,0,:]
        if len(pts)<8:continue
        contact=seed_near[pts[:,1],pts[:,0]]>0
        if np.count_nonzero(contact)<2:continue
        shift=int(np.flatnonzero(contact)[0]);pts=np.roll(pts,-shift,axis=0);contact=np.roll(contact,-shift)
        pts=np.vstack([pts,pts[:1]]);contact=np.r_[contact,True]
        for a,b in _v7_runs(~contact):
            if a==0 or b>=len(pts) or b-a>max(12,round(32*unit)):continue
            path=pts[a-1:b+1]
            chord=float(np.linalg.norm(path[-1]-path[0]))
            if chord<3 or len(path)>chord*1.85:continue
            cv2.polylines(observed,[path],False,255,1)
            curve_count+=1
    report['observed_curve_joins']=curve_count
    # Fill one-pixel seams, but do not close genuine multi-pixel door openings.
    # No blanket close: even narrow real openings stay open.
    repair = np.zeros_like(observed)
    near_evidence = cv2.dilate(evidence,np.ones((3,3),np.uint8))
    # Spatial endpoint buckets avoid quadratic all-pairs matching.
    max_gap = max(32,round(64*unit))
    if cv2.countNonZero(roi):
        _,_,roi_stats,_=cv2.connectedComponentsWithStats(roi)
        sizes=np.max(roi_stats[1:,:2+2][:,2:4],axis=1)
        if len(sizes): max_gap=max(max_gap,min(160,int(np.percentile(sizes,90))+8))
    bucket = {}
    ends = []
    for length,p,q in anchors:
        direction = (q-p)/length
        for point,outward in [(p,-direction),(q,direction)]:
            idx=len(ends); ends.append((point,outward,length))
            key=tuple(np.floor(point/max_gap).astype(int))
            bucket.setdefault(key,[]).append(idx)
    accepted=[]; seen=set()
    for p,q in through_roi:
        cv2.line(repair,tuple(p),tuple(q),255,1)
        accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                         'end':(q/[w/W,h/H]).round(2).tolist(),'kind':'crossing_roi'})
    for i,(p,dp,lp) in enumerate(ends):
        if time.perf_counter()-started > budget_seconds:
            report['repair_budget_exhausted']=True; break
        cell=np.floor(p/max_gap).astype(int)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in bucket.get((cell[0]+dx,cell[1]+dy),[]):
                    if j<=i: continue
                    q,dq,lq=ends[j]; v=q-p; gap=float(np.linalg.norm(v))
                    if gap<2 or gap>max_gap: continue
                    direction=v/gap
                    if np.dot(dp,direction)<0.985 or np.dot(dq,-direction)<0.985: continue
                    if min(lp,lq)<4: continue
                    n=max(3,int(gap)+1)
                    xs=np.clip(np.rint(np.linspace(p[0],q[0],n)).astype(int),0,w-1)
                    ys=np.clip(np.rint(np.linspace(p[1],q[1],n)).astype(int),0,h-1)
                    masked=roi[ys,xs]>0
                    support=near_evidence[ys,xs]>0
                    missing=observed[ys,xs]==0
                    if missing.mean()<0.15: continue
                    # Evidence outside OCR must be continuous enough; a blank
                    # unmasked doorway is never treated as an occlusion.
                    # Repair only the actual masked gap, not the low-contrast
                    # corridor background that happened to produce gradients.
                    supported=((observed[ys,xs]>0)|masked)[1:-1]
                    uncovered=max((b-a for a,b in _v7_runs(~supported)),default=0)
                    if masked.mean()<0.15: continue
                    kind='occlusion'
                    if supported.mean()<0.88 or uncovered>max(2,round(3*unit)): continue
                    if kind=='faint_line' and support[1:-1].mean()<0.92: continue
                    key=tuple(np.rint(np.r_[p,q]/3).astype(int))
                    if key in seen: continue
                    seen.add(key)
                    cv2.line(repair,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),255,1)
                    accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                                     'end':(q/[w/W,h/H]).round(2).tolist(),'kind':kind})
    # A short divider often ends under a word immediately before a T-junction.
    # Requiring a second collinear stub rejects it. Continue to an observed
    # transverse wall only when the missing run is mostly inside the OCR ROI.
    near_observed=cv2.dilate(observed,np.ones((3,3),np.uint8))
    strong_near=cv2.dilate(((np.abs(gx)+np.abs(gy)>=16).astype(np.uint8)*255),np.ones((3,3),np.uint8))
    t_join_count=0
    for p,direction,length in ends:
        if time.perf_counter()-started>budget_seconds:
            report['repair_budget_exhausted']=True;break
        if length<max(6,7*unit):continue
        steps=np.arange(1,min(max_gap,96)+1)
        xs=np.rint(p[0]+steps*direction[0]).astype(int)
        ys=np.rint(p[1]+steps*direction[1]).astype(int)
        inside=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
        xs=xs[inside];ys=ys[inside];steps=steps[inside]
        hits=np.flatnonzero((near_observed[ys,xs]>0)&(roi[ys,xs]==0)&(steps>=5))
        for end in hits[:12]:
            if end<5:continue
            covered=roi[ys[:end],xs[:end]]>0
            if covered.mean()<0.65:continue
            validated=covered|(strong_near[ys[:end],xs[:end]]>0)
            uncovered=max((b-a for a,b in _v7_runs(~validated)),default=0)
            if uncovered>3:continue
            q=np.array([xs[end],ys[end]])
            normal=np.array([-direction[1],direction[0]])
            transverse=[]
            for side in (-1,1):
                z=np.rint(q+side*4*normal).astype(int)
                transverse.append(0<=z[0]<w and 0<=z[1]<h and near_observed[z[1],z[0]]>0)
            if not all(transverse):continue
            cv2.line(repair,tuple(np.rint(p).astype(int)),tuple(q),255,1)
            accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                             'end':(q/[w/W,h/H]).round(2).tolist(),'kind':'masked_T_join'})
            t_join_count+=1;break
    # Masked corners are not collinear gaps. Intersect two outward rays;
    # both rays must meet forward inside the occlusion and within a short span.
    corner_count=0
    for i,(p,dp,lp) in enumerate(ends):
        if time.perf_counter()-started>budget_seconds:
            report['repair_budget_exhausted']=True;break
        if lp<max(12,18*unit):continue
        cell=np.floor(p/max_gap).astype(int)
        for bx in (-1,0,1):
            for by in (-1,0,1):
                for j in bucket.get((cell[0]+bx,cell[1]+by),[]):
                    if j<=i:continue
                    q,dq,lq=ends[j]
                    if lq<max(12,18*unit):continue
                    delta=q-p;distance=float(np.linalg.norm(delta))
                    if distance<4 or distance>max(32,32*unit):continue
                    determinant=dp[0]*dq[1]-dp[1]*dq[0]
                    if abs(determinant)<.35:continue
                    t=(delta[0]*dq[1]-delta[1]*dq[0])/determinant
                    u=(delta[0]*dp[1]-delta[1]*dp[0])/determinant
                    if t< -1 or u< -1 or t+u>max(48,48*unit):continue
                    mid=p+max(0,t)*dp
                    mx,my=np.rint(mid).astype(int)
                    if not (0<=mx<w and 0<=my<h and roi[my,mx]>0):continue
                    path=np.vstack([np.linspace(p,mid,max(2,int(max(t,0))+1)),
                                    np.linspace(mid,q,max(2,int(max(u,0))+1))])
                    xx=np.clip(np.rint(path[:,0]).astype(int),0,w-1);yy=np.clip(np.rint(path[:,1]).astype(int),0,h-1)
                    covered=roi[yy,xx]>0
                    if covered.mean()<.65:continue
                    supported=covered|(strong_near[yy,xx]>0)
                    if max((b-a for a,b in _v7_runs(~supported)),default=0)>3:continue
                    cv2.polylines(repair,[np.rint(np.array([p,mid,q])).astype(np.int32)],False,255,1)
                    accepted.append({'start':(p/[w/W,h/H]).round(2).tolist(),
                                     'corner':(mid/[w/W,h/H]).round(2).tolist(),
                                     'end':(q/[w/W,h/H]).round(2).tolist(),'kind':'masked_corner'})
                    corner_count+=1
    report['corner_repairs']=corner_count
    report['t_join_repairs']=t_join_count
    merged = observed | repair
    # Collapse duplicate multichannel strokes, without broad gradient unions.
    skeleton = _v7_skeleton(merged)
    merged=_v7_barrier_raster(skeleton)
    report.update({'retained_segments':len(anchors),'repairs':accepted,
                   'repair_count':len(accepted),'core_seconds':round(time.perf_counter()-started,4),
                   'wall_pixels_working':int(np.count_nonzero(merged))})
    masks={'observed':observed,'repairs':repair,'occlusion_mask':roi,'skeleton':skeleton,'walls':merged}
    if (w,h)!=(W,H):
        masks={k:cv2.resize(v,(W,H),interpolation=cv2.INTER_NEAREST) for k,v in masks.items()}
        masks['skeleton']=_v7_skeleton(masks['walls'])
        masks['walls']=_v7_barrier_raster(masks['skeleton'])
        report['core_seconds']=round(time.perf_counter()-started,4)
    return (masks['walls']>0).astype(np.uint8),masks,report


def _v7_write_image(path, image):
    # OpenCV imwrite is unreliable with Chinese paths on some Windows builds.
    path=Path(path)
    ok,encoded=cv2.imencode(path.suffix,image)
    if not ok: raise IOError('Could not encode image: '+str(path))
    encoded.tofile(str(path))


def _restore_unsupported_narrow_repairs(image, walls, masks, bg_mask=None):
    """Retract only short added barriers across a uniform, rail-bounded passage.

    Observed wall pixels and OCR/icon occlusions are never erased. Background
    is a veto/appearance cue, never a reason to carve through a physical wall.
    """
    result = walls.copy()
    observed = (masks['observed'] > 0).astype(np.uint8)
    protected = cv2.dilate(observed, np.ones((3, 3), np.uint8)) > 0
    occluded = masks['occlusion_mask'] > 0
    added = (walls > 0) & (masks['repairs'] > 0) & ~protected & ~occluded
    n, labels, stats, _ = cv2.connectedComponentsWithStats(added.astype(np.uint8), 8)
    h, w = walls.shape
    max_span = int(np.clip(round(min(h, w) * .025), 8, 24))
    records = []
    started = time.perf_counter()
    for cid in range(1, n):
        if time.perf_counter() - started > 1.5:
            break
        x, y, sw, sh, area = map(int, stats[cid])
        span = max(sw, sh)
        if span < 3 or span > max_span or min(sw, sh) > 3:
            continue
        pad = max(6, span)
        x0, y0, x1, y1 = max(0,x-pad), max(0,y-pad), min(w,x+sw+pad), min(h,y+sh+pad)
        cut = labels[y0:y1, x0:x1] == cid
        local = result[y0:y1, x0:x1].copy()
        _, before = cv2.connectedComponents((local == 0).astype(np.uint8), connectivity=4)
        touched = cv2.dilate(cut.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
        ids = np.unique(before[touched & (local == 0)])
        ids = ids[ids > 0]
        if len(ids) != 2:
            continue
        # Both sides must continue beyond the short closure, between parallel rails.
        horizontal = sw >= sh
        rails = observed[y0:y1, x0:x1]
        cy, cx = y-y0+sh//2, x-x0+sw//2
        extent = max(4, span//2)
        if horizontal:
            bands = [rails[max(0,cy-extent):cy+extent+1, max(0,x-x0-3):x-x0+1],
                     rails[max(0,cy-extent):cy+extent+1, x-x0+sw-1:x-x0+sw+3]]
            samples = [(cx,cy-extent),(cx,cy+extent)]
        else:
            bands = [rails[max(0,y-y0-3):y-y0+1, max(0,cx-extent):cx+extent+1],
                     rails[y-y0+sh-1:y-y0+sh+3, max(0,cx-extent):cx+extent+1]]
            samples = [(cx-extent,cy),(cx+extent,cy)]
        axis = 1 if horizontal else 0
        if any(b.size == 0 or np.mean(np.any(b > 0, axis=axis)) < .65 for b in bands):
            continue
        if any(not (0 <= px < local.shape[1] and 0 <= py < local.shape[0]) for px,py in samples):
            continue
        side_ids = [int(before[py,px]) for px,py in samples]
        if 0 in side_ids or side_ids[0] == side_ids[1]:
            continue
        if bg_mask is not None and np.any(bg_mask[y0:y1,x0:x1][touched] > 0):
            continue
        # Same local floor appearance on both sides; reject original dark strokes.
        patch = image[y0:y1,x0:x1]
        colors = [np.median(patch[before == k],axis=0) for k in side_ids]
        floor = (colors[0] + colors[1]) * .5
        if np.linalg.norm(colors[0]-colors[1]) > 24:
            continue
        if np.mean(np.linalg.norm(patch[cut].astype(float)-floor,axis=1) > 32) > .15:
            continue
        local[cut] = 0
        _, after = cv2.connectedComponents((local == 0).astype(np.uint8), connectivity=4)
        if after[samples[0][1],samples[0][0]] != after[samples[1][1],samples[1][0]]:
            continue
        result[y0:y1,x0:x1][cut] = 0
        records.append({'bbox':[x,y,sw,sh], 'removed_pixels':area,
                        'reason':'unsupported_added_barrier_between_parallel_rails'})
    return result, {'restored_count':len(records), 'records':records,
                    'seconds':round(time.perf_counter()-started,4)}


def _physical_region_contacts(res_matrix, walls, rid, neighbours, metric_by_id, gap=18, deadline=None):
    """Local four-connected doorway evidence; cannot detour through other rooms."""
    contacts = set()
    a = metric_by_id[rid]['bbox']
    for nid in sorted(neighbours):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if nid not in metric_by_id:
            continue
        b = metric_by_id[nid]['bbox']
        x0 = max(0, max(a[0], b[0])-gap)
        y0 = max(0, max(a[1], b[1])-gap)
        x1 = min(res_matrix.shape[1], min(a[0]+a[2], b[0]+b[2])+gap)
        y1 = min(res_matrix.shape[0], min(a[1]+a[3], b[1]+b[3])+gap)
        if x1 <= x0 or y1 <= y0:
            continue
        ids = res_matrix[y0:y1,x0:x1]
        allowed = ((walls[y0:y1,x0:x1] == 0) & ((ids == rid) | (ids == nid) | (ids == 1))).astype(np.uint8)
        # Require a real opening of at least 3 pixels, not a diagonal crack.
        allowed = cv2.erode(allowed,np.ones((3,3),np.uint8))
        _, cc = cv2.connectedComponents(allowed, connectivity=4)
        left = np.unique(cc[ids == rid]); right = np.unique(cc[ids == nid])
        if np.intersect1d(left[left>0],right[right>0]).size:
            contacts.add(int(nid))
    return contacts


def extract_walls_with_repair_v7(image_path, output_dir, ocr_data, bg_mask=None, yolo_boxes=None):
    started=time.perf_counter()
    image=cv2.imdecode(np.fromfile(str(image_path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: return None
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    walls,masks,report=extract_wall_evidence_v7(image,ocr_data,yolo_boxes,
        max_dim=int(os.environ.get('MAP_WALL_MAX_DIM','2048')),
        budget_seconds=float(os.environ.get('MAP_WALL_BUDGET_SECONDS','18')))
    if os.environ.get('MAP_NARROW_REPAIR_GUARD','1') != '0':
        walls, narrow_report = _restore_unsupported_narrow_repairs(image, walls, masks, bg_mask)
        masks['walls'] = walls.astype(np.uint8) * 255
        report['narrow_passage_guard'] = narrow_report
    # Background masks remain navigation obstacles in the existing caller.
    # Their color-cluster contours are not assumed to be physical walls.
    report['background_policy']='handled_by_existing_pipeline_after_wall_extraction'
    _v7_write_image(output_dir/'debug_cleaned_walls_0721_4.jpg',masks['walls'])
    if os.environ.get('MAP_WALL_DEBUG','1')!='0':
        for name,mask in masks.items():
            _v7_write_image(output_dir/('wall_v7_'+name+'.png'),mask)
        overlay=image.copy()
        overlay[(masks['walls']>0)&(masks['repairs']==0)]=(0,90,255)
        overlay[(masks['walls']>0)&(masks['repairs']>0)]=(255,0,255)
        _v7_write_image(output_dir/'wall_v7_overlay.jpg',overlay)
    report['stage_seconds']=round(time.perf_counter()-started,4)
    report['within_20_seconds']=report['stage_seconds']<=20
    (output_dir/'wall_v7_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[Wall V7] {:.3f}s, {} evidence-gated repairs'.format(report['stage_seconds'],report['repair_count']))
    return walls

"""Preserve raw enclosed small spaces when partition-only closing consumes them."""
import cv2
import numpy as np

def partition_close_v7(wall_matrix, door_size, min_area):
    walls=(wall_matrix>0).astype(np.uint8)
    h,w=walls.shape
    closed=cv2.morphologyEx(walls,cv2.MORPH_CLOSE,np.ones((door_size,door_size),np.uint8))
    n,raw_labels,stats,_=cv2.connectedComponentsWithStats(1-walls,connectivity=8)
    _,closed_labels=cv2.connectedComponents(1-closed,connectivity=8)
    preserved=[]
    for i in range(1,n):
        x,y,sw,sh,area=map(int,stats[i])
        if area<min_area or area>h*w*.08 or x==0 or y==0 or x+sw>=w or y+sh>=h:continue
        region=raw_labels[y:y+sh,x:x+sw]==i
        survivors=np.unique(closed_labels[y:y+sh,x:x+sw][region]);survivors=survivors[survivors>0]
        # If closing creates multiple subspaces, keep that partition. Otherwise
        # recover the original enclosure without crossing any actual wall.
        if len(survivors)<=1:
            local=closed[y:y+sh,x:x+sw]
            removed=int(np.count_nonzero(local[region]))
            if removed:
                local[region]=0
                preserved.append({'box':[x,y,sw,sh],'area':area,'restored_pixels':removed,'fully_lost':len(survivors)==0})
    return closed,{'door_size':int(door_size),'preserved_count':len(preserved),'preserved':preserved}


def extract_walls_with_repair(image_path, output_dir, ocr_data, bg_mask=None, yolo_boxes=None):
    engine=os.environ.get("MAP_WALL_ENGINE","v7").lower()
    if engine=="legacy":
        return extract_walls_with_repair_legacy(image_path, output_dir, ocr_data, bg_mask, yolo_boxes)
    if engine=="v6":
        return extract_walls_with_repair_v6(image_path, output_dir, ocr_data, bg_mask, yolo_boxes)
    return extract_walls_with_repair_v7(image_path, output_dir, ocr_data, bg_mask, yolo_boxes)


def _env_flag(name, default=True):
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _robust_location_scale(values):
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0, 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    # 1.4826 * MAD 是高斯分布下與標準差相容的 robust scale。
    scale = max(1.0, 1.4826 * mad)
    return max(1.0, median), scale


def _build_fast_region_adjacency(res_matrix, metrics_list, radius):
    """只在各 region 的局部 ROI 做膨脹，建立尺度自適應的 Region Adjacency Graph。"""
    h, w = res_matrix.shape[:2]
    ids = {int(m['id']) for m in metrics_list}
    adj_map = {rid: set() for rid in ids}
    radius = int(max(2, radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    for m in metrics_list:
        rid = int(m['id'])
        lx, ly, sw, sh = map(int, m.get('bbox', [0, 0, w, h]))
        pad = radius + 2
        x1, y1, x2, y2 = _clip_box(lx - pad, ly - pad, lx + sw + pad, ly + sh + pad, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        local_ids = res_matrix[y1:y2, x1:x2]
        local_region = (local_ids == rid).astype(np.uint8)
        if not np.any(local_region):
            continue
        dilated = cv2.dilate(local_region, kernel, iterations=1)
        neighbours = np.unique(local_ids[dilated > 0])
        for n_id in neighbours:
            n_id = int(n_id)
            if n_id > 1 and n_id != rid and n_id in adj_map:
                adj_map[rid].add(n_id)
                adj_map[n_id].add(rid)
    return adj_map


def _fast_region_shape_features(res_matrix, metric):
    """在單一 bbox 內計算 geometry；不做全圖逐候選 skeleton，成本維持很低。"""
    h, w = res_matrix.shape[:2]
    rid = int(metric['id'])
    lx, ly, sw, sh = map(int, metric.get('bbox', [0, 0, w, h]))
    lx, ly, x2, y2 = _clip_box(lx, ly, lx + sw, ly + sh, w, h)
    sw, sh = max(1, x2 - lx), max(1, y2 - ly)
    local = (res_matrix[ly:y2, lx:x2] == rid).astype(np.uint8)
    pixel_area = int(np.count_nonzero(local))
    bbox_area = max(1, sw * sh)
    rectangularity = min(1.0, pixel_area / float(bbox_area))
    aspect = max(sw, sh) / float(max(1, min(sw, sh)))
    span = max(sw / float(max(1, w)), sh / float(max(1, h)))

    solidity = 1.0
    compactness = 1.0
    perimeter = 0.0
    if pixel_area >= 3:
        contours, _ = cv2.findContours(local * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 1.0:
                solidity = float(np.clip(pixel_area / hull_area, 0.0, 1.0))
            if perimeter > 1.0:
                compactness = float(np.clip((4.0 * math.pi * pixel_area) / (perimeter * perimeter), 0.0, 1.0))

    return {
        'area_px': pixel_area,
        'rectangularity': rectangularity,
        'solidity': solidity,
        'compactness': compactness,
        'aspect': aspect,
        'map_span': span,
        'bbox': [lx, ly, sw, sh],
        'perimeter': perimeter,
    }


def _select_primary_public_region_safe(
    res_matrix,
    metrics_list,
    corridor_mask,
    corridor_color_details,
    door_size,
):
    """Select one primary circulation seed without merging same-colour regions.

    Colour is deliberately only one vote.  The seed must also look plausible in
    the region-adjacency graph and must remain supported before *and* after the
    colour mask is smoothed.  This prevents a common failure on mall maps where
    every shop shares one fill colour and thin white separators are crossed by
    morphology.

    The function is scale-relative and uses robust statistics from the current
    map; it contains no coordinates or colours tailored to a particular image.
    """
    h, w = res_matrix.shape[:2]
    if not metrics_list:
        return None, corridor_mask, None, {"reason": "no_regions", "records": []}

    ids = [int(m['id']) for m in metrics_list]
    metric_by_id = {int(m['id']): m for m in metrics_list}
    adjacency_radius = int(np.clip(
        round(max(4.0, float(door_size) * 0.70)),
        4,
        max(5, min(18, round(min(h, w) * 0.018))),
    ))
    adj_map = _build_fast_region_adjacency(res_matrix, metrics_list, adjacency_radius)
    features = {
        rid: _fast_region_shape_features(res_matrix, metric_by_id[rid])
        for rid in ids
    }

    areas = np.asarray(
        [max(1, int(features[rid]['area_px'] or metric_by_id[rid].get('area', 1))) for rid in ids],
        dtype=np.float32,
    )
    area_median = max(1.0, float(np.median(areas)))
    degrees = np.asarray([len(adj_map.get(rid, ())) for rid in ids], dtype=np.float32)
    max_degree = max(1.0, float(degrees.max()) if degrees.size else 1.0)
    degree_q70 = float(np.percentile(degrees, 70)) if degrees.size else 2.0
    degree_gate = int(np.clip(math.ceil(degree_q70), 3, 8))
    largest_rid = ids[int(np.argmax(areas))]

    details = corridor_color_details or {}
    labels_color = details.get('labels_2d')
    candidate_ids = [int(v) for v in details.get('candidate_ids', [])]
    mask_candidates = []
    if isinstance(labels_color, np.ndarray) and labels_color.shape == (h, w):
        for color_id in candidate_ids:
            raw_mask = (labels_color == color_id).astype(np.uint8) * 255
            smooth_mask = _build_corridor_cluster_mask(labels_color, color_id)
            mask_candidates.append((color_id, raw_mask, smooth_mask))
    elif corridor_mask is not None:
        fallback = ((corridor_mask > 0).astype(np.uint8) * 255)
        mask_candidates.append((None, fallback, fallback))

    max_label = max(int(res_matrix.max()) + 1, max(ids) + 1)
    records = []
    best = None
    for color_id, raw_mask, smooth_mask in mask_candidates:
        raw_counts = np.bincount(
            res_matrix[raw_mask > 0].ravel(), minlength=max_label
        ).astype(np.int64)
        smooth_counts = np.bincount(
            res_matrix[smooth_mask > 0].ravel(), minlength=max_label
        ).astype(np.int64)
        total_raw_region_pixels = max(1, int(sum(raw_counts[rid] for rid in ids)))

        # A colour present in many regular regions is probably a room palette,
        # not proof that those disconnected regions form one corridor.
        regular_capture_count = 0
        for rid in ids:
            feat = features[rid]
            area = max(1, int(feat['area_px']))
            raw_overlap = float(raw_counts[rid]) / float(area)
            if (
                raw_overlap >= 0.52
                and feat['rectangularity'] >= 0.76
                and feat['solidity'] >= 0.88
            ):
                regular_capture_count += 1
        diffuse_room_colour = regular_capture_count / float(max(1, len(ids)))

        for rid in ids:
            feat = features[rid]
            area = max(1, int(feat['area_px']))
            area_ratio = area / area_median
            degree = len(adj_map.get(rid, ()))
            raw_overlap = float(raw_counts[rid]) / float(area)
            smooth_overlap = float(smooth_counts[rid]) / float(area)
            stable_overlap = min(raw_overlap, smooth_overlap)
            colour_mass_share = float(raw_counts[rid]) / float(total_raw_region_pixels)
            irregularity = float(np.clip(
                max(1.0 - feat['rectangularity'], 1.0 - feat['solidity']), 0.0, 1.0
            ))
            long_open_shape = bool(
                feat['aspect'] >= 2.4 and feat['map_span'] >= 0.14
            )
            topology_gate = bool(
                degree >= degree_gate
                or area_ratio >= 2.8
                or (degree >= 2 and (irregularity >= 0.24 or long_open_shape))
            )
            colour_gate = bool(
                raw_overlap >= 0.14
                and smooth_overlap >= 0.22
                and raw_counts[rid] >= max(48, int(area_median * 0.018))
            )
            obvious_regular_room = bool(
                feat['rectangularity'] >= 0.82
                and feat['solidity'] >= 0.92
                and degree <= 2
                and area_ratio < 4.0
            )

            score = (
                0.26 * min(1.0, raw_overlap)
                + 0.12 * min(1.0, stable_overlap)
                + 0.23 * min(1.0, degree / max_degree)
                + 0.15 * min(1.0, math.log1p(area_ratio) / math.log(5.0))
                + 0.10 * irregularity
                + 0.14 * min(1.0, colour_mass_share * 3.0)
                - 0.20 * min(1.0, diffuse_room_colour * 2.5)
                - (0.32 if obvious_regular_room else 0.0)
            )
            if rid == largest_rid:
                score += 0.04

            eligible = bool(colour_gate and topology_gate and not obvious_regular_room)
            record = {
                'rid': int(rid),
                'color_id': None if color_id is None else int(color_id),
                'score': round(float(score), 6),
                'eligible': bool(eligible),
                'area_px': int(area),
                'area_ratio_to_median': round(float(area_ratio), 4),
                'degree': int(degree),
                'degree_gate': int(degree_gate),
                'raw_overlap': round(float(raw_overlap), 6),
                'smoothed_overlap': round(float(smooth_overlap), 6),
                'stable_overlap': round(float(stable_overlap), 6),
                'colour_mass_share': round(float(colour_mass_share), 6),
                'diffuse_room_colour_ratio': round(float(diffuse_room_colour), 6),
                'rectangularity': round(float(feat['rectangularity']), 5),
                'solidity': round(float(feat['solidity']), 5),
                'aspect': round(float(feat['aspect']), 4),
                'map_span': round(float(feat['map_span']), 5),
                'obvious_regular_room': bool(obvious_regular_room),
            }
            records.append(record)
            if eligible and (best is None or score > best['score_raw']):
                best = {
                    **record,
                    'score_raw': float(score),
                    'mask': smooth_mask,
                }

    fallback_used = False
    if best is None:
        # Colour may be unavailable on monochrome architectural plans.  A
        # topology-only fallback is allowed, but only for a region that is
        # clearly more connected or much larger than a typical room.
        topology_candidates = []
        for rid in ids:
            feat = features[rid]
            area = max(1, int(feat['area_px']))
            area_ratio = area / area_median
            degree = len(adj_map.get(rid, ()))
            obvious_regular_room = bool(
                feat['rectangularity'] >= 0.84
                and feat['solidity'] >= 0.94
                and degree <= 2
                and area_ratio < 4.0
            )
            eligible = bool(
                not obvious_regular_room
                and (degree >= degree_gate or area_ratio >= 3.2)
            )
            if not eligible:
                continue
            irregularity = max(1.0 - feat['rectangularity'], 1.0 - feat['solidity'])
            score = (
                0.52 * min(1.0, degree / max_degree)
                + 0.34 * min(1.0, math.log1p(area_ratio) / math.log(5.0))
                + 0.14 * min(1.0, irregularity * 2.5)
            )
            topology_candidates.append((score, rid))
        if topology_candidates:
            topology_candidates.sort(reverse=True)
            score, rid = topology_candidates[0]
            best = {
                'rid': int(rid),
                'color_id': None,
                'score': round(float(score), 6),
                'score_raw': float(score),
                'mask': ((res_matrix == int(rid)).astype(np.uint8) * 255),
            }
            fallback_used = True

    report_records = sorted(records, key=lambda rec: rec['score'], reverse=True)
    report = {
        'version': 'public_seed_ensemble_v3',
        'mode': 'topology_colour_stability',
        'selected_region_id': int(best['rid']) if best is not None else None,
        'selected_color_id': (
            None if best is None or best.get('color_id') is None else int(best['color_id'])
        ),
        'selected_score': float(best['score']) if best is not None else None,
        'topology_only_fallback': bool(fallback_used),
        'adjacency_radius_px': int(adjacency_radius),
        'degree_gate': int(degree_gate),
        'region_count': int(len(ids)),
        'records': report_records[:max(80, len(ids) * 2)],
    }
    if best is None:
        report['reason'] = 'no_candidate_passed_colour_topology_or_fallback_guards'
        return None, corridor_mask, None, report
    return int(best['rid']), best.get('mask'), best.get('color_id'), report


def _dominant_raw_free_component(res_matrix, metric, raw_free_labels):
    rid = int(metric['id'])
    h, w = res_matrix.shape[:2]
    lx, ly, sw, sh = map(int, metric.get('bbox', [0, 0, w, h]))
    x1, y1, x2, y2 = _clip_box(lx, ly, lx + sw, ly + sh, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0
    local_region = (res_matrix[y1:y2, x1:x2] == rid)
    values = raw_free_labels[y1:y2, x1:x2][local_region]
    values = values[values > 0]
    if values.size == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[int(np.argmax(counts))])


def _recover_multi_public_spaces(
    res_matrix,
    metrics_list,
    wall_matrix,
    adj_map,
    seed_ids,
    output_dir=None,
    time_budget_seconds=8.0,
):
    """
    從已分割 regions 推定公共空間；RAG 只篩候選，實際入口才是接受證據。

    只使用：
      - robust area / rectangularity / solidity / aspect
      - Region Adjacency Graph
      - 原始 wall-free connectivity（hard barrier gate）
      - 加入候選後能新服務多少房間（Delta-Reach 的快速近似）

    不對每個候選重建 skeleton / A*，因此 2K 圖通常只需亞秒到數秒；另有硬時間預算。
    """
    start = time.perf_counter()
    budget = float(np.clip(float(time_budget_seconds), 0.5, 25.0))
    enabled = _env_flag('MAP_ENABLE_PUBLIC_SPACE_RECOVERY', True)
    metric_by_id = {int(m['id']): m for m in metrics_list}
    valid_ids = set(metric_by_id)
    seed_ids = {int(v) for v in seed_ids if v is not None and int(v) in valid_ids}

    report = {
        'version': 'physical_multi_public_v8',
        'enabled': bool(enabled),
        'time_budget_seconds': round(budget, 3),
        'seed_public_ids': sorted(seed_ids),
        'accepted_ids': [],
        'accepted_roles': {},
        'accepted_modes': {},
        'gateway_targets': {},
        'candidate_count': 0,
        'timed_out': False,
        'elapsed_seconds': 0.0,
        'parameters': {},
        'records': [],
    }
    if not enabled or not metrics_list or not seed_ids:
        report['reason'] = 'disabled_or_no_public_seed'
        report['elapsed_seconds'] = round(time.perf_counter() - start, 4)
        return set(), report

    # 以「關門後的 region」做語意分割，但用「關門前的 repaired wall-free」判斷是否真的有硬牆阻隔。
    raw_free = (wall_matrix == 0).astype(np.uint8)
    _, raw_free_labels = cv2.connectedComponents(raw_free, connectivity=8)

    non_seed_areas = [max(1, int(m.get('area', 1))) for m in metrics_list if int(m['id']) not in seed_ids]
    area_median, area_scale = _robust_location_scale(non_seed_areas)
    q75 = float(np.percentile(non_seed_areas, 75)) if non_seed_areas else area_median
    room_area_cap = max(area_median * 2.8, q75 * 1.8)

    degrees = np.asarray([len(adj_map.get(int(m['id']), ())) for m in metrics_list if int(m['id']) not in seed_ids], dtype=np.float32)
    auto_neighbor_threshold = int(np.clip(math.ceil(float(np.percentile(degrees, 85))) if degrees.size else 4, 4, 8))
    env_neighbor = int(os.environ.get('MAP_PUBLIC_RECOVERY_MIN_NEIGHBORS', '0') or 0)
    neighbor_threshold = env_neighbor if env_neighbor > 0 else auto_neighbor_threshold

    room_like_count = max(1, sum(1 for m in metrics_list if int(m['id']) not in seed_ids and int(m.get('area', 1)) <= room_area_cap))
    auto_min_gain = int(np.clip(round(room_like_count * 0.04), 2, 5))
    env_gain = int(os.environ.get('MAP_PUBLIC_RECOVERY_MIN_GAIN', '0') or 0)
    min_gain = env_gain if env_gain > 0 else auto_min_gain
    area_multiplier = float(np.clip(float(os.environ.get('MAP_PUBLIC_RECOVERY_AREA_MULTIPLIER', '3.2')), 2.0, 8.0))
    max_candidates = int(np.clip(int(os.environ.get('MAP_PUBLIC_RECOVERY_MAX_CANDIDATES', '18')), 4, 40))

    report['parameters'] = {
        'area_median_px': round(float(area_median), 2),
        'area_mad_scale_px': round(float(area_scale), 2),
        'room_area_cap_px': round(float(room_area_cap), 2),
        'neighbor_threshold': int(neighbor_threshold),
        'min_reach_gain': int(min_gain),
        'area_multiplier': round(float(area_multiplier), 3),
        'max_candidates': int(max_candidates),
    }

    features = {}
    free_cc_by_id = {}
    for m in metrics_list:
        if time.perf_counter() - start > budget * 0.62:
            report['timed_out'] = True
            break
        rid = int(m['id'])
        feat = _fast_region_shape_features(res_matrix, m)
        area = max(1, int(feat['area_px'] or m.get('area', 1)))
        feat['area_ratio_to_room_median'] = area / float(max(area_median, 1.0))
        feat['area_robust_z'] = (area - area_median) / float(max(area_scale, 1.0))
        features[rid] = feat
        free_cc_by_id[rid] = _dominant_raw_free_component(res_matrix, m, raw_free_labels)

    if len(features) < len(metrics_list):
        # 預算已經過半時不再做更多候選 geometry；保守返回，避免互動等待被拖長。
        report['reason'] = 'budget_guard_during_feature_extraction'

    room_like_ids = {
        rid for rid, feat in features.items()
        if rid not in seed_ids and feat['area_px'] <= room_area_cap
    }

    candidates = []
    for rid, feat in features.items():
        if rid in seed_ids:
            continue
        room_neighbours = set(adj_map.get(rid, ())) & room_like_ids
        neighbour_count = len(room_neighbours)
        large_outlier = (
            feat['area_ratio_to_room_median'] >= area_multiplier
            or feat['area_robust_z'] >= 3.0
        )
        irregular_open = (
            feat['area_ratio_to_room_median'] >= 1.8
            and (feat['rectangularity'] <= 0.72 or feat['solidity'] <= 0.84)
            and neighbour_count >= 2
        )
        long_connector = (
            feat['aspect'] >= 2.6
            and feat['map_span'] >= 0.16
            and feat['area_ratio_to_room_median'] >= 1.25
            and neighbour_count >= 2
        )
        high_adjacency = neighbour_count >= neighbor_threshold
        regular_room_geometry = bool(
            feat['rectangularity'] >= 0.82
            and feat['solidity'] >= 0.92
            and feat['area_ratio_to_room_median'] < 4.0
            and neighbour_count < max(6, neighbor_threshold + 2)
        )
        if large_outlier or irregular_open or long_connector or high_adjacency:
            priority = (
                neighbour_count * 5.0
                + min(12.0, feat['area_ratio_to_room_median'])
                + (2.0 if irregular_open else 0.0)
                + (1.5 if long_connector else 0.0)
            )
            candidates.append((priority, rid, room_neighbours, {
                'large_outlier': bool(large_outlier),
                'irregular_open': bool(irregular_open),
                'long_connector': bool(long_connector),
                'high_adjacency': bool(high_adjacency),
                'regular_room_geometry': bool(regular_room_geometry),
            }))

    candidates.sort(reverse=True)
    candidates = candidates[:max_candidates]
    report['candidate_count'] = len(candidates)

    # Proximity is only a candidate filter. Count actual entrances for acceptance.
    physical_adj = {}
    for _, rid, _, _ in candidates:
        if time.perf_counter() - start > budget:
            report['timed_out'] = True
            break
        physical_adj[rid] = _physical_region_contacts(
            res_matrix, wall_matrix, rid, adj_map.get(rid, ()), metric_by_id, deadline=start+budget)
    public_ids = set(seed_ids)
    accepted = set()
    public_free_ccs = {free_cc_by_id.get(rid, 0) for rid in public_ids if free_cc_by_id.get(rid, 0) > 0}

    # 目前已由 public network 直接服務到的「room-like」regions。
    def reachable_room_ids():
        reached = set()
        for pid in public_ids:
            reached.update(set(adj_map.get(pid, ())) & room_like_ids)
        return reached - public_ids

    # 最多三輪即可涵蓋 public-area chain；避免任意長的 fixed-point iteration。
    pending = list(candidates)
    for round_idx in range(3):
        if not pending:
            break
        if time.perf_counter() - start > budget:
            report['timed_out'] = True
            break
        changed = False
        next_pending = []
        currently_reached = reachable_room_ids()

        for priority, rid, initial_room_neighbours, gates in pending:
            if time.perf_counter() - start > budget:
                report['timed_out'] = True
                next_pending.append((priority, rid, initial_room_neighbours, gates))
                break

            feat = features[rid]
            room_neighbours = (set(adj_map.get(rid, ())) & room_like_ids) - public_ids
            reach_gain_ids = room_neighbours - currently_reached
            reach_gain = len(reach_gain_ids)
            physical_neighbours = physical_adj.get(rid, set())
            adjacent_public = physical_neighbours & public_ids
            physical_room_neighbours = (physical_neighbours & room_like_ids) - public_ids
            cc_id = free_cc_by_id.get(rid, 0)
            same_raw_free_component = bool(cc_id > 0 and cc_id in public_free_ccs)

            # V8：同一自由空間還不夠，必須有不穿過其他房間的局部實體入口。
            no_hard_barrier = same_raw_free_component and bool(adjacent_public)
            bridge_public = len(adjacent_public) >= 2
            enough_topology_benefit = (
                reach_gain >= min_gain
                or len(room_neighbours) >= neighbor_threshold
                or bridge_public
                or (gates['long_connector'] and reach_gain >= 1)
            )

            # 「語意閘道 override」只允許非常強的公共空間證據通過：
            # 1) RAG 直接碰到既有 public region；2) 一次服務很多新房間；
            # 3) high-adjacency + 大型/開放幾何。避免一般大店面因面積大就被誤判。
            gateway_gain_gate = max(4, int(min_gain) * 2)
            gateway_neighbour_gate = max(6, int(neighbor_threshold) + 2)
            strong_public_geometry = bool(
                gates['high_adjacency']
                and (gates['large_outlier'] or gates['irregular_open'] or gates['long_connector'])
                and feat['area_ratio_to_room_median'] >= max(2.5, area_multiplier * 0.9)
            )
            overwhelming_topology = bool(
                reach_gain >= gateway_gain_gate
                and len(room_neighbours) >= gateway_neighbour_gate
            )
            semantic_gateway_override = False  # Geometry cannot authorize crossing a wall.

            # 大多數店舖是高 rectangularity/solidity 的封閉區。除非同時有極強的
            # 連通與服務增益，這類候選不得因面積或局部鄰接被恢復成公共空間。
            regular_room_guard = bool(gates.get('regular_room_geometry', False))
            overwhelming_room_exception = bool(
                no_hard_barrier
                and len(room_neighbours) >= max(6, neighbor_threshold + 2)
                and reach_gain >= max(3, min_gain)
            )

            physical_service = len(physical_room_neighbours)
            circulation_evidence = bool(
                (bridge_public and (gates['long_connector'] or gates['irregular_open']))
                or (physical_service >= max(2, min_gain)
                    and (gates['long_connector'] or gates['irregular_open']
                         or (gates['high_adjacency'] and physical_service >= neighbor_threshold)))
            )
            regular_connector_exception = bool(gates['long_connector']
                                               and physical_service >= max(2, min_gain))
            accepted_now = bool(no_hard_barrier and circulation_evidence
                                and (not regular_room_guard or overwhelming_room_exception
                                     or regular_connector_exception))
            if accepted_now:
                accepted.add(rid)
                public_ids.add(rid)
                if cc_id > 0 and no_hard_barrier:
                    public_free_ccs.add(cc_id)
                currently_reached.update(reach_gain_ids)
                changed = True

                if feat['solidity'] < 0.78 or (feat['rectangularity'] < 0.62 and feat['area_ratio_to_room_median'] >= 2.0):
                    role = 'public_open_area'
                else:
                    role = 'public_circulation'
                report['accepted_roles'][str(rid)] = role
                if semantic_gateway_override:
                    reason = 'topology_gateway_override'
                    report['accepted_modes'][str(rid)] = 'semantic_gateway_override'
                    report['gateway_targets'][str(rid)] = sorted(int(v) for v in adjacent_public)
                else:
                    reason = 'physical_entrances_and_circulation_geometry'
                    report['accepted_modes'][str(rid)] = 'physical_public_connected'
                    report['gateway_targets'][str(rid)] = sorted(adjacent_public)
            else:
                role = None
                if regular_room_guard and not overwhelming_room_exception:
                    reason = 'regular_room_geometry_guard'
                elif not no_hard_barrier:
                    reason = 'no_physical_public_entrance'
                else:
                    reason = 'insufficient_physical_room_entrances_or_circulation_geometry'
                next_pending.append((priority, rid, initial_room_neighbours, gates))

            if len(report['records']) < max_candidates * 3:
                report['records'].append({
                    'round': int(round_idx + 1),
                    'rid': int(rid),
                    'accepted': bool(accepted_now),
                    'reason': reason,
                    'role': role,
                    'priority': round(float(priority), 3),
                    'room_neighbor_count': int(len(room_neighbours)),
                    'reach_gain': int(reach_gain),
                    'reach_gain_ids': sorted(int(v) for v in reach_gain_ids)[:20],
                    'adjacent_public_ids': sorted(int(v) for v in adjacent_public),
                    'physical_room_entrances': int(physical_service),
                    'physical_public_contacts': sorted(adjacent_public),
                    'same_raw_free_component': bool(same_raw_free_component),
                    'semantic_gateway_override': bool(semantic_gateway_override),
                    'regular_room_guard': bool(regular_room_guard),
                    'overwhelming_room_exception': bool(overwhelming_room_exception),
                    'gateway_gain_gate': int(gateway_gain_gate),
                    'gateway_neighbour_gate': int(gateway_neighbour_gate),
                    'raw_free_component': int(cc_id),
                    'geometry_gates': gates,
                    'area_ratio_to_room_median': round(float(feat['area_ratio_to_room_median']), 3),
                    'area_robust_z': round(float(feat['area_robust_z']), 3),
                    'rectangularity': round(float(feat['rectangularity']), 3),
                    'solidity': round(float(feat['solidity']), 3),
                    'aspect': round(float(feat['aspect']), 3),
                    'map_span': round(float(feat['map_span']), 3),
                })

        pending = next_pending
        if not changed:
            break

    report['accepted_ids'] = sorted(int(v) for v in accepted)
    report['elapsed_seconds'] = round(time.perf_counter() - start, 4)

    if output_dir is not None:
        output_dir = Path(output_dir)
        # Debug：灰色 = 原本 public seed；橘色 = 被 topology/reachability 救回的公共區。
        debug = np.zeros((res_matrix.shape[0], res_matrix.shape[1], 3), dtype=np.uint8)
        if seed_ids:
            debug[np.isin(res_matrix, np.asarray(sorted(seed_ids), dtype=np.int32))] = (190, 190, 190)
        if accepted:
            debug[np.isin(res_matrix, np.asarray(sorted(accepted), dtype=np.int32))] = (0, 165, 255)
        debug_path = output_dir / 'debug_public_space_recovery.jpg'
        debug_ok = safe_imwrite(str(debug_path), debug)
        report['debug_image_path'] = str(debug_path)
        report['debug_image_written'] = bool(debug_ok)
        print(f"[公共空間恢復] debug={debug_path}，written={debug_ok}")
        with open(output_dir / 'public_space_recovery_report_fast_v1.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return accepted, report


def _restore_supported_floor_v10(mask, details, bg_mask):
    """Conservative palette evidence check for an incomplete architectural shell.

    Restore supported colour components themselves, never their convex hull.
    Border/background colours and small printed swatches cannot expand a shell.
    """
    report = {'restored_pixels': 0, 'supported_clusters': []}
    if not details or details.get('labels_2d') is None or np.all(mask):
        return mask, report
    palette = np.asarray(details['labels_2d'])
    if palette.shape != mask.shape:
        return mask, report
    h, w = mask.shape
    total = h*w
    radius = max(2, round(min(h,w)*.004))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
    result = mask.copy()
    shell_distance=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5)
    for k in np.unique(palette):
        raw = palette == k
        if bg_mask is not None and np.mean(np.asarray(bg_mask)[raw]>0)>.5:
            continue
        if np.count_nonzero(raw & (mask>0)) < total*.015:
            continue
        merged = cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(merged,8)
        accepted=[]; outside=0
        for i in range(1,n):
            x,y,bw,bh,a=stats[i]
            if a<total*.002:
                continue
            region=cc[y:y+bh,x:x+bw]==i
            overlap=np.count_nonzero(region & (mask[y:y+bh,x:x+bw]>0))
            border_count=int(x==0)+int(y==0)+int(x+bw>=w)+int(y+bh>=h)
            if border_count and (border_count>1 or overlap/a<.05):
                continue
            # A peripheral shop must touch the established shell locally.
            near=shell_distance[y:y+bh,x:x+bw] <= min(h,w)*.08
            if overlap/a<.02 and not np.any(region&near):
                continue
            accepted.append(i); outside+=int(a-overlap)
        # Only intervene on a substantial omission, not normal contour noise.
        if outside<total*.025:
            continue
        added=np.isin(cc,accepted).astype(np.uint8)
        outlines,_=cv2.findContours(added,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(added,outlines,-1,1,-1)
        result[added>0]=255
        report['supported_clusters'].append(int(k))
    report['restored_pixels']=int(np.count_nonzero((result>0)&(mask==0)))
    return result,report


def build_corridor_transfers_v10(payload, regions, id_labels, max_gap_px=None):
    """Suggest nearby road-end transfers; never add them as physical road edges.

    Reject straight probes through another room or exterior. A proposal does
    not prove an entrance exists; the planner must pause for confirmation.
    """
    from scipy.spatial import cKDTree
    public={int(k) for k,v in id_labels.items() if v.get('portal')}
    nodes=payload.get('nodes',{})
    graph=nx.Graph()
    for e in payload.get('edges',[]):
        if not e.get('semantic_only') and e.get('edge_type')!='recovery_gateway':
            graph.add_edge(e['source'],e['target'])
    comps={n:i for i,c in enumerate(nx.connected_components(graph)) for n in c}
    h,w=regions.shape
    groups={}
    for n in graph:
        x,y=map(lambda v:int(round(v)),nodes[n]['coordinates'])
        if 0<=x<w and 0<=y<h and int(regions[y,x]) in public:
            groups.setdefault(int(regions[y,x]),[]).append((n,x,y))
    limit=float(max_gap_px or max(12,min(h,w)*.06))
    result=[]
    ids=sorted(groups)
    for ai,a in enumerate(ids):
        for b in ids[ai+1:]:
            aa,bb=groups[a],groups[b]
            tree=cKDTree([(v[1],v[2]) for v in bb])
            ds,js=tree.query([(v[1],v[2]) for v in aa],k=min(8,len(bb)))
            choices=sorted(zip(np.asarray(ds).reshape(-1),np.repeat(np.arange(len(aa)),min(8,len(bb))),np.asarray(js).reshape(-1)))
            for dist,i,j in choices:
                if dist>limit:break
                u,x,y=aa[int(i)];v,xx,yy=bb[int(j)]
                if comps[u]==comps[v]:continue
                count=max(2,int(np.ceil(dist*2))+1)
                xs=np.rint(np.linspace(x,xx,count)).astype(int);ys=np.rint(np.linspace(y,yy,count)).astype(int)
                labels=regions[ys,xs]
                if not np.all(np.isin(labels,[1,a,b])):continue
                # A wall strip may hide a door, a wide blocked zone is not an interface.
                if np.count_nonzero(labels==1)*dist/count > max(4,min(h,w)*.015):continue
                result.append({'id':f'T_{a}_{b}','source':u,'target':v,'from_corridor':a,'to_corridor':b,
                    'boundary_hint':[(x+xx)/2,(y+yy)/2],'distance_px':float(dist),
                    'semantic_only':True,'physical_connection_confirmed':False,
                    'navigation_notice':'請在此交界附近尋找入口；確認已進入下一走道後，再繼續導航。'})
                break
    return result


def plan_staged_route(graph_payload, start_node, end_node, confirmed_transfers=(), rejected_transfers=()):
    """Plan from road waypoint IDs (or attached R_* room IDs).

    Return a preview itinerary and only release walking stages up to the first
    unconfirmed transfer. Call again after confirmation; reject IDs to replan.
    Distances of search transfers are excluded from physical walking distance.
    """
    payload=graph_payload.get('graph',graph_payload)
    nodes=payload['nodes']
    def resolve(n):
        if n not in nodes:raise ValueError('Unknown navigation node: '+str(n))
        if nodes[n].get('type')=='room_center':
            n=nodes[n].get('attachment_node')
        if not n or n not in nodes:raise ValueError('Room has no valid road attachment')
        return n
    start_node,end_node=resolve(start_node),resolve(end_node)
    g=nx.Graph();g.add_nodes_from(n for n,v in nodes.items() if v.get('type')!='room_center')
    for e in payload.get('edges',[]):
        if e.get('semantic_only') or e.get('edge_type')=='recovery_gateway':continue
        g.add_edge(e['source'],e['target'],weight=float(e['distance_px']),transfer=None)
    confirmed=set(confirmed_transfers);rejected=set(rejected_transfers)
    # Prefer a fully physical route even if substantially longer.
    penalty=1+sum(float(e.get('distance_px',0)) for e in payload.get('edges',[]))
    for t in payload.get('corridor_transfers',[]):
        if t['id'] in rejected or t['source'] not in g or t['target'] not in g:continue
        if g.has_edge(t['source'],t['target']):continue
        g.add_edge(t['source'],t['target'],weight=penalty+float(t['distance_px']),transfer=t)
    if not nx.has_path(g,start_node,end_node):
        return {'status':'no_route','stages':[],'reason':'沒有可行道路或有依據的交界候選，需補充入口位置。'}
    path=nx.shortest_path(g,start_node,end_node,weight='weight')
    stages=[];walk=[path[0]];distance=0.;blocked=False;active=[]
    def flush():
        nonlocal walk
        if walk:
            stages.append({'type':'walk','nodes':walk,'coordinates':[nodes[n]['coordinates'] for n in walk]})
            walk=[]
    for u,v in zip(path,path[1:]):
        edge=g[u][v];t=edge['transfer']
        if t:
            flush()
            stages.append({'type':'find_entrance','transfer_id':t['id'],'from_node':u,'to_node':v,
                'boundary_hint':t['boundary_hint'],'requires_confirmation':t['id'] not in confirmed,
                'instruction':t['navigation_notice'],'physical_path':None})
            walk=[v]
        else:
            walk.append(v);distance+=edge['weight']
    flush()
    for stage in stages:
        if blocked:break
        active.append(stage)
        if stage['type']=='find_entrance' and stage['requires_confirmation']:blocked=True
    return {'status':'awaiting_entrance_confirmation' if blocked else 'ready',
        'stages':stages,'active_stages':active,'walking_distance_px':distance,
        'contains_unverified_transfer':any(s['type']=='find_entrance' for s in stages)}


def _build_map_interior_mask(wall_matrix, bg_mask=None):
    """Select subdivided building shells, not every large printed rectangle.

    Nested enclosed spaces supply architectural support. Page frames are not
    candidates. No colour, legend location, filename or room number is assumed.
    """
    h, w = wall_matrix.shape
    total = h * w
    radius = max(1, round(min(h, w) * .0015))
    ink = cv2.dilate((wall_matrix > 0).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fallback = {'version':'architectural_shell_v9','fallback_full_image':True}
    if hierarchy is None:
        return np.full((h,w),255,np.uint8), fallback
    tree = hierarchy[0]
    areas = np.array([abs(cv2.contourArea(c)) for c in contours])
    candidates = []
    for i,c in enumerate(contours):
        area = areas[i]
        if not total*.045 <= area <= total*.92:
            continue
        x,y,bw,bh=cv2.boundingRect(c)
        if x<=2 or y<=2 or x+bw>=w-2 or y+bh>=h-2:
            continue
        # Holes within a shell are usable enclosed spaces; ignore text strokes.
        child=int(tree[i,2]); enclosed=[]
        while child>=0:
            a=areas[child]
            if total*.00045 <= a <= area*.45:
                enclosed.append(a)
            child=int(tree[child,0])
        if len(enclosed)<4:
            continue
        score=float(sum(np.sqrt(enclosed)))
        candidates.append((score,i,len(enclosed),float(area)))
    if not candidates:
        fallback['reason']='no_supported_architectural_shell'
        return np.full((h,w),255,np.uint8),fallback
    candidates.sort(reverse=True)
    best=candidates[0]
    # Multiple detached buildings are retained when they have comparable
    # subdivision support; a legend containing tiny icon boxes cannot qualify.
    chosen=[v for v in candidates if v[0]>=best[0]*.28 and v[2]>=max(4,best[2]*.15)]
    mask=np.zeros((h,w),np.uint8)
    for _,i,_,_ in chosen:
        cv2.drawContours(mask,[cv2.convexHull(contours[i])],-1,255,-1)
    mask=cv2.dilate(mask,np.ones((2*radius+1,)*2,np.uint8))
    return mask,{'version':'architectural_shell_v9','fallback_full_image':False,
        'selected_contour_count':len(chosen),'interior_ratio':float(np.mean(mask>0)),
        'shells':[{'contour':i,'support':n,'score':round(s,2),'area':a} for s,i,n,a in chosen]}



def _region_metric_from_matrix(res_matrix, rid):
    ys, xs = np.where(res_matrix == int(rid))
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx, cy = float(xs.mean()), float(ys.mean())
    mean_x = float(np.mean(np.abs(xs.astype(np.float32) - cx)))
    mean_y = float(np.mean(np.abs(ys.astype(np.float32) - cy)))
    return {
        "id": int(rid),
        "area": int(xs.size),
        "max_dist": max(mean_x, mean_y),
        "min_dist": min(mean_x, mean_y),
        "centroid": [cx, cy],
        "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
    }


def _merge_nested_decorative_regions(
    res_matrix,
    metrics_list,
    protected_ids,
    probe_radius,
    output_dir=None,
):
    """合併被商標框、圖示外框切出的內嵌假房間。

    真正相鄰房間通常只在一側接觸；商標內框則會在上/下/左/右至少三個方向被
    同一個外層 region 包圍。此處只合併後者，且永不合併進公共走道。
    """
    protected = {int(v) for v in protected_ids if v is not None}
    report = {
        "version": "nested_decoration_merge_v2",
        "merged_count": 0,
        "records": [],
    }
    if not metrics_list or not _env_flag("MAP_ENABLE_UNVERIFIED_NESTED_MERGE", False):
        report["reason"] = "preserve_enclosed_rooms_without_decoration_evidence"
        return res_matrix, metrics_list, report

    H, W = res_matrix.shape[:2]
    map_area = float(max(1, H * W))
    areas = np.asarray([max(1, int(m.get("area", 1))) for m in metrics_list], dtype=np.float32)
    area_median = float(np.median(areas)) if areas.size else 1.0
    probe_radius = int(np.clip(int(probe_radius), 5, 24))
    removed_ids = set()

    for metric in sorted(metrics_list, key=lambda item: int(item.get("area", 0))):
        child_id = int(metric["id"])
        if child_id in protected or child_id in removed_ids:
            continue
        child_area = int(np.count_nonzero(res_matrix == child_id))
        if child_area <= 0:
            continue
        lx, ly, sw, sh = map(int, metric.get("bbox", [0, 0, W, H]))
        # 商場 logo 白框有時接近典型小店面的面積；面積只作寬鬆候選 gate，
        # 真正決策仍由「同一 parent 三面以上包圍」與 parent/child 比例負責。
        if child_area > max(area_median * 1.35, map_area * 0.020):
            continue
        if max(sw / float(max(W, 1)), sh / float(max(H, 1))) > 0.35:
            continue

        best = None
        for radius in sorted(set([max(4, probe_radius // 2), probe_radius, min(28, probe_radius * 2)])):
            x1, y1, x2, y2 = _clip_box(
                lx - radius - 2, ly - radius - 2,
                lx + sw + radius + 2, ly + sh + radius + 2,
                W, H,
            )
            local_ids = res_matrix[y1:y2, x1:x2]
            child = (local_ids == child_id).astype(np.uint8)
            if not np.any(child):
                continue
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            )
            ring = (cv2.dilate(child, kernel) > 0) & (child == 0)
            surrounding = local_ids[ring & (local_ids > 1)]
            if surrounding.size < 20:
                continue
            ids, counts = np.unique(surrounding, return_counts=True)
            order = np.argsort(counts)[::-1]
            parent_id = int(ids[int(order[0])])
            parent_count = int(counts[int(order[0])])
            dominant_ratio = parent_count / float(max(1, int(counts.sum())))
            if parent_id in protected or parent_id == child_id or dominant_ratio < 0.76:
                continue

            py, px = np.where(ring & (local_ids == parent_id))
            if px.size == 0:
                continue
            center_x = (lx + sw * 0.5) - x1
            center_y = (ly + sh * 0.5) - y1
            dx = px.astype(np.float32) - center_x
            dy = py.astype(np.float32) - center_y
            sectors = set()
            for sx, sy in zip(dx.tolist(), dy.tolist()):
                if abs(sx) >= abs(sy):
                    sectors.add("right" if sx >= 0 else "left")
                else:
                    sectors.add("down" if sy >= 0 else "up")
            if len(sectors) < 3:
                continue

            parent_metric = next(
                (m for m in metrics_list if int(m["id"]) == parent_id), None
            )
            if parent_metric is None:
                continue
            px0, py0, pw, ph = map(int, parent_metric.get("bbox", [0, 0, 0, 0]))
            tolerance = radius + 3
            contained = bool(
                lx >= px0 - tolerance
                and ly >= py0 - tolerance
                and lx + sw <= px0 + pw + tolerance
                and ly + sh <= py0 + ph + tolerance
            )
            parent_area = int(np.count_nonzero(res_matrix == parent_id))
            # L 形外層店面扣掉大型白色 logo 框後，外層可通行像素甚至可能略少於
            # 內框；三面包圍 + bbox containment 已是更可靠的 nested 證據。
            if not contained or parent_area <= 0 or child_area / float(parent_area) > 1.25:
                continue

            candidate = {
                "child_id": child_id,
                "parent_id": parent_id,
                "child_area": child_area,
                "parent_area_before": parent_area,
                "dominant_surround_ratio": round(float(dominant_ratio), 4),
                "surrounding_sectors": sorted(sectors),
                "probe_radius_px": int(radius),
            }
            if best is None or dominant_ratio > best[0]:
                best = (dominant_ratio, candidate)

        if best is None:
            continue
        record = best[1]
        parent_id = int(record["parent_id"])
        res_matrix[res_matrix == child_id] = parent_id
        removed_ids.add(child_id)
        report["records"].append(record)

    new_metrics = []
    for metric in metrics_list:
        rid = int(metric["id"])
        if rid in removed_ids:
            continue
        recomputed = _region_metric_from_matrix(res_matrix, rid)
        if recomputed is not None:
            new_metrics.append(recomputed)

    report["merged_count"] = int(len(removed_ids))
    report["removed_ids"] = sorted(int(v) for v in removed_ids)
    if output_dir is not None:
        output_dir = Path(output_dir)
        with open(output_dir / "nested_room_merge_report_v2.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return res_matrix, new_metrics, report


# =========================================
# 模組 C：空間分割與走道縫合 (🌟回傳值更新，輸出房間資料供後續導航網格使用)
# =========================================
def _partition_floor_ownership_v9(res_matrix, walls, door_size):
    """Bounded simultaneous 4-neighbour expansion into partition-only pixels.

    Real walls and rejected exterior remain blocked. This restores doorway
    pixels consumed by partition closing without merging room identities.
    """
    owner=res_matrix.copy()
    owner[walls>0]=1
    free=(walls==0)
    for _ in range(max(2,int(door_size)+2)):
        empty=(owner==1)&free
        if not np.any(empty):break
        spread=np.zeros_like(owner)
        np.maximum(spread[1:],owner[:-1],out=spread[1:])
        np.maximum(spread[:-1],owner[1:],out=spread[:-1])
        np.maximum(spread[:,1:],owner[:,:-1],out=spread[:,1:])
        np.maximum(spread[:,:-1],owner[:,1:],out=spread[:,:-1])
        add=empty&(spread>1)
        if not np.any(add):break
        owner[add]=spread[add]
    return owner


def _public_partition_classifier_v9(res_matrix, metrics, walls, details, door_size):
    """Joint shape/colour/topology classification with spatial entrance evidence.

    A second entrance to the SAME public region counts: it may close a loop.
    A corridor palette is a supporting vote, never an instruction to merge.
    All contact edges come from four-connected traversable pixels.
    """
    start=time.perf_counter(); h,w=res_matrix.shape
    byid={int(m['id']):m for m in metrics};ids=sorted(byid)
    if not ids:return res_matrix,metrics,set(),{},{}
    # Regions have already passed the architectural envelope filter.
    owner=_partition_floor_ownership_v9(res_matrix,walls,door_size)
    live=[]
    for rid in ids:
        m=byid[rid];x,y,bw,bh=map(int,m['bbox']);pad=int(door_size)+3
        x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(w,x+bw+pad),min(h,y+bh+pad)
        yy,xx=np.nonzero(owner[y0:y1,x0:x1]==rid)
        if not len(xx):continue
        xx=xx+x0;yy=yy+y0;cx=float(xx.mean());cy=float(yy.mean())
        mx=float(np.abs(xx-cx).mean());my=float(np.abs(yy-cy).mean())
        m.update(area=int(len(xx)),centroid=[cx,cy],max_dist=max(mx,my),min_dist=min(mx,my),
                 bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)])
        live.append(rid)
    ids=live
    if not ids:return owner,[],set(),{}, {'version':'joint_public_v9','reason':'no_interior_regions','records':[]}
    metrics=[byid[r] for r in ids]
    near=_build_fast_region_adjacency(owner,metrics,max(4,min(18,door_size)))
    features={r:_fast_region_shape_features(owner,byid[r]) for r in ids}
    median=max(1,float(np.median([features[r]['area_px'] for r in ids])))
    # Collect actual interface coordinates in one raster scan. No repeated A*.
    contacts={r:{} for r in ids}
    for dy,dx in [(0,1),(1,0)]:
        a=owner[:h-dy,:w-dx];b=owner[dy:,dx:]
        valid=(a>1)&(b>1)&(a!=b)
        yy,xx=np.nonzero(valid)
        if not len(xx):continue
        lo=np.minimum(a[valid],b[valid]);hi=np.maximum(a[valid],b[valid])
        key=lo.astype(np.int64)*(int(owner.max())+1)+hi
        order=np.argsort(key);cuts=np.r_[0,np.flatnonzero(np.diff(key[order]))+1,len(order)]
        for l,u in zip(cuts[:-1],cuts[1:]):
            idx=order[l:u];r=int(lo[idx[0]]);n=int(hi[idx[0]])
            if r not in contacts or n not in contacts:continue
            pts=np.column_stack((xx[idx],yy[idx])).astype(np.int32)
            contacts[r].setdefault(n,[]).append(pts);contacts[n].setdefault(r,[]).append(pts)
    for r in ids:
        contacts[r]={n:np.concatenate(p) for n,p in contacts[r].items()}
    palette=details.get('labels_2d') if details else None
    hist={};colour_count=int(palette.max())+1 if isinstance(palette,np.ndarray) and palette.shape==owner.shape else 0
    for r in ids:
        x,y,bw,bh=byid[r]['bbox'];local=owner[y:y+bh,x:x+bw]==r
        if colour_count:
            hist[r]=np.bincount(palette[y:y+bh,x:x+bw][local],minlength=colour_count).astype(float)
            hist[r]/=max(1,hist[r].sum())
    records={}
    for r in ids:
        f=features[r];area=f['area_px']/median;degree=len(near[r])
        # Perimeter^2 / area measures branching/elongation without a bbox-axis
        # assumption, so bent connectors can qualify as well as straight ones.
        x,y,bw,bh=f['bbox']
        shape=(owner[y:y+bh,x:x+bw]==r).astype(np.uint8)
        cs,_=cv2.findContours(shape,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(cs,key=cv2.contourArea)
        simple=cv2.approxPolyDP(contour,max(2,door_size*.3),True)
        perimeter=cv2.arcLength(simple,True)
        stretch=perimeter**2/max(1,16*f['area_px'])
        irregular=(f['solidity']<.80 or f['rectangularity']<.65)
        score=(.55*degree/max(1,max(map(len,near.values())))+
               .25*min(1,np.log1p(area)/np.log(8))+
               .20*min(1,max(0,stretch-1)/5))
        eligible=degree>=4 and area>=2 and (irregular or stretch>=2.8 or (f['aspect']>=2.8 and degree>=6) or (degree>=10 and area>=8 and f['rectangularity']<.75))
        records[r]={'rid':r,'area_ratio':round(area,3),'degree':degree,
          'stretch':round(stretch,3),'rectangularity':round(f['rectangularity'],3),
          'solidity':round(f['solidity'],3),'seed_score':round(float(score),5),
          'seed_eligible':bool(eligible),'physical_neighbours':sorted(contacts[r])}
    eligible=[r for r in ids if records[r]['seed_eligible']]
    if not eligible:
        return owner,metrics,set(),{}, {'version':'joint_public_v9','reason':'no_structural_seed','records':list(records.values())}
    seed=max(eligible,key=lambda r:records[r]['seed_score']); public={seed}
    roles={seed:'public_circulation'}
    # Independent public structures can exist behind closed door symbols.
    # Their semantic classification is separate from physical graph reachability.
    seed_degree=records[seed]['degree']
    regular_ids=[r for r in ids if features[r]['solidity']>.90 and features[r]['rectangularity']>.80]
    palette_diffuse=(float(np.median([np.minimum(hist[r],hist[seed]).sum() for r in regular_ids]))
                     if colour_count and regular_ids else 1.0)
    for r in eligible:
        rec=records[r]
        if r!=seed and rec['degree']>=max(8,seed_degree*.30) and rec['area_ratio']>=4 and ((rec['stretch']>=2.8 and rec['solidity']<.85) or (rec['degree']>=10 and rec['area_ratio']>=8 and rec['rectangularity']<.75)):
            public.add(r);roles[r]='public_open_area'
            rec['accepted_reason']='independent_branched_public_structure'
    # Candidate contacts must be spatially separated relative to corridor width,
    # not just different region IDs. Retain all region IDs throughout.
    for _ in range(3):
        changed=False
        for r in ids:
            if r in public:continue
            f=features[r];rec=records[r];pubcontacts=[p for n,p in contacts[r].items() if n in public]
            pts=np.concatenate(pubcontacts) if pubcontacts else np.empty((0,2),int)
            width=max(3,2*f['area_px']/max(1,f['perimeter']))
            separated=False;spread=0.
            if len(pts)>=4:
                # Distinct boundary clusters, rather than endpoints along one
                # broad doorway. Closing small interface gaps consolidates noise.
                x,y,bw,bh=f['bbox'];pad=max(3,int(round(width*.2)))
                x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
                mask=np.zeros((y1-y0,x1-x0),np.uint8)
                mask[pts[:,1]-y0,pts[:,0]-x0]=1
                mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
                n,cc,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
                centers=centers[1:][stats[1:,cv2.CC_STAT_AREA]>=10]
                if len(centers)>=2:
                    spread=float(np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=2)))
                    separated=spread>=max(door_size*2,width*2.5)
            sim=float(np.minimum(hist[r],hist[seed]).sum()) if colour_count else 0.
            long_shape=rec['stretch']>=2.6 and rec['area_ratio']>=1.2 and (f['solidity']<.82 or f['aspect']>=3.0)
            # Palette-specific rooms remain protected by shape and actual gates.
            connector=separated and long_shape and (not colour_count or palette_diffuse>=.6 or sim>=max(.35,palette_diffuse+.20))
            extension=(len(contacts[r].keys()&public)>=1 and long_shape and sim>=.65 and palette_diffuse<.6 and rec['degree']>=4)
            rec.update({'public_entrance_separation':round(spread,2),'separated_public_entrances':bool(separated),
                        'seed_palette_similarity':round(sim,3)})
            if connector or extension:
                public.add(r);roles[r]='public_circulation';changed=True
                rec['accepted_reason']='separated_public_entrances' if connector else 'supported_corridor_extension'
        if not changed:break
    records[seed]['accepted_reason']='structural_primary_seed'
    for r in ids:records[r]['public']=r in public
    report={'version':'joint_public_v9','seed':seed,'public_ids':sorted(public),'records':list(records.values()),
            'elapsed_seconds':round(time.perf_counter()-start,4)}
    return owner,metrics,public,roles,report


def _preserve_public_clearance_v9(public_mask, wall_matrix, collision):
    """Recover connectivity lost solely to automatic collision padding.

    Never erase observed/repaired walls. Only public floor with >=1.5 pixels of
    raw clearance can supply a thin medial band across a padding-induced split.
    """
    free=((public_mask>0)&(wall_matrix==0)).astype(np.uint8)
    distance=cv2.distanceTransform((wall_matrix==0).astype(np.uint8),cv2.DIST_L2,5)
    free[distance<1.5]=0
    nr,raw=cv2.connectedComponents(free,connectivity=4)
    ns,safe=cv2.connectedComponents((free&(collision==0)).astype(np.uint8),connectivity=4)
    if ns<=1:return collision,{'restored_pixels':0,'split_components':[]}
    pairs=np.unique(np.column_stack((raw[safe>0],safe[safe>0])),axis=0)
    counts=np.bincount(pairs[:,0],minlength=nr)
    split=np.flatnonzero(counts>1);split=split[split>0]
    if not len(split):return collision,{'restored_pixels':0,'split_components':[]}
    candidates=np.isin(raw,split)
    skel=_skeletonize_uint8((candidates.astype(np.uint8)*255))
    band=cv2.dilate((skel>0).astype(np.uint8),np.ones((3,3),np.uint8))>0
    restore=band&candidates&(free>0)&(collision>0)
    result=collision.copy();result[restore]=0
    return result,{'restored_pixels':int(restore.sum()),'split_components':split.tolist()}


class RoomSegmenter:
    def __init__(self, output_dir, yolo_model_path=None, area_ratio=1/8000, door_ratio=0.002):
        self.output_dir = Path(output_dir)
        self.area_ratio = area_ratio
        self.door_ratio = door_ratio
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None
        self.last_public_recovery_report = {}
        self.last_public_classifier_report = {}

    def _fallback_yolo_detections(self, original_img_path):
        if not self.yolo_model_path:
            return []
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.yolo_model_path)
        result = self._yolo_model.predict(source=original_img_path, conf=0.15, imgsz=896, verbose=False)[0]
        detections = []
        for obj in result.boxes:
            coords = obj.xyxy[0].detach().cpu().tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in coords]
            cls_id = int(obj.cls[0].detach().cpu().item())
            label = self._yolo_model.names[cls_id] if isinstance(self._yolo_model.names, (list, tuple)) else self._yolo_model.names.get(cls_id, str(cls_id))
            detections.append({
                'box': [x1, y1, x2, y2],
                'center': [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))],
                'conf': float(obj.conf[0].detach().cpu().item()),
                'label': str(label),
            })
        return detections

    def process(
        self,
        original_img_path,
        wall_matrix,
        corridor_mask,
        ocr_data,
        yolo_detections=None,
        save_csv=True,
        corridor_color_details=None,
        bg_mask=None,
        structural_wall_matrix=None,
    ):
        h, w = wall_matrix.shape[:2]
        min_area = int((h * w) * self.area_ratio)
        door_size = max(1, int(np.sqrt(h**2 + w**2) * self.door_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (door_size, door_size))
        if os.environ.get("MAP_PARTITION_GUARD", "1") != "0":
            closed, partition_guard_report = partition_close_v7(wall_matrix, door_size, min_area)
            (self.output_dir / "partition_guard_v7.json").write_text(
                json.dumps(partition_guard_report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            closed = cv2.morphologyEx(wall_matrix, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((1 - closed).astype(np.uint8), connectivity=8)

        map_interior_mask, map_interior_report = _build_map_interior_mask(
            structural_wall_matrix if structural_wall_matrix is not None else wall_matrix, bg_mask=bg_mask
        )
        map_interior_mask, palette_report = _restore_supported_floor_v10(
            map_interior_mask, corridor_color_details, bg_mask)
        map_interior_report["palette_envelope_check"] = palette_report
        safe_imwrite(
            str(self.output_dir / "debug_map_interior_mask_v2.jpg"),
            map_interior_mask,
        )
        rejected_outside = []

        res_matrix = np.ones((h, w), dtype=np.int32)
        metrics_list = []
        current_id = 2
        for i in range(1, num_labels):
            lx, ly, sw, sh, area = map(int, stats[i])
            if lx <= 2 or ly <= 2 or lx + sw >= w - 2 or ly + sh >= h - 2 or area < min_area:
                continue
            local_component = labels[ly:ly+sh, lx:lx+sw] == i
            local_interior = map_interior_mask[ly:ly+sh, lx:lx+sw] > 0
            interior_ratio = float(np.mean(local_interior[local_component])) if np.any(local_component) else 0.0
            if interior_ratio < 0.58:
                rejected_outside.append({
                    "component": int(i),
                    "bbox": [lx, ly, sw, sh],
                    "area": int(area),
                    "map_interior_overlap": round(interior_ratio, 5),
                })
                continue
            ys, xs = np.nonzero(local_component)
            if len(xs) == 0:
                continue
            cx, cy = map(float, centroids[i])
            gx = xs.astype(np.float32) + lx
            gy = ys.astype(np.float32) + ly
            mean_x = float(np.mean(np.abs(gx - cx)))
            mean_y = float(np.mean(np.abs(gy - cy)))
            metrics_list.append({
                'id': current_id,
                'area': area,
                'max_dist': max(mean_x, mean_y),
                'min_dist': min(mean_x, mean_y),
                'centroid': [cx, cy],
                'bbox': [lx, ly, sw, sh],
                'map_interior_overlap': interior_ratio,
            })
            target = res_matrix[ly:ly+sh, lx:lx+sw]
            target[local_component & (wall_matrix[ly:ly+sh, lx:lx+sw] == 0)] = current_id
            current_id += 1

        map_interior_report["rejected_component_count"] = int(len(rejected_outside))
        map_interior_report["rejected_components"] = rejected_outside[:80]
        with open(self.output_dir / "map_interior_report_v2.json", "w", encoding="utf-8") as f:
            json.dump(map_interior_report, f, ensure_ascii=False, indent=2)

        if not metrics_list:
            return res_matrix, [], {}

        # V9: rejected exterior is also blocked for navigation and floor recovery.
        wall_matrix[map_interior_mask == 0] = 1
        res_matrix, metrics_list, public_v9, roles_v9, joint_report = _public_partition_classifier_v9(
            res_matrix, metrics_list, wall_matrix, corridor_color_details or {}, door_size)
        main_cid = joint_report.get('seed')
        corridor_rids = sorted(public_v9)
        joint_report.update(classifier_mode='joint_v9', primary_public_region_id=main_cid,
                            independent_regions_merged=False,
                            same_colour_regions_preserved=len(metrics_list)-len(public_v9))
        self.last_public_classifier_report = joint_report
        (self.output_dir / 'corridor_topology_report_fast_v4.json').write_text(
            json.dumps(joint_report, ensure_ascii=False, indent=2), encoding='utf-8')
        recovery_debug=np.zeros((h,w,3),np.uint8)
        for rid in public_v9:
            recovery_debug[res_matrix==rid]=(190,190,190) if rid==main_cid else (0,165,255)
        safe_imwrite(str(self.output_dir/'debug_public_space_recovery.jpg'),recovery_debug)
        (self.output_dir / 'public_partition_v9.json').write_text(
            json.dumps(joint_report, ensure_ascii=False, indent=2), encoding='utf-8')

        # 商標白框、App 圓章與圖示外框可能在房間內形成第二個封閉區。若一個小 region
        # 在至少三個方向都被同一房間包圍，將它併回外層房間；公共走道永不參與合併。
        protected_public_ids = set(int(v) for v in corridor_rids)
        if main_cid is not None:
            protected_public_ids.add(int(main_cid))
        res_matrix, metrics_list, nested_merge_report = _merge_nested_decorative_regions(
            res_matrix=res_matrix,
            metrics_list=metrics_list,
            protected_ids=protected_public_ids,
            probe_radius=max(5, int(round(door_size * 0.55))),
            output_dir=self.output_dir,
        )
        if nested_merge_report.get("merged_count", 0):
            print(
                f"[房間分割] 已合併 {nested_merge_report['merged_count']} 個"
                "被商標/圖示外框切出的內嵌假房間。"
            )

        existing_ids = {int(m['id']) for m in metrics_list}
        public_seed_ids = ({int(main_cid)} if main_cid is not None else set()) & existing_ids
        recovered_public_ids = (set(public_v9) - public_seed_ids) & existing_ids
        adjacency_radius = max(4, min(18, door_size))
        recovery_budget = 8.0
        recovery_report = {
            'version': 'joint_public_v9', 'accepted_ids': sorted(recovered_public_ids),
            'accepted_roles': {str(r): roles_v9[r] for r in recovered_public_ids},
            'accepted_modes': {str(r): 'structural_public_seed' for r in recovered_public_ids},
            'gateway_targets': {}, 'elapsed_seconds': joint_report.get('elapsed_seconds', 0),
            'time_budget_seconds': recovery_budget,
        }
        self.last_public_recovery_report = recovery_report
        (self.output_dir / 'public_space_recovery_report_fast_v1.json').write_text(
            json.dumps(recovery_report, ensure_ascii=False, indent=2), encoding='utf-8')

        id_labels = {
            str(m['id']): {
                "names": [], "objects": [], "portal": False, "shape": [],
                "space_type": "room", "recovered_public_space": False,
            }
            for m in metrics_list
        }
        virtual_room_id = current_id
        id_labels[str(virtual_room_id)] = {
            "names": [], "objects": [], "portal": False, "shape": [],
            "space_type": "unknown", "recovered_public_space": False,
        }

        # portal 是現有導航模組的相容性契約：所有 public circulation/open area 都仍透過 portal=True 進 graph。
        for rid in sorted(public_seed_ids):
            if str(rid) in id_labels:
                id_labels[str(rid)]["portal"] = True
                id_labels[str(rid)]["space_type"] = "public_circulation"
        for rid in sorted(recovered_public_ids):
            if str(rid) in id_labels:
                id_labels[str(rid)]["portal"] = True
                id_labels[str(rid)]["space_type"] = recovery_report.get('accepted_roles', {}).get(str(rid), 'public_open_area')
                id_labels[str(rid)]["recovered_public_space"] = True
                id_labels[str(rid)]["public_recovery_mode"] = recovery_report.get('accepted_modes', {}).get(str(rid), 'raw_free_connected')
                id_labels[str(rid)]["recovery_gateway_to"] = recovery_report.get('gateway_targets', {}).get(str(rid), [])
                id_labels[str(rid)]["recovery_gateway_search_px"] = int(adjacency_radius)

        print(
            f"[公共空間恢復] seed={sorted(public_seed_ids)}，"
            f"recovered={sorted(recovered_public_ids)}，"
            f"耗時={recovery_report.get('elapsed_seconds', 0.0):.3f}s / budget={recovery_report.get('time_budget_seconds', recovery_budget):.1f}s"
        )

        top20 = max(1, int(len(metrics_list) * 0.2))
        top30 = max(1, int(len(metrics_list) * 0.3))
        for m in sorted(metrics_list, key=lambda x: x['max_dist'], reverse=True)[:top20]:
            id_labels[str(m['id'])]["shape"].append("長寬")
        for m in sorted(metrics_list, key=lambda x: x['min_dist'])[:top20]:
            id_labels[str(m['id'])]["shape"].append("短窄")
        for m in sorted(metrics_list, key=lambda x: x['area'], reverse=True)[:top20]:
            id_labels[str(m['id'])]["shape"].append("大")
        for m in sorted(metrics_list, key=lambda x: x['area'])[:top30]:
            id_labels[str(m['id'])]["shape"].append("小")

        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        random_colors = [(255, 220, 200), (200, 255, 200), (200, 255, 255), (200, 200, 255)]
        for m in metrics_list:
            rid = int(m['id'])
            if id_labels[str(rid)].get("recovered_public_space", False):
                color = (160, 220, 255)
            elif id_labels[str(rid)]["portal"]:
                color = (200, 200, 200)
            else:
                color = random.choice(random_colors)
            vis_img[res_matrix == rid] = color
            cv2.putText(vis_img, str(rid), (int(m['centroid'][0] - 15), int(m['centroid'][1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        metric_centers = np.array([[m['centroid'][0], m['centroid'][1]] for m in metrics_list], dtype=np.float32)
        metric_ids = [int(m['id']) for m in metrics_list]
        def nearest_metric_id(x, y, max_radius=None):
            d2 = np.sum((metric_centers - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
            best_idx = int(np.argmin(d2))
            if max_radius is not None and float(d2[best_idx]) > float(max_radius) ** 2:
                return None
            return metric_ids[best_idx]

        for item in ocr_data:
            tx, ty = map(int, item['center'])
            if 0 <= ty < h and 0 <= tx < w:
                target_id = int(res_matrix[ty, tx])
                if target_id == 1:
                    box = item.get('box', [tx, ty, tx, ty])
                    text_span = max(
                        abs(int(box[2]) - int(box[0])),
                        abs(int(box[3]) - int(box[1])),
                    ) if len(box) >= 4 else 0
                    local_radius = float(np.clip(
                        max(22, text_span * 1.4),
                        22,
                        max(34, min(h, w) * 0.035),
                    ))
                    target_id = nearest_metric_id(tx, ty, max_radius=local_radius)
                if target_id is None:
                    continue
                if str(target_id) in id_labels:
                    id_labels[str(target_id)]["names"].append(item['text'])

        if yolo_detections is None:
            yolo_detections = self._fallback_yolo_detections(original_img_path)
        for det in yolo_detections:
            ox, oy = map(int, det.get('center', [0, 0]))
            conf = float(det.get('conf', 0.0))
            label = str(det.get('label', det.get('class_id', 'object')))
            if not (0 <= oy < h and 0 <= ox < w):
                continue
            target_id = int(res_matrix[oy, ox]) if conf >= 0.40 else virtual_room_id
            if target_id == 1:
                box = det.get('box', [ox, oy, ox, oy])
                object_span = max(
                    abs(int(box[2]) - int(box[0])),
                    abs(int(box[3]) - int(box[1])),
                ) if len(box) >= 4 else 0
                local_radius = float(np.clip(
                    max(24, object_span * 0.8),
                    24,
                    max(40, min(h, w) * 0.04),
                ))
                target_id = nearest_metric_id(ox, oy, max_radius=local_radius)
            if target_id is None:
                continue
            if str(target_id) in id_labels:
                id_labels[str(target_id)]["objects"].append(f"{label}({conf:.2f})")

        if save_csv:
            np.savetxt(self.output_dir / "_0721_4.csv", res_matrix, fmt="%d", delimiter=",")
        safe_imwrite(str(self.output_dir / "debug_0721_4.jpg"), vis_img)
        with open(self.output_dir / "room_data_0721_4.json", 'w', encoding='utf-8') as f:
            json.dump(id_labels, f, ensure_ascii=False, indent=4)
        print("[完成] 完美融合版 JSON 已生成。")
        return res_matrix, metrics_list, id_labels

# =========================================
# 🌟 模組 D：航點導航圖生成器 (捷運路網與十字射線正交版)
# =========================================
# 已移除：舊版 WaypointGraphGenerator（原本會被下方 V8 完全覆蓋）

        # =========================================
# 🌟 模組 D：航點導航圖生成器 V13
#    1) 尺度自適應：由走道寬度推導碰撞邊界、補償半徑、A* 網格解析度
#    2) 骨架角點 + 稀疏 skeleton sampling，避免長直/緩彎走道缺 seed
#    3) component 不再只接最大路網；改成全域 component-pair 合併（MST-like）
#    4) 橋接只允許四方向，成本同時考慮距離、轉彎、離走道距離與房間核心
#    5) strict -> adaptive band -> wall-topology fallback；仍不可達者 connect-or-prune
#    6) Topology-first cycle preservation：reference skeleton + bridge bypass + persistent-hole census
# =========================================
class WaypointGraphGenerator:
    def __init__(self, output_dir, scale="1 pixel = 0.05 meters"):
        self.output_dir = Path(output_dir)
        self.scale = scale

    def generate(self, wall_matrix, res_matrix, metrics_list, id_labels):
        graph_generation_start = time.perf_counter()
        print("[系統] 正在生成 LLM 專用導航拓樸圖 V14（長直線補網 + 房間直達投影）...")
        if res_matrix is None or not metrics_list:
            print("[警告] 缺少空間分配矩陣，無法生成導航圖。")
            return

        # -------------------------------------------------
        # 1. 導航遮罩、牆壁碰撞遮罩與牆距（V11：尺度自適應）
        # -------------------------------------------------
        # V8：實體入口確認的公共區域直接納入 primary；舊式語意恢復才走 auxiliary。
        # 重要：既有 V13.1 的尺度估計、骨架、環路修補只看 primary public regions，
        # recovered open area 絕不能回頭改寫主路網的 corridor width / skeleton / cycle census。
        corridor_ids = [int(rid) for rid, data in id_labels.items() if data.get("portal", False)]
        recovered_corridor_ids = [
            int(rid) for rid, data in id_labels.items()
            if data.get("portal", False) and data.get("recovered_public_space", False)
            and data.get("public_recovery_mode") not in {"physical_public_connected", "structural_public_seed"}
        ]
        primary_corridor_ids = [cid for cid in corridor_ids if cid not in set(recovered_corridor_ids)]
        # 極端保底：若舊資料沒有 recovered flag，仍沿用全部 portal，維持舊行為。
        if not primary_corridor_ids:
            primary_corridor_ids = list(corridor_ids)

        primary_corridor_mask = np.zeros_like(res_matrix, dtype=np.uint8)
        for cid in primary_corridor_ids:
            primary_corridor_mask[res_matrix == cid] = 255

        recovered_corridor_mask = np.zeros_like(res_matrix, dtype=np.uint8)
        for cid in recovered_corridor_ids:
            recovered_corridor_mask[res_matrix == cid] = 255

        # pure_corridor_mask 是舊 V13.1 的契約；V7 明確固定為 primary mask。
        # 因此 recovery 不會破壞原有環狀結構。
        pure_corridor_mask = primary_corridor_mask.copy()

        H, W = res_matrix.shape[:2]
        map_short_side = max(1, min(H, W))

        # 不再把 9x9 / 25x25 寫死。先由走道本身估計典型寬度，
        # 之後所有 collision margin、橋接 band 與 A* 解析度都由此尺度推導。
        original_seed_ids = [int(rid) for rid, data in id_labels.items()
                             if data.get("portal", False) and not data.get("recovered_public_space", False)]
        width_seed_mask = (np.isin(res_matrix, original_seed_ids).astype(np.uint8)*255
                           if original_seed_ids else pure_corridor_mask)
        corridor_binary = (width_seed_mask > 0).astype(np.uint8)
        if np.any(corridor_binary):
            corridor_inside_distance = cv2.distanceTransform(corridor_binary, cv2.DIST_L2, 5)
            corridor_seed_skeleton = _skeletonize_uint8(width_seed_mask)
            width_samples = corridor_inside_distance[corridor_seed_skeleton > 0]
            width_samples = width_samples[np.isfinite(width_samples) & (width_samples > 0)]
            if width_samples.size:
                corridor_half_width = float(np.median(width_samples))
            else:
                positive = corridor_inside_distance[corridor_inside_distance > 0]
                corridor_half_width = float(np.percentile(positive, 85)) if positive.size else 6.0
        else:
            corridor_inside_distance = np.zeros((H, W), dtype=np.float32)
            corridor_seed_skeleton = np.zeros((H, W), dtype=np.uint8)
            corridor_half_width = max(4.0, map_short_side * 0.01)

        corridor_width_est = float(np.clip(
            corridor_half_width * 2.0,
            8.0,
            max(12.0, map_short_side * 0.20),
        ))

        # Width calibration uses the seed; clearance must cover every actual road.
        corridor_inside_distance = cv2.distanceTransform(
            (pure_corridor_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

        auto_wall_radius = int(np.clip(round(corridor_width_est * 0.075), 2, 7))
        env_wall_radius = int(os.environ.get("MAP_WALL_COLLISION_RADIUS_PX", "0") or 0)
        wall_collision_radius = env_wall_radius if env_wall_radius > 0 else auto_wall_radius

        wall_uint8 = ((wall_matrix > 0).astype(np.uint8) * 255)
        wall_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (wall_collision_radius * 2 + 1, wall_collision_radius * 2 + 1),
        )
        wall_collision = cv2.dilate(wall_uint8, wall_kernel, iterations=1)
        clearance_report = {'restored_pixels': 0, 'explicit_radius_preserved': env_wall_radius > 0}
        if env_wall_radius <= 0:
            wall_collision, clearance_report = _preserve_public_clearance_v9(
                pure_corridor_mask, wall_matrix, wall_collision)
        (self.output_dir / 'public_clearance_v9.json').write_text(
            json.dumps(clearance_report, indent=2), encoding='utf-8')

        # -------------------------------------------------
        # FAST V6：Recovered public-space semantic gateways
        # -------------------------------------------------
        # 只對 topology_gateway_override 的 recovered public region 開局部 virtual doorway。
        # 不修改原 wall_matrix，只修改導航 collision mask，因此不會污染房間/牆體語意。
        semantic_gateway_mask = np.zeros_like(pure_corridor_mask, dtype=np.uint8)
        semantic_gateway_records = []

        def _multiple_pairs_between_region_sets(mask_a, mask_b, *, max_pairs, min_separation_px, max_gap_px):
            """找多個彼此分散的 semantic gateway 候選。

            這些點只表示「兩個公共區域在此附近可能存在可通行入口」，不是已偵測到的實體門。
            因此只利用 region 鄰近關係與幾何距離做 NMS，不會把 gateway 寫回 wall matrix。
            """
            a = (mask_a > 0)
            b = (mask_b > 0)
            if not np.any(a) or not np.any(b):
                return []

            # 只在 recovered region 邊界搜尋，避免內部像素壟斷 nearest-distance 排序。
            a_u8 = a.astype(np.uint8) * 255
            eroded = cv2.erode(a_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            boundary = (a_u8 > 0) & (eroded == 0)
            ys, xs = np.where(boundary)
            if xs.size == 0:
                return []

            dist_to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
            gaps = dist_to_b[ys, xs]
            order = np.argsort(gaps)
            selected = []
            min_sep = max(8.0, float(min_separation_px))
            max_gap = max(1.0, float(max_gap_px))

            # 邊界可能有數千點；先只掃最靠近另一公共區的前段即可。
            scan_limit = min(int(order.size), max(500, int(max_pairs) * 500))
            for oi in order[:scan_limit]:
                gap = float(gaps[int(oi)])
                if gap > max_gap:
                    break
                ax, ay = int(xs[int(oi)]), int(ys[int(oi)])

                # NMS：同一入口附近不要重複放多條紅線。
                if any(math.hypot(ax - q[0][0], ay - q[0][1]) < min_sep for q in selected):
                    continue

                search_r = int(max(4, math.ceil(gap) + 4))
                x1, y1, x2, y2 = _clip_box(
                    ax - search_r, ay - search_r,
                    ax + search_r + 1, ay + search_r + 1,
                    W, H,
                )
                bys, bxs = np.where(b[y1:y2, x1:x2])
                if bxs.size == 0:
                    continue
                gx = bxs.astype(np.float32) + x1
                gy = bys.astype(np.float32) + y1
                d2 = (gx - ax) ** 2 + (gy - ay) ** 2
                j = int(np.argmin(d2))
                bx, by = int(round(float(gx[j]))), int(round(float(gy[j])))
                actual_gap = float(math.hypot(bx - ax, by - ay))
                if actual_gap > max_gap:
                    continue

                # 兩端都做 separation，避免 recovered 端雖分散、primary 端卻全部擠在同一點。
                if any(
                    math.hypot(bx - q[1][0], by - q[1][1]) < min_sep * 0.70
                    for q in selected
                ):
                    continue

                selected.append(((ax, ay), (bx, by), actual_gap))
                if len(selected) >= int(max_pairs):
                    break
            return selected

        recovered_gateway_ids = [
            int(rid) for rid, data in id_labels.items()
            if data.get('portal', False)
            and data.get('recovered_public_space', False)
            and int(rid) in recovered_corridor_ids
        ]
        gateway_points_per_region = int(np.clip(
            int(os.environ.get('MAP_PUBLIC_GATEWAY_POINTS_PER_REGION', '3')),
            1, 6,
        ))
        for rid in recovered_gateway_ids:
            data = id_labels.get(str(rid), {})
            target_ids = [
                int(v) for v in data.get('recovery_gateway_to', [])
                if str(v) in id_labels and id_labels[str(v)].get('portal', False) and int(v) != rid
            ]
            if not target_ids:
                target_ids = [cid for cid in primary_corridor_ids if cid != rid]
            if not target_ids:
                continue

            rag_radius = int(data.get('recovery_gateway_search_px', 18) or 18)
            auto_gateway_gap = max(float(rag_radius) * 1.35, corridor_width_est * 0.30)
            env_gateway_gap = float(os.environ.get('MAP_PUBLIC_GATEWAY_MAX_GAP_PX', '0') or 0.0)
            gateway_max_gap = env_gateway_gap if env_gateway_gap > 0 else float(np.clip(auto_gateway_gap, 10.0, 30.0))
            gateway_separation = float(np.clip(
                float(os.environ.get('MAP_PUBLIC_GATEWAY_MIN_SEPARATION_PX', '0') or 0.0)
                or max(corridor_width_est * 1.10, 42.0),
                24.0, 140.0,
            ))

            a_mask = (res_matrix == rid)
            b_mask = np.isin(res_matrix, np.asarray(target_ids, dtype=np.int32))
            recovery_mode = str(data.get('public_recovery_mode', 'raw_free_connected'))
            if recovery_mode != 'semantic_gateway_override':
                other_rooms = (res_matrix > 1) & ~a_mask & ~b_mask
                pairs = _raw_free_gateway_pairs(
                    a_mask, b_mask, wall_matrix, other_rooms,
                    gateway_max_gap, gateway_separation, gateway_points_per_region
                )
            else:
                pairs = _multiple_pairs_between_region_sets(
                    a_mask,
                    b_mask,
                    max_pairs=gateway_points_per_region,
                    min_separation_px=gateway_separation,
                    max_gap_px=gateway_max_gap,
                )
            if not pairs:
                semantic_gateway_records.append({
                    'recovered_id': int(rid), 'target_ids': target_ids, 'accepted': False,
                    'reason': 'no_semantic_gateway_candidate_within_gap',
                    'max_gap_px': round(gateway_max_gap, 2),
                    'semantic_only': True,
                    'physical_entrance_confirmed': False,
                })
                continue

            gateway_thickness = int(np.clip(round(corridor_width_est * 0.22), 7, 19))
            recovery_mode = str(data.get('public_recovery_mode', 'raw_free_connected'))
            accepted_for_region = 0
            for gateway_index, (p_rec, p_pub, gap_px) in enumerate(pairs, start=1):
                # raw-free connected 模式仍需物理 free-space 證據；semantic override 則只表示「附近找入口」。
                # Raw-free pairs already passed local four-direction reachability
                # against actual walls; do not reject them again with dilated walls.
                target_id = int(res_matrix[p_pub[1], p_pub[0]]) if (0 <= int(p_pub[0]) < W and 0 <= int(p_pub[1]) < H) else None
                # 只有通過 gate 的候選才畫進紅色 semantic-gateway debug。
                cv2.line(semantic_gateway_mask, p_rec, p_pub, 255, gateway_thickness)
                cv2.circle(semantic_gateway_mask, p_rec, max(3, gateway_thickness // 2), 255, -1)
                cv2.circle(semantic_gateway_mask, p_pub, max(3, gateway_thickness // 2), 255, -1)

                record = {
                    'gateway_id': f'GW_{rid}_{gateway_index}',
                    'gateway_index': int(gateway_index),
                    'recovered_id': int(rid),
                    'target_id': int(target_id) if target_id in target_ids else None,
                    'target_ids': target_ids,
                    'accepted': True,
                    'recovered_point': [int(p_rec[0]), int(p_rec[1])],
                    'public_point': [int(p_pub[0]), int(p_pub[1])],
                    'gap_px': round(gap_px, 2),
                    'max_gap_px': round(gateway_max_gap, 2),
                    'min_separation_px': round(gateway_separation, 2),
                    'thickness_px': int(gateway_thickness),
                    'recovery_mode': recovery_mode,
                    # 關鍵語意：這不是門偵測結果，只是一個可用於規劃的「近似跨區入口候選」。
                    'semantic_only': True,
                    'physical_entrance_confirmed': False,
                    'gateway_kind': 'approximate_cross_region_access',
                    'navigation_notice': '抵達跨區連接附近後，請在附近尋找實際可通行的入口；圖上的跨區連線只是位置提示，不代表入口的精確位置。',
                }
                semantic_gateway_records.append(record)
                accepted_for_region += 1

            print(
                f"[公共空間閘道 V8] recovered={rid}，候選={len(pairs)}，"
                f"accepted={accepted_for_region}，max={gateway_points_per_region}。"
            )

        # V7：此處只「記錄」gateway，不修改 primary wall_collision / pure_corridor_mask。
        # 否則 recovery 會在主 V13.1 執行前改變尺度與拓樸參考，正是環路被破壞的來源。
        if np.any(semantic_gateway_mask):
            print(
                f"[公共空間閘道] 已辨識 {sum(1 for r in semantic_gateway_records if r.get('accepted'))} "
                "個局部 virtual doorway；延後到 primary graph 完成後再融合。"
            )

        debug_gateway = np.zeros((H, W, 3), dtype=np.uint8)
        debug_gateway[primary_corridor_mask > 0] = (190, 190, 190)
        debug_gateway[recovered_corridor_mask > 0] = (255, 180, 80)
        debug_gateway[semantic_gateway_mask > 0] = (0, 0, 255)
        gateway_debug_path = self.output_dir / 'debug_semantic_public_gateways.jpg'
        gateway_debug_ok = safe_imwrite(str(gateway_debug_path), debug_gateway)
        print(f"[公共空間閘道] debug={gateway_debug_path}，written={gateway_debug_ok}")
        with open(self.output_dir / 'semantic_public_gateway_report.json', 'w', encoding='utf-8') as f:
            json.dump(semantic_gateway_records, f, ensure_ascii=False, indent=2)

        wall_free = cv2.bitwise_not(wall_collision)

        auto_corridor_expand = int(np.clip(
            round(corridor_width_est * 0.35),
            6,
            max(12, round(map_short_side * 0.035)),
        ))
        env_corridor_expand = int(os.environ.get("MAP_CORRIDOR_EXPAND_RADIUS_PX", "0") or 0)
        corridor_expand_radius = env_corridor_expand if env_corridor_expand > 0 else auto_corridor_expand
        corridor_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (corridor_expand_radius * 2 + 1, corridor_expand_radius * 2 + 1),
        )
        corridor_expansion = cv2.dilate(pure_corridor_mask, corridor_kernel, iterations=1)
        valid_routing_mask = cv2.bitwise_and(corridor_expansion, wall_free)

        # corridor_distance[y, x] = 該點離「已知走道」最近距離。
        # V11 的補償搜尋可在缺失的走道標記附近活動，但會依這個距離加成本，
        # 因此不再是「超過固定 12px 就永遠無法橋接」。
        outside_corridor = (pure_corridor_mask == 0).astype(np.uint8)
        corridor_distance = cv2.distanceTransform(outside_corridor, cv2.DIST_L2, 5)

        wall_distance = cv2.distanceTransform(wall_free, cv2.DIST_L2, 5)
        min_attachment_clearance = float(np.clip(corridor_width_est * 0.04, 1.0, 4.0))
        safe_attachment_mask = (
            (valid_routing_mask > 0)
            & (wall_distance >= min_attachment_clearance)
        ).astype(np.uint8) * 255

        print(
            "[路網尺度] "
            f"corridor_width≈{corridor_width_est:.1f}px，"
            f"wall_collision_radius={wall_collision_radius}px，"
            f"corridor_expand_radius={corridor_expand_radius}px"
        )

        def in_bounds(x, y):
            return 0 <= int(x) < W and 0 <= int(y) < H

        def point_is_safe(pt, clearance=0.0, mask=None):
            x, y = int(pt[0]), int(pt[1])
            use_mask = valid_routing_mask if mask is None else mask
            return (
                in_bounds(x, y)
                and use_mask[y, x] > 0
                and wall_collision[y, x] == 0
                and wall_distance[y, x] >= float(clearance)
            )

        def local_line_mask(p1, p2, thickness=3):
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            pad = max(2, int(thickness) + 1)
            x1, y1, x2, y2 = _clip_box(
                min(p1[0], p2[0]) - pad, min(p1[1], p2[1]) - pad,
                max(p1[0], p2[0]) + pad + 1, max(p1[1], p2[1]) + pad + 1,
                W, H
            )
            if x2 <= x1 or y2 <= y1:
                return None
            lm = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.line(lm, (p1[0] - x1, p1[1] - y1), (p2[0] - x1, p2[1] - y1), 255, max(1, int(thickness)))
            return x1, y1, x2, y2, lm

        def line_mask_ratio(p1, p2, mask, thickness=3):
            data = local_line_mask(p1, p2, thickness)
            if data is None:
                return 0.0
            x1, y1, x2, y2, lm = data
            total = cv2.countNonZero(lm)
            if total == 0:
                return 0.0
            return cv2.countNonZero(cv2.bitwise_and(lm, mask[y1:y2, x1:x2])) / float(total)

        def line_hits_wall(p1, p2, thickness=3):
            data = local_line_mask(p1, p2, thickness)
            if data is None:
                return True
            x1, y1, x2, y2, lm = data
            return cv2.countNonZero(cv2.bitwise_and(lm, wall_collision[y1:y2, x1:x2])) > 0

        def line_is_safe(p1, p2, mask=None, min_ratio=0.985, thickness=3):
            use_mask = valid_routing_mask if mask is None else mask
            return (not line_hits_wall(p1, p2, thickness=max(2, thickness))) and line_mask_ratio(p1, p2, use_mask, thickness=thickness) >= min_ratio

        # -------------------------------------------------
        # 2. 房間中心點
        # -------------------------------------------------
        room_coords = {}
        for m in metrics_list:
            rid = str(m['id'])
            if id_labels.get(rid, {}).get("portal", False):
                continue
            room_coords[rid] = (
                int(round(m['centroid'][0])),
                int(round(m['centroid'][1]))
            )

        # V14：房間投影的「直達性」證據。
        #
        # 舊版只比較房間中心到路網的歐氏距離，因此隔壁店家的路線即使更近，
        # 連線仍可能穿過另一個房間。導航語意上，正常門口應由本房間直接穿過
        # 一道外牆進入公共走道；因此所有投影入口共用這個檢查器：
        #   1) 線段不得穿過其他非 portal 房間；
        #   2) 記錄是否只穿過一個顯著牆帶，供品質診斷；
        #   3) 完全找不到直達候選時，才保留舊版最近點 fallback。
        navigation_room_ids = {
            int(rid) for rid in room_coords
            if str(rid).lstrip('-').isdigit()
        }

        def _count_significant_runs(flags, minimum_length):
            runs = 0
            run_len = 0
            for flag in np.asarray(flags, dtype=bool).tolist():
                if flag:
                    run_len += 1
                else:
                    if run_len >= minimum_length:
                        runs += 1
                    run_len = 0
            if run_len >= minimum_length:
                runs += 1
            return int(runs)

        def room_direct_connection_evidence(room_id, center, target):
            x0, y0 = map(int, center)
            x1, y1 = map(int, target)
            sample_count = max(abs(x1 - x0), abs(y1 - y0)) + 1
            if sample_count <= 1:
                return {
                    "direct_connection": True,
                    "single_wall_connection": True,
                    "crosses_other_room": False,
                    "crossed_room_ids": [],
                    "wall_crossing_runs": 0,
                }

            xs = np.clip(
                np.rint(np.linspace(x0, x1, sample_count)).astype(np.int32),
                0, W - 1,
            )
            ys = np.clip(
                np.rint(np.linspace(y0, y1, sample_count)).astype(np.int32),
                0, H - 1,
            )
            labels = res_matrix[ys, xs]
            own_id = int(room_id) if str(room_id).lstrip('-').isdigit() else None
            other_ids = sorted({
                int(value) for value in np.unique(labels)
                if int(value) in navigation_room_ids and int(value) != own_id
            })

            # 1px 雜點不視為一堵牆；門框/房間外牆則會形成連續牆帶。
            min_wall_run = max(2, int(round(corridor_width_est * 0.025)))
            wall_runs = _count_significant_runs(wall_matrix[ys, xs] > 0, min_wall_run)
            # 「不跨越其他房間」是硬優先條件；牆帶數保留為診斷資訊。
            # 掃描圖、文字殘影或雙線門框可能把同一道物理牆切成兩段，因此不能因
            # wall_runs>1 就改選一條確定穿越其他房間的線。
            direct = not other_ids
            return {
                "direct_connection": bool(direct),
                "single_wall_connection": bool(direct and wall_runs <= 1),
                "crosses_other_room": bool(other_ids),
                "crossed_room_ids": [str(v) for v in other_ids],
                "wall_crossing_runs": int(wall_runs),
            }

        def select_room_projection_candidate(room_id, center, candidates):
            """先取最近的直達候選；沒有時才退回候選提供的舊版 score。"""
            if not candidates:
                return None, None, False
            nearest = sorted(candidates, key=lambda c: (float(c["distance"]), c.get("tie", 0)))
            for candidate in nearest:
                evidence = room_direct_connection_evidence(room_id, center, candidate["point"])
                if evidence["direct_connection"]:
                    return candidate, evidence, False

            fallback = min(
                candidates,
                key=lambda c: (float(c.get("fallback_score", c["distance"])), float(c["distance"])),
            )
            evidence = room_direct_connection_evidence(room_id, center, fallback["point"])
            return fallback, evidence, True

        def store_attachment_visibility(info, evidence, fallback_used):
            evidence = evidence or {
                "direct_connection": False,
                "single_wall_connection": False,
                "crosses_other_room": False,
                "crossed_room_ids": [],
                "wall_crossing_runs": None,
            }
            info["direct_connection"] = bool(evidence.get("direct_connection", False))
            info["single_wall_connection"] = bool(evidence.get("single_wall_connection", False))
            info["crosses_other_room"] = bool(evidence.get("crosses_other_room", False))
            info["crossed_room_ids"] = list(evidence.get("crossed_room_ids", []))
            info["wall_crossing_runs"] = evidence.get("wall_crossing_runs")
            info["nearest_projection_fallback_used"] = bool(fallback_used)

        # -------------------------------------------------
        # FAST V9：Room -> public-space ownership
        # -------------------------------------------------
        # 目的：房間應優先掛到「自己所屬的公共空間路網」，而不是全圖最近的一條路。
        #
        # V8 的 recovery reattach 是從 recovered mask 往外 dilation 找鄰居；
        # 對靠近 primary/recovery 邊界的房間容易誤判，而且未被挑中的 recovery 房間
        # 仍會保留一開始投影到 primary graph 的 attachment。
        #
        # V9 改成先做 ownership：對每個 public region 建 distance transform，
        # 再用「房間邊界的 robust distance + boundary support」決定 owner。
        # 這不是固定色彩規則，也不需要門偵測；只是回答：
        #   這個房間在幾何上主要面向哪一塊 public free-space？
        public_region_ids = sorted(set(int(v) for v in primary_corridor_ids + recovered_corridor_ids))
        public_distance_maps = {}
        for pid in public_region_ids:
            pmask = (res_matrix == int(pid)).astype(np.uint8)
            if np.any(pmask):
                public_distance_maps[int(pid)] = cv2.distanceTransform(
                    (pmask == 0).astype(np.uint8), cv2.DIST_L2, 5
                )

        owner_probe_radius = float(np.clip(corridor_width_est * 0.72, 16.0, 52.0))
        owner_support_band = float(np.clip(corridor_width_est * 0.12, 3.0, 10.0))
        room_public_owner = {}
        room_public_owner_report = []

        for room_id, center in room_coords.items():
            room_mask = (res_matrix == int(room_id)).astype(np.uint8)
            if not np.any(room_mask):
                room_public_owner[room_id] = None
                continue

            eroded = cv2.erode(room_mask, np.ones((3, 3), np.uint8), iterations=1)
            boundary = (room_mask > 0) & (eroded == 0)
            by, bx = np.where(boundary)
            if bx.size == 0:
                by, bx = np.where(room_mask > 0)

            candidates = []
            for pid, dist_map in public_distance_maps.items():
                vals = dist_map[by, bx]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                q05 = float(np.percentile(vals, 5))
                q20 = float(np.percentile(vals, 20))
                support_limit = q05 + owner_support_band
                support_ratio = float(np.mean(vals <= support_limit))

                # q20 比單一最近點穩健：一個角落擦到另一條走道，不應壓過
                # 一整段房間邊界真正面向的 public region。
                score = q20 + 0.20 * q05 - 0.45 * owner_support_band * support_ratio
                candidates.append({
                    "public_id": int(pid),
                    "score": float(score),
                    "q05_gap_px": q05,
                    "q20_gap_px": q20,
                    "support_ratio": support_ratio,
                    "recovered": bool(int(pid) in set(recovered_corridor_ids)),
                })

            candidates.sort(key=lambda r: (r["score"], r["q20_gap_px"], -r["support_ratio"]))
            owner = None
            confidence = "unresolved"
            if candidates:
                best = candidates[0]
                # 至少要有一段房間邊界真的靠近該 public region；
                # 避免很遠的巨大 open area 因面積大而搶走房間。
                if best["q05_gap_px"] <= owner_probe_radius:
                    owner = int(best["public_id"])
                    if len(candidates) == 1:
                        confidence = "single_candidate"
                    else:
                        margin = float(candidates[1]["score"] - best["score"])
                        confidence = "strong" if margin >= owner_support_band * 0.70 else "close_boundary"

            room_public_owner[room_id] = owner
            if owner is not None and room_id in id_labels:
                id_labels[room_id]["navigation_public_owner"] = int(owner)
                id_labels[room_id]["navigation_public_owner_kind"] = (
                    "recovered_public" if int(owner) in set(recovered_corridor_ids) else "primary_public"
                )

            room_public_owner_report.append({
                "room_id": str(room_id),
                "center": [int(center[0]), int(center[1])],
                "owner_public_id": int(owner) if owner is not None else None,
                "owner_kind": (
                    "recovered_public" if owner is not None and int(owner) in set(recovered_corridor_ids)
                    else "primary_public" if owner is not None
                    else "unresolved"
                ),
                "confidence": confidence,
                "candidates": [
                    {
                        "public_id": int(r["public_id"]),
                        "score": round(float(r["score"]), 3),
                        "q05_gap_px": round(float(r["q05_gap_px"]), 3),
                        "q20_gap_px": round(float(r["q20_gap_px"]), 3),
                        "support_ratio": round(float(r["support_ratio"]), 4),
                        "recovered": bool(r["recovered"]),
                    }
                    for r in candidates[:4]
                ],
            })

        with open(self.output_dir / "room_public_owner_report_v9.json", "w", encoding="utf-8") as f:
            json.dump(room_public_owner_report, f, ensure_ascii=False, indent=2)

        recovered_owned_count = sum(
            1 for v in room_public_owner.values() if v is not None and int(v) in set(recovered_corridor_ids)
        )
        primary_owned_count = sum(
            1 for v in room_public_owner.values() if v is not None and int(v) in set(primary_corridor_ids)
        )
        unresolved_owned_count = sum(1 for v in room_public_owner.values() if v is None)
        print(
            f"[房間歸屬 V9] primary={primary_owned_count}，recovery={recovered_owned_count}，"
            f"unresolved={unresolved_owned_count}，probe≈{owner_probe_radius:.1f}px"
        )

        # Ownership debug：
        #   灰 = primary public space
        #   橘 = recovered public space
        #   綠 = primary-owned room
        #   青 = recovery-owned room
        #   紅 = unresolved room
        owner_debug = np.zeros((H, W, 3), dtype=np.uint8)
        owner_debug[primary_corridor_mask > 0] = (190, 190, 190)
        owner_debug[recovered_corridor_mask > 0] = (0, 180, 255)
        for room_id, owner in room_public_owner.items():
            room_pixels = (res_matrix == int(room_id))
            if owner is None:
                owner_debug[room_pixels] = (0, 0, 180)
            elif int(owner) in set(recovered_corridor_ids):
                owner_debug[room_pixels] = (255, 220, 80)
            else:
                owner_debug[room_pixels] = (80, 200, 80)
            cx, cy = room_coords[room_id]
            cv2.putText(
                owner_debug,
                f"R{room_id}->P{owner if owner is not None else '?'}",
                (max(0, cx - 28), max(12, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        owner_debug_path = self.output_dir / "debug_room_public_owner_v9.jpg"
        safe_imwrite(str(owner_debug_path), owner_debug)

        # -------------------------------------------------
        # 3. 走道骨架候選點：角點 + 端點 + junction
        # -------------------------------------------------
        try:
            skeleton = cv2.ximgproc.thinning(pure_corridor_mask)
        except AttributeError:
            skeleton = _skeletonize_uint8(pure_corridor_mask)

        corner_pts = []
        corners = cv2.goodFeaturesToTrack(
            skeleton,
            maxCorners=220,
            qualityLevel=0.01,
            minDistance=26
        )
        if corners is not None:
            corner_pts.extend(tuple(map(int, pt[0])) for pt in corners)

        sk01 = (skeleton > 0).astype(np.uint8)
        neigh = cv2.filter2D(
            sk01,
            cv2.CV_16S,
            np.ones((3, 3), np.int16),
            borderType=cv2.BORDER_CONSTANT
        ) - sk01.astype(np.int16)
        structural_mask = ((sk01 > 0) & ((neigh == 1) | (neigh >= 3))).astype(np.uint8) * 255
        structural_mask = cv2.dilate(structural_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n_cc, _, cc_stats, cc_centroids = cv2.connectedComponentsWithStats((structural_mask > 0).astype(np.uint8), 8)
        for i in range(1, n_cc):
            if cc_stats[i, cv2.CC_STAT_AREA] >= 1:
                corner_pts.append((int(round(cc_centroids[i][0])), int(round(cc_centroids[i][1]))))

        # V11：只靠 corner / endpoint 在「長直走道、緩彎走道、色塊斷裂」時可能沒有足夠 seed，
        # 導致本來同一條走道被投射成數個互不相交的短線。沿 skeleton 做稀疏且尺度自適應的補點。
        skeleton_sample_spacing = int(np.clip(
            round(corridor_width_est * 0.75),
            24,
            max(32, round(map_short_side * 0.055)),
        ))
        sk_yx = np.argwhere(skeleton > 0)
        if sk_yx.size:
            buckets = {}
            for sy, sx in sk_yx:
                key = (int(sx) // skeleton_sample_spacing, int(sy) // skeleton_sample_spacing)
                clearance = float(corridor_inside_distance[int(sy), int(sx)])
                old = buckets.get(key)
                if old is None or clearance > old[0]:
                    buckets[key] = (clearance, (int(sx), int(sy)))
            corner_pts.extend(rec[1] for rec in buckets.values())

        # 去除非常接近的候選點；半徑也隨走道寬度縮放，避免高解析度地圖點過密。
        candidate_dedupe_radius = int(np.clip(round(corridor_width_est * 0.22), 10, 20))
        corner_pts = _dedupe_points_by_radius(corner_pts, radius=candidate_dedupe_radius)
        if not corner_pts:
            print("[警告] 找不到走道轉角或骨架結構點，無法建立路網！")
            return

        def snap_vals(vals, tolerance=18):
            sorted_v = sorted(set(int(v) for v in vals))
            mapping, group = {}, []
            for v in sorted_v:
                if not group or v - group[-1] <= tolerance:
                    group.append(v)
                else:
                    med = int(round(float(np.median(group))))
                    for gv in group:
                        mapping[gv] = med
                    group = [v]
            if group:
                med = int(round(float(np.median(group))))
                for gv in group:
                    mapping[gv] = med
            return mapping

        snap_tolerance = int(np.clip(round(corridor_width_est * 0.28), 8, 22))
        snap_x = snap_vals([p[0] for p in corner_pts], tolerance=snap_tolerance)
        snap_y = snap_vals([p[1] for p in corner_pts], tolerance=snap_tolerance)
        snapped_pts = []
        for x, y in corner_pts:
            sx, sy = snap_x.get(x, x), snap_y.get(y, y)
            if point_is_safe((sx, sy)):
                snapped_pts.append((sx, sy))
            elif point_is_safe((x, y)):
                snapped_pts.append((x, y))
        snapped_pts = list(set(snapped_pts))

        # -------------------------------------------------
        # 4. 從候選點投射水平/垂直走道線
        # -------------------------------------------------
        raw_h_segs, raw_v_segs = [], []
        min_raw_len = int(np.clip(round(corridor_width_est * 0.28), 12, 28))
        for cx, cy in snapped_pts:
            if not point_is_safe((cx, cy)):
                continue

            rx = cx
            while rx < W and valid_routing_mask[cy, rx] > 0:
                rx += 1
            lx = cx
            while lx >= 0 and valid_routing_mask[cy, lx] > 0:
                lx -= 1
            if (rx - 1) - (lx + 1) >= min_raw_len:
                raw_h_segs.append((cy, lx + 1, rx - 1))

            dy = cy
            while dy < H and valid_routing_mask[dy, cx] > 0:
                dy += 1
            uy = cy
            while uy >= 0 and valid_routing_mask[uy, cx] > 0:
                uy -= 1
            if (dy - 1) - (uy + 1) >= min_raw_len:
                raw_v_segs.append((cx, uy + 1, dy - 1))

        raw_segment_count = len(raw_h_segs) + len(raw_v_segs)

        # -------------------------------------------------
        # 5. 平行路線刪減
        #    僅當兩線之間沒有牆、而且中間大多是同一片走道時才合併。
        #    代表線不使用座標平均，而是選牆距最大、遮罩最完整的位置。
        # -------------------------------------------------
        def interval_overlap(a1, a2, b1, b2):
            return max(0, min(a2, b2) - max(a1, b1) + 1)

        def merge_spans(spans, gap=8):
            if not spans:
                return []
            spans = sorted((int(a), int(b)) for a, b in spans if b >= a)
            out = []
            a, b = spans[0]
            for na, nb in spans[1:]:
                if na <= b + gap:
                    b = max(b, nb)
                else:
                    out.append((a, b))
                    a, b = na, nb
            out.append((a, b))
            return out

        def same_open_corridor(seg_a, seg_b, is_horizontal, threshold):
            """
            V8：只合併「非常接近、重疊度高、且中間整帶都可通行」的重複平行線。

            V7 的 threshold=52 加上傳遞式 connected-component 分組，會出現 A 接近 B、
            B 接近 C，最後 A/B/C 全部被合併的鏈式效應；寬走道兩側或環狀結構因此可能
            被壓成單線。V8 把合併限制為局部重複線，不碰真正的平行分支。
            """
            fa, a1, a2 = map(int, seg_a)
            fb, b1, b2 = map(int, seg_b)
            fixed_gap = abs(fa - fb)
            if fixed_gap > int(threshold):
                return False

            len_a = max(1, a2 - a1 + 1)
            len_b = max(1, b2 - b1 + 1)
            ov1, ov2 = max(a1, b1), min(a2, b2)
            overlap = ov2 - ov1 + 1
            overlap_ratio = overlap / float(max(1, min(len_a, len_b)))
            if overlap < 30 or overlap_ratio < 0.78:
                return False

            # 兩條線端點差異過大時通常是不同支線，不視為同一路線的重複取樣。
            endpoint_drift = abs(a1 - b1) + abs(a2 - b2)
            if endpoint_drift > max(34, int(0.30 * max(len_a, len_b))):
                return False

            if is_horizontal:
                y1, y2 = sorted((fa, fb))
                strip_valid = valid_routing_mask[y1:y2 + 1, ov1:ov2 + 1]
                strip_wall = wall_collision[y1:y2 + 1, ov1:ov2 + 1]
            else:
                x1, x2 = sorted((fa, fb))
                strip_valid = valid_routing_mask[ov1:ov2 + 1, x1:x2 + 1]
                strip_wall = wall_collision[ov1:ov2 + 1, x1:x2 + 1]

            if strip_valid.size == 0:
                return False
            wall_ratio = float(np.mean(strip_wall > 0))
            valid_ratio = float(np.mean(strip_valid > 0))
            if wall_ratio > 0.001 or valid_ratio < 0.86:
                return False

            # 在重疊區抽樣多個橫切面。至少 80% 的橫切面必須從一條線完整通到另一條線。
            sample_count = min(9, max(5, overlap // 35))
            sample_positions = np.linspace(ov1, ov2, sample_count).astype(int)
            pass_count = 0
            for p in sample_positions:
                if is_horizontal:
                    cross_valid = valid_routing_mask[min(fa, fb):max(fa, fb) + 1, p]
                    cross_wall = wall_collision[min(fa, fb):max(fa, fb) + 1, p]
                else:
                    cross_valid = valid_routing_mask[p, min(fa, fb):max(fa, fb) + 1]
                    cross_wall = wall_collision[p, min(fa, fb):max(fa, fb) + 1]
                if cross_valid.size and np.all(cross_valid > 0) and not np.any(cross_wall > 0):
                    pass_count += 1
            return pass_count >= int(math.ceil(sample_count * 0.80))

        def split_safe_runs(fixed, p1, p2, is_horizontal, min_len=18):
            if p2 < p1:
                return []
            if is_horizontal:
                vals = (
                    (valid_routing_mask[fixed, p1:p2 + 1] > 0)
                    & (wall_collision[fixed, p1:p2 + 1] == 0)
                )
            else:
                vals = (
                    (valid_routing_mask[p1:p2 + 1, fixed] > 0)
                    & (wall_collision[p1:p2 + 1, fixed] == 0)
                )

            runs = []
            start = None
            for idx, ok in enumerate(vals.tolist() + [False]):
                if ok and start is None:
                    start = idx
                elif not ok and start is not None:
                    end = idx - 1
                    if end - start + 1 >= min_len:
                        runs.append((fixed, p1 + start, p1 + end))
                    start = None
            return runs

        def representative_score(fixed, spans, is_horizontal):
            valid_count = 0
            total = 0
            clear_values = []
            for p1, p2 in spans:
                if is_horizontal:
                    if not (0 <= fixed < H):
                        continue
                    v = valid_routing_mask[fixed, p1:p2 + 1] > 0
                    c = wall_distance[fixed, p1:p2 + 1]
                else:
                    if not (0 <= fixed < W):
                        continue
                    v = valid_routing_mask[p1:p2 + 1, fixed] > 0
                    c = wall_distance[p1:p2 + 1, fixed]
                total += int(v.size)
                valid_count += int(np.count_nonzero(v))
                if np.any(v):
                    clear_values.extend(c[v].tolist())
            if total == 0:
                return -1e9
            ratio = valid_count / float(total)
            clearance = float(np.median(clear_values)) if clear_values else 0.0
            return ratio * 1000.0 + min(clearance, 40.0) * 12.0

        def collapse_parallel_segments(segs, is_horizontal, threshold=16):
            """
            V8 非傳遞式群組：
            - 群組總寬不能超過 threshold；
            - 新線必須和群組內每一條線都通過 same_open_corridor；
            - 代表座標只能從原始線座標中挑選，不再在中間憑空建立新線。
            """
            if not segs:
                return []
            segs = sorted(set(tuple(map(int, s)) for s in segs), key=lambda s: (s[0], s[1], s[2]))
            groups = []
            for seg in segs:
                placed = False
                for group in groups:
                    fixed_values = [g[0] for g in group]
                    new_min = min(min(fixed_values), seg[0])
                    new_max = max(max(fixed_values), seg[0])
                    if new_max - new_min > threshold:
                        continue
                    if all(same_open_corridor(seg, g, is_horizontal, threshold) for g in group):
                        group.append(seg)
                        placed = True
                        break
                if not placed:
                    groups.append([seg])

            collapsed = []
            for group in groups:
                if len(group) == 1:
                    collapsed.extend(split_safe_runs(group[0][0], group[0][1], group[0][2], is_horizontal, min_len=18))
                    continue

                spans = merge_spans([(s[1], s[2]) for s in group], gap=6)
                # 只在原本存在的座標中選最佳代表線，避免平均/搜尋到另一條結構中間。
                candidate_fixed = sorted(set(s[0] for s in group))
                best_fixed = max(candidate_fixed, key=lambda f: representative_score(f, spans, is_horizontal))
                for p1, p2 in spans:
                    collapsed.extend(split_safe_runs(best_fixed, p1, p2, is_horizontal, min_len=18))

            by_fixed = {}
            for fixed, p1, p2 in collapsed:
                by_fixed.setdefault(fixed, []).append((p1, p2))
            result = []
            for fixed, spans in by_fixed.items():
                for p1, p2 in merge_spans(spans, gap=4):
                    result.append((fixed, p1, p2))
            return sorted(set(result))

        def exact_coordinate_cleanup(segs):
            """只清除完全同座標/同方向的重疊線，不合併不同平行分支。"""
            by_fixed = {}
            for fixed, p1, p2 in set(tuple(map(int, s)) for s in segs):
                by_fixed.setdefault(fixed, []).append((p1, p2))
            out = []
            for fixed, spans in by_fixed.items():
                for p1, p2 in merge_spans(spans, gap=3):
                    out.append((fixed, p1, p2))
            return sorted(out)

        # 先嘗試 16px 的保守合併；若整體刪除超過 48%，自動降低強度。
        # 這個保留率防線可避免左側環狀結構被一次壓成單線。
        raw_h_unique = exact_coordinate_cleanup(raw_h_segs)
        raw_v_unique = exact_coordinate_cleanup(raw_v_segs)
        base_unique_count = len(raw_h_unique) + len(raw_v_unique)
        parallel_threshold_used = 0
        h_segs, v_segs = raw_h_unique, raw_v_unique
        for candidate_threshold in (16, 12, 8):
            cand_h = collapse_parallel_segments(raw_h_unique, is_horizontal=True, threshold=candidate_threshold)
            cand_v = collapse_parallel_segments(raw_v_unique, is_horizontal=False, threshold=candidate_threshold)
            candidate_count = len(cand_h) + len(cand_v)
            retention = candidate_count / float(max(1, base_unique_count))
            if retention >= 0.52:
                h_segs, v_segs = cand_h, cand_v
                parallel_threshold_used = candidate_threshold
                break

        reduced_segment_count = len(h_segs) + len(v_segs)
        parallel_retention_ratio = reduced_segment_count / float(max(1, base_unique_count))
        h_by_y = {}
        v_by_x = {}
        for seg in h_segs:
            h_by_y.setdefault(seg[0], []).append(seg)
        for seg in v_segs:
            v_by_x.setdefault(seg[0], []).append(seg)

        # -------------------------------------------------
        # 6. 建立基礎道路節點：線端點、交叉點，以及仍落在線上的骨架候選點
        # -------------------------------------------------
        grid_nodes = set()
        for y, x1, x2 in h_segs:
            if point_is_safe((x1, y)):
                grid_nodes.add((x1, y))
            if point_is_safe((x2, y)):
                grid_nodes.add((x2, y))
        for x, y1, y2 in v_segs:
            if point_is_safe((x, y1)):
                grid_nodes.add((x, y1))
            if point_is_safe((x, y2)):
                grid_nodes.add((x, y2))

        for y, hx1, hx2 in h_segs:
            for x, vy1, vy2 in v_segs:
                if hx1 <= x <= hx2 and vy1 <= y <= vy2 and point_is_safe((x, y)):
                    grid_nodes.add((x, y))

        def point_on_any_segment(pt, tolerance=1):
            x, y = map(int, pt)
            for sy in range(y - tolerance, y + tolerance + 1):
                for _, x1, x2 in h_by_y.get(sy, ()): 
                    if x1 <= x <= x2:
                        return True
            for sx in range(x - tolerance, x + tolerance + 1):
                for _, y1, y2 in v_by_x.get(sx, ()): 
                    if y1 <= y <= y2:
                        return True
            return False

        for pt in snapped_pts:
            if point_is_safe(pt) and point_on_any_segment(pt):
                grid_nodes.add(pt)

        # -------------------------------------------------
        # 7. 房間安全抵達點
        #    幾何投影後，沿著該路線搜尋最近且位於 safe_attachment_mask 的點。
        # -------------------------------------------------
        room_attachments = {}

        def nearest_safe_point_on_routes(room_id, rx, ry):
            candidates = {}

            def consider(pt):
                x, y = int(pt[0]), int(pt[1])
                if not point_is_safe((x, y), clearance=min_attachment_clearance, mask=safe_attachment_mask):
                    return
                d = math.hypot(rx - x, ry - y)
                key = (x, y)
                old = candidates.get(key)
                if old is None or d < old["distance"]:
                    candidates[key] = {
                        "point": key,
                        "distance": float(d),
                        "fallback_score": float(d),
                    }

            for y, x1, x2 in h_segs:
                px = int(np.clip(rx, x1, x2))
                max_offset = max(px - x1, x2 - px)
                found = False
                for offset in range(max_offset + 1):
                    for x in (px - offset, px + offset):
                        if x1 <= x <= x2 and point_is_safe((x, y), clearance=min_attachment_clearance, mask=safe_attachment_mask):
                            consider((x, y))
                            found = True
                    if found:
                        break

            for x, y1, y2 in v_segs:
                py = int(np.clip(ry, y1, y2))
                max_offset = max(py - y1, y2 - py)
                found = False
                for offset in range(max_offset + 1):
                    for y in (py - offset, py + offset):
                        if y1 <= y <= y2 and point_is_safe((x, y), clearance=min_attachment_clearance, mask=safe_attachment_mask):
                            consider((x, y))
                            found = True
                    if found:
                        break

            candidate, evidence, fallback_used = select_room_projection_candidate(
                room_id,
                (rx, ry),
                list(candidates.values()),
            )
            if candidate is None:
                return None, float('inf'), evidence, fallback_used
            return candidate["point"], float(candidate["distance"]), evidence, fallback_used

        for rid, center in room_coords.items():
            owner_public_id = room_public_owner.get(rid)
            owner_is_recovered = (
                owner_public_id is not None and int(owner_public_id) in set(recovered_corridor_ids)
            )

            # V9：recovery-owned room 不允許先掛到 primary graph。
            # 其 attachment 先保留 pending，等自己的 local recovery graph 完成後再處理。
            if owner_is_recovered:
                attach_pt, attach_dist = None, None
                attachment_evidence, attachment_fallback = None, False
                initial_method = f"pending_recovered_public_{int(owner_public_id)}"
            else:
                attach_pt, attach_dist, attachment_evidence, attachment_fallback = nearest_safe_point_on_routes(
                    rid, *center
                )
                if attach_pt is None:
                    initial_method = "pending_graph_fallback"
                elif attachment_fallback:
                    initial_method = "safe_route_projection_fallback_nearest"
                else:
                    initial_method = "safe_route_projection_direct"

            info = {
                "room_center": center,
                "attachment_point": attach_pt,
                "attachment_node": None,
                "distance_px": round(float(attach_dist), 2) if attach_pt is not None else None,
                "safe_clearance_px": (
                    round(float(wall_distance[attach_pt[1], attach_pt[0]]), 2)
                    if attach_pt is not None else None
                ),
                "attachment_method": initial_method,
                "public_owner_id": int(owner_public_id) if owner_public_id is not None else None,
                "public_owner_kind": (
                    "recovered_public" if owner_is_recovered
                    else "primary_public" if owner_public_id is not None
                    else "unresolved"
                ),
            }
            store_attachment_visibility(info, attachment_evidence, attachment_fallback)
            room_attachments[rid] = info
            if attach_pt is not None:
                grid_nodes.add(attach_pt)

        def _skeleton_chains_from_mask(safe_mask):
            """在單一 recovered region 內把 1px skeleton 壓縮成 polyline chains。"""
            ys, xs = np.where(safe_mask > 0)
            if xs.size == 0:
                return [], np.zeros_like(safe_mask)
            pad = 3
            x1, y1, x2, y2 = _clip_box(
                int(xs.min()) - pad, int(ys.min()) - pad,
                int(xs.max()) + pad + 1, int(ys.max()) + pad + 1,
                W, H
            )
            local_safe = safe_mask[y1:y2, x1:x2]
            # Recovery graph 需要真正 1px skeleton。既有 _skeletonize_uint8 的 morphology fallback
            # 在沒有 opencv-contrib/ximgproc 時可能留下 2px ridge，會讓 degree 爆增成大量假 junction。
            try:
                from skimage.morphology import skeletonize as _sk_skeletonize
                local_skel = (_sk_skeletonize(local_safe > 0).astype(np.uint8) * 255)
            except Exception:
                try:
                    local_skel = cv2.ximgproc.thinning(((local_safe > 0).astype(np.uint8) * 255))
                except AttributeError:
                    local_skel = _skeletonize_uint8(local_safe)
            syx = np.argwhere(local_skel > 0)
            if syx.size == 0:
                return [], np.zeros_like(safe_mask)

            pixels = {(int(x), int(y)) for y, x in syx}
            offsets = (
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),            (1, 0),
                (-1, 1),  (0, 1),   (1, 1),
            )
            neigh = {}
            for p in pixels:
                px, py = p
                ns = []
                for dx, dy in offsets:
                    q = (px + dx, py + dy)
                    if q in pixels:
                        ns.append(q)
                neigh[p] = ns

            keys = {p for p, ns in neigh.items() if len(ns) != 2}
            # 純 cycle 沒有 degree!=2 的點；人工挑一點作 trace anchor。
            if not keys and pixels:
                keys.add(next(iter(pixels)))

            # V14：8-neighbour skeleton 在同一個實體交叉口周圍，常會產生一小團
            # degree>=3 像素。舊版把團內每條 1px 邊都轉成 recovery route，最後形成
            # 小型方格/鋸齒。先將相鄰 structural pixels 收斂成一個代表點，後續仍保留
            # 每一條真正離開交叉口的 branch，但不保留交叉口內部的像素級網格。
            key_representative = {}
            if keys:
                key_mask = np.zeros_like(local_skel, dtype=np.uint8)
                for kx, ky in keys:
                    key_mask[ky, kx] = 1
                key_count, key_labels, _, key_centroids = cv2.connectedComponentsWithStats(
                    key_mask, 8
                )
                for label_idx in range(1, key_count):
                    cyx = np.argwhere(key_labels == label_idx)
                    if cyx.size == 0:
                        continue
                    cx, cy = key_centroids[label_idx]
                    best_idx = int(np.argmin(
                        (cyx[:, 1].astype(np.float32) - float(cx)) ** 2
                        + (cyx[:, 0].astype(np.float32) - float(cy)) ** 2
                    ))
                    representative = (
                        int(cyx[best_idx, 1]),
                        int(cyx[best_idx, 0]),
                    )
                    for ky, kx in cyx:
                        key_representative[(int(kx), int(ky))] = representative

            visited_edges = set()
            chains = []

            def edge_key(a, b):
                return (a, b) if a <= b else (b, a)

            # 從 structural/key pixels 往外 trace。
            for start in list(keys):
                for nxt in neigh.get(start, ()):
                    ek = edge_key(start, nxt)
                    if ek in visited_edges:
                        continue
                    visited_edges.add(ek)
                    chain = [start, nxt]
                    prev, cur = start, nxt
                    guard = 0
                    while cur not in keys and guard < len(pixels) + 4:
                        guard += 1
                        options = [q for q in neigh.get(cur, ()) if q != prev]
                        if not options:
                            break
                        q = options[0]
                        ek2 = edge_key(cur, q)
                        if ek2 in visited_edges:
                            break
                        visited_edges.add(ek2)
                        chain.append(q)
                        prev, cur = cur, q
                    if len(chain) >= 2:
                        chains.append(chain)

            # 仍未 trace 的 closed fragments / junction-cluster內部邊也補上。
            for a, ns in neigh.items():
                for b in ns:
                    ek = edge_key(a, b)
                    if ek in visited_edges:
                        continue
                    visited_edges.add(ek)
                    chains.append([a, b])

            global_skel = np.zeros_like(safe_mask)
            gy, gx = np.where(local_skel > 0)
            global_skel[gy + y1, gx + x1] = 255

            out = []
            epsilon = float(np.clip(corridor_width_est * 0.08, 2.5, 7.0))
            for chain in chains:
                chain = list(chain)
                if chain[0] in key_representative:
                    chain[0] = key_representative[chain[0]]
                if chain[-1] in key_representative:
                    chain[-1] = key_representative[chain[-1]]
                # 同一 junction cluster 內的殘餘 1px 邊不是導航分支。
                if chain[0] == chain[-1] and len(chain) <= 3:
                    continue
                raw = [(int(px + x1), int(py + y1)) for px, py in chain]
                if len(raw) >= 2:
                    out.append(raw)
            return out, global_skel

        # -------------------------------------------------
        # 8. 建立 NetworkX 基礎路網
        # -------------------------------------------------
        G = nx.Graph()
        nodes_data, wp_node_map = {}, {}
        wp_counter = 0

        def add_waypoint(pt, node_type="waypoint"):
            nonlocal wp_counter
            pt = (int(pt[0]), int(pt[1]))
            if pt in wp_node_map:
                return wp_node_map[pt]
            wid = f"W_{wp_counter}"
            wp_counter += 1
            wp_node_map[pt] = wid
            nodes_data[wid] = {
                "type": node_type,
                "name": f"走道點_{wp_counter}",
                "coordinates": [pt[0], pt[1]]
            }
            G.add_node(wid, pos=pt)
            return wid

        def add_graph_edge(u, v, edge_type="route", mask_ratio=None, **metadata):
            if u == v:
                return
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']

            # V15 strict-H/V invariant：所有「實體導航道路」只能水平或垂直。
            # recovery_gateway 是語意跨區提示，不代表實際道路，因此不套用此限制。
            physical_edge_types = {"route", "recovery_route", "shortcut", "component_bridge"}
            if edge_type in physical_edge_types and not (
                int(p1[0]) == int(p2[0]) or int(p1[1]) == int(p2[1])
            ):
                print(
                    f"[Graph V15] 拒絕斜向 {edge_type} edge: "
                    f"{tuple(map(int, p1))} -> {tuple(map(int, p2))}"
                )
                return

            dist = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            attrs = {"edge_type": edge_type, "weight": dist}
            if mask_ratio is not None:
                attrs["mask_ratio"] = round(float(mask_ratio), 3)
            # V8：保留 topology / semantic-gateway metadata 到 canonical graph。
            # None 不寫入，避免舊 JSON 被大量空欄位污染。
            attrs.update({k: v for k, v in metadata.items() if v is not None})
            if G.has_edge(u, v):
                old = G.edges[u, v]
                # 保留較高語意等級的邊；同級邊允許補上 topology metadata。
                priority = {"route": 0, "recovery_route": 0, "shortcut": 1, "component_bridge": 2, "recovery_gateway": 3}
                if priority.get(edge_type, 0) >= priority.get(old.get("edge_type", "route"), 0):
                    old.update(attrs)
                old["weight"] = min(float(old.get("weight", dist)), dist)
            else:
                G.add_edge(u, v, **attrs)

        for pt in sorted(grid_nodes):
            add_waypoint(pt)

        nodes_by_y = {}
        nodes_by_x = {}
        for pt in grid_nodes:
            nodes_by_y.setdefault(pt[1], []).append(pt)
            nodes_by_x.setdefault(pt[0], []).append(pt)
        for pts in nodes_by_y.values():
            pts.sort(key=lambda p: p[0])
        for pts in nodes_by_x.values():
            pts.sort(key=lambda p: p[1])

        for y, x1, x2 in h_segs:
            pts = [pt for pt in nodes_by_y.get(y, ()) if x1 <= pt[0] <= x2]
            for p1, p2 in zip(pts[:-1], pts[1:]):
                if line_is_safe(p1, p2, min_ratio=0.995, thickness=3):
                    add_graph_edge(add_waypoint(p1), add_waypoint(p2), edge_type="route")

        for x, y1, y2 in v_segs:
            pts = [pt for pt in nodes_by_x.get(x, ()) if y1 <= pt[1] <= y2]
            for p1, p2 in zip(pts[:-1], pts[1:]):
                if line_is_safe(p1, p2, min_ratio=0.995, thickness=3):
                    add_graph_edge(add_waypoint(p1), add_waypoint(p2), edge_type="route")

        # 刪除未落在任何有效線段上的孤立候選點，避免後續被橘線錯誤強接。
        for nid in list(G.nodes()):
            if G.degree(nid) == 0:
                pt = tuple(G.nodes[nid]['pos'])
                G.remove_node(nid)
                nodes_data.pop(nid, None)
                wp_node_map.pop(pt, None)

        # V9: every accepted public region must have real road geometry. Sparse
        # global corner sampling can miss a narrow component beside a wide hall.
        coverage_raster = np.zeros((H, W), np.uint8)
        for u, v in G.edges():
            cv2.line(coverage_raster, tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos']), 255, 1)
        missing_public_graph_records = []
        coverage_start = time.perf_counter()
        for public_id in primary_corridor_ids:
            region = (res_matrix == public_id)
            if np.count_nonzero(region & (coverage_raster > 0)) >= 3:
                continue
            if time.perf_counter() - coverage_start > 4.0:
                missing_public_graph_records.append({'rid': int(public_id), 'status': 'budget_exhausted'})
                continue
            local_safe = (region & (wall_collision == 0)).astype(np.uint8) * 255
            chains, _ = _skeleton_chains_from_mask(local_safe)
            added = 0
            for chain in chains:
                points = _rectilinear_chain_path(chain, local_safe)
                if not points or len(points) < 2:
                    continue
                length = sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(points[:-1],points[1:]))
                if length < max(4, corridor_width_est*.06):
                    continue
                # Keep verified H/V centerlines, including independent components.
                previous = add_waypoint(points[0])
                for point in points[1:]:
                    current = add_waypoint(point)
                    add_graph_edge(previous,current,edge_type='route',public_region_id=int(public_id))
                    previous = current
                    added += 1
            missing_public_graph_records.append({'rid':int(public_id),'added_edges':added,
                'status':'local_skeleton_installed' if added else 'no_safe_skeleton'})
        (self.output_dir/'public_graph_seed_coverage_v9.json').write_text(
            json.dumps(missing_public_graph_records,indent=2),encoding='utf-8')

        # V8：任何房間都不能因為投影點失效而被刪除。
        # 若原始投影沒有成功掛到路網，改投影到「目前真正存在的圖邊」，必要時切分該邊。
        def project_point_to_segment(point, a, b):
            px, py = point
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            denom = float(vx * vx + vy * vy)
            if denom <= 1e-9:
                return (int(ax), int(ay))
            t = ((px - ax) * vx + (py - ay) * vy) / denom
            t = float(np.clip(t, 0.0, 1.0))
            return (int(round(ax + t * vx)), int(round(ay + t * vy)))

        def attach_room_to_existing_graph(room_id, center, primary_only=False):
            if G.number_of_edges() == 0:
                return None
            candidates = []
            # 優先使用原始 route；若沒有才容許人工邊。
            edge_priority = {"route": 0, "shortcut": 1, "component_bridge": 2}
            for u, v, edata in list(G.edges(data=True)):
                et = str(edata.get("edge_type", "route"))
                if primary_only and et.startswith("recovery_"):
                    continue
                p1 = tuple(G.nodes[u]['pos'])
                p2 = tuple(G.nodes[v]['pos'])
                proj = project_point_to_segment(center, p1, p2)
                if not point_is_safe(proj, clearance=min_attachment_clearance, mask=safe_attachment_mask):
                    continue
                d = math.hypot(center[0] - proj[0], center[1] - proj[1])
                score = d + edge_priority.get(et, 3) * 1000.0
                candidates.append({
                    "point": proj,
                    "distance": float(d),
                    "fallback_score": float(score),
                    "tie": int(edge_priority.get(et, 3)),
                    "u": u,
                    "v": v,
                    "edge_data": dict(edata),
                })

            selected, evidence, fallback_used = select_room_projection_candidate(
                room_id, center, candidates
            )
            if selected is None:
                safe_nodes = [
                    n for n in G.nodes()
                    if point_is_safe(G.nodes[n]['pos'], clearance=min_attachment_clearance, mask=safe_attachment_mask)
                    and (
                        not primary_only
                        or nodes_data.get(n, {}).get("type") not in {"recovery_waypoint", "recovery_gateway"}
                    )
                ]
                if not safe_nodes:
                    safe_nodes = [
                        n for n in G.nodes()
                        if (
                            not primary_only
                            or nodes_data.get(n, {}).get("type") not in {"recovery_waypoint", "recovery_gateway"}
                        )
                    ]
                if not safe_nodes:
                    return None
                node_candidates = [
                    {
                        "point": tuple(G.nodes[n]['pos']),
                        "distance": math.hypot(
                            G.nodes[n]['pos'][0] - center[0],
                            G.nodes[n]['pos'][1] - center[1],
                        ),
                        "node": n,
                    }
                    for n in safe_nodes
                ]
                selected, evidence, fallback_used = select_room_projection_candidate(
                    room_id, center, node_candidates
                )
                if selected is None:
                    return None
                pt = tuple(selected["point"])
                nid = selected["node"]
                method = (
                    "nearest_graph_node_fallback_nearest"
                    if fallback_used else "nearest_graph_node_direct"
                )
                return pt, nid, float(selected["distance"]), method, evidence, fallback_used

            d = float(selected["distance"])
            u, v = selected["u"], selected["v"]
            proj, edata = tuple(selected["point"]), selected["edge_data"]
            method_suffix = "fallback_nearest" if fallback_used else "direct"
            if proj == tuple(G.nodes[u]['pos']):
                return proj, u, d, f"graph_edge_endpoint_{method_suffix}", evidence, fallback_used
            if proj == tuple(G.nodes[v]['pos']):
                return proj, v, d, f"graph_edge_endpoint_{method_suffix}", evidence, fallback_used

            old_type = edata.get("edge_type", "route")
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            aid = add_waypoint(proj, node_type="attachment_waypoint")
            add_graph_edge(u, aid, edge_type=old_type)
            add_graph_edge(aid, v, edge_type=old_type)
            return proj, aid, d, f"graph_edge_projection_{method_suffix}", evidence, fallback_used

        for rid, info in room_attachments.items():
            owner_public_id = info.get("public_owner_id")
            if owner_public_id is not None and int(owner_public_id) in set(recovered_corridor_ids):
                # V9：自己的 recovery graph 尚未建立，禁止 generic fallback 偷掛到 primary。
                info["attachment_point"] = None
                info["attachment_node"] = None
                info["attachment_method"] = f"pending_recovered_public_{int(owner_public_id)}"
                continue

            pt = info.get("attachment_point")
            nid = wp_node_map.get(tuple(pt)) if pt is not None else None
            if nid is not None and nid in G and G.degree(nid) > 0:
                info["attachment_node"] = nid
                continue

            fallback = attach_room_to_existing_graph(rid, info["room_center"])
            if fallback is None:
                # 房間中心節點仍會保留；只有道路 attachment 暫時為空。
                info["attachment_point"] = None
                info["attachment_node"] = None
                info["attachment_method"] = "unattached_no_road_graph"
                continue
            fpt, fnid, fdist, method, evidence, fallback_used = fallback
            info["attachment_point"] = tuple(fpt)
            info["attachment_node"] = fnid
            info["distance_px"] = round(float(fdist), 2)
            info["safe_clearance_px"] = round(float(wall_distance[fpt[1], fpt[0]]), 2)
            info["attachment_method"] = method
            store_attachment_visibility(info, evidence, fallback_used)

        # -------------------------------------------------
        # 9. A* / 正交橋接共用的多層可通行遮罩（V11）
        #
        # 原 V10 的 strict / +6px / +12px 都是固定像素，且最近鄰縮圖可能把窄通道直接吃掉。
        # V11 將「地圖尺度、走道寬度、房間核心、牆壁」分開建模：
        #   - shortcut A*：只在保守的 corridor band 內活動。
        #   - component bridge：strict -> adaptive_near -> adaptive_far。
        #   - 只有高可信 component 仍無法接合時，才使用 wall-free fallback。
        # -------------------------------------------------
        map_scale_guess = int(np.clip(round(map_short_side / 240.0), 2, 6))
        corridor_scale_guess = int(np.clip(round(corridor_width_est / 8.0), 2, 6))
        astar_scale = max(2, min(map_scale_guess, corridor_scale_guess))
        # 控制方向感知搜尋網格的最長邊；最終橋仍回到原解析度做 exact
        # projection + line safety 驗證，因此這是計算解析度最佳化，不是縮小輸出。
        bridge_grid_max_dim = int(np.clip(
            int(os.environ.get("MAP_BRIDGE_GRID_MAX_DIM", "640")), 384, 1200
        ))
        astar_scale = int(np.clip(
            max(astar_scale, int(math.ceil(max(H, W) / float(bridge_grid_max_dim)))),
            2, 8,
        ))

        def make_astar_mask(dilate_px):
            if dilate_px <= 0:
                m = valid_routing_mask.copy()
            else:
                r = max(1, int(round(dilate_px)))
                k = r * 2 + 1
                m = cv2.dilate(
                    valid_routing_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                )
            return cv2.bitwise_and(m, wall_free)

        ds_w = max(1, int(math.ceil(W / astar_scale)))
        ds_h = max(1, int(math.ceil(H / astar_scale)))

        def downsample_route_mask(route_mask):
            """
            面積式縮圖而不是 INTER_NEAREST：
            - walk fraction >= 10% 可保留窄走道；
            - 只要 cell 內出現明顯 wall collision 就封鎖，避免縮圖後穿牆。
            """
            walk_fraction = cv2.resize(
                (route_mask > 0).astype(np.float32),
                (ds_w, ds_h),
                interpolation=cv2.INTER_AREA,
            )
            wall_fraction = cv2.resize(
                (wall_collision > 0).astype(np.float32),
                (ds_w, ds_h),
                interpolation=cv2.INTER_AREA,
            )
            return ((walk_fraction >= 0.10) & (wall_fraction <= 0.02)).astype(np.uint8)

        # 一般 shortcut 仍採保守走道範圍，避免捷徑跑進房間。
        shortcut_relax_radius = int(np.clip(
            round(corridor_width_est * 0.22),
            4,
            max(6, round(map_short_side * 0.018)),
        ))
        astar_modes = []
        for mode_name, route_mask, penalty in [
            ("strict", make_astar_mask(0), 0.0),
            (f"adaptive_relaxed_{shortcut_relax_radius}", make_astar_mask(shortcut_relax_radius), float(shortcut_relax_radius) * 4.0),
        ]:
            ds = downsample_route_mask(route_mask)
            clear_ds = cv2.distanceTransform(ds, cv2.DIST_L2, 3)
            astar_modes.append((mode_name, route_mask, penalty, ds, clear_ds))

        # 房間核心只用於 component bridge 的「語意保護」。
        room_region_ids = [
            int(rid)
            for rid, data in id_labels.items()
            if not data.get("portal", False)
        ]
        semantic_room_mask = (
            np.isin(res_matrix, np.asarray(room_region_ids, dtype=np.int32))
            if room_region_ids else np.zeros_like(res_matrix, dtype=bool)
        ).astype(np.uint8) * 255
        room_core_radius = int(np.clip(round(corridor_width_est * 0.10), 2, 10))
        if room_core_radius > 0:
            room_core_mask = cv2.erode(
                semantic_room_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (room_core_radius * 2 + 1, room_core_radius * 2 + 1),
                ),
                iterations=1,
            )
        else:
            room_core_mask = semantic_room_mask.copy()
        # 已確認為走道的像素永遠優先，不因房間標籤誤差被擋掉。
        room_core_mask[valid_routing_mask > 0] = 0

        near_bridge_radius = int(np.clip(
            max(corridor_expand_radius + 3, round(corridor_width_est * 0.65)),
            8,
            max(16, round(map_short_side * 0.055)),
        ))
        far_bridge_radius = int(np.clip(
            max(near_bridge_radius + 4, round(corridor_width_est * 1.25), round(map_short_side * 0.020)),
            near_bridge_radius,
            max(near_bridge_radius, round(map_short_side * 0.085)),
        ))

        def corridor_band_mask(radius_px, block_room_core=True):
            band = ((corridor_distance <= float(radius_px)) | (valid_routing_mask > 0)).astype(np.uint8) * 255
            band = cv2.bitwise_and(band, wall_free)
            if block_room_core:
                # 只封鎖「深房間」；走道邊界誤差仍有 repair 空間。
                band[(room_core_mask > 0) & (valid_routing_mask == 0)] = 0
            return band

        corridor_dist_ds = cv2.resize(
            corridor_distance.astype(np.float32),
            (ds_w, ds_h),
            interpolation=cv2.INTER_LINEAR,
        )
        room_fraction_ds = cv2.resize(
            (room_core_mask > 0).astype(np.float32),
            (ds_w, ds_h),
            interpolation=cv2.INTER_AREA,
        )

        def make_bridge_mode(name, route_mask, mode_rank, *, off_corridor_weight, room_weight, mode_penalty):
            ds = downsample_route_mask(route_mask)
            clear_ds = cv2.distanceTransform(ds, cv2.DIST_L2, 3)
            # 全圖皆可通行時 OpenCV 會以極大 float 表示「沒有最近障礙」；
            # 先截斷再乘尺度，避免 overflow warning，且不改變 clearance penalty=0 的結果。
            clear_ds = np.minimum(clear_ds, float(max(ds.shape) * 2)) * float(astar_scale)
            return {
                "name": str(name),
                "mode_rank": int(mode_rank),
                "route_mask": route_mask,
                "ds": ds,
                "clear_ds": clear_ds,
                "off_corridor_weight": float(off_corridor_weight),
                "room_weight": float(room_weight),
                "mode_penalty": float(mode_penalty),
            }

        bridge_modes = [
            make_bridge_mode(
                "strict",
                valid_routing_mask,
                0,
                off_corridor_weight=0.0,
                room_weight=0.0,
                mode_penalty=0.0,
            ),
            make_bridge_mode(
                f"adaptive_near_{near_bridge_radius}",
                corridor_band_mask(near_bridge_radius, block_room_core=True),
                1,
                off_corridor_weight=0.45,
                room_weight=8.0,
                mode_penalty=corridor_width_est * 0.35,
            ),
            make_bridge_mode(
                f"adaptive_far_{far_bridge_radius}",
                corridor_band_mask(far_bridge_radius, block_room_core=True),
                2,
                off_corridor_weight=0.90,
                room_weight=12.0,
                mode_penalty=corridor_width_est * 0.90,
            ),
        ]

        # 最後手段：只要 wall topology 判定可以走，就允許搜尋；但房間核心與離走道距離
        # 都會被大幅加成本。這不是第一層補償，避免把商店/房間當成捷徑。
        fallback_bridge_modes = [
            make_bridge_mode(
                "wall_free_topology_fallback",
                wall_free,
                3,
                off_corridor_weight=2.20,
                room_weight=18.0,
                mode_penalty=corridor_width_est * 2.50,
            )
        ]

        default_axis_expansions = int(np.clip(ds_w * ds_h * 6, 420000, 1800000))
        axis_bridge_max_expansions = int(
            os.environ.get("MAP_AXIS_BRIDGE_MAX_EXPANSIONS", str(default_axis_expansions))
        )
        bridge_turn_penalty_px = float(np.clip(
            corridor_width_est * 0.90,
            20.0,
            max(40.0, map_short_side * 0.08),
        ))

        print(
            "[橋接參數] "
            f"astar_scale={astar_scale}px/cell，"
            f"near={near_bridge_radius}px，far={far_bridge_radius}px，"
            f"turn_penalty={bridge_turn_penalty_px:.1f}px，"
            f"max_expansions={axis_bridge_max_expansions}"
        )

        astar_cache = {}
        astar_calls = 0
        max_astar_calls = int(os.environ.get("MAP_MAX_ASTAR_CALLS", "140"))

        def nearest_valid_cell(cell, ds_mask, max_radius=8):
            cx, cy = cell
            h, w = ds_mask.shape
            if 0 <= cx < w and 0 <= cy < h and ds_mask[cy, cx] > 0:
                return (cx, cy)
            best = None
            for r in range(1, max_radius + 1):
                for yy in range(max(0, cy - r), min(h, cy + r + 1)):
                    for xx in range(max(0, cx - r), min(w, cx + r + 1)):
                        if ds_mask[yy, xx] == 0:
                            continue
                        d = (xx - cx) ** 2 + (yy - cy) ** 2
                        if best is None or d < best[0]:
                            best = (d, xx, yy)
                if best is not None:
                    return (best[1], best[2])
            return None

        def simplify_passable_path(points, route_mask):
            if len(points) <= 2:
                return points
            out = [points[0]]
            i = 0
            while i < len(points) - 1:
                best_j = i + 1
                upper = min(len(points) - 1, i + 40)
                for j in range(upper, i, -1):
                    if line_is_safe(points[i], points[j], mask=route_mask, min_ratio=0.985, thickness=3):
                        best_j = j
                        break
                out.append(points[best_j])
                i = best_j

            # 長直線每 90px 插入中繼點，避免單一邊太長而難以後續吸附/轉向。
            dense = [out[0]]
            for p1, p2 in zip(out[:-1], out[1:]):
                d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                pieces = max(1, int(math.ceil(d / 90.0)))
                for k in range(1, pieces + 1):
                    t = k / float(pieces)
                    dense.append((
                        int(round(p1[0] + (p2[0] - p1[0]) * t)),
                        int(round(p1[1] + (p2[1] - p1[1]) * t))
                    ))
            return dense

        def astar_path(start_pt, goal_pt, mode_name, route_mask, ds, clear_ds, max_expansions=90000):
            nonlocal astar_calls
            cache_key = (tuple(start_pt), tuple(goal_pt), mode_name)
            reverse_key = (tuple(goal_pt), tuple(start_pt), mode_name)
            if cache_key in astar_cache:
                return astar_cache[cache_key]
            if reverse_key in astar_cache:
                rev = astar_cache[reverse_key]
                return None if rev is None else (list(reversed(rev[0])), rev[1])

            if astar_calls >= max_astar_calls:
                astar_cache[cache_key] = None
                return None
            astar_calls += 1
            ds_h, ds_w = ds.shape
            s0 = (int(round(start_pt[0] / astar_scale)), int(round(start_pt[1] / astar_scale)))
            g0 = (int(round(goal_pt[0] / astar_scale)), int(round(goal_pt[1] / astar_scale)))
            s = nearest_valid_cell(s0, ds)
            g = nearest_valid_cell(g0, ds)
            if s is None or g is None:
                astar_cache[cache_key] = None
                return None

            import heapq
            open_heap = [(0.0, 0.0, s)]
            came_from = {}
            g_score = {s: 0.0}
            closed = set()
            nbrs = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))
            ]
            expansions = 0
            found = False
            while open_heap and expansions < max_expansions:
                _, curr_g, cur = heapq.heappop(open_heap)
                if cur in closed:
                    continue
                closed.add(cur)
                expansions += 1
                if cur == g:
                    found = True
                    break
                cx, cy = cur
                for dx, dy, step_cost in nbrs:
                    nx_, ny_ = cx + dx, cy + dy
                    if not (0 <= nx_ < ds_w and 0 <= ny_ < ds_h) or ds[ny_, nx_] == 0:
                        continue
                    if dx != 0 and dy != 0:
                        # 防止對角線切牆角。
                        if ds[cy, nx_] == 0 or ds[ny_, cx] == 0:
                            continue
                    clearance = float(clear_ds[ny_, nx_])
                    move_cost = step_cost * (1.0 + 1.6 / (clearance + 1.0))
                    ng = curr_g + move_cost
                    nxt = (nx_, ny_)
                    if ng >= g_score.get(nxt, float('inf')):
                        continue
                    g_score[nxt] = ng
                    came_from[nxt] = cur
                    h = math.hypot(g[0] - nx_, g[1] - ny_)
                    heapq.heappush(open_heap, (ng + h, ng, nxt))

            if not found:
                astar_cache[cache_key] = None
                return None

            cells = [g]
            cur = g
            while cur != s:
                cur = came_from[cur]
                cells.append(cur)
            cells.reverse()

            points = [
                (
                    int(np.clip(cx * astar_scale + astar_scale // 2, 0, W - 1)),
                    int(np.clip(cy * astar_scale + astar_scale // 2, 0, H - 1))
                )
                for cx, cy in cells
            ]
            points[0] = tuple(map(int, start_pt))
            points[-1] = tuple(map(int, goal_pt))
            points = simplify_passable_path(points, route_mask)
            path_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:]))
            result = (points, float(path_len))
            astar_cache[cache_key] = result
            return result

        # -------------------------------------------------
        # 一般 A* 僅保留給後續捷徑檢查；component 橋接改由 V10 四方向多起點搜尋處理。
        # -------------------------------------------------
        def best_astar_path(start_pt, goal_pt):
            direct_dist = math.hypot(goal_pt[0] - start_pt[0], goal_pt[1] - start_pt[1])
            if line_is_safe(start_pt, goal_pt, mask=valid_routing_mask, min_ratio=0.985, thickness=3):
                return (direct_dist, [tuple(map(int, start_pt)), tuple(map(int, goal_pt))], direct_dist, "strict_direct", valid_routing_mask)
            best = None
            for mode_name, route_mask, penalty, ds, clear_ds in astar_modes:
                result = astar_path(start_pt, goal_pt, mode_name, route_mask, ds, clear_ds)
                if result is None:
                    continue
                points, path_len = result
                score = path_len + penalty
                if best is None or score < best[0]:
                    best = (score, points, path_len, mode_name, route_mask)
                if mode_name == "strict":
                    break
            return best

        def add_polyline_to_graph(points, edge_type, interior_node_type="astar_waypoint"):
            if not points or len(points) < 2:
                return []
            node_ids = [add_waypoint(points[0])]
            for p in points[1:-1]:
                node_ids.append(add_waypoint(p, node_type=interior_node_type))
            node_ids.append(add_waypoint(points[-1]))
            for a, b in zip(node_ids[:-1], node_ids[1:]):
                add_graph_edge(a, b, edge_type=edge_type)
            return node_ids

        # -------------------------------------------------
        # 10. V10 不連通 component：任意路段 ↔ 任意路段的正交橋接
        #
        # 規則：
        # 1) 仍以最大道路 component 作為主路網。
        # 2) 起點與終點不限定既有節點；可落在任一水平/垂直道路邊的中間。
        # 3) 搜尋只允許上、下、左、右四方向，禁止任何斜線。
        # 4) 成本優先順序：轉彎數 → 遮罩嚴格度 → 路徑長度。
        # 5) 接觸既有道路邊時，自動切分該邊並建立 bridge_contact_waypoint。
        # -------------------------------------------------
        bridge_records = []
        unconnected_records = []
        pruned_component_records = []

        def component_sort_key(nodes):
            """最大路網先以節點數判定；節點數相同時，以道路總長度判定。"""
            node_set = set(nodes)
            total_length = 0.0
            for u, v, edata in G.edges(data=True):
                if u in node_set and v in node_set:
                    total_length += float(edata.get("weight", 0.0))
            return len(node_set), total_length

        def is_axis_segment(p1, p2):
            return int(p1[0]) == int(p2[0]) or int(p1[1]) == int(p2[1])

        def component_axis_edges(component_nodes):
            """只回傳 component 中的水平/垂直邊；斜邊不作為橋接接觸面。"""
            node_set = set(component_nodes)
            records = []
            for u, v, edata in list(G.edges(data=True)):
                if u not in node_set or v not in node_set:
                    continue
                p1 = tuple(map(int, G.nodes[u]["pos"]))
                p2 = tuple(map(int, G.nodes[v]["pos"]))
                if not is_axis_segment(p1, p2):
                    continue
                records.append((u, v, p1, p2, dict(edata)))
            return records

        def rasterize_component_on_ds(component_nodes, ds_mask):
            """
            把 component 的既有正交道路邊畫到縮小網格。

            V10 直接「round 後再與 ds_mask 相交」，在 5px/cell 的情況下，
            一條真實道路邊可能剛好落到被縮圖判為 0 的 cell，整個 component 便得到
            no_contact_cells。V11 允許最多 2 個 cell 的局部 contact snap，但仍必須落在
            該 mode 的可通行 cell 上。
            """
            contact = np.zeros_like(ds_mask, dtype=np.uint8)
            ds_h, ds_w = ds_mask.shape

            for _, _, p1, p2, _ in component_axis_edges(component_nodes):
                c1 = (
                    int(np.clip(round(p1[0] / astar_scale), 0, ds_w - 1)),
                    int(np.clip(round(p1[1] / astar_scale), 0, ds_h - 1)),
                )
                c2 = (
                    int(np.clip(round(p2[0] / astar_scale), 0, ds_w - 1)),
                    int(np.clip(round(p2[1] / astar_scale), 0, ds_h - 1)),
                )
                cv2.line(contact, c1, c2, 1, 1)

            for nid in component_nodes:
                if nid not in G:
                    continue
                px, py = G.nodes[nid]["pos"]
                cx = int(np.clip(round(px / astar_scale), 0, ds_w - 1))
                cy = int(np.clip(round(py / astar_scale), 0, ds_h - 1))
                contact[cy, cx] = 1

            exact = ((contact > 0) & (ds_mask > 0)).astype(np.uint8)
            if np.any(exact):
                return exact

            # 抗縮圖 alias：只向外找非常小的鄰域，找到第一層合法 cell 就停止。
            for radius in (1, 2):
                k = radius * 2 + 1
                expanded = cv2.dilate(
                    contact,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                    iterations=1,
                )
                snapped = ((expanded > 0) & (ds_mask > 0)).astype(np.uint8)
                if np.any(snapped):
                    return snapped

            return exact

        def compress_orthogonal_points(points):
            """只刪除重複點與共線中繼點；絕不把 L 型路徑簡化成斜線。"""
            cleaned = []
            for p in points:
                q = (int(p[0]), int(p[1]))
                if not cleaned or q != cleaned[-1]:
                    cleaned.append(q)
            if len(cleaned) <= 2:
                return cleaned

            changed = True
            while changed and len(cleaned) > 2:
                changed = False
                out = [cleaned[0]]
                for idx in range(1, len(cleaned) - 1):
                    a = out[-1]
                    b = cleaned[idx]
                    c = cleaned[idx + 1]
                    if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                        changed = True
                        continue
                    out.append(b)
                out.append(cleaned[-1])
                cleaned = out
            return cleaned

        def count_axis_turns(points):
            dirs = []
            for a, b in zip(points[:-1], points[1:]):
                dx = int(b[0]) - int(a[0])
                dy = int(b[1]) - int(a[1])
                if dx and dy:
                    return 10 ** 9
                if dx:
                    dirs.append((1 if dx > 0 else -1, 0))
                elif dy:
                    dirs.append((0, 1 if dy > 0 else -1))
            return sum(1 for a, b in zip(dirs[:-1], dirs[1:]) if a != b)

        def refine_axis_polyline(points, route_mask):
            """
            以目前折線的所有 x/y 座標建立小型 Hanan grid，再做方向感知最短路徑。
            這會消除縮小網格映射造成的 2~5px 階梯狀假轉彎，但仍只使用水平/垂直線。
            """
            points = compress_orthogonal_points(points)
            if len(points) <= 1:
                return points

            start = tuple(points[0])
            goal = tuple(points[-1])
            xs = sorted(set(int(p[0]) for p in points))
            ys = sorted(set(int(p[1]) for p in points))

            # 防止極端情況候選格過大；保留首尾與均勻抽樣座標。
            def limit_values(values, must_keep, limit=36):
                values = sorted(set(values).union(must_keep))
                if len(values) <= limit:
                    return values
                idxs = np.linspace(0, len(values) - 1, limit).astype(int)
                kept = {values[i] for i in idxs}
                kept.update(must_keep)
                return sorted(kept)

            xs = limit_values(xs, {start[0], goal[0]})
            ys = limit_values(ys, {start[1], goal[1]})

            candidates = []
            index = {}
            for y in ys:
                for x in xs:
                    pt = (int(x), int(y))
                    safe = (
                        0 <= x < W and 0 <= y < H
                        and route_mask[y, x] > 0
                        and wall_collision[y, x] == 0
                    )
                    if pt in (start, goal) or safe:
                        index[pt] = len(candidates)
                        candidates.append(pt)

            if start not in index or goal not in index:
                return points

            adjacency = [[] for _ in candidates]
            by_x = {}
            by_y = {}
            for idx, (x, y) in enumerate(candidates):
                by_x.setdefault(x, []).append((y, idx))
                by_y.setdefault(y, []).append((x, idx))

            # 同一直線上只連相鄰候選點；更遠點可由多段共線邊自動合併。
            for group in by_x.values():
                group.sort()
                for (_, a_idx), (_, b_idx) in zip(group[:-1], group[1:]):
                    a, b = candidates[a_idx], candidates[b_idx]
                    if line_is_safe(a, b, mask=route_mask, min_ratio=0.94, thickness=3):
                        dist = abs(b[1] - a[1])
                        adjacency[a_idx].append((b_idx, 1, dist))
                        adjacency[b_idx].append((a_idx, 1, dist))
            for group in by_y.values():
                group.sort()
                for (_, a_idx), (_, b_idx) in zip(group[:-1], group[1:]):
                    a, b = candidates[a_idx], candidates[b_idx]
                    if line_is_safe(a, b, mask=route_mask, min_ratio=0.94, thickness=3):
                        dist = abs(b[0] - a[0])
                        adjacency[a_idx].append((b_idx, 0, dist))
                        adjacency[b_idx].append((a_idx, 0, dist))

            import heapq
            start_idx = index[start]
            goal_idx = index[goal]
            start_state = (start_idx, -1)
            best = {start_state: (0, 0.0)}
            parent = {}
            heap = [(0, 0.0, start_idx, -1)]
            goal_state = None

            while heap:
                turns, length, node_idx, prev_dir = heapq.heappop(heap)
                state = (node_idx, prev_dir)
                if best.get(state) != (turns, length):
                    continue
                if node_idx == goal_idx:
                    goal_state = state
                    break
                for nxt_idx, move_dir, edge_len in adjacency[node_idx]:
                    nt = turns + (0 if prev_dir in (-1, move_dir) else 1)
                    nl = length + float(edge_len)
                    nxt_state = (nxt_idx, move_dir)
                    cost = (nt, nl)
                    if cost >= best.get(nxt_state, (10 ** 9, float('inf'))):
                        continue
                    best[nxt_state] = cost
                    parent[nxt_state] = state
                    heapq.heappush(heap, (nt, nl, nxt_idx, move_dir))

            if goal_state is None:
                return points

            states = [goal_state]
            while states[-1] != start_state:
                states.append(parent[states[-1]])
            states.reverse()
            refined = [candidates[state[0]] for state in states]
            return compress_orthogonal_points(refined)

        def nearest_axis_edge_projection(point, component_nodes):
            """將任意點投影到 component 中最近的水平/垂直道路邊。"""
            px, py = map(int, point)
            best = None
            for u, v, p1, p2, edata in component_axis_edges(component_nodes):
                if p1[1] == p2[1]:
                    x = int(max(min(p1[0], p2[0]), min(px, max(p1[0], p2[0]))))
                    proj = (x, int(p1[1]))
                else:
                    y = int(max(min(p1[1], p2[1]), min(py, max(p1[1], p2[1]))))
                    proj = (int(p1[0]), y)
                d = math.hypot(proj[0] - px, proj[1] - py)
                rec = (float(d), u, v, proj, edata)
                if best is None or rec[0] < best[0]:
                    best = rec

            if best is not None:
                return best

            # 沒有邊時退回最近節點。
            candidates = [n for n in component_nodes if n in G]
            if not candidates:
                return None
            nid = min(
                candidates,
                key=lambda n: math.hypot(
                    G.nodes[n]["pos"][0] - px,
                    G.nodes[n]["pos"][1] - py,
                ),
            )
            proj = tuple(map(int, G.nodes[nid]["pos"]))
            return (math.hypot(proj[0] - px, proj[1] - py), nid, nid, proj, {})

        def orthogonal_join(a, b, route_mask):
            """
            將縮小網格 contact 接回真實道路邊。
            先試 0/1 轉彎；若縮圖 alias 讓 L 型剛好撞到邊界，再在局部搜尋 2 轉彎 H-V-H / V-H-V。
            """
            a = tuple(map(int, a))
            b = tuple(map(int, b))
            if a == b:
                return [a]

            def polyline_is_safe(pts):
                return all(
                    is_axis_segment(x, y)
                    and line_is_safe(
                        x,
                        y,
                        mask=route_mask,
                        min_ratio=0.92,
                        thickness=3,
                    )
                    for x, y in zip(pts[:-1], pts[1:])
                )

            candidates = []
            if is_axis_segment(a, b):
                candidates.append([a, b])
            else:
                candidates.extend([
                    [a, (a[0], b[1]), b],
                    [a, (b[0], a[1]), b],
                ])

            valid = []
            for pts in candidates:
                pts = compress_orthogonal_points(pts)
                if polyline_is_safe(pts):
                    length = sum(
                        abs(y[0] - x[0]) + abs(y[1] - x[1])
                        for x, y in zip(pts[:-1], pts[1:])
                    )
                    valid.append((length, len(pts), pts))
            if valid:
                return min(valid, key=lambda z: (z[0], z[1]))[2]

            # 局部 2-turn fallback。contact snap 最多只有幾個 A* cell，
            # 因此不需要再啟動一次全圖搜尋。
            margin = int(np.clip(round(corridor_width_est * 0.45), 8, 48))
            step = max(1, astar_scale)
            x_lo = max(0, min(a[0], b[0]) - margin)
            x_hi = min(W - 1, max(a[0], b[0]) + margin)
            y_lo = max(0, min(a[1], b[1]) - margin)
            y_hi = min(H - 1, max(a[1], b[1]) + margin)

            for xmid in range(x_lo, x_hi + 1, step):
                pts = compress_orthogonal_points([
                    a,
                    (xmid, a[1]),
                    (xmid, b[1]),
                    b,
                ])
                if polyline_is_safe(pts):
                    length = sum(
                        abs(y[0] - x[0]) + abs(y[1] - x[1])
                        for x, y in zip(pts[:-1], pts[1:])
                    )
                    valid.append((length, len(pts), pts))

            for ymid in range(y_lo, y_hi + 1, step):
                pts = compress_orthogonal_points([
                    a,
                    (a[0], ymid),
                    (b[0], ymid),
                    b,
                ])
                if polyline_is_safe(pts):
                    length = sum(
                        abs(y[0] - x[0]) + abs(y[1] - x[1])
                        for x, y in zip(pts[:-1], pts[1:])
                    )
                    valid.append((length, len(pts), pts))

            return min(valid, key=lambda z: (z[0], z[1]))[2] if valid else None


        def exact_contact_path(raw_points, small_component, main_component, route_mask):
            """把縮小網格端點接到兩個 component 的實際道路邊中間。"""
            if not raw_points:
                return None
            raw_points = compress_orthogonal_points(raw_points)
            source_rec = nearest_axis_edge_projection(raw_points[0], small_component)
            target_rec = nearest_axis_edge_projection(raw_points[-1], main_component)
            if source_rec is None or target_rec is None:
                return None

            source_proj = source_rec[3]
            target_proj = target_rec[3]
            prefix = orthogonal_join(source_proj, raw_points[0], route_mask)
            suffix = orthogonal_join(raw_points[-1], target_proj, route_mask)
            if prefix is None or suffix is None:
                return None

            points = prefix[:-1] + raw_points + suffix[1:]
            points = compress_orthogonal_points(points)
            points = refine_axis_polyline(points, route_mask)
            if len(points) < 1:
                return None
            if any(not is_axis_segment(a, b) for a, b in zip(points[:-1], points[1:])):
                return None
            if any(
                not line_is_safe(a, b, mask=route_mask, min_ratio=0.94, thickness=3)
                for a, b in zip(points[:-1], points[1:])
            ):
                return None
            return {
                "points": points,
                "source_projection": source_rec,
                "target_projection": target_rec,
            }

        def multisource_axis_min_turn_path(source_component, target_component, mode):
            """
            V11 方向感知多起點 Dijkstra。

            重要差異：
            - 仍只允許四方向，所以輸出一定是水平/垂直折線。
            - 不再把「轉彎數」放在所有成本之前。V10 會為了少 1 個轉彎，
              寧可走非常遠的 relaxed 區域；V11 改成「實際行走成本 + 轉彎懲罰」。
            - 離已知走道越遠、進入房間核心越深，成本越高。
            """
            mode_name = mode["name"]
            route_mask = mode["route_mask"]
            ds = mode["ds"]
            clear_ds = mode["clear_ds"]

            source_mask = rasterize_component_on_ds(source_component, ds)
            target_mask = rasterize_component_on_ds(target_component, ds)
            source_yx = np.argwhere(source_mask > 0)
            if source_yx.size == 0 or not np.any(target_mask > 0):
                return {"status": "no_contact_cells", "mode_name": mode_name}

            import heapq

            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            best_cost = {}
            came_from = {}
            open_heap = []
            sequence = 0

            for sy, sx in source_yx:
                state = (int(sx), int(sy), -1)
                # (weighted_score, turns, geometric_length)
                best_cost[state] = (0.0, 0, 0.0)
                heapq.heappush(open_heap, (0.0, 0, 0.0, sequence, state))
                sequence += 1

            goal_state = None
            goal_cost = None
            expansions = 0
            ds_h, ds_w = ds.shape
            corridor_norm = max(4.0, corridor_width_est)

            while open_heap and expansions < axis_bridge_max_expansions:
                weighted, turns, geometric_length, _, state = heapq.heappop(open_heap)
                if best_cost.get(state) != (weighted, turns, geometric_length):
                    continue
                expansions += 1
                cx, cy, prev_dir = state

                if target_mask[cy, cx] > 0:
                    goal_state = state
                    goal_cost = (weighted, turns, geometric_length)
                    break

                for dir_idx, (dx, dy) in enumerate(directions):
                    nx_, ny_ = cx + dx, cy + dy
                    if not (0 <= nx_ < ds_w and 0 <= ny_ < ds_h):
                        continue
                    if ds[ny_, nx_] == 0:
                        continue

                    added_turn = 0 if prev_dir in (-1, dir_idx) else 1
                    new_turns = turns + added_turn
                    step_px = float(astar_scale)

                    clearance_px = float(clear_ds[ny_, nx_])
                    clearance_penalty = 0.30 * min(
                        2.5,
                        corridor_norm / max(clearance_px, 1.0),
                    )

                    off_distance_px = float(corridor_dist_ds[ny_, nx_])
                    off_penalty = float(mode["off_corridor_weight"]) * min(
                        4.0,
                        off_distance_px / corridor_norm,
                    )

                    room_fraction = float(room_fraction_ds[ny_, nx_])
                    room_penalty = float(mode["room_weight"]) * room_fraction

                    step_weight = step_px * (
                        1.0
                        + clearance_penalty
                        + off_penalty
                        + room_penalty
                    )
                    turn_weight = bridge_turn_penalty_px if added_turn else 0.0

                    new_weighted = weighted + step_weight + turn_weight
                    new_length = geometric_length + step_px
                    next_state = (nx_, ny_, dir_idx)
                    new_cost = (new_weighted, new_turns, new_length)
                    old_cost = best_cost.get(next_state)
                    if old_cost is not None and new_cost >= old_cost:
                        continue

                    best_cost[next_state] = new_cost
                    came_from[next_state] = state
                    heapq.heappush(
                        open_heap,
                        (new_weighted, new_turns, new_length, sequence, next_state),
                    )
                    sequence += 1

            if goal_state is None:
                return {
                    "status": "search_limit" if expansions >= axis_bridge_max_expansions else "no_path",
                    "expansions": int(expansions),
                    "mode_name": mode_name,
                }

            states = [goal_state]
            while states[-1] in came_from:
                states.append(came_from[states[-1]])
            states.reverse()

            raw_points = [
                (
                    int(np.clip(st[0] * astar_scale + astar_scale // 2, 0, W - 1)),
                    int(np.clip(st[1] * astar_scale + astar_scale // 2, 0, H - 1)),
                )
                for st in states
            ]
            exact = exact_contact_path(raw_points, source_component, target_component, route_mask)
            if exact is None:
                return {
                    "status": "contact_projection_failed",
                    "expansions": int(expansions),
                    "mode_name": mode_name,
                }

            points = exact["points"]
            path_len = sum(
                abs(b[0] - a[0]) + abs(b[1] - a[1])
                for a, b in zip(points[:-1], points[1:])
            )
            turn_count = count_axis_turns(points)
            if turn_count >= 10 ** 9:
                return {
                    "status": "non_axis_path_rejected",
                    "expansions": int(expansions),
                    "mode_name": mode_name,
                }

            # exact projection 可能使最終幾何長度與縮圖搜索略有差異；
            # 因此最後再以 exact path 的長度與轉彎數校正一次評分。
            weighted_score = (
                float(goal_cost[0] if goal_cost is not None else path_len)
                + float(mode["mode_penalty"])
                + max(0, turn_count - int(goal_cost[1] if goal_cost is not None else turn_count))
                  * bridge_turn_penalty_px
            )

            return {
                "status": "ok",
                "points": points,
                "path_len": float(path_len),
                "path_cost": float(weighted_score),
                "turn_count": int(turn_count),
                "mode_name": mode_name,
                "mode_rank": int(mode["mode_rank"]),
                "route_mask": route_mask,
                "source_projection": exact["source_projection"],
                "target_projection": exact["target_projection"],
                "expansions": int(expansions),
            }

        def best_axis_component_bridge(source_component, target_component, modes=None):
            """
            依安全層級逐級放寬。只要某一層找到路徑，就不再去更寬鬆的層，
            避免「有 strict 路可走卻因少一個轉彎而跑去 room/free-space」。
            """
            if modes is None:
                modes = bridge_modes
            search_limit_seen = False
            status_log = []

            for mode in modes:
                result = multisource_axis_min_turn_path(
                    source_component,
                    target_component,
                    mode,
                )
                status_log.append({
                    "mode": mode["name"],
                    "status": result.get("status"),
                    "expansions": int(result.get("expansions", 0)),
                })
                if result.get("status") != "ok":
                    search_limit_seen = search_limit_seen or result.get("status") == "search_limit"
                    continue

                result["score"] = (
                    int(result["mode_rank"]),
                    float(result["path_cost"]),
                    int(result["turn_count"]),
                    float(result["path_len"]),
                )
                result["mode_attempts"] = status_log
                return result

            return {
                "status": "search_limit" if search_limit_seen else "no_axis_path",
                "mode_attempts": status_log,
            }

        def global_multisource_axis_bridge(comps, mode):
            """一次搜尋所有 component，取代逐 pair 重跑全圖 Dijkstra。

            舊流程對每個 component pair 都各自掃描同一張 passable grid；碎片數為 C
            時，一輪最壞會做 O(C^2) 次搜尋。這裡把每個 component 的道路接觸像素
            同時放入同一個方向感知 Dijkstra。不同來源的 wavefront 相遇時，即得到
            一個候選橋接；整張 grid 每個方向狀態至多被最佳化一次，因此昂貴部分由
            O(C^2 * grid) 降為 O(grid log grid)。安全遮罩、四方向限制、牆距／房間
            懲罰、exact contact projection 與少轉彎規則均沿用原實作。
            """
            mode_name = mode["name"]
            route_mask = mode["route_mask"]
            ds = mode["ds"]
            clear_ds = mode["clear_ds"]
            ds_h_local, ds_w_local = ds.shape

            # 每個 component 在本 mode 只 rasterize 一次；舊版會對每個 pair 重做。
            contact_masks = []
            overlap_cells = {}
            for comp_index, component in enumerate(comps):
                contact = rasterize_component_on_ds(component, ds)
                if not np.any(contact > 0):
                    contact_masks.append(contact)
                    continue

                # 線網通常本來就是 1px；若遇到較厚接觸區，只取邊界可減少大量
                # 等價起點，不影響任意 component 間的最短離開路徑。
                eroded = cv2.erode(contact, np.ones((3, 3), np.uint8), iterations=1)
                boundary = ((contact > 0) & (eroded == 0)).astype(np.uint8)
                if not np.any(boundary):
                    boundary = (contact > 0).astype(np.uint8)
                contact_masks.append(boundary)

                for sy, sx in np.argwhere(boundary > 0):
                    overlap_cells.setdefault((int(sx), int(sy)), []).append(comp_index)

            # FAST reachability gate：若兩個 component 的 contact 根本不在同一個
            # passable connected component，方向感知 Dijkstra 不可能把它們連起來。
            # 先以 O(grid) connected-components 判斷，可直接略過 strict/far 中大量
            # 注定失敗的全圖搜尋；並把只有單一來源的 passable island 排除。
            _, passable_labels = cv2.connectedComponents(
                (ds > 0).astype(np.uint8), connectivity=4
            )
            owners_by_passable = {}
            passable_ids_by_component = []
            for comp_index, contact in enumerate(contact_masks):
                ids_here = set(map(int, np.unique(passable_labels[contact > 0])))
                ids_here.discard(0)
                passable_ids_by_component.append(ids_here)
                for passable_id in ids_here:
                    owners_by_passable.setdefault(passable_id, set()).add(comp_index)

            joinable_passable_ids = {
                pid for pid, owners in owners_by_passable.items() if len(owners) >= 2
            }
            active_components = [
                idx for idx, ids_here in enumerate(passable_ids_by_component)
                if ids_here & joinable_passable_ids
            ]
            if len(active_components) < 2:
                return {
                    "status": "disconnected_passable_components",
                    "mode_name": mode_name,
                    "expansions": 0,
                    "active_components": int(len(active_components)),
                }

            searchable = np.isin(
                passable_labels,
                np.asarray(sorted(joinable_passable_ids), dtype=passable_labels.dtype),
            )

            directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
            opposite_dir = {0: 2, 1: 3, 2: 0, 3: 1}
            corridor_norm = max(4.0, corridor_width_est)

            # state=(x,y,進入方向)；來源 component 與 parent 分開保存。
            best_cost = {}
            state_owner = {}
            came_from = {}
            open_heap = []
            sequence = 0
            direct_overlap_candidates = {}

            for cell, owners in overlap_cells.items():
                if not searchable[cell[1], cell[0]]:
                    continue
                unique_owners = sorted(set(owners).intersection(active_components))
                if not unique_owners:
                    continue
                if len(unique_owners) > 1:
                    for oi in range(len(unique_owners)):
                        for oj in range(oi + 1, len(unique_owners)):
                            pair = (unique_owners[oi], unique_owners[oj])
                            direct_overlap_candidates[pair] = cell

                # 同一 cell 只需一個零成本 state；其他 owner 由上面的 direct
                # overlap 候選保留，避免 heap 內完全相同 state 相互覆寫。
                owner = unique_owners[0]
                state = (cell[0], cell[1], -1)
                if state in best_cost:
                    continue
                best_cost[state] = (0.0, 0, 0.0)
                state_owner[state] = owner
                heapq.heappush(open_heap, (0.0, 0, 0.0, sequence, state))
                sequence += 1

            # 每對相鄰 wavefront 只留加權成本最低的碰撞。這些 pair 正是 passable
            # grid 上 component Voronoi 圖的邊，包含全域最短安全合併候選。
            collisions = {}
            collision_parent = list(range(len(comps)))

            def collision_find(x):
                while collision_parent[x] != x:
                    collision_parent[x] = collision_parent[collision_parent[x]]
                    x = collision_parent[x]
                return x

            def collision_union(a, b):
                ra, rb = collision_find(int(a)), collision_find(int(b))
                if ra != rb:
                    collision_parent[rb] = ra

            for owner_pair in direct_overlap_candidates:
                collision_union(owner_pair[0], owner_pair[1])

            def collision_coverage_complete():
                for passable_id in joinable_passable_ids:
                    roots = {
                        collision_find(owner)
                        for owner in owners_by_passable.get(passable_id, set())
                    }
                    if len(roots) > 1:
                        return False
                return True

            def register_collision(a_state, a_cost, a_owner, b_state, b_cost, b_owner, a_tail=None):
                if a_owner == b_owner:
                    return
                pair = tuple(sorted((int(a_owner), int(b_owner))))
                a_dir = int(a_state[2])
                if a_tail is not None:
                    a_dir = int(a_tail[2])
                b_dir = int(b_state[2])
                join_turn = 0
                if a_dir >= 0 and b_dir >= 0 and a_dir != opposite_dir[b_dir]:
                    join_turn = 1
                weighted = float(a_cost[0]) + float(b_cost[0]) + join_turn * bridge_turn_penalty_px
                turns = int(a_cost[1]) + int(b_cost[1]) + join_turn
                length = float(a_cost[2]) + float(b_cost[2])
                rec = {
                    "weighted": weighted,
                    "turns": turns,
                    "length": length,
                    "a_state": a_state,
                    "b_state": b_state,
                    "a_owner": int(a_owner),
                    "b_owner": int(b_owner),
                    "a_tail": a_tail,
                }
                old = collisions.get(pair)
                if old is None or (weighted, turns, length) < (old["weighted"], old["turns"], old["length"]):
                    collisions[pair] = rec
                collision_union(a_owner, b_owner)

            expansions = 0
            coverage_complete_at = None
            extra_after_coverage = max(5000, int(ds_w_local * ds_h_local * 0.04))
            while open_heap and expansions < axis_bridge_max_expansions:
                weighted, turns, geometric_length, _, state = heapq.heappop(open_heap)
                current_cost = (weighted, turns, geometric_length)
                if best_cost.get(state) != current_cost:
                    continue
                expansions += 1
                cx, cy, prev_dir = state
                owner = state_owner[state]

                # 不同方向可能在同一 cell 相遇；檢查所有已知方向狀態。
                for other_dir in (-1, 0, 1, 2, 3):
                    other_state = (cx, cy, other_dir)
                    if other_state == state or other_state not in best_cost:
                        continue
                    other_owner = state_owner[other_state]
                    if other_owner != owner:
                        register_collision(
                            state, current_cost, owner,
                            other_state, best_cost[other_state], other_owner,
                        )

                for dir_idx, (dx, dy) in enumerate(directions):
                    nx_cell, ny_cell = cx + dx, cy + dy
                    if not (0 <= nx_cell < ds_w_local and 0 <= ny_cell < ds_h_local):
                        continue
                    if not searchable[ny_cell, nx_cell]:
                        continue

                    added_turn = 0 if prev_dir in (-1, dir_idx) else 1
                    new_turns = turns + added_turn
                    step_px = float(astar_scale)
                    clearance_px = float(clear_ds[ny_cell, nx_cell])
                    clearance_penalty = 0.30 * min(2.5, corridor_norm / max(clearance_px, 1.0))
                    off_distance_px = float(corridor_dist_ds[ny_cell, nx_cell])
                    off_penalty = float(mode["off_corridor_weight"]) * min(4.0, off_distance_px / corridor_norm)
                    room_fraction = float(room_fraction_ds[ny_cell, nx_cell])
                    room_penalty = float(mode["room_weight"]) * room_fraction
                    step_weight = step_px * (1.0 + clearance_penalty + off_penalty + room_penalty)
                    new_weighted = weighted + step_weight + (bridge_turn_penalty_px if added_turn else 0.0)
                    new_length = geometric_length + step_px
                    next_state = (nx_cell, ny_cell, dir_idx)
                    new_cost = (new_weighted, new_turns, new_length)

                    # 先和此 cell 的其他 owner 波前碰撞，再決定是否更新本方向 state。
                    for other_dir in (-1, 0, 1, 2, 3):
                        other_state = (nx_cell, ny_cell, other_dir)
                        if other_state not in best_cost:
                            continue
                        other_owner = state_owner[other_state]
                        if other_owner == owner:
                            continue
                        register_collision(
                            state, new_cost, owner,
                            other_state, best_cost[other_state], other_owner,
                            a_tail=(nx_cell, ny_cell, dir_idx),
                        )

                    old_cost = best_cost.get(next_state)
                    if old_cost is not None:
                        # 不同 owner 到達相同方向 state 時已在上方登記碰撞；state
                        # 仍保留成本較低的 owner，符合 multi-source Voronoi 搜尋。
                        if new_cost >= old_cost:
                            continue
                    best_cost[next_state] = new_cost
                    state_owner[next_state] = owner
                    came_from[next_state] = state
                    heapq.heappush(
                        open_heap,
                        (new_weighted, new_turns, new_length, sequence, next_state),
                    )
                    sequence += 1

                # 足以形成每個可達來源群的 component forest 後，再多收集少量
                # exact-projection 備選，避免把大片已知空白區的所有方向 state 掃完。
                if expansions % 512 == 0 and collision_coverage_complete():
                    if coverage_complete_at is None:
                        coverage_complete_at = expansions
                    elif expansions - coverage_complete_at >= extra_after_coverage:
                        break

            def state_path_cells(state):
                cells = [(int(state[0]), int(state[1]))]
                guard = 0
                while state in came_from and guard <= len(came_from) + 1:
                    guard += 1
                    state = came_from[state]
                    cells.append((int(state[0]), int(state[1])))
                cells.reverse()
                return cells

            def cell_to_point(cell):
                return (
                    int(np.clip(cell[0] * astar_scale + astar_scale // 2, 0, W - 1)),
                    int(np.clip(cell[1] * astar_scale + astar_scale // 2, 0, H - 1)),
                )

            candidates = []
            for (owner_a, owner_b), cell in direct_overlap_candidates.items():
                candidates.append({
                    "weighted": 0.0,
                    "turns": 0,
                    "length": 0.0,
                    "a_owner": owner_a,
                    "b_owner": owner_b,
                    "raw_cells": [cell],
                })

            for rec in collisions.values():
                a_cells = state_path_cells(rec["a_state"])
                if rec.get("a_tail") is not None:
                    tail_cell = (int(rec["a_tail"][0]), int(rec["a_tail"][1]))
                    if not a_cells or a_cells[-1] != tail_cell:
                        a_cells.append(tail_cell)
                b_cells = state_path_cells(rec["b_state"])
                raw_cells = a_cells + list(reversed(b_cells[:-1]))
                candidates.append({**rec, "raw_cells": raw_cells})

            if not candidates:
                return {
                    "status": "search_limit" if expansions >= axis_bridge_max_expansions else "no_axis_path",
                    "mode_name": mode_name,
                    "expansions": int(expansions),
                    "active_components": int(len(active_components)),
                    "frontier_collisions": 0,
                }

            # exact projection/refinement 只處理每對 component 的最佳碰撞，不再重跑搜尋。
            valid = []
            for rec in sorted(candidates, key=lambda r: (r["weighted"], r["turns"], r["length"])):
                owner_a = int(rec["a_owner"])
                owner_b = int(rec["b_owner"])
                raw_points = compress_orthogonal_points([cell_to_point(c) for c in rec["raw_cells"]])
                exact = exact_contact_path(raw_points, comps[owner_a], comps[owner_b], route_mask)
                if exact is None:
                    continue
                points = exact["points"]
                path_len = float(sum(
                    abs(b[0] - a[0]) + abs(b[1] - a[1])
                    for a, b in zip(points[:-1], points[1:])
                ))
                turn_count = int(count_axis_turns(points))
                if turn_count >= 10 ** 9:
                    continue
                path_cost = (
                    float(rec["weighted"])
                    + float(mode["mode_penalty"])
                    + max(0, turn_count - int(rec["turns"])) * bridge_turn_penalty_px
                )
                valid.append({
                    "status": "ok",
                    "points": points,
                    "path_len": path_len,
                    "path_cost": float(path_cost),
                    "turn_count": turn_count,
                    "mode_name": mode_name,
                    "mode_rank": int(mode["mode_rank"]),
                    "route_mask": route_mask,
                    "source_projection": exact["source_projection"],
                    "target_projection": exact["target_projection"],
                    "expansions": int(expansions),
                    "source_component_index": owner_a,
                    "target_component_index": owner_b,
                    "active_components": int(len(active_components)),
                    "frontier_collisions": int(len(candidates)),
                    "search_engine": "global_multisource_directional_dijkstra",
                })

            if not valid:
                return {
                    "status": "contact_projection_failed",
                    "mode_name": mode_name,
                    "expansions": int(expansions),
                    "active_components": int(len(active_components)),
                    "frontier_collisions": int(len(candidates)),
                }

            valid.sort(key=lambda r: (r["path_cost"], r["turn_count"], r["path_len"]))
            # 同一安全 mode 內，以 Kruskal 規則挑出 component-level forest。
            # 這些橋各自都已完整通過 exact projection 與安全驗證；批次安裝只省掉
            # 「裝一條後又重掃整張地圖」的重工，不會加入額外或較寬鬆的路徑。
            dsu_parent = list(range(len(comps)))

            def dsu_find(x):
                while dsu_parent[x] != x:
                    dsu_parent[x] = dsu_parent[dsu_parent[x]]
                    x = dsu_parent[x]
                return x

            batch_results = []
            for rec in valid:
                ra = dsu_find(int(rec["source_component_index"]))
                rb = dsu_find(int(rec["target_component_index"]))
                if ra == rb:
                    continue
                dsu_parent[rb] = ra
                batch_results.append(rec)
                if len(batch_results) >= len(comps) - 1:
                    break

            result = dict(batch_results[0])
            result["batch_results"] = batch_results
            result["batch_bridge_count"] = int(len(batch_results))
            return result


        def split_axis_edge_at_projection(component_nodes, projection_record):
            """在任意道路邊中間建立接觸節點並切分原邊。"""
            _, u, v, proj, edata = projection_record
            proj = tuple(map(int, proj))

            if u == v:
                return u, proj, False
            if u not in G or v not in G or not G.has_edge(u, v):
                # 邊可能因同一輪另一端切分而改變；重新投影到更新後的 component。
                # 批次橋接時，同一 component 可能已新增 split/contact node；從任一
                # 尚存舊節點展開到目前 live component，避免第二條安全橋因 stale set 失敗。
                seed = next((n for n in component_nodes if n in G), None)
                live_component = (
                    set(nx.node_connected_component(G, seed))
                    if seed is not None else set(component_nodes)
                )
                refreshed = nearest_axis_edge_projection(proj, live_component)
                if refreshed is None:
                    return None
                _, u, v, proj, edata = refreshed
                proj = tuple(map(int, proj))
                if u == v:
                    return u, proj, False

            p1 = tuple(map(int, G.nodes[u]["pos"]))
            p2 = tuple(map(int, G.nodes[v]["pos"]))
            if proj == p1:
                return u, proj, False
            if proj == p2:
                return v, proj, False

            old = dict(G.edges[u, v])
            old_type = old.get("edge_type", "route")
            G.remove_edge(u, v)
            nid = add_waypoint(proj, node_type="bridge_contact_waypoint")
            add_graph_edge(u, nid, edge_type=old_type, mask_ratio=old.get("mask_ratio"))
            add_graph_edge(nid, v, edge_type=old_type, mask_ratio=old.get("mask_ratio"))
            return nid, proj, True

        def install_axis_component_bridge(result, small_component, main_component):
            source_info = split_axis_edge_at_projection(
                small_component, result["source_projection"]
            )
            target_info = split_axis_edge_at_projection(
                main_component, result["target_projection"]
            )
            if source_info is None or target_info is None:
                return None

            source_id, source_pt, source_created = source_info
            target_id, target_pt, target_created = target_info
            points = list(result["points"])
            points[0] = tuple(source_pt)
            points[-1] = tuple(target_pt)
            points = compress_orthogonal_points(points)

            if any(not is_axis_segment(a, b) for a, b in zip(points[:-1], points[1:])):
                return None

            node_ids = [source_id]
            for p in points[1:-1]:
                node_ids.append(add_waypoint(p, node_type="bridge_turn_waypoint"))
            node_ids.append(target_id)
            for a, b in zip(node_ids[:-1], node_ids[1:]):
                if a != b:
                    add_graph_edge(a, b, edge_type="component_bridge")

            return {
                "source": source_id,
                "target": target_id,
                "source_point": tuple(source_pt),
                "target_point": tuple(target_pt),
                "source_contact_created": bool(source_created),
                "target_contact_created": bool(target_created),
                "points": points,
                "node_ids": node_ids,
            }

        def approximate_component_distance(comp_a, comp_b):
            """僅供警告訊息使用；實際橋接端點仍由多起點搜尋決定。"""
            a = [G.nodes[n]["pos"] for n in comp_a if n in G]
            b = [G.nodes[n]["pos"] for n in comp_b if n in G]
            if not a or not b:
                return None
            aa = np.asarray(a, dtype=np.float32)
            bb = np.asarray(b, dtype=np.float32)
            best = float("inf")
            block = 256
            for i in range(0, len(aa), block):
                d2 = np.sum((aa[i:i + block, None, :] - bb[None, :, :]) ** 2, axis=2)
                best = min(best, float(np.min(d2)))
            return math.sqrt(best)

        def component_bbox_gap(comp_a, comp_b):
            """只用 bbox 做便宜的 pair 排序；真正可通行性仍由 Dijkstra 判定。"""
            pts_a = [G.nodes[n]["pos"] for n in comp_a if n in G]
            pts_b = [G.nodes[n]["pos"] for n in comp_b if n in G]
            if not pts_a or not pts_b:
                return float("inf")
            ax = [p[0] for p in pts_a]
            ay = [p[1] for p in pts_a]
            bx = [p[0] for p in pts_b]
            by = [p[1] for p in pts_b]
            dx = max(0, min(bx) - max(ax), min(ax) - max(bx))
            dy = max(0, min(by) - max(ay), min(ay) - max(by))
            return float(math.hypot(dx, dy))

        def component_total_length(component_nodes):
            node_set = set(component_nodes)
            total = 0.0
            for u, v, edata in G.edges(data=True):
                if u in node_set and v in node_set:
                    total += float(edata.get("weight", 0.0))
            return total

        def component_attachment_count(component_nodes):
            node_set = set(component_nodes)
            return sum(
                1
                for info in room_attachments.values()
                if info.get("attachment_node") in node_set
            )

        def candidate_component_pairs(comps):
            """
            不再只嘗試 small -> largest。
            先建立所有 component pair，再以 bbox gap 排序；component 很多時保留：
            - 全域最近 pair
            - 最大 component 與每一塊的 pair
            這相當於 graph-of-components 的 MST 候選，而不是 star-only 拓樸。
            """
            pairs = []
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    pairs.append((component_bbox_gap(comps[i], comps[j]), i, j))
            pairs.sort(key=lambda rec: rec[0])
            if len(comps) <= 12:
                return pairs

            pair_limit = max(24, int(os.environ.get("MAP_BRIDGE_PAIR_CANDIDATES", "48")))
            selected = {(i, j): (d, i, j) for d, i, j in pairs[:pair_limit]}
            for d, i, j in pairs:
                if i == 0 or j == 0:
                    selected[(i, j)] = (d, i, j)
            return sorted(selected.values(), key=lambda rec: rec[0])

        def try_global_component_bridge(comps, modes):
            """
            以 mode 安全層級為外層、component pair 為內層：
            只要較嚴格的 mode 存在任一可行 pair，就不會跑到更寬鬆的 mode。
            """
            bridge_engine = str(
                os.environ.get("MAP_COMPONENT_BRIDGE_ENGINE", "global_multisource")
            ).strip().lower()

            # FAST V2：預設一次搜尋所有 components。保留 pairwise engine 作為
            # 回歸比對開關，但不再讓一般執行承受 O(C^2) 次全圖 Dijkstra。
            if bridge_engine not in {"pairwise", "legacy"}:
                mode_failures = []
                for mode in modes:
                    result = global_multisource_axis_bridge(comps, mode)
                    if result.get("status") == "ok":
                        success_attempt = {
                            "mode": mode["name"],
                            "status": "ok",
                            "expansions": int(result.get("expansions", 0)),
                            "active_components": int(result.get("active_components", 0)),
                            "frontier_collisions": int(result.get("frontier_collisions", 0)),
                            "search_engine": result.get("search_engine"),
                            "batch_bridge_count": int(result.get("batch_bridge_count", 1)),
                        }
                        batch = []
                        for batch_result in result.get("batch_results", [result]):
                            batch_result["score"] = (
                                int(batch_result["mode_rank"]),
                                float(batch_result["path_cost"]),
                                int(batch_result["turn_count"]),
                                float(batch_result["path_len"]),
                            )
                            batch_result["mode_attempts"] = mode_failures + [success_attempt]
                            bi = int(batch_result["source_component_index"])
                            bj = int(batch_result["target_component_index"])
                            batch.append({
                                "source_component": comps[bi],
                                "target_component": comps[bj],
                                "source_component_index": bi,
                                "target_component_index": bj,
                                "rough_distance_px": float(component_bbox_gap(comps[bi], comps[bj])),
                                "result": batch_result,
                            })

                        first = batch[0]
                        return {
                            "status": "ok",
                            **first,
                            "batch": batch,
                            "failed_attempts": mode_failures,
                        }
                    mode_failures.append({
                        "mode": mode["name"],
                        "status": result.get("status", "no_axis_path"),
                        "expansions": int(result.get("expansions", 0)),
                        "active_components": int(result.get("active_components", 0)),
                        "frontier_collisions": int(result.get("frontier_collisions", 0)),
                        "search_engine": "global_multisource_directional_dijkstra",
                    })
                return {
                    "status": "no_axis_path",
                    "failed_attempts": mode_failures,
                }

            pair_candidates = candidate_component_pairs(comps)
            all_failures = []

            for mode in modes:
                successes = []
                mode_failures = []
                for rough_distance, i, j in pair_candidates:
                    comp_a = comps[i]
                    comp_b = comps[j]
                    result = best_axis_component_bridge(
                        comp_a,
                        comp_b,
                        modes=[mode],
                    )
                    if result.get("status") == "ok":
                        # rough_distance 僅做極小 tie-break，不改變安全層級。
                        global_score = (
                            float(result["path_cost"]) + 0.02 * float(rough_distance),
                            int(result["turn_count"]),
                            float(result["path_len"]),
                        )
                        successes.append(
                            (global_score, rough_distance, i, j, comp_a, comp_b, result)
                        )
                    else:
                        mode_failures.append({
                            "source_component_index": int(i),
                            "target_component_index": int(j),
                            "source_component_size": int(len(comp_a)),
                            "target_component_size": int(len(comp_b)),
                            "bbox_gap_px": round(float(rough_distance), 2),
                            "reason": result.get("status", "no_axis_path"),
                            "mode_attempts": result.get("mode_attempts", []),
                        })

                if successes:
                    successes.sort(key=lambda rec: rec[0])
                    _, rough_distance, i, j, comp_a, comp_b, result = successes[0]
                    return {
                        "status": "ok",
                        "source_component": comp_a,
                        "target_component": comp_b,
                        "source_component_index": int(i),
                        "target_component_index": int(j),
                        "rough_distance_px": float(rough_distance),
                        "result": result,
                        "failed_attempts": all_failures + mode_failures,
                    }

                all_failures.extend(mode_failures)

            return {
                "status": "no_axis_path",
                "failed_attempts": all_failures,
            }

        def remove_unreachable_component(component_nodes, reason):
            node_set = set(component_nodes)
            record = {
                "component_size": int(len(node_set)),
                "route_length_px": round(float(component_total_length(node_set)), 2),
                "room_attachment_count": int(component_attachment_count(node_set)),
                "reason": str(reason),
                "action": "pruned_from_navigation_graph",
            }
            for nid in list(node_set):
                if nid not in G:
                    continue
                pt = tuple(G.nodes[nid]["pos"])
                G.remove_node(nid)
                nodes_data.pop(nid, None)
                if wp_node_map.get(pt) == nid:
                    wp_node_map.pop(pt, None)
            pruned_component_records.append(record)
            return record

        initial_component_count = nx.number_connected_components(G) if G.number_of_nodes() else 0
        component_merge_start = time.perf_counter()
        bridge_safety = 0
        max_bridge_rounds = max(24, initial_component_count * 4)
        unresolved_policy = str(
            os.environ.get("MAP_UNREACHABLE_COMPONENT_POLICY", "keep")
        ).strip().lower()

        while G.number_of_nodes() > 0 and not nx.is_connected(G):
            bridge_safety += 1
            if bridge_safety > max_bridge_rounds:
                print("[警告] V11 component 合併超過安全輪數，停止搜尋。")
                break

            comps = sorted(
                (set(c) for c in nx.connected_components(G)),
                key=component_sort_key,
                reverse=True,
            )

            # 第一階段：只在 corridor-aware 的 strict / adaptive band 中搜尋。
            chosen = try_global_component_bridge(comps, bridge_modes)

            # 第二階段：如果所有 corridor-aware pair 都失敗，再讓 wall topology 裁決。
            # 這一步仍禁止穿 wall；只是允許穿過「語意分割漏標」的 free-space。
            used_fallback = False
            if chosen.get("status") != "ok":
                fallback = try_global_component_bridge(comps, fallback_bridge_modes)
                if fallback.get("status") == "ok":
                    chosen = fallback
                    used_fallback = True
                else:
                    # wall-free 仍找不到路，表示在目前 wall topology 下這些 component
                    # 真的是物理不連通。此時「強畫一條橘線」只會穿牆。
                    # 導航圖預設採 connect-or-prune：保留最大可導航 component，
                    # 其餘 unreachable candidate 移除，房間稍後會重新吸附到主路網。
                    failed_attempts = (
                        chosen.get("failed_attempts", [])
                        + fallback.get("failed_attempts", [])
                    )
                    unconnected_records = failed_attempts

                    if unresolved_policy in {"prune", "connect_or_prune", "single"}:
                        primary = comps[0]
                        removed = []
                        for component in comps[1:]:
                            removed.append(
                                remove_unreachable_component(
                                    component,
                                    reason="no_wall_safe_path_after_adaptive_bridge",
                                )
                            )
                        print(
                            f"[V11] wall-safe 橋接仍失敗；已保留最大路網 "
                            f"({len(primary)} nodes) 並移除 {len(removed)} 個不可達候選 component。"
                        )
                        # G 此時理論上只剩 primary；下一輪/while 條件會自然結束。
                        continue

                    print(
                        "[警告] 仍有高層拓樸互不連通，且 MAP_UNREACHABLE_COMPONENT_POLICY=keep；"
                        "保留多 component 供人工檢查。"
                    )
                    break

            batch_items = chosen.get("batch") or [chosen]
            installed_count = 0
            for batch_index, item in enumerate(batch_items, start=1):
                result = item["result"]
                source_component = item["source_component"]
                target_component = item["target_component"]
                installed = install_axis_component_bridge(
                    result,
                    source_component,
                    target_component,
                )
                if installed is None:
                    unconnected_records.append({
                        "source_component_size": int(len(source_component)),
                        "target_component_size": int(len(target_component)),
                        "reason": "batch_bridge_installation_failed",
                        "mode_attempts": result.get("mode_attempts", []),
                    })
                    continue

                installed_count += 1
                points = installed["points"]
                bridge_records.append({
                    "source": installed["source"],
                    "target": installed["target"],
                    "source_component_size": int(len(source_component)),
                    "target_component_size": int(len(target_component)),
                    "source_contact_point": list(map(int, installed["source_point"])),
                    "target_contact_point": list(map(int, installed["target_point"])),
                    "source_contact_node_created": installed["source_contact_created"],
                    "target_contact_node_created": installed["target_contact_created"],
                    "rough_component_gap_px": round(float(item["rough_distance_px"]), 2),
                    "path_distance_px": round(float(result["path_len"]), 2),
                    "weighted_path_cost": round(float(result.get("path_cost", result["path_len"])), 2),
                    "turn_count": int(result["turn_count"]),
                    "astar_mode": result["mode_name"],
                    "fallback_used": bool(used_fallback),
                    "bridge_strategy": "global_multisource_mst_four_direction_adaptive_connect_or_prune",
                    "bridge_batch_index": int(batch_index),
                    "bridge_batch_size": int(len(batch_items)),
                    "path": [list(map(int, p)) for p in points],
                    "new_turn_nodes": [list(map(int, p)) for p in points[1:-1]],
                    "graph_node_ids": installed["node_ids"],
                    "mode_attempts": result.get("mode_attempts", []),
                })
                print(
                    f"[完成] V11 批次合併路網 {batch_index}/{len(batch_items)}："
                    f"{len(source_component)} nodes <-> {len(target_component)} nodes，"
                    f"接觸點 {installed['source_point']} -> {installed['target_point']}，"
                    f"轉彎 {result['turn_count']} 次，距離 {result['path_len']:.1f}px，"
                    f"模式 {result['mode_name']}"
                    + ("（wall-topology fallback）" if used_fallback else "")
                    + "。"
                )

            if installed_count == 0:
                print("[警告] V11 找到橋接候選，但本批 edge split 全部安裝失敗；停止本輪。")
                break

        component_merge_elapsed = time.perf_counter() - component_merge_start
        print(
            f"[計時] 路網 component 合併：{component_merge_elapsed:.3f} 秒，"
            f"initial_components={initial_component_count}，bridges={len(bridge_records)}"
        )

        # -------------------------------------------------
        # 11. V13.1 Topology-First Cycle Preservation + Canonical Graph Fusion
        #
        # V12 從「幾何上接近的 edge pair」猜 shortcut，因此會有兩個根本限制：
        #   1) 大環的缺失支路可能跨越數百/數千 pixel，根本不會進入 near-edge 候選。
        #   2) 小型密集區的 10~20px 微小 gap 反而會優先吃掉 closure budget。
        #
        # V13 改成拓樸優先：
        #   A. 從實際 walkable mask 建立 reference skeleton，不先看導航 graph。
        #   B. 找 reference skeleton 中「實際存在、但目前 graph 沒覆蓋」的長支路；
        #      只要該支路兩端能掛回既有 graph，就以四方向路徑補回，無距離上限。
        #   C. 對目前 graph 的 bridge-chain 做 physical bypass test：把唯一道路暫時封住，
        #      若 walkable mask 中仍能從 chain 一端走到另一端，表示真實地圖有第二條路，
        #      graph 卻把它錯畫成單一路徑；此時補回 alternate path。
        #   D. 對 walkable mask 中跨多個 clearance 尺度仍存在的 enclosed hole 做 cycle census；
        #      若 graph 沒有 cycle 包圍該 hole，直接從 reference skeleton 的實際 cycle 還原。
        #
        # 最終新增線仍只允許 H/V；菱形、圓弧或斜走道會以安全的 rectilinear staircase 表示。
        # -------------------------------------------------
        topology_cycle_records = []
        topology_unresolved_records = []
        topology_stage_start = time.perf_counter()

        # -------------
        # 11.1 Topology reference mask
        # -------------
        topology_close_radius = int(np.clip(round(corridor_width_est * 0.08), 1, 8))
        topology_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (topology_close_radius * 2 + 1, topology_close_radius * 2 + 1),
        )
        topology_walkable_mask = cv2.morphologyEx(
            valid_routing_mask,
            cv2.MORPH_CLOSE,
            topology_close_kernel,
        )
        # closing 只能補走道標記的小缺口；wall 仍然具有最高優先權，絕不穿牆。
        topology_walkable_mask = cv2.bitwise_and(topology_walkable_mask, wall_free)

        # corridor expansion 可能跨過一條牆後，在牆的另一側產生「沒有任何 corridor seed」的假 walkable band。
        # 尤其外牆會因此在地圖外側形成一圈假的環。V13 只保留真正包含 pure_corridor seed 的
        # connected component；這是 topology reference 與一般 relaxed routing 最大的差別。
        topo_count, topo_labels, topo_stats, _ = cv2.connectedComponentsWithStats(
            (topology_walkable_mask > 0).astype(np.uint8), connectivity=8
        )
        topo_keep = np.zeros(topo_count, dtype=bool)
        pure_seed = (pure_corridor_mask > 0)
        for topo_id in range(1, topo_count):
            area = int(topo_stats[topo_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            seed_pixels = int(np.count_nonzero(pure_seed & (topo_labels == topo_id)))
            seed_gate = max(3, int(round(area * 0.002)))
            if seed_pixels >= seed_gate:
                topo_keep[topo_id] = True
        if np.any(topo_keep):
            topology_walkable_mask = (topo_keep[topo_labels].astype(np.uint8) * 255)
        topology_walkable_mask = cv2.bitwise_and(topology_walkable_mask, wall_free)
        reference_skeleton = _skeletonize_uint8(topology_walkable_mask)

        # V13 的 loop 判定由 persistent-hole census 與 bridge-bypass 負責；
        # reference skeleton 本身只提供「實際走道中心線」幾何，不再建立昂貴的 pixel-level cycle graph。
        reference_skeleton_node_count = int(cv2.countNonZero(reference_skeleton))

        # reference skeleton 的 coverage tolerance 必須接近「同一條寬走道內的中心線偏移」，
        # 但不能大到把真正平行的另一側環路一起吞掉。
        represented_radius = int(np.clip(
            round(corridor_width_est * 0.48),
            5,
            max(10, round(map_short_side * 0.035)),
        ))
        reference_support_radius = int(np.clip(
            round(corridor_width_est * 0.62),
            represented_radius + 2,
            max(represented_radius + 4, round(map_short_side * 0.05)),
        ))
        reference_min_branch_pixels = int(max(
            10,
            round(corridor_width_est * 0.55),
            round(map_short_side * 0.006),
        ))

        # -------------
        # 11.2 Persistent enclosed-hole detection
        # -------------
        def _v13_enclosed_components(walk_mask, min_area):
            inv = (walk_mask == 0).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
            if count <= 1:
                return [], labels, set()
            border_ids = set(map(int, np.unique(np.concatenate([
                labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
            ]))))
            records = []
            for cid in range(1, count):
                if cid in border_ids:
                    continue
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area < int(min_area):
                    continue
                x = int(stats[cid, cv2.CC_STAT_LEFT])
                y = int(stats[cid, cv2.CC_STAT_TOP])
                w = int(stats[cid, cv2.CC_STAT_WIDTH])
                h = int(stats[cid, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[cid]
                records.append({
                    'label': int(cid),
                    'area': area,
                    'bbox': [x, y, w, h],
                    'centroid': [int(round(cx)), int(round(cy))],
                })
            return records, labels, border_ids

        v13_min_hole_area = int(max(
            80,
            corridor_width_est * corridor_width_est * 0.45,
            H * W * 0.00012,
        ))

        # 0 / 5% / 10% 典型走道寬度的 clearance。真正的環通常在多個尺度仍保持 enclosed；
        # 文字、icon 或單像素小洞通常不會通過這個 persistence gate。
        persistence_radii = sorted(set([
            0,
            int(max(1, round(corridor_width_est * 0.05))),
            int(max(1, round(corridor_width_est * 0.10))),
        ]))
        persistence_maps = []
        for radius in persistence_radii:
            if radius <= 0:
                wm = topology_walkable_mask.copy()
            else:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                wm = cv2.erode(topology_walkable_mask, k, iterations=1)
            holes, labels, border_ids = _v13_enclosed_components(wm, max(24, int(v13_min_hole_area * 0.45)))
            persistence_maps.append((radius, wm, holes, labels, border_ids))

        base_holes, _, _ = _v13_enclosed_components(topology_walkable_mask, v13_min_hole_area)
        persistent_holes = []
        for idx, hole in enumerate(base_holes):
            cx, cy = map(int, hole['centroid'])
            survives = 0
            scales = []
            for radius, wm, holes_at_scale, labels, border_ids in persistence_maps:
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                cid = int(labels[cy, cx])
                if cid > 0 and cid not in border_ids:
                    area = int(np.sum(labels == cid))
                    if area >= max(24, int(v13_min_hole_area * 0.35)):
                        survives += 1
                        scales.append(int(radius))
            # 至少兩個尺度存在；只有單一尺度時多半是 morphology/文字造成的小洞。
            if survives >= min(2, len(persistence_maps)):
                rec = dict(hole)
                rec['hole_id'] = f'H_{idx}'
                rec['persistence_count'] = int(survives)
                rec['persistence_radii_px'] = scales
                persistent_holes.append(rec)

        # 同一個實體 obstacle 可能因 wall-collision / morphology 形成數個同心 enclosed component。
        # 以 bbox 高重疊 + centroid 接近做 nested-hole 去重，避免同一個環被算成 H_0/H_1/H_2 多次。
        if len(persistent_holes) > 1:
            ordered = sorted(persistent_holes, key=lambda h: (int(h['area']), h['bbox'][2]*h['bbox'][3]))
            deduped = []
            centroid_tol = float(max(4.0, corridor_width_est * 0.22))
            for hole in ordered:
                hx,hy,hw,hh = map(int,hole['bbox'])
                hcx,hcy = map(float,hole['centroid'])
                duplicate = False
                for kept in deduped:
                    kx,ky,kw,kh = map(int,kept['bbox'])
                    kcx,kcy = map(float,kept['centroid'])
                    if math.hypot(hcx-kcx,hcy-kcy) > centroid_tol:
                        continue
                    ix1,iy1=max(hx,kx),max(hy,ky)
                    ix2,iy2=min(hx+hw,kx+kw),min(hy+hh,ky+kh)
                    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                    smaller=max(1,min(hw*hh,kw*kh))
                    if inter/float(smaller) >= 0.72:
                        duplicate=True
                        break
                if not duplicate:
                    deduped.append(hole)
            persistent_holes = deduped
            for idx,hole in enumerate(persistent_holes):
                hole['hole_id']=f'H_{idx}'

        # -------------
        # 11.3 Current graph geometry helpers
        # -------------
        v13_axis_edge_cache = {"signature": None, "records": []}
        v13_cycle_basis_cache = {"signature": None, "basis": []}

        def v13_graph_signature():
            # 所有 topology 安裝／edge split 都會改變至少一個計數；以常數時間
            # signature 讓同一輪的 projection 與 hole checks 共用昂貴圖論結果。
            return (int(G.number_of_nodes()), int(G.number_of_edges()))

        def v13_axis_edge_records():
            signature = v13_graph_signature()
            if v13_axis_edge_cache["signature"] == signature:
                return v13_axis_edge_cache["records"]
            records = []
            for u, v, edata in list(G.edges(data=True)):
                if u not in G or v not in G:
                    continue
                p1 = tuple(map(int, G.nodes[u]['pos']))
                p2 = tuple(map(int, G.nodes[v]['pos']))
                if not is_axis_segment(p1, p2):
                    continue
                records.append({
                    'u': u,
                    'v': v,
                    'p1': p1,
                    'p2': p2,
                    'edata': dict(edata),
                    'horizontal': bool(p1[1] == p2[1]),
                    'length': float(abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])),
                })
            v13_axis_edge_cache["signature"] = signature
            v13_axis_edge_cache["records"] = records
            return records

        def v13_cycle_basis_cached():
            signature = v13_graph_signature()
            if v13_cycle_basis_cache["signature"] != signature:
                try:
                    basis = nx.cycle_basis(G)
                except nx.NetworkXError:
                    basis = []
                v13_cycle_basis_cache["signature"] = signature
                v13_cycle_basis_cache["basis"] = basis
            return v13_cycle_basis_cache["basis"]

        def v13_rasterize_graph(thickness=3):
            mask = np.zeros((H, W), dtype=np.uint8)
            for u, v in list(G.edges()):
                if u not in G or v not in G:
                    continue
                p1 = tuple(map(int, G.nodes[u]['pos']))
                p2 = tuple(map(int, G.nodes[v]['pos']))
                # V14 topology backfill 可使用經 free-space 視線驗證的斜向長直線；
                # coverage census 必須把它納入，否則同一條已補路段會被重複修補。
                cv2.line(mask, p1, p2, 255, max(1, int(thickness)))
            return mask

        def v13_project_to_edge(point, edge):
            px, py = map(int, point)
            p1, p2 = edge['p1'], edge['p2']
            if edge['horizontal']:
                x = int(max(min(p1[0], p2[0]), min(px, max(p1[0], p2[0]))))
                return (x, int(p1[1]))
            y = int(max(min(p1[1], p2[1]), min(py, max(p1[1], p2[1]))))
            return (int(p1[0]), y)

        def v13_projection_candidates(point, max_distance=None, limit=6):
            point = tuple(map(int, point))
            candidates = []
            for edge in v13_axis_edge_records():
                proj = v13_project_to_edge(point, edge)
                d = float(math.hypot(proj[0] - point[0], proj[1] - point[1]))
                if max_distance is not None and d > float(max_distance):
                    continue
                if not in_bounds(proj[0], proj[1]) or wall_collision[proj[1], proj[0]] > 0:
                    continue
                candidates.append((d, edge, proj))
            candidates.sort(key=lambda rec: rec[0])
            return candidates[:max(1, int(limit))]

        v13_distance_cache = {}
        def v13_endpoint_lengths(node):
            if node not in v13_distance_cache:
                v13_distance_cache[node] = nx.single_source_dijkstra_path_length(G, node, weight='weight')
            return v13_distance_cache[node]

        def v13_graph_distance_between_projections(edge_a, pa, edge_b, pb):
            best = float('inf')
            for na in (edge_a['u'], edge_a['v']):
                if na not in G:
                    continue
                da = float(abs(pa[0] - G.nodes[na]['pos'][0]) + abs(pa[1] - G.nodes[na]['pos'][1]))
                lengths = v13_endpoint_lengths(na)
                for nb in (edge_b['u'], edge_b['v']):
                    if nb not in G or nb not in lengths:
                        continue
                    db = float(abs(pb[0] - G.nodes[nb]['pos'][0]) + abs(pb[1] - G.nodes[nb]['pos'][1]))
                    best = min(best, da + float(lengths[nb]) + db)
            return best

        def v13_point_on_axis_segment(pt, a, b):
            x, y = map(int, pt)
            ax, ay = map(int, a)
            bx, by = map(int, b)
            if ay == by == y:
                return min(ax, bx) <= x <= max(ax, bx)
            if ax == bx == x:
                return min(ay, by) <= y <= max(ay, by)
            return False

        def v13_ensure_graph_node(pt, node_type='topology_cycle_waypoint'):
            pt = tuple(map(int, pt))
            existing = wp_node_map.get(pt)
            if existing in G:
                return existing, False

            # 若點落在既有 axis edge 中間，先 split，確保交叉點在圖論上真的相連。
            for u, v, edata in list(G.edges(data=True)):
                if u not in G or v not in G:
                    continue
                p1 = tuple(map(int, G.nodes[u]['pos']))
                p2 = tuple(map(int, G.nodes[v]['pos']))
                if not is_axis_segment(p1, p2) or not v13_point_on_axis_segment(pt, p1, p2):
                    continue
                if pt == p1:
                    return u, False
                if pt == p2:
                    return v, False
                old = dict(edata)
                G.remove_edge(u, v)
                nid = add_waypoint(pt, node_type=node_type)
                add_graph_edge(u, nid, edge_type=old.get('edge_type', 'route'), mask_ratio=old.get('mask_ratio'))
                add_graph_edge(nid, v, edge_type=old.get('edge_type', 'route'), mask_ratio=old.get('mask_ratio'))
                return nid, True

            return add_waypoint(pt, node_type=node_type), True

        def v13_segment_intersections(a, b):
            a = tuple(map(int, a))
            b = tuple(map(int, b))
            pts = {a, b}
            horizontal = a[1] == b[1]
            if not horizontal and a[0] != b[0]:
                return [a, b]

            for edge in v13_axis_edge_records():
                p1, p2 = edge['p1'], edge['p2']
                if horizontal:
                    y = a[1]
                    xlo, xhi = sorted((a[0], b[0]))
                    if edge['horizontal']:
                        if p1[1] != y:
                            continue
                        for p in (p1, p2):
                            if xlo <= p[0] <= xhi:
                                pts.add((int(p[0]), int(y)))
                    else:
                        x = p1[0]
                        ylo, yhi = sorted((p1[1], p2[1]))
                        if xlo <= x <= xhi and ylo <= y <= yhi:
                            pts.add((int(x), int(y)))
                else:
                    x = a[0]
                    ylo, yhi = sorted((a[1], b[1]))
                    if not edge['horizontal']:
                        if p1[0] != x:
                            continue
                        for p in (p1, p2):
                            if ylo <= p[1] <= yhi:
                                pts.add((int(x), int(p[1])))
                    else:
                        y = p1[1]
                        xlo, xhi = sorted((p1[0], p2[0]))
                        if ylo <= y <= yhi and xlo <= x <= xhi:
                            pts.add((int(x), int(y)))

            if horizontal:
                return sorted(pts, key=lambda p: (p[0] - a[0]) * (1 if b[0] >= a[0] else -1))
            return sorted(pts, key=lambda p: (p[1] - a[1]) * (1 if b[1] >= a[1] else -1))

        def v13_install_path(points, strategy):
            points = compress_straight_points(points)
            if len(points) < 2:
                return None

            new_edges = []
            created_nodes = []
            full_node_ids = []
            for seg_a, seg_b in zip(points[:-1], points[1:]):
                if seg_a == seg_b:
                    continue
                cut_points = v13_segment_intersections(seg_a, seg_b)
                seg_node_ids = []
                for pt in cut_points:
                    nid, created = v13_ensure_graph_node(pt)
                    seg_node_ids.append(nid)
                    if created:
                        created_nodes.append(nid)
                if full_node_ids:
                    if seg_node_ids and full_node_ids[-1] == seg_node_ids[0]:
                        full_node_ids.extend(seg_node_ids[1:])
                    else:
                        full_node_ids.extend(seg_node_ids)
                else:
                    full_node_ids.extend(seg_node_ids)

                for na, nb in zip(seg_node_ids[:-1], seg_node_ids[1:]):
                    if na == nb:
                        continue
                    if G.has_edge(na, nb):
                        # 已有道路就沿用，不把原本 route 整段染成 shortcut。
                        continue
                    add_graph_edge(na, nb, edge_type='shortcut')
                    if G.has_edge(na, nb):
                        G.edges[na, nb]['topology_cycle'] = True
                        G.edges[na, nb]['topology_strategy'] = str(strategy)
                        G.edges[na, nb]['movement'] = 'straight_visibility_segment_preferred'
                        new_edges.append((na, nb))

            if not new_edges:
                return None
            v13_distance_cache.clear()
            return {
                'node_ids': full_node_ids,
                'created_nodes': created_nodes,
                'new_edges': new_edges,
                'new_edge_count': int(len(new_edges)),
            }

        # -------------
        # 11.4 Global four-direction search without gap limit
        # -------------
        v13_turn_penalty_px = float(max(4.0, corridor_width_est * 0.18))
        v14_backfill_turn_multiplier = float(np.clip(
            float(os.environ.get("MAP_BACKFILL_STRAIGHT_TURN_MULTIPLIER", "2.4")),
            1.0,
            8.0,
        ))
        v13_max_raw_turns = max(16, int(os.environ.get('MAP_V13_MAX_RAW_TURNS', '96')))
        v13_max_search_expansions = max(50000, int(os.environ.get('MAP_V13_MAX_SEARCH_EXPANSIONS', '650000')))

        def v13_downsample_mask(mask, scale):
            scale = max(1, int(scale))
            w = max(1, int(math.ceil(W / scale)))
            h = max(1, int(math.ceil(H / scale)))
            walk_fraction = cv2.resize((mask > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            wall_fraction = cv2.resize((wall_collision > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            return ((walk_fraction >= 0.12) & (wall_fraction <= 0.02)).astype(np.uint8)

        def v13_nearest_valid(cell, ds_mask, max_radius):
            cx, cy = map(int, cell)
            h, w = ds_mask.shape[:2]
            if 0 <= cx < w and 0 <= cy < h and ds_mask[cy, cx] > 0:
                return (cx, cy)
            for radius in range(1, max(1, int(max_radius)) + 1):
                x1, x2 = max(0, cx - radius), min(w - 1, cx + radius)
                y1, y2 = max(0, cy - radius), min(h - 1, cy + radius)
                candidates = []
                for x in range(x1, x2 + 1):
                    for y in (y1, y2):
                        if 0 <= y < h and ds_mask[y, x] > 0:
                            candidates.append((abs(x - cx) + abs(y - cy), x, y))
                for y in range(y1 + 1, y2):
                    for x in (x1, x2):
                        if 0 <= x < w and ds_mask[y, x] > 0:
                            candidates.append((abs(x - cx) + abs(y - cy), x, y))
                if candidates:
                    candidates.sort()
                    return (candidates[0][1], candidates[0][2])
            return None

        def v13_path_segments_safe(points, route_mask, ratio=0.76):
            """
            驗證 V13 rectilinear centerline。這裡刻意使用 1px centerline，而不是 2~3px brush：
            菱形/斜走道在 H/V staircase 表示時，粗 brush 會跨出窄 support tube 而造成假失敗；
            真正的安全邊界已由 wall_collision dilation 保證。
            """
            if not points or len(points) < 2:
                return False
            for a, b in zip(points[:-1], points[1:]):
                if not is_axis_segment(a, b):
                    return False
                data = local_line_mask(a, b, thickness=1)
                if data is None:
                    return False
                x1,y1,x2,y2,lm = data
                if cv2.countNonZero(cv2.bitwise_and(lm, wall_collision[y1:y2,x1:x2])) > 0:
                    return False
                total = cv2.countNonZero(lm)
                if total <= 0:
                    return False
                inside = cv2.countNonZero(cv2.bitwise_and(lm, route_mask[y1:y2,x1:x2]))
                if inside / float(total) < float(ratio):
                    return False
            return True

        def compress_straight_points(points):
            """刪除重複點與任意方向共線點；供導航補網的長直線表示使用。"""
            cleaned = []
            for point in points or []:
                pt = tuple(map(int, point))
                if not cleaned or pt != cleaned[-1]:
                    cleaned.append(pt)
            if len(cleaned) <= 2:
                return cleaned
            out = [cleaned[0]]
            for point, nxt in zip(cleaned[1:-1], cleaned[2:]):
                prev = out[-1]
                cross = (
                    (point[0] - prev[0]) * (nxt[1] - point[1])
                    - (point[1] - prev[1]) * (nxt[0] - point[0])
                )
                if cross == 0:
                    continue
                out.append(point)
            out.append(cleaned[-1])
            return out

        def straight_visibility_segment_safe(a, b, route_mask, min_ratio=0.985):
            """
            V15 strict-H/V：只允許水平或垂直的可視直線。

            舊版只檢查「兩點之間是否可直視」，因此在斜向走道中會直接把原本的
            四方向折線跨接成斜線。導航路網的正式邊必須是 rectilinear，所以任何
            x、y 同時改變的線段一律拒絕。
            """
            a = tuple(map(int, a))
            b = tuple(map(int, b))
            if not is_axis_segment(a, b):
                return False
            data = local_line_mask(a, b, thickness=1)
            if data is None:
                return False
            x1, y1, x2, y2, line = data
            if cv2.countNonZero(cv2.bitwise_and(line, wall_collision[y1:y2, x1:x2])) > 0:
                return False
            total = cv2.countNonZero(line)
            if total <= 0:
                return False
            inside = cv2.countNonZero(cv2.bitwise_and(line, route_mask[y1:y2, x1:x2]))
            return inside / float(total) >= float(min_ratio)

        def simplify_backfill_to_straight_segments(points, route_mask):
            """
            V15 strict-H/V 低轉彎化簡。

            1) 先用既有 Hanan-grid refinement 尋找較少轉彎的四方向折線；
            2) 再只合併「同 x 或同 y」而且整段安全的節點；
            3) 絕不以任意角度 straight visibility 跨接兩點。

            因此輸出的每一段都必定是水平或垂直，不會再出現 recovery graph
            中大量青色斜線，同時仍盡量保留長直線與少轉彎特性。
            """
            points = compress_orthogonal_points(points or [])
            if len(points) <= 1:
                return points

            refined = refine_axis_polyline(points, route_mask)
            if refined and v13_path_segments_safe(refined, route_mask, ratio=0.76):
                points = compress_orthogonal_points(refined)

            if len(points) <= 2:
                # 兩點路徑也必須守住 H/V invariant；正常情況下 v13_four_direction_path
                # 已保證這件事，這裡再做一次保險。
                if len(points) == 2 and not is_axis_segment(points[0], points[1]):
                    return compress_orthogonal_points(points)
                return points

            simplified = [points[0]]
            anchor = 0
            while anchor < len(points) - 1:
                chosen = anchor + 1
                # 只尋找與 anchor 共 x / 共 y 的最遠安全點；禁止斜向 shortcut。
                for idx in range(len(points) - 1, anchor, -1):
                    if not is_axis_segment(points[anchor], points[idx]):
                        continue
                    if straight_visibility_segment_safe(points[anchor], points[idx], route_mask):
                        chosen = idx
                        break
                simplified.append(points[chosen])
                anchor = chosen

            simplified = compress_orthogonal_points(simplified)
            # 最後的 invariant：所有實體 navigation segment 必須四方向。
            if any(not is_axis_segment(a, b) for a, b in zip(simplified[:-1], simplified[1:])):
                return compress_orthogonal_points(points)
            return simplified

        def count_navigation_turns(points):
            points = compress_orthogonal_points(points or [])
            if any(not is_axis_segment(a, b) for a, b in zip(points[:-1], points[1:])):
                return 10 ** 9
            return int(count_axis_turns(points))

        topology_search_budget = float(np.clip(float(os.environ.get('MAP_TOPOLOGY_SEARCH_SECONDS','8.0')), .1, 60.0))
        topology_search_deadline = time.perf_counter() + topology_search_budget
        path_search_deadline = topology_search_deadline
        topology_search_exhausted = False
        topology_search_status_counts = Counter()

        def v13_four_direction_path(start_pt, goal_pt, route_mask, turn_penalty=None):
            if path_search_deadline is not None and time.perf_counter() >= path_search_deadline:
                return {'status': 'time_budget'}
            start_pt = tuple(map(int, start_pt))
            goal_pt = tuple(map(int, goal_pt))
            if start_pt == goal_pt:
                return {'status': 'ok', 'points': [start_pt], 'path_len': 0.0, 'turn_count': 0, 'scale': 1, 'expansions': 0}

            route_mask = cv2.bitwise_and(((route_mask > 0).astype(np.uint8) * 255), wall_free)
            if cv2.countNonZero(route_mask) == 0:
                return {'status': 'empty_mask'}

            fast = _compiled_axis_path(start_pt, goal_pt, route_mask,
                v13_turn_penalty_px if turn_penalty is None else turn_penalty,
                deadline=path_search_deadline)
            topology_search_status_counts[str(fast.get('status','unknown'))] += 1
            if fast.get('status') != 'backend_unavailable':
                return fast

            # 先試簡單 0/1/2-turn；成功時可以避免大圖搜尋。
            simple = orthogonal_join(start_pt, goal_pt, route_mask)
            if simple is not None:
                refined = refine_axis_polyline(simple, route_mask)
                refined = compress_orthogonal_points(refined)
                if v13_path_segments_safe(refined, route_mask, ratio=0.90):
                    plen = float(sum(abs(b[0]-a[0]) + abs(b[1]-a[1]) for a, b in zip(refined[:-1], refined[1:])))
                    return {
                        'status': 'ok', 'points': refined, 'path_len': plen,
                        'turn_count': int(count_axis_turns(refined)), 'scale': 1, 'expansions': 0,
                    }

            penalty = float(v13_turn_penalty_px if turn_penalty is None else turn_penalty)
            search_scales = []
            for scale in (max(2, min(int(astar_scale), 5)), 2, 1):
                if scale not in search_scales:
                    search_scales.append(scale)

            last_status = 'no_path'
            total_expansions = 0
            for scale in search_scales:
                # Full-map coordinates are retained; this keeps installation/intersection logic simple.
                ds = v13_downsample_mask(route_mask, scale)
                ds_h, ds_w = ds.shape[:2]
                s0 = (int(round(start_pt[0] / scale)), int(round(start_pt[1] / scale)))
                g0 = (int(round(goal_pt[0] / scale)), int(round(goal_pt[1] / scale)))
                snap_radius = max(3, int(math.ceil(corridor_width_est / max(scale, 1))))
                s = v13_nearest_valid(s0, ds, snap_radius)
                g = v13_nearest_valid(g0, ds, snap_radius)
                if s is None or g is None:
                    last_status = 'no_contact_cells'
                    continue

                clear_ds = cv2.distanceTransform(ds, cv2.DIST_L2, 3)
                dirs = [(1,0), (0,1), (-1,0), (0,-1)]
                open_heap = []
                g_best = {}
                parent = {}
                seq = 0
                start_state = (s[0], s[1], -1)
                g_best[start_state] = (0.0, 0, 0.0)
                h0 = (abs(g[0]-s[0]) + abs(g[1]-s[1])) * float(scale)
                heapq.heappush(open_heap, (h0, 0.0, 0, 0.0, seq, start_state))
                seq += 1
                goal_state = None
                expansions = 0
                scale_limit = min(
                    v13_max_search_expansions,
                    max(70000, int(ds_w * ds_h * 2.2)),
                )

                while open_heap and expansions < scale_limit:
                    _, weighted, turns, plen, _, state = heapq.heappop(open_heap)
                    if g_best.get(state) != (weighted, turns, plen):
                        continue
                    expansions += 1
                    if (expansions % 256 == 0 and path_search_deadline is not None
                            and time.perf_counter() >= path_search_deadline):
                        return {'status': 'time_budget', 'expansions': expansions}
                    cx, cy, prev_dir = state
                    if (cx, cy) == g:
                        goal_state = state
                        break
                    for dir_idx, (dx, dy) in enumerate(dirs):
                        nx_, ny_ = cx + dx, cy + dy
                        if not (0 <= nx_ < ds_w and 0 <= ny_ < ds_h) or ds[ny_, nx_] == 0:
                            continue
                        add_turn = 0 if prev_dir in (-1, dir_idx) else 1
                        new_turns = turns + add_turn
                        if new_turns > v13_max_raw_turns:
                            continue
                        clearance_px = max(1.0, float(clear_ds[ny_, nx_]) * float(scale))
                        clearance_penalty = min(1.25, corridor_width_est / clearance_px) * 0.10
                        step = float(scale)
                        nweighted = weighted + step * (1.0 + clearance_penalty) + (penalty if add_turn else 0.0)
                        nplen = plen + step
                        nxt = (nx_, ny_, dir_idx)
                        cost = (nweighted, new_turns, nplen)
                        if cost >= g_best.get(nxt, (float('inf'), 10**9, float('inf'))):
                            continue
                        g_best[nxt] = cost
                        parent[nxt] = state
                        heuristic = (abs(g[0]-nx_) + abs(g[1]-ny_)) * float(scale)
                        heapq.heappush(open_heap, (nweighted + heuristic, nweighted, new_turns, nplen, seq, nxt))
                        seq += 1

                total_expansions += expansions
                if goal_state is None:
                    last_status = 'search_limit' if expansions >= scale_limit else 'no_path'
                    continue

                states = [goal_state]
                while states[-1] in parent:
                    states.append(parent[states[-1]])
                states.reverse()
                centers = [
                    (
                        int(np.clip(st[0] * scale + scale // 2, 0, W - 1)),
                        int(np.clip(st[1] * scale + scale // 2, 0, H - 1)),
                    )
                    for st in states
                ]
                centers = compress_orthogonal_points(centers)
                if not centers:
                    last_status = 'empty_reconstruction'
                    continue

                prefix = orthogonal_join(start_pt, centers[0], route_mask)
                suffix = orthogonal_join(centers[-1], goal_pt, route_mask)
                if prefix is None or suffix is None:
                    last_status = 'endpoint_refine_failed'
                    continue
                points = prefix[:-1] + centers + suffix[1:]
                points = compress_orthogonal_points(points)
                refined = refine_axis_polyline(points, route_mask)
                refined = compress_orthogonal_points(refined)

                # 優先使用少轉彎 refined；若它因 Hanan 抽樣失敗，保留原始 staircase。
                chosen = refined if v13_path_segments_safe(refined, route_mask, ratio=0.76) else points
                if not v13_path_segments_safe(chosen, route_mask, ratio=0.72):
                    last_status = 'full_resolution_validation_failed'
                    continue

                plen = float(sum(abs(b[0]-a[0]) + abs(b[1]-a[1]) for a, b in zip(chosen[:-1], chosen[1:])))
                return {
                    'status': 'ok',
                    'points': chosen,
                    'path_len': plen,
                    'turn_count': int(count_axis_turns(chosen)),
                    'scale': int(scale),
                    'expansions': int(total_expansions),
                }

            return {'status': last_status, 'expansions': int(total_expansions)}

        # -------------
        # 11.5 Reference-skeleton missing-branch completion
        # -------------
        def v13_graph_cycle_encloses_quick(point):
            if G.number_of_nodes() < 3 or G.number_of_edges() < 3:
                return False
            basis = v13_cycle_basis_cached()
            px, py = map(float, point)
            for cycle in basis:
                if len(cycle) < 3:
                    continue
                poly = np.asarray([G.nodes[n]['pos'] for n in cycle if n in G], dtype=np.int32)
                if len(poly) < 3:
                    continue
                try:
                    if cv2.pointPolygonTest(poly.reshape((-1,1,2)), (px,py), False) >= 0:
                        return True
                except cv2.error:
                    continue
            return False

        def v13_missing_reference_components():
            graph_mask = v13_rasterize_graph(thickness=3)
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (represented_radius * 2 + 1, represented_radius * 2 + 1),
            )
            represented = cv2.dilate(graph_mask, k, iterations=1)
            # 先找所有 reference skeleton coverage gap；後面再用 persistent-hole gate 判斷是否屬於真環。
            missing = cv2.bitwise_and(reference_skeleton, cv2.bitwise_not(represented))
            count, labels, stats, _ = cv2.connectedComponentsWithStats((missing > 0).astype(np.uint8), connectivity=8)
            # G is unchanged during this census. Evaluate each hole once,
            # instead of rebuilding every cycle polygon for every missing component.
            unclosed_holes = [hole for hole in persistent_holes
                              if not v13_graph_cycle_encloses_quick(hole['centroid'])]
            records = []
            relevant_missing = np.zeros_like(missing)
            for cid in range(1, count):
                pixels = int(stats[cid, cv2.CC_STAT_AREA])
                if pixels < reference_min_branch_pixels:
                    continue
                comp = (labels == cid).astype(np.uint8) * 255
                degree = _skeleton_degree(comp)
                endpoint_mask = ((comp > 0) & (degree <= 1)).astype(np.uint8) * 255
                endpoints = _cc_centroids(cv2.dilate(endpoint_mask, np.ones((3,3), np.uint8)), min_area=1)
                endpoints = _dedupe_points_by_radius(
                    endpoints,
                    radius=max(3, int(round(corridor_width_est * 0.10))),
                )
                if len(endpoints) < 2:
                    # 沒有兩個接點的 closed missing loop 交給 persistent-hole fallback。
                    continue
                # 最長缺失支路優先；對多分支 component 保留最遠的 endpoint pairs。
                pairs = []
                for i in range(len(endpoints)):
                    for j in range(i+1, len(endpoints)):
                        d = math.hypot(endpoints[i][0]-endpoints[j][0], endpoints[i][1]-endpoints[j][1])
                        pairs.append((float(d), endpoints[i], endpoints[j]))
                pairs.sort(key=lambda rec: rec[0], reverse=True)
                x = int(stats[cid, cv2.CC_STAT_LEFT])
                y = int(stats[cid, cv2.CC_STAT_TOP])
                w = int(stats[cid, cv2.CC_STAT_WIDTH])
                h = int(stats[cid, cv2.CC_STAT_HEIGHT])
                # Reference-gap stage 只服務「實體有 persistent hole、但 graph 尚未形成 cycle」的區域。
                # 這避免普通死巷或遠離任何 loop 的 coverage gap 被錯補成假環。
                related_holes = []
                for hole in unclosed_holes:
                    hx,hy,hw,hh = map(int,hole['bbox'])
                    margin = int(max(reference_support_radius * 2, corridor_width_est * 1.5))
                    ax1,ay1,ax2,ay2 = x,y,x+w,y+h
                    bx1,by1,bx2,by2 = hx-margin,hy-margin,hx+hw+margin,hy+hh+margin
                    if not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2):
                        related_holes.append(hole['hole_id'])
                if not related_holes:
                    continue
                records.append({
                    'cid': int(cid), 'pixels': pixels, 'mask': comp,
                    'bbox': [x,y,w,h], 'pairs': pairs[:12],
                    'related_hole_ids': related_holes,
                })
                relevant_missing[comp > 0] = 255
            records.sort(key=lambda rec: (rec['pixels'], max(rec['bbox'][2], rec['bbox'][3])), reverse=True)
            return records, relevant_missing

        reference_support_cache = {}

        def v13_support_mask_for_component(comp_mask, endpoint_links=(), cache_key=None):
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (reference_support_radius * 2 + 1, reference_support_radius * 2 + 1),
            )
            support_key = cache_key if cache_key is not None else hashlib.blake2b(comp_mask, digest_size=12).digest()
            base_support = reference_support_cache.get(support_key)
            if base_support is None:
                yy, xx = np.where(comp_mask > 0)
                base_support = np.zeros_like(comp_mask)
                if len(xx):
                    x0,y0,x1,y1 = _clip_box(int(xx.min())-reference_support_radius,
                        int(yy.min())-reference_support_radius,
                        int(xx.max())+reference_support_radius+1,
                        int(yy.max())+reference_support_radius+1,W,H)
                    base_support[y0:y1,x0:x1] = cv2.dilate(comp_mask[y0:y1,x0:x1],k)
                if len(reference_support_cache) >= 24:
                    reference_support_cache.clear()
                reference_support_cache[support_key] = base_support
            support = base_support.copy()
            connector = np.zeros_like(support)
            thick = max(3, reference_support_radius * 2 + 1)
            for a, b in endpoint_links:
                cv2.line(connector, tuple(map(int,a)), tuple(map(int,b)), 255, thick)
            support = cv2.bitwise_or(support, connector)
            return cv2.bitwise_and(support, topology_walkable_mask)

        reference_search_cache = {}

        def v13_try_reference_component(record):
            record_key = hashlib.blake2b(record['mask'], digest_size=12).digest()
            attempted_pairs = set()
            max_contact = float(max(
                represented_radius * 2.8,
                corridor_width_est * 1.6,
                map_short_side * 0.03,
            ))
            for _, ep_a, ep_b in record['pairs']:
                cand_a = v13_projection_candidates(ep_a, max_distance=max_contact, limit=5)
                cand_b = v13_projection_candidates(ep_b, max_distance=max_contact, limit=5)
                if not cand_a or not cand_b:
                    continue
                combos = []
                for da, edge_a, pa in cand_a:
                    for db, edge_b, pb in cand_b:
                        if pa == pb:
                            continue
                        if edge_a['u'] == edge_b['u'] and edge_a['v'] == edge_b['v'] and abs(pa[0]-pb[0])+abs(pa[1]-pb[1]) < reference_min_branch_pixels:
                            continue
                        combos.append((da+db, da, db, edge_a, pa, edge_b, pb))
                combos.sort(key=lambda rec: rec[0])
                for _, da, db, edge_a, pa, edge_b, pb in combos[:12]:
                    support_key = (record_key, ep_a, pa, ep_b, pb)
                    if support_key in attempted_pairs:
                        continue
                    attempted_pairs.add(support_key)
                    result = reference_search_cache.get(support_key)
                    if result is not None and result.get('status') != 'ok':
                        continue
                    support = v13_support_mask_for_component(
                        record['mask'], endpoint_links=[(ep_a, pa), (ep_b, pb)],
                        cache_key=record_key,
                    )
                    if result is None:
                        result = v13_four_direction_path(
                            pa,
                            pb,
                            support,
                            turn_penalty=max(2.0, v13_turn_penalty_px * v14_backfill_turn_multiplier),
                        )
                        if len(reference_search_cache) >= 1024:
                            reference_search_cache.clear()
                        reference_search_cache[support_key] = result
                    if result.get('status') != 'ok':
                        continue
                    if result['path_len'] < max(8.0, corridor_width_est * 0.35):
                        continue
                    straight_points = simplify_backfill_to_straight_segments(
                        result['points'], support
                    )
                    straight_length = float(sum(
                        math.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b in zip(straight_points[:-1], straight_points[1:])
                    ))
                    old_distance = v13_graph_distance_between_projections(edge_a, pa, edge_b, pb)
                    if not math.isfinite(old_distance):
                        continue
                    installed = v13_install_path(straight_points, 'V14_reference_gap_straight_visibility')
                    if installed is None:
                        continue
                    return {
                        'status': 'ok',
                        'repair_type': 'reference_skeleton_gap',
                        'source_point': list(map(int, pa)),
                        'target_point': list(map(int, pb)),
                        'reference_endpoint_a': list(map(int, ep_a)),
                        'reference_endpoint_b': list(map(int, ep_b)),
                        'reference_component_pixels': int(record['pixels']),
                        'contact_distance_a_px': round(float(da), 2),
                        'contact_distance_b_px': round(float(db), 2),
                        'old_graph_distance_px': round(float(old_distance), 2),
                        'new_path_distance_px': round(straight_length, 2),
                        'turn_count': int(count_navigation_turns(straight_points)),
                        'axis_search_turn_count_before_straightening': int(result['turn_count']),
                        'search_scale_px_per_cell': int(result.get('scale', 1)),
                        'search_expansions': int(result.get('expansions', 0)),
                        'path': [list(map(int,p)) for p in straight_points],
                        'new_edge_count': int(installed['new_edge_count']),
                        'graph_node_ids': installed['node_ids'],
                    }
            return {'status': 'no_safe_reference_path'}

        initial_missing_components, initial_missing_skeleton = v13_missing_reference_components()
        reference_round_limit = max(
            12,
            min(160, len(initial_missing_components) * 3 + len(persistent_holes) * 4 + 20),
        )
        reference_repairs = 0
        for _ in range(reference_round_limit):
            if time.perf_counter() >= topology_search_deadline:
                topology_search_exhausted = True
                break
            components, _ = v13_missing_reference_components()
            if not components:
                break
            success = None
            for record in components:
                if time.perf_counter() >= topology_search_deadline:
                    topology_search_exhausted = True
                    break
                success = v13_try_reference_component(record)
                if success.get('status') == 'ok':
                    break
            if not success or success.get('status') != 'ok':
                break
            topology_cycle_records.append(success)
            reference_repairs += 1
            v13_distance_cache.clear()
            print(
                f"[V13 topology] 補回 reference branch #{reference_repairs}："
                f"{success['source_point']} -> {success['target_point']}，"
                f"new={success['new_path_distance_px']:.1f}px，turns={success['turn_count']}。"
            )

        # -------------
        # 11.6 Bridge-chain physical bypass test
        # -------------
        def v13_bridge_chains():
            if G.number_of_edges() == 0:
                return []
            bridge_pairs = []
            try:
                bridge_pairs = list(nx.bridges(G))
            except nx.NetworkXError:
                return []
            if not bridge_pairs:
                return []
            B = nx.Graph()
            B.add_edges_from(bridge_pairs)
            visited = set()
            chains = []

            def edge_key(a,b):
                return tuple(sorted((a,b)))

            boundary = [n for n in B.nodes() if B.degree(n) != 2]
            for start_node in boundary:
                for nxt in list(B.neighbors(start_node)):
                    key = edge_key(start_node, nxt)
                    if key in visited:
                        continue
                    chain = [start_node, nxt]
                    visited.add(key)
                    prev, cur = start_node, nxt
                    while B.degree(cur) == 2:
                        options = [n for n in B.neighbors(cur) if n != prev]
                        if not options:
                            break
                        nn = options[0]
                        k = edge_key(cur, nn)
                        if k in visited:
                            break
                        visited.add(k)
                        chain.append(nn)
                        prev, cur = cur, nn
                    if len(chain) >= 2:
                        chains.append(chain)

            # 理論上 bridge subgraph 不可能形成純 cycle；保險起見補上尚未 trace 的邊。
            for a,b in bridge_pairs:
                if edge_key(a,b) not in visited:
                    chains.append([a,b])

            records = []
            for chain in chains:
                length = 0.0
                for a,b in zip(chain[:-1], chain[1:]):
                    if G.has_edge(a,b):
                        length += float(G.edges[a,b].get('weight', math.hypot(
                            G.nodes[b]['pos'][0]-G.nodes[a]['pos'][0],
                            G.nodes[b]['pos'][1]-G.nodes[a]['pos'][1],
                        )))
                records.append({'nodes': chain, 'length': float(length)})
            records.sort(key=lambda rec: rec['length'], reverse=True)
            return records

        def v13_chain_mask(chain_nodes):
            mask = np.zeros((H,W), dtype=np.uint8)
            half_width_samples = []
            for a,b in zip(chain_nodes[:-1], chain_nodes[1:]):
                if a not in G or b not in G:
                    continue
                p1 = tuple(map(int,G.nodes[a]['pos']))
                p2 = tuple(map(int,G.nodes[b]['pos']))
                cv2.line(mask, p1, p2, 255, 3)
                for p in (p1,p2):
                    x,y = p
                    if 0 <= x < W and 0 <= y < H:
                        d = float(corridor_inside_distance[y,x])
                        if d > 0:
                            half_width_samples.append(d)
            local_half = float(np.median(half_width_samples)) if half_width_samples else corridor_width_est * 0.5
            block_radius = int(np.clip(
                round(max(corridor_width_est * 0.58, local_half * 1.20)),
                5,
                max(10, round(corridor_width_est * 1.05)),
            ))
            return mask, block_radius

        def v13_bridge_bypass_candidate(chain_rec):
            chain = chain_rec['nodes']
            if len(chain) < 2 or chain[0] not in G or chain[-1] not in G:
                return None
            start_pt = tuple(map(int,G.nodes[chain[0]]['pos']))
            goal_pt = tuple(map(int,G.nodes[chain[-1]]['pos']))
            if start_pt == goal_pt:
                return None
            chain_mask, block_radius = v13_chain_mask(chain)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_radius*2+1, block_radius*2+1))
            forbidden = cv2.dilate(chain_mask, k, iterations=1)
            alternate_mask = cv2.bitwise_and(topology_walkable_mask, cv2.bitwise_not(forbidden))

            # 端點必須能離開被封鎖的 chain；只在 junction 端點附近開小 gate。
            gate_radius = int(max(3, round(block_radius * 1.05)))
            gate = np.zeros((H,W), dtype=np.uint8)
            cv2.circle(gate, start_pt, gate_radius, 255, -1)
            cv2.circle(gate, goal_pt, gate_radius, 255, -1)
            alternate_mask = cv2.bitwise_or(alternate_mask, cv2.bitwise_and(gate, topology_walkable_mask))

            # 快速 connectivity precheck，避免每條死巷都進 heading-aware A*。
            pre_scale = max(2, min(5, int(astar_scale)))
            pre = v13_downsample_mask(alternate_mask, pre_scale)
            s = v13_nearest_valid((round(start_pt[0]/pre_scale), round(start_pt[1]/pre_scale)), pre, max(3, int(corridor_width_est/pre_scale)))
            g = v13_nearest_valid((round(goal_pt[0]/pre_scale), round(goal_pt[1]/pre_scale)), pre, max(3, int(corridor_width_est/pre_scale)))
            if s is None or g is None:
                return None
            cc_count, cc_labels = cv2.connectedComponents(pre, connectivity=4)
            if cc_count <= 1 or cc_labels[s[1],s[0]] == 0 or cc_labels[s[1],s[0]] != cc_labels[g[1],g[0]]:
                return None

            result = v13_four_direction_path(
                start_pt,
                goal_pt,
                alternate_mask,
                turn_penalty=max(2.0, v13_turn_penalty_px * v14_backfill_turn_multiplier),
            )
            if result.get('status') != 'ok':
                return None
            if result['path_len'] < max(10.0, corridor_width_est * 0.45):
                return None
            straight_points = simplify_backfill_to_straight_segments(
                result['points'], alternate_mask
            )

            # Alternate path 必須真的離開原 chain band，否則只是同一條寬走道內左右挪幾 pixel。
            inv_chain = (chain_mask == 0).astype(np.uint8)
            chain_distance = cv2.distanceTransform(inv_chain, cv2.DIST_L2, 5)
            sampled_dist = []
            for p in straight_points:
                x,y = map(int,p)
                if 0 <= x < W and 0 <= y < H:
                    sampled_dist.append(float(chain_distance[y,x]))
            max_deviation = max(sampled_dist) if sampled_dist else 0.0
            if max_deviation < max(block_radius * 1.35, corridor_width_est * 0.75):
                return None

            installed = v13_install_path(straight_points, 'V14_bridge_bypass_straight_visibility')
            if installed is None:
                return None
            return {
                'status': 'ok',
                'repair_type': 'bridge_chain_physical_bypass',
                'bridge_chain_node_count': int(len(chain)),
                'bridge_chain_length_px': round(float(chain_rec['length']),2),
                'source_point': list(map(int,start_pt)),
                'target_point': list(map(int,goal_pt)),
                'block_radius_px': int(block_radius),
                'alternate_path_distance_px': round(float(sum(
                    math.hypot(b[0]-a[0], b[1]-a[1])
                    for a,b in zip(straight_points[:-1], straight_points[1:])
                )),2),
                'max_deviation_from_original_chain_px': round(float(max_deviation),2),
                'turn_count': int(count_navigation_turns(straight_points)),
                'axis_search_turn_count_before_straightening': int(result['turn_count']),
                'search_scale_px_per_cell': int(result.get('scale',1)),
                'search_expansions': int(result.get('expansions',0)),
                'path': [list(map(int,p)) for p in straight_points],
                'new_edge_count': int(installed['new_edge_count']),
                'graph_node_ids': installed['node_ids'],
            }

        initial_bridge_chain_count = len(v13_bridge_chains())
        bridge_round_limit = max(8, min(96, initial_bridge_chain_count * 2 + 12))
        bridge_repairs = 0
        for _ in range(bridge_round_limit):
            if time.perf_counter() >= topology_search_deadline:
                topology_search_exhausted = True
                break
            chains = v13_bridge_chains()
            if not chains:
                break
            repaired = None
            # 長 chain 優先；大環通常正是由一整段錯誤 bridge-chain 表現出來。
            for chain_rec in chains:
                if time.perf_counter() >= topology_search_deadline:
                    topology_search_exhausted = True
                    break
                repaired = v13_bridge_bypass_candidate(chain_rec)
                if repaired is not None:
                    break
            if repaired is None:
                break
            topology_cycle_records.append(repaired)
            bridge_repairs += 1
            v13_distance_cache.clear()
            print(
                f"[V13 topology] bridge bypass #{bridge_repairs}："
                f"{repaired['source_point']} -> {repaired['target_point']}，"
                f"alternate={repaired['alternate_path_distance_px']:.1f}px，"
                f"deviation={repaired['max_deviation_from_original_chain_px']:.1f}px。"
            )

        # -------------
        # 11.7 Persistent-hole cycle census and full-loop fallback
        # -------------
        def v13_graph_cycle_encloses(point):
            if G.number_of_nodes() < 3 or G.number_of_edges() < 3:
                return False, None
            basis = v13_cycle_basis_cached()
            px, py = map(float, point)
            for cycle in basis:
                if len(cycle) < 3:
                    continue
                poly = np.asarray([G.nodes[n]['pos'] for n in cycle if n in G], dtype=np.int32)
                if len(poly) < 3:
                    continue
                contour = poly.reshape((-1,1,2))
                try:
                    inside = cv2.pointPolygonTest(contour, (px,py), False)
                except cv2.error:
                    inside = -1
                if inside >= 0:
                    return True, cycle
            return False, None

        def v13_reference_cycle_for_hole(hole):
            """
            從 reference skeleton 的 2-core 取得包圍指定 persistent hole 的實際 loop。

            不直接使用 cycle_basis 的單一 fundamental cycle，因為斜線 skeleton 在 8-neighbor
            pixel graph 中容易出現大量局部小 cycle。先做 k-core(k=2) 去掉樹枝，再把 core
            rasterize 成窄帶，最後用 contour 取得穩定、依幾何順序排列的 closed loop。
            """
            x,y,w,h = map(int,hole['bbox'])
            margin = int(np.clip(
                max(corridor_width_est * 2.0, max(w,h) * 0.22),
                corridor_width_est,
                map_short_side * 0.32,
            ))
            x1,y1,x2,y2 = _clip_box(x-margin,y-margin,x+w+margin,y+h+margin,W,H)
            roi = (reference_skeleton[y1:y2,x1:x2] > 0).astype(np.uint8)
            ys,xs = np.where(roi>0)
            if len(xs) < 8 or len(xs) > 40000:
                return None

            pixel_set = set((int(px),int(py)) for px,py in zip(xs,ys))
            P = nx.Graph()
            P.add_nodes_from(pixel_set)
            # 全 8-neighbor：k-core 會去掉 dead-end；後面的 contour 負責消除局部 diagonal 小環干擾。
            forward_neighbors = [(1,0),(0,1),(1,1),(-1,1)]
            for px,py in pixel_set:
                for dx,dy in forward_neighbors:
                    q=(px+dx,py+dy)
                    if q in pixel_set:
                        P.add_edge((px,py),q)
            if P.number_of_edges() < 3:
                return None
            try:
                core = nx.k_core(P,k=2)
            except nx.NetworkXError:
                return None
            if core.number_of_nodes() < 8:
                return None

            core_mask=np.zeros_like(roi,dtype=np.uint8)
            for px,py in core.nodes():
                if 0<=px<core_mask.shape[1] and 0<=py<core_mask.shape[0]:
                    core_mask[py,px]=255
            # 把一像素 core 變成可追 contour 的窄帶；寬度保持很小，不改變真正拓樸。
            core_band=cv2.dilate(core_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
            contours,_=cv2.findContours(core_band,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            local_cx=float(hole['centroid'][0]-x1)
            local_cy=float(hole['centroid'][1]-y1)
            candidates=[]
            for contour in contours:
                if contour is None or len(contour)<8:
                    continue
                try:
                    inside=cv2.pointPolygonTest(contour,(local_cx,local_cy),False)
                except cv2.error:
                    inside=-1
                if inside<0:
                    continue
                perimeter=float(cv2.arcLength(contour,True))
                area=abs(float(cv2.contourArea(contour)))
                if area < max(16.0, float(hole['area'])*0.20):
                    continue
                pts=[(int(p[0][0]+x1),int(p[0][1]+y1)) for p in contour]
                candidates.append((perimeter,area,pts))
            if not candidates:
                return None
            # 選最短、仍能包圍 hole 的 core contour，代表最靠近該 obstacle 的 reference loop。
            candidates.sort(key=lambda rec:(rec[0],rec[1]))
            return candidates[0][2]

        def v13_arc_support_mask(arc_points, graph_connector=None):
            raw = np.zeros((H,W),dtype=np.uint8)
            if len(arc_points) == 1:
                cv2.circle(raw, tuple(map(int,arc_points[0])), 1, 255, -1)
            else:
                for a,b in zip(arc_points[:-1],arc_points[1:]):
                    cv2.line(raw, tuple(map(int,a)), tuple(map(int,b)), 255, 2)
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (reference_support_radius*2+1,reference_support_radius*2+1),
            )
            support = cv2.dilate(raw,k,iterations=1)
            if graph_connector is not None:
                a,b = graph_connector
                connector = np.zeros((H,W),dtype=np.uint8)
                cv2.line(connector,tuple(map(int,a)),tuple(map(int,b)),255,max(3,reference_support_radius*2+1))
                support = cv2.bitwise_or(support,connector)
            return cv2.bitwise_and(support,topology_walkable_mask)

        def v13_restore_full_reference_cycle(hole):
            cycle = v13_reference_cycle_for_hole(hole)
            if not cycle or len(cycle) < 12:
                return None

            # 找 reference cycle 上距目前 graph 最近的 anchor。
            step = max(1, len(cycle)//140)
            anchor_options = []
            max_anchor_distance = float(max(corridor_width_est*2.2, represented_radius*3.0, map_short_side*0.04))
            for idx in range(0,len(cycle),step):
                pt = cycle[idx]
                projections = v13_projection_candidates(pt,max_distance=max_anchor_distance,limit=2)
                for d,edge,proj in projections:
                    anchor_options.append((d,idx,pt,edge,proj))
            if not anchor_options:
                return None
            anchor_options.sort(key=lambda rec: rec[0])

            for anchor_distance, anchor_idx, anchor_ref, anchor_edge, anchor_proj in anchor_options[:10]:
                seq = cycle[anchor_idx:] + cycle[:anchor_idx]
                if len(seq) < 12:
                    continue
                opposite_idx = len(seq)//2
                opposite = tuple(map(int,seq[opposite_idx]))
                arc1 = seq[:opposite_idx+1]
                arc2 = [seq[0]] + list(reversed(seq[opposite_idx:]))
                support1 = v13_arc_support_mask(arc1,graph_connector=(anchor_ref,anchor_proj))
                support2 = v13_arc_support_mask(arc2,graph_connector=(anchor_ref,anchor_proj))
                result1 = v13_four_direction_path(
                    anchor_proj,
                    opposite,
                    support1,
                    turn_penalty=max(2.0, v13_turn_penalty_px * v14_backfill_turn_multiplier * 0.85),
                )
                if result1.get('status') != 'ok':
                    continue
                result2 = v13_four_direction_path(
                    anchor_proj,
                    opposite,
                    support2,
                    turn_penalty=max(2.0, v13_turn_penalty_px * v14_backfill_turn_multiplier * 0.85),
                )
                if result2.get('status') != 'ok':
                    continue
                straight1 = simplify_backfill_to_straight_segments(result1['points'], support1)
                straight2 = simplify_backfill_to_straight_segments(result2['points'], support2)

                # 兩條 arc 需要具有明顯不同的中段，否則只是同一路徑重複兩次。
                set1 = set(tuple(map(int,p)) for p in straight1[1:-1])
                set2 = set(tuple(map(int,p)) for p in straight2[1:-1])
                overlap = len(set1 & set2) / float(max(1,min(len(set1),len(set2)))) if set1 and set2 else 0.0
                if overlap > 0.55:
                    continue

                install1 = v13_install_path(straight1,'V14_persistent_hole_straight_arc_A')
                if install1 is None:
                    continue
                install2 = v13_install_path(straight2,'V14_persistent_hole_straight_arc_B')
                if install2 is None:
                    # 第一條已加入；它仍是合法走道 coverage，不刪除，但記錄 partial，後續 census 會再檢查。
                    topology_unresolved_records.append({
                        'type':'persistent_hole_partial_restore',
                        'hole_id':hole['hole_id'],
                        'reason':'second_arc_install_failed',
                    })
                    return None

                return {
                    'status':'ok',
                    'repair_type':'persistent_hole_full_cycle_restore',
                    'hole_id':hole['hole_id'],
                    'hole_centroid':hole['centroid'],
                    'hole_area_px':int(hole['area']),
                    'anchor_point':list(map(int,anchor_proj)),
                    'anchor_reference_point':list(map(int,anchor_ref)),
                    'anchor_distance_px':round(float(anchor_distance),2),
                    'opposite_point':list(map(int,opposite)),
                    'arc_a_distance_px':round(float(sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(straight1[:-1],straight1[1:]))),2),
                    'arc_b_distance_px':round(float(sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(straight2[:-1],straight2[1:]))),2),
                    'arc_a_turn_count':int(count_navigation_turns(straight1)),
                    'arc_b_turn_count':int(count_navigation_turns(straight2)),
                    'arc_overlap_ratio':round(float(overlap),3),
                    'path_a':[list(map(int,p)) for p in straight1],
                    'path_b':[list(map(int,p)) for p in straight2],
                    'new_edge_count':int(install1['new_edge_count']+install2['new_edge_count']),
                }
            return None

        hole_status_before = {}
        for hole in persistent_holes:
            represented, _ = v13_graph_cycle_encloses(hole['centroid'])
            hole_status_before[hole['hole_id']] = bool(represented)

        hole_fallback_repairs = 0
        for hole in persistent_holes:
            if time.perf_counter() >= topology_search_deadline:
                topology_search_exhausted = True
                break
            represented, _ = v13_graph_cycle_encloses(hole['centroid'])
            if represented:
                continue
            restored = v13_restore_full_reference_cycle(hole)
            if restored is None:
                topology_unresolved_records.append({
                    'type':'persistent_hole_unrepresented',
                    'hole_id':hole['hole_id'],
                    'centroid':hole['centroid'],
                    'area_px':int(hole['area']),
                })
                continue
            topology_cycle_records.append(restored)
            hole_fallback_repairs += 1
            v13_distance_cache.clear()
            print(
                f"[V13 topology] persistent hole {hole['hole_id']} 完整環還原："
                f"A={restored['arc_a_distance_px']:.1f}px，B={restored['arc_b_distance_px']:.1f}px。"
            )

        # -------------------------------------------------
        # 11.8 V13.1 local graph fusion
        # -------------------------------------------------
        # V13 的 cycle path 已經在 NetworkX 中，但某些 rectilinear staircase 與原始
        # route 只差 1~數 px；debug 圖看起來像兩張路網疊在一起。這一層不重建環，
        # 只把「局部近接、同一走道內、可用單一 H/V 短線安全接合」的 topology node
        # 熔接到既有 route/component_bridge edge，並保證每個 topology subgraph 至少
        # 有兩個 base-graph contact（只要物理上存在可安全的接點）。
        v13_fusion_gap_px = int(np.clip(round(corridor_width_est * 0.08), 2, 8))
        topology_fusion_records = []

        def v13_is_topology_node(nid):
            node_type = str(nodes_data.get(nid, {}).get('type', ''))
            return node_type.startswith('topology_')

        def v13_base_edge_records():
            records = []
            for edge in v13_axis_edge_records():
                et = str(edge['edata'].get('edge_type', 'route'))
                # Original route / component bridge segments remain base edges even when
                # one endpoint became a topology split node.
                if et != 'shortcut':
                    records.append(edge)
                    continue
                if not v13_is_topology_node(edge['u']) and not v13_is_topology_node(edge['v']):
                    records.append(edge)
            return records

        def v13_fuse_node_to_base(nid, max_gap):
            if nid not in G:
                return None
            pt = tuple(map(int, G.nodes[nid]['pos']))
            # Already has a direct base contact.
            if any(nb in G and not v13_is_topology_node(nb) for nb in G.neighbors(nid)):
                return None

            candidates = []
            for edge in v13_base_edge_records():
                if nid in (edge['u'], edge['v']):
                    continue
                proj = v13_project_to_edge(pt, edge)
                # Connector itself must be a single H/V segment; otherwise skip instead
                # of creating a diagonal visual bridge.
                if not is_axis_segment(pt, proj):
                    continue
                d = float(abs(proj[0] - pt[0]) + abs(proj[1] - pt[1]))
                if d <= 0.0 or d > float(max_gap):
                    continue
                if not in_bounds(proj[0], proj[1]) or wall_collision[proj[1], proj[0]] > 0:
                    continue
                if not v13_path_segments_safe([pt, proj], topology_walkable_mask, ratio=0.60):
                    continue
                candidates.append((d, edge, proj))
            if not candidates:
                return None
            candidates.sort(key=lambda rec: rec[0])
            d, edge, proj = candidates[0]

            # Split the base edge at the projection.  v13_ensure_graph_node reuses an
            # existing node at the same coordinate whenever possible.
            target_id, created = v13_ensure_graph_node(proj, node_type='topology_fusion_waypoint')
            if target_id == nid or target_id not in G:
                return None
            add_graph_edge(nid, target_id, edge_type='route')
            if not G.has_edge(nid, target_id):
                return None
            G.edges[nid, target_id]['topology_cycle'] = True
            G.edges[nid, target_id]['topology_fusion'] = True
            G.edges[nid, target_id]['topology_strategy'] = 'V13_1_local_graph_fusion'
            v13_distance_cache.clear()
            record = {
                'source_node': nid,
                'source_point': list(map(int, pt)),
                'target_node': target_id,
                'target_point': list(map(int, proj)),
                'gap_px': round(float(d), 2),
                'target_edge_before_split': [edge['u'], edge['v']],
                'target_node_created': bool(created),
            }
            topology_fusion_records.append(record)
            return record

        # A. Fuse every visually-near topology node that currently has no base contact.
        for nid in list(G.nodes()):
            if v13_is_topology_node(nid):
                v13_fuse_node_to_base(nid, v13_fusion_gap_px)

        # B. Component-level invariant: every repaired topology component should have
        # at least two independent base contacts when such local contacts exist.  This
        # prevents a restored loop from becoming a decorative/dangling overlay.
        topology_nodes_now = [n for n in G.nodes() if v13_is_topology_node(n)]
        topo_subgraph = G.subgraph(topology_nodes_now).copy()
        topology_component_contact_report = []
        for comp in list(nx.connected_components(topo_subgraph)) if topo_subgraph.number_of_nodes() else []:
            comp = set(comp)
            def _contacts():
                return {
                    nb for n in comp for nb in G.neighbors(n)
                    if nb in G and nb not in comp and not v13_is_topology_node(nb)
                }
            contacts_before = _contacts()
            if len(contacts_before) < 2:
                # Try nearest low-degree topology nodes first; at most two connectors per
                # component so this cannot turn a corridor into a ladder network.
                candidates = sorted(
                    (n for n in comp if n in G),
                    key=lambda n: (G.degree(n), G.nodes[n]['pos'][1], G.nodes[n]['pos'][0]),
                )
                for nid in candidates:
                    if len(_contacts()) >= 2:
                        break
                    v13_fuse_node_to_base(nid, v13_fusion_gap_px)
            contacts_after = _contacts()
            topology_component_contact_report.append({
                'node_count': int(len(comp)),
                'base_contacts_before': int(len(contacts_before)),
                'base_contacts_after': int(len(contacts_after)),
                'integrated': bool(len(contacts_after) >= 2),
            })

        # C. Promote every validated topology repair into the canonical road layer.
        # The provenance remains in topology_cycle/topology_strategy, but the planner and
        # debug renderer now see ONE road graph instead of a green base graph plus a purple
        # shortcut overlay.  This does not change geometry or connectivity; it only makes
        # the repaired loop an ordinary bidirectional road after topology validation.
        topology_promoted_edge_count = 0
        for _u, _v, _edata in G.edges(data=True):
            if not _edata.get('topology_cycle'):
                continue
            old_type = str(_edata.get('edge_type', 'route'))
            if old_type != 'route':
                _edata['topology_original_edge_type'] = old_type
                _edata['edge_type'] = 'route'
                topology_promoted_edge_count += 1

        try:
            topology_graph_connected_after_fusion = bool(G.number_of_nodes() > 0 and nx.is_connected(G))
        except nx.NetworkXError:
            topology_graph_connected_after_fusion = False
        orphan_topology_components = int(sum(
            1 for rec in topology_component_contact_report if not rec['integrated']
        ))
        print(
            '[V13.1 fusion] '
            f'local_fusion_edges={len(topology_fusion_records)}，'
            f'topology_components={len(topology_component_contact_report)}，'
            f'orphan_components={orphan_topology_components}，'
            f'graph_connected={topology_graph_connected_after_fusion}。'
        )

        hole_status_after = {}
        for hole in persistent_holes:
            represented, cycle = v13_graph_cycle_encloses(hole['centroid'])
            hole_status_after[hole['hole_id']] = bool(represented)
            hole['represented_before'] = bool(hole_status_before.get(hole['hole_id'],False))
            hole['represented_after'] = bool(represented)
            hole['representing_cycle_node_count'] = int(len(cycle)) if cycle else 0

        final_missing_components, final_missing_skeleton = v13_missing_reference_components()
        try:
            final_bridge_count = len(list(nx.bridges(G))) if G.number_of_edges() else 0
        except nx.NetworkXError:
            final_bridge_count = 0

        # reference debug：白=walkable；綠=reference skeleton；紅=最終仍未被 graph coverage 的 reference branch；
        # 黃圈=persistent hole centroid。
        topology_debug = cv2.cvtColor(topology_walkable_mask,cv2.COLOR_GRAY2BGR)
        topology_debug[reference_skeleton>0] = (0,180,0)
        topology_debug[final_missing_skeleton>0] = (0,0,255)
        for hole in persistent_holes:
            cx,cy = map(int,hole['centroid'])
            cv2.circle(topology_debug,(cx,cy),7,(0,255,255),2)
            cv2.putText(
                topology_debug,
                hole['hole_id'],
                (cx+8,cy-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0,165,255),
                1,
                cv2.LINE_AA,
            )
        safe_imwrite(str(self.output_dir/'debug_topology_reference_v13.jpg'),topology_debug)

        print(
            '[V13 topology summary] '
            f'persistent_holes={len(persistent_holes)}，'
            f'reference_repairs={reference_repairs}，'
            f'bridge_bypass_repairs={bridge_repairs}，'
            f'hole_fallback_repairs={hole_fallback_repairs}，'
            f'final_missing_reference_components={len(final_missing_components)}，'
            f'final_graph_bridges={final_bridge_count}。'
        )

        # 為舊 JSON/前端保留 shortcuts 欄位名稱；內容已是 V13 topology repair，不再是 V12 near-edge heuristic。
        shortcut_records = topology_cycle_records
        topology_stage_elapsed = time.perf_counter() - topology_stage_start
        print(f"[計時] V13 topology 保留與融合：{topology_stage_elapsed:.3f} 秒")

        # -------------------------------------------------
        # FAST V8：Rectilinear recovered public-space graph augmentation
        # -------------------------------------------------
        # 設計原則：
        #   1) primary V13.1 到這裡已完全結束，原有環路/尺度/bridge 結果凍結；
        #   2) 每個 recovered region 只在自己的 safe mask 內建立局部 medial-skeleton graph；
        #   3) 最後才透過 semantic gateway 接到 primary graph；
        #   4) 不重新執行 global component bridge / cycle census，因此不會因新增 open area
        #      反過來刪減或改寫原有主路網。
        recovery_graph_start = time.perf_counter()
        recovery_graph_budget = float(np.clip(
            float(os.environ.get("MAP_RECOVERY_GRAPH_BUDGET_SECONDS", "8.0")),
            0.5, 20.0
        ))
        recovery_graph_records = []
        # V13.1 已完成時的 base graph snapshot；後續 recovery 只能加法式擴充，不能刪改這些節點。
        primary_graph_nodes_snapshot = set(G.nodes())
        recovery_graph_node_ids = set()
        recovery_graph_edge_count = 0
        recovery_gateway_edge_count = 0
        recovery_room_reattach_count = 0
        recovery_graph_timed_out = False

        recovery_phase_deadline = recovery_graph_start + recovery_graph_budget * 0.60
        path_search_deadline = recovery_phase_deadline

        def _recovery_budget_ok():
            return time.perf_counter() <= recovery_phase_deadline

        def _polyline_safe_ratio(polyline, safe_mask):
            if len(polyline) < 2:
                return 0.0
            good = 0
            total = 0
            for a, b in zip(polyline[:-1], polyline[1:]):
                lm = local_line_mask(a, b, thickness=2)
                if lm is None:
                    continue
                x1, y1, x2, y2, local = lm
                count = cv2.countNonZero(local)
                if count <= 0:
                    continue
                total += count
                good += cv2.countNonZero(
                    cv2.bitwise_and(local, safe_mask[y1:y2, x1:x2])
                )
            return good / float(max(total, 1))

        def _split_nearest_edge_for_point(point, allowed_edge_prefix=None, access_mask=None):
            """把 point 投影到最近既有 graph edge；需要時切分該 edge。"""
            if G.number_of_edges() == 0:
                return None
            best = None
            candidates = []
            px, py = map(int, point)
            for u, v, edata in list(G.edges(data=True)):
                et = str(edata.get("edge_type", "route"))
                if allowed_edge_prefix == "primary" and et.startswith("recovery_"):
                    continue
                if allowed_edge_prefix == "recovery" and not et.startswith("recovery_"):
                    continue
                a, b = tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos'])
                proj = project_point_to_segment((px, py), a, b)
                d = math.hypot(proj[0]-px, proj[1]-py)
                candidates.append((d,u,v,proj,dict(edata)))
            candidates.sort(key=lambda item:item[0])
            if access_mask is not None:
                _, access_components = cv2.connectedComponents((access_mask > 0).astype(np.uint8), connectivity=4)
                access_id = int(access_components[py,px])
                candidates = [c for c in candidates if access_id > 0
                    and access_components[int(c[3][1]),int(c[3][0])] == access_id]
            for candidate in candidates[:24]:
                if access_mask is None:
                    best = candidate
                    break
                result = _compiled_axis_path(point,candidate[3],access_mask,v13_turn_penalty_px)
                if result.get('status') == 'ok':
                    best = candidate
                    break
            if best is None:
                return None
            d, u, v, proj, edata = best
            if proj == tuple(G.nodes[u]['pos']):
                return u, proj, d
            if proj == tuple(G.nodes[v]['pos']):
                return v, proj, d
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            nid = add_waypoint(proj, node_type=(
                "recovery_waypoint" if allowed_edge_prefix == "recovery"
                else "waypoint"
            ))
            old_type = edata.get("edge_type", "route")
            add_graph_edge(u, nid, edge_type=old_type)
            add_graph_edge(nid, v, edge_type=old_type)
            return nid, proj, d

        # recovered region -> graph node ids / local safe mask / skeleton
        recovery_nodes_by_rid = {}
        recovery_safe_masks = {}
        recovery_skeleton_masks = {}

        def _add_recovery_axis_polyline(points, rec_nodes, *, topology_cycle=False, recovery_cycle=False):
            """V15：只把水平/垂直 navigation polyline 加進 recovery graph。"""
            pts = compress_orthogonal_points(points or [])
            if len(pts) < 2:
                return 0

            # Recovery graph 是實體導航道路，不接受任何斜邊。
            # 若上游意外送進非 H/V segment，整條拒絕，避免 debug/JSON 再出現斜線。
            if any(not is_axis_segment(a, b) for a, b in zip(pts[:-1], pts[1:])):
                print("[Recovery graph V15] 略過含斜線的非法 polyline；等待四方向搜尋重新建立。")
                return 0

            added = 0
            prev = None
            for pt in pts:
                nid = add_waypoint(pt, node_type="recovery_waypoint")
                recovery_graph_node_ids.add(nid)
                rec_nodes.add(nid)
                if prev is not None and prev != nid:
                    before = G.number_of_edges()
                    add_graph_edge(
                        prev, nid,
                        edge_type="recovery_route",
                        topology_cycle=bool(topology_cycle),
                        recovery_cycle=bool(recovery_cycle),
                        movement="four_direction_rectilinear",
                    )
                    if G.number_of_edges() > before:
                        added += 1
                prev = nid
            return added

        def _recovery_cycle_rank(node_ids):
            nodes = {n for n in node_ids if n in G}
            if not nodes:
                return 0
            sub = G.subgraph(nodes)
            components = nx.number_connected_components(sub) if sub.number_of_nodes() else 0
            return max(0, int(sub.number_of_edges() - sub.number_of_nodes() + components))

        def _recovery_persistent_holes(rec_safe):
            """以 morphology-stable hole census 當 recovery topology reference。

            小文字/圖示洞不算；只保留相對於走道尺度仍顯著的 enclosed obstacle/room island。
            """
            close_r = int(np.clip(round(corridor_width_est * 0.05), 1, 5))
            stable = cv2.morphologyEx(
                rec_safe,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1)),
            )
            contours, hierarchy = cv2.findContours(stable, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                return []
            min_hole_area = float(max(180.0, corridor_width_est * corridor_width_est * 0.28))
            holes = []
            for idx, contour in enumerate(contours):
                parent = int(hierarchy[0][idx][3])
                if parent < 0:
                    continue
                area = float(abs(cv2.contourArea(contour)))
                if area < min_hole_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                holes.append({
                    "contour": contour,
                    "area": area,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "centroid": (
                        int(round(x + w / 2.0)),
                        int(round(y + h / 2.0)),
                    ),
                })
            holes.sort(key=lambda r: r["area"], reverse=True)
            return holes

        def _hole_ring_candidate_paths(rec_safe, hole):
            """對單一 persistent hole 建立四方向封閉候選環；四段全成功才回傳。"""
            hole_mask = np.zeros_like(rec_safe)
            cv2.drawContours(hole_mask, [hole["contour"]], -1, 255, -1)
            inner_r = int(np.clip(round(corridor_width_est * 0.14), 3, 14))
            outer_r = int(np.clip(round(corridor_width_est * 0.78), inner_r + 8, 72))
            inner = cv2.dilate(
                hole_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_r * 2 + 1, inner_r * 2 + 1)),
            )
            outer = cv2.dilate(
                hole_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_r * 2 + 1, outer_r * 2 + 1)),
            )
            ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
            ring = cv2.bitwise_and(ring, rec_safe)
            ys, xs = np.where(ring > 0)
            if xs.size < 20:
                return None

            cx, cy = hole["centroid"]
            x, y, w, h = hole["bbox"]
            radial = max(inner_r + 3, int(round((inner_r + outer_r) * 0.52)))
            targets = [
                (cx, y - radial),
                (x + w + radial, cy),
                (cx, y + h + radial),
                (x - radial, cy),
            ]
            anchors = []
            gx = xs.astype(np.float32)
            gy = ys.astype(np.float32)
            for tx, ty in targets:
                d2 = (gx - float(tx)) ** 2 + (gy - float(ty)) ** 2
                order = np.argsort(d2)
                chosen = None
                for oi in order[: min(160, len(order))]:
                    pt = (int(xs[int(oi)]), int(ys[int(oi)]))
                    if all(math.hypot(pt[0]-q[0], pt[1]-q[1]) >= max(6.0, corridor_width_est * 0.18) for q in anchors):
                        chosen = pt
                        break
                if chosen is None:
                    return None
                anchors.append(chosen)

            staged = []
            # 在 ring support 內依序走 top->right->bottom->left->top；因此形成真正閉環。
            for a, b in zip(anchors, anchors[1:] + anchors[:1]):
                result = v13_four_direction_path(
                    a, b, ring,
                    turn_penalty=max(2.0, v13_turn_penalty_px * v14_backfill_turn_multiplier * 0.85),
                )
                if result.get("status") != "ok":
                    return None
                pts = simplify_backfill_to_straight_segments(result.get("points", []), ring)
                if len(pts) < 2:
                    return None
                staged.append(pts)
            return staged

        recovery_straight_turn_multiplier = float(np.clip(
            float(os.environ.get("MAP_RECOVERY_STRAIGHT_TURN_MULTIPLIER", "2.4")),
            1.0,
            8.0,
        ))

        def _straight_recovery_chain_paths(chain, route_mask):
            """以整條 topology chain 為單位求少轉彎路徑；失敗才遞迴拆段。"""
            chain = [tuple(map(int, p)) for p in chain]
            if len(chain) < 2:
                return [], {"attempts": 0, "fallback_splits": 0, "turns": 0}

            # closed chain 以四個弧長近似錨點保留環路，其餘 chain 直接首尾連線。
            initial_pieces = [chain]
            if chain[0] == chain[-1] and len(chain) >= 8:
                idxs = sorted(set([
                    0,
                    len(chain) // 4,
                    len(chain) // 2,
                    (3 * len(chain)) // 4,
                    len(chain) - 1,
                ]))
                initial_pieces = [
                    chain[a:b + 1]
                    for a, b in zip(idxs[:-1], idxs[1:])
                    if b > a
                ]

            paths = []
            attempts = 0
            fallback_splits = 0

            def solve(piece, depth=0):
                nonlocal attempts, fallback_splits
                if len(piece) < 2 or piece[0] == piece[-1]:
                    return
                direct = _rectilinear_chain_path(piece, route_mask)
                if direct is not None and len(direct) >= 2:
                    paths.append(direct)
                    return
                attempts += 1
                result = v13_four_direction_path(
                    piece[0],
                    piece[-1],
                    route_mask,
                    turn_penalty=max(
                        2.0,
                        v13_turn_penalty_px * recovery_straight_turn_multiplier,
                    ),
                )
                if result.get("status") == "ok":
                    routed = simplify_backfill_to_straight_segments(
                        result.get("points", []), route_mask
                    )
                    if len(routed) >= 2:
                        paths.append(routed)
                        return

                # 狹長、真正多彎的走道若不能首尾直化，只在必要處逐層拆分；
                # 深度上限避免退回逐骨架像素照描。
                if depth < 4 and len(piece) >= 4:
                    fallback_splits += 1
                    mid = len(piece) // 2
                    solve(piece[:mid + 1], depth + 1)
                    solve(piece[mid:], depth + 1)

            for piece in initial_pieces:
                solve(piece)

            total_turns = sum(count_navigation_turns(path) for path in paths)
            return paths, {
                "attempts": int(attempts),
                "fallback_splits": int(fallback_splits),
                "turns": int(total_turns),
            }

        # global wall collision 保持 primary 版本；recovery 只取其 region 內真正 free pixels。
        for rec_rid in recovered_corridor_ids:
            if not _recovery_budget_ok():
                recovery_graph_timed_out = True
                break

            rec_region = (res_matrix == int(rec_rid)).astype(np.uint8) * 255
            rec_safe = cv2.bitwise_and(rec_region, wall_free)
            # 小 close 只補 segmentation 細裂縫，不跨牆。
            rec_safe = cv2.morphologyEx(
                rec_safe, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            rec_safe = cv2.bitwise_and(rec_safe, wall_free)
            recovery_safe_masks[int(rec_rid)] = rec_safe

            chains, rec_skel = _skeleton_chains_from_mask(rec_safe)
            recovery_skeleton_masks[int(rec_rid)] = rec_skel
            rec_nodes = set()
            added_edges_before = G.number_of_edges()
            recovery_chain_route_count = 0
            recovery_chain_fallback_splits = 0
            recovery_chain_turn_count = 0
            recovery_chain_attempt_count = 0

            # V14：medial skeleton 只提供「分岔與環路」拓樸，不逐折點照描。
            # 每條 structural chain 優先首尾直化，只有整段不可走時才遞迴拆分。
            for chain in chains:
                if not _recovery_budget_ok():
                    recovery_graph_timed_out = True
                    break
                clean_chain = [
                    tuple(map(int, pt)) for pt in chain
                    if in_bounds(pt[0], pt[1]) and rec_safe[int(pt[1]), int(pt[0])] > 0
                ]
                if len(clean_chain) < 2:
                    continue
                routed_paths, route_stats = _straight_recovery_chain_paths(clean_chain, rec_safe)
                recovery_chain_attempt_count += int(route_stats["attempts"])
                recovery_chain_fallback_splits += int(route_stats["fallback_splits"])
                recovery_chain_turn_count += int(route_stats["turns"])
                for routed in routed_paths:
                    if _add_recovery_axis_polyline(routed, rec_nodes) > 0:
                        recovery_chain_route_count += 1

            # 清掉因 chain simplification 造成的 recovery isolated nodes。
            for nid in list(rec_nodes):
                if nid in G and G.degree(nid) == 0:
                    pos = tuple(G.nodes[nid]['pos'])
                    G.remove_node(nid)
                    nodes_data.pop(nid, None)
                    wp_node_map.pop(pos, None)
                    rec_nodes.discard(nid)
                    recovery_graph_node_ids.discard(nid)

            recovery_nodes_by_rid[int(rec_rid)] = rec_nodes
            added_edges = max(0, G.number_of_edges() - added_edges_before)
            recovery_graph_edge_count += added_edges
            recovery_graph_records.append({
                "recovered_id": int(rec_rid),
                "safe_area_px": int(cv2.countNonZero(rec_safe)),
                "skeleton_px": int(cv2.countNonZero(rec_skel)),
                "chain_count": int(len(chains)),
                "straightened_chain_route_count": int(recovery_chain_route_count),
                "straight_path_attempt_count": int(recovery_chain_attempt_count),
                "fallback_split_count": int(recovery_chain_fallback_splits),
                "straight_path_turn_count": int(recovery_chain_turn_count),
                "straight_turn_penalty_multiplier": round(float(recovery_straight_turn_multiplier), 3),
                "road_node_count": int(len(rec_nodes)),
                "road_edge_count": int(added_edges),
                "gateway_edges": 0,
                "reattached_rooms": [],
            })

        record_by_rid = {int(r["recovered_id"]): r for r in recovery_graph_records}

        # 同一 recovered region 若因簡化/障礙物被切成數塊，只做「區域內」四方向安全縫合。
        # V8 不再用任意角度直線；所有 stitch 都走 v13_four_direction_path。
        for rid, rec_nodes in recovery_nodes_by_rid.items():
            if not _recovery_budget_ok():
                recovery_graph_timed_out = True
                break
            rec_nodes = {n for n in rec_nodes if n in G}
            if len(rec_nodes) < 2:
                continue
            rec_safe = recovery_safe_masks.get(rid)
            local_stitch_count = 0
            while _recovery_budget_ok():
                local_sub = G.subgraph(rec_nodes)
                comps = [set(c) for c in nx.connected_components(local_sub)]
                if len(comps) <= 1:
                    break
                comps.sort(key=len, reverse=True)
                base = comps[0]
                candidate_pairs = []
                max_join = max(corridor_width_est * 2.4, 96.0)
                for other in comps[1:]:
                    for u in base:
                        pu = tuple(G.nodes[u]['pos'])
                        for v in other:
                            pv = tuple(G.nodes[v]['pos'])
                            d = math.hypot(pu[0] - pv[0], pu[1] - pv[1])
                            if d <= max_join:
                                candidate_pairs.append((d, u, v, pu, pv))
                candidate_pairs.sort(key=lambda z: z[0])
                stitched = False
                for _, u, v, pu, pv in candidate_pairs[:16]:
                    if not _recovery_budget_ok():
                        break
                    result = v13_four_direction_path(
                        pu, pv, rec_safe,
                        turn_penalty=max(2.0, v13_turn_penalty_px * recovery_straight_turn_multiplier),
                    )
                    if result.get("status") != "ok":
                        continue
                    straight_stitch = simplify_backfill_to_straight_segments(
                        result.get("points", []), rec_safe
                    )
                    before = G.number_of_edges()
                    _add_recovery_axis_polyline(straight_stitch, rec_nodes)
                    delta = max(0, G.number_of_edges() - before)
                    if delta > 0:
                        recovery_graph_edge_count += delta
                        local_stitch_count += delta
                        stitched = True
                        break
                if not stitched or local_stitch_count >= 16:
                    break
            recovery_nodes_by_rid[rid] = rec_nodes
            if rid in record_by_rid:
                record_by_rid[rid]["local_stitch_edges"] = int(local_stitch_count)

        # V8 recovery topology census：以 persistent holes 作最低 cycle requirement。
        # 若 skeleton->rectilinear 轉換已保住足夠環路，就完全不加線；不足時才局部補環。
        for rid, rec_nodes in recovery_nodes_by_rid.items():
            if not _recovery_budget_ok():
                recovery_graph_timed_out = True
                break
            rec_nodes = {n for n in rec_nodes if n in G}
            rec_safe = recovery_safe_masks.get(rid)
            if rec_safe is None or not rec_nodes:
                continue
            holes = _recovery_persistent_holes(rec_safe)
            required_rank = int(len(holes))
            rank_before = _recovery_cycle_rank(rec_nodes)
            cycle_repairs = 0
            if rank_before < required_rank:
                for hole_index, hole in enumerate(holes, start=1):
                    if not _recovery_budget_ok() or _recovery_cycle_rank(rec_nodes) >= required_rank:
                        break
                    staged = _hole_ring_candidate_paths(rec_safe, hole)
                    if not staged:
                        continue
                    before_rank = _recovery_cycle_rank(rec_nodes)
                    before_edges = G.number_of_edges()
                    for pts in staged:
                        _add_recovery_axis_polyline(
                            pts,
                            rec_nodes,
                            topology_cycle=True,
                            recovery_cycle=True,
                        )
                    after_rank = _recovery_cycle_rank(rec_nodes)
                    if after_rank > before_rank:
                        cycle_repairs += 1
                        recovery_graph_edge_count += max(0, G.number_of_edges() - before_edges)
            rank_after = _recovery_cycle_rank(rec_nodes)
            recovery_nodes_by_rid[rid] = rec_nodes
            if rid in record_by_rid:
                record_by_rid[rid].update({
                    "persistent_hole_count": int(required_rank),
                    "cycle_rank_before_repair": int(rank_before),
                    "cycle_rank_after_repair": int(rank_after),
                    "cycle_repairs": int(cycle_repairs),
                    "cycle_requirement_met": bool(rank_after >= required_rank),
                    "movement": "straight_visibility_segment_preferred",
                })
            print(
                f"[Recovery topology V8] rid={rid}，holes={required_rank}，"
                f"cycle_rank={rank_before}->{rank_after}，repairs={cycle_repairs}。"
            )

        # Reserve the last 40% for integration and room attachments, even if
        # optional local cycle refinement exhausted its own time slice.
        recovery_phase_deadline = recovery_graph_start + recovery_graph_budget
        path_search_deadline = recovery_phase_deadline
        # 只在 local recovery graph 建完後才接 semantic gateway，primary graph 已不會再被重建/裁剪。
        integrated_recovered_ids = {
            rid for rid, nodes in recovery_nodes_by_rid.items()
            if nodes & primary_graph_nodes_snapshot
        }
        for grecord in semantic_gateway_records:
            if not _recovery_budget_ok():
                recovery_graph_timed_out = True
                break
            if not grecord.get("accepted"):
                continue
            rid = int(grecord.get("recovered_id", -1))
            rec_nodes = recovery_nodes_by_rid.get(rid, set())
            if not rec_nodes:
                continue
            p_rec = tuple(map(int, grecord.get("recovered_point", ())))
            p_pub = tuple(map(int, grecord.get("public_point", ())))
            if len(p_rec) != 2 or len(p_pub) != 2:
                continue

            # recovery side：gateway 本身可以是語意跨區線，但 gateway -> recovery local graph
            # 必須仍是實際可走的 H/V 路徑；不允許用斜線硬接 skeleton。
            rec_safe = recovery_safe_masks.get(rid)
            if rec_safe is None:
                grecord["integration_status"] = "recovery_safe_mask_missing"
                continue
            access = _nearest_mask_point(p_rec, rec_safe, max(8, wall_collision_radius * 3))
            if access is None:
                grecord["integration_status"] = "recovery_contact_eroded"
                continue
            p_rec = access
            grecord["recovery_safe_contact"] = list(access)
            nearest_candidates = sorted(
                (
                    (math.hypot(G.nodes[n]['pos'][0] - p_rec[0], G.nodes[n]['pos'][1] - p_rec[1]), n)
                    for n in rec_nodes if n in G
                ),
                key=lambda z: z[0],
            )[:10]
            best_rec = None
            for d, nid in nearest_candidates:
                if not _recovery_budget_ok():
                    break
                pt = tuple(G.nodes[nid]['pos'])
                result = v13_four_direction_path(
                    p_rec, pt, rec_safe,
                    turn_penalty=max(2.0, v13_turn_penalty_px * recovery_straight_turn_multiplier),
                )
                if result.get("status") == "ok":
                    best_rec = (
                        float(d),
                        nid,
                        pt,
                        simplify_backfill_to_straight_segments(result.get("points", []), rec_safe),
                    )
                    break
            if best_rec is None:
                grecord["integration_status"] = "recovery_side_no_safe_axis_access"
                continue

            rec_anchor = add_waypoint(p_rec, node_type="recovery_gateway")
            recovery_graph_node_ids.add(rec_anchor)
            rec_nodes.add(rec_anchor)
            if rec_anchor != best_rec[1]:
                _add_recovery_axis_polyline(best_rec[3], rec_nodes)

            # primary side：投影到「primary graph」最近邊，不改變任何 primary topology，只做 edge split。
            primary_access_mask = cv2.bitwise_and(valid_routing_mask, wall_free)
            primary_access_mask[(res_matrix > 1) & (primary_corridor_mask == 0)] = 0
            primary_contact = _nearest_mask_point(
                p_pub, primary_access_mask, max(8, wall_collision_radius * 3)
            )
            if primary_contact is None:
                grecord["integration_status"] = "primary_contact_eroded"
                continue
            primary_projection = _split_nearest_edge_for_point(
                primary_contact, allowed_edge_prefix="primary", access_mask=primary_access_mask
            )
            if primary_projection is None:
                grecord["integration_status"] = "no_reachable_primary_edge"
                continue
            primary_anchor, projected_pub, primary_gap = primary_projection
            # Keep the uncertain transition local. The remainder to the existing
            # road is a verified orthogonal route inside primary public space.
            access_path = _compiled_axis_path(
                projected_pub, primary_contact, primary_access_mask, v13_turn_penalty_px
            )['points']
            previous = primary_anchor
            for point in access_path[1:]:
                contact_node = add_waypoint(point)
                add_graph_edge(previous, contact_node, edge_type="route")
                previous = contact_node
            primary_anchor = previous
            grecord["primary_safe_contact"] = list(primary_contact)

            # 跨區線只是一條「規劃語意邊」：可以用於 shortest path，但不是實體門偵測結果。
            # main.py 看到這種 edge 時必須提醒使用者在附近尋找真正入口。
            add_graph_edge(
                primary_anchor,
                rec_anchor,
                edge_type="recovery_gateway",
                semantic_transition=True,
                semantic_only=True,
                physical_connection_confirmed=False,
                gateway_kind="approximate_cross_region_access",
                gateway_id=grecord.get("gateway_id"),
                recovered_id=int(rid),
                target_id=grecord.get("target_id"),
                navigation_notice=grecord.get(
                    "navigation_notice",
                    "抵達跨區連接附近後，請在附近尋找實際可通行的入口；圖上的跨區連線只是位置提示，不代表入口的精確位置。",
                ),
            )
            recovery_gateway_edge_count += 1
            integrated_recovered_ids.add(rid)
            grecord["integration_status"] = "integrated_after_primary_freeze"
            grecord["primary_graph_projection"] = [int(projected_pub[0]), int(projected_pub[1])]
            grecord["primary_projection_distance_px"] = round(float(primary_gap), 2)
            grecord["recovery_anchor_node"] = rec_anchor
            grecord["primary_anchor_node"] = primary_anchor
            if rid in record_by_rid:
                record_by_rid[rid]["gateway_edges"] += 1

        # 未接到 primary 的 recovered graph 不應污染 canonical graph。
        for rid, rec_nodes in recovery_nodes_by_rid.items():
            if rid in integrated_recovered_ids:
                continue
            for nid in list(rec_nodes):
                if nid not in G:
                    continue
                # 只移除 recovery-only component；若意外已碰 primary node則保留。
                if (nid not in primary_graph_nodes_snapshot
                        and nodes_data.get(nid, {}).get("type") in {"recovery_waypoint", "recovery_gateway"}):
                    G.remove_node(nid)
                    pos = tuple(nodes_data.get(nid, {}).get("coordinates", ()))
                    nodes_data.pop(nid, None)
                    if len(pos) == 2 and wp_node_map.get(pos) == nid:
                        wp_node_map.pop(pos, None)
                    recovery_graph_node_ids.discard(nid)
            if rid in record_by_rid:
                record_by_rid[rid]["discarded_reason"] = "no_primary_gateway"

        # 移除「沒有接到任何 primary base node」的 recovery-only orphan component。
        # 這保證 augmentation 不會把 canonical graph 從 connected 變成 disconnected。
        primary_connected_union = set()
        seen_primary = set()
        for base_nid in primary_graph_nodes_snapshot:
            if base_nid not in G or base_nid in seen_primary:
                continue
            comp = set(nx.node_connected_component(G, base_nid))
            primary_connected_union.update(comp)
            seen_primary.update(comp)
        orphan_recovery_nodes = {
            nid for nid in recovery_graph_node_ids
            if nid in G and nid not in primary_connected_union
        }
        if orphan_recovery_nodes:
            for nid in list(orphan_recovery_nodes):
                if nid not in G:
                    continue
                pos = tuple(G.nodes[nid]['pos'])
                G.remove_node(nid)
                nodes_data.pop(nid, None)
                if wp_node_map.get(pos) == nid:
                    wp_node_map.pop(pos, None)
                recovery_graph_node_ids.discard(nid)
            print(f"[Recovery graph V8] 移除未接回 primary 的 orphan recovery nodes={len(orphan_recovery_nodes)}。")

        # 將 recovery 周圍房間的 attachment 改到 local recovery graph。
        # 這一步只改語意抵達點，不重新跑 primary V13。
        for rid in sorted(integrated_recovered_ids):
            if not _recovery_budget_ok():
                recovery_graph_timed_out = True
                break
            rec_nodes = recovery_nodes_by_rid.get(rid, set()) & set(G.nodes())
            if not rec_nodes:
                continue
            rec_mask = (res_matrix == rid).astype(np.uint8) * 255

            # V9：不再用「recovery mask dilation 剛好掃到誰」決定房間。
            # ownership 已在 graph 建立前固定；每個 recovered public region 只接
            # 自己的房間，primary-side 房間即使幾何上很近也不會被搶走。
            owned_room_ids = {
                str(room_id) for room_id, owner in room_public_owner.items()
                if owner is not None
                and int(owner) == int(rid)
                and str(room_id) in room_attachments
            }

            for room_id in owned_room_ids:
                info = room_attachments.get(room_id)
                if not info:
                    continue
                center = tuple(info["room_center"])
                # 只搜尋「這一個 recovered region 自己的」recovery edges。
                # 不能搜尋全圖 recovery_*，否則未來有 26/60/75 多個 recovery 區時會跨區吸附。
                candidates = []
                current_rec_nodes = {n for n in rec_nodes if n in G}
                max_attach = float(max(corridor_width_est * 2.8, map_short_side * 0.12))
                for u, v, edata in list(G.edges(data=True)):
                    if edata.get("edge_type") != "recovery_route":
                        continue
                    if u not in current_rec_nodes or v not in current_rec_nodes:
                        continue
                    p1 = tuple(G.nodes[u]['pos'])
                    p2 = tuple(G.nodes[v]['pos'])
                    proj = project_point_to_segment(center, p1, p2)
                    d = math.hypot(proj[0] - center[0], proj[1] - center[1])
                    if d <= max_attach:
                        candidates.append({
                            "point": proj,
                            "distance": float(d),
                            "fallback_score": float(d),
                            "u": u,
                            "v": v,
                            "edge_data": dict(edata),
                        })
                selected, evidence, fallback_used = select_room_projection_candidate(
                    room_id, center, candidates
                )
                if selected is None:
                    continue

                d = float(selected["distance"])
                u, v = selected["u"], selected["v"]
                proj, edata = tuple(selected["point"]), selected["edge_data"]

                if proj == tuple(G.nodes[u]['pos']):
                    aid = u
                elif proj == tuple(G.nodes[v]['pos']):
                    aid = v
                else:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                    aid = add_waypoint(proj, node_type="attachment_waypoint")
                    recovery_graph_node_ids.add(aid)
                    rec_nodes.add(aid)
                    recovery_nodes_by_rid[rid] = rec_nodes
                    edge_meta = {
                        k: edata.get(k) for k in ("topology_cycle", "recovery_cycle", "movement")
                        if k in edata
                    }
                    add_graph_edge(u, aid, edge_type=edata.get("edge_type", "recovery_route"), **edge_meta)
                    add_graph_edge(aid, v, edge_type=edata.get("edge_type", "recovery_route"), **edge_meta)

                info["attachment_node"] = aid
                info["attachment_point"] = tuple(map(int, proj))
                info["distance_px"] = round(float(d), 2)
                ax, ay = map(int, proj)
                info["safe_clearance_px"] = round(float(wall_distance[ay, ax]), 2) if in_bounds(ax, ay) else None
                method_suffix = "fallback_nearest" if fallback_used else "direct"
                info["attachment_method"] = f"owned_recovered_public_{rid}_projection_{method_suffix}"
                info["public_owner_id"] = int(rid)
                info["public_owner_kind"] = "recovered_public"
                store_attachment_visibility(info, evidence, fallback_used)
                recovery_room_reattach_count += 1
                if rid in record_by_rid:
                    record_by_rid[rid]["reattached_rooms"].append(str(room_id))

        recovery_graph_elapsed = time.perf_counter() - recovery_graph_start
        # 更新 gateway report，讓 debug/report 能明確區分「辨識到」與「已融合到 canonical graph」。
        with open(self.output_dir / "semantic_public_gateway_report.json", "w", encoding="utf-8") as f:
            json.dump(semantic_gateway_records, f, ensure_ascii=False, indent=2)

        # Recovery debug：白=region safe mask；青=local recovery graph；紅=semantic gateway。
        recovery_debug = cv2.cvtColor(
            (1 - (wall_matrix > 0).astype(np.uint8)) * 255,
            cv2.COLOR_GRAY2BGR
        )
        recovery_union = np.zeros((H, W), dtype=np.uint8)
        for rid in recovered_corridor_ids:
            recovery_union = cv2.bitwise_or(
                recovery_union,
                recovery_safe_masks.get(rid, np.zeros((H, W), dtype=np.uint8))
            )
        recovery_debug[recovery_union > 0] = (245, 245, 245)
        for u, v, edata in G.edges(data=True):
            et = str(edata.get("edge_type", ""))
            if et == "recovery_route":
                cv2.line(recovery_debug, tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos']), (255, 180, 0), 2)
            elif et == "recovery_gateway":
                cv2.line(recovery_debug, tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos']), (0, 0, 255), 3)
        safe_imwrite(str(self.output_dir / "debug_recovery_local_graph_v8.jpg"), recovery_debug)

        print(
            "[Recovery graph V8] "
            f"regions={sorted(integrated_recovered_ids)}，"
            f"nodes={len(recovery_graph_node_ids)}，"
            f"route_edges≈{recovery_graph_edge_count}，gateway_edges={recovery_gateway_edge_count}，"
            f"reattached_rooms={recovery_room_reattach_count}，"
            f"elapsed={recovery_graph_elapsed:.3f}s/{recovery_graph_budget:.1f}s，"
            f"timed_out={recovery_graph_timed_out}。"
        )

        # -------------------------------------------------
        # 12. 共線 degree=2 壓縮；attachment 節點必須保留
        # -------------------------------------------------
        protected_attachment_nodes = {
            info.get("attachment_node")
            for info in room_attachments.values()
            if info.get("attachment_node") in G
        }
        protected_attachment_nodes.update({
            nid for nid in G.nodes()
            if nodes_data.get(nid, {}).get("type") == "recovery_gateway"
        })

        def dominant_edge_type(type_a, type_b):
            priority = {"route": 0, "recovery_route": 0, "shortcut": 1, "component_bridge": 2, "recovery_gateway": 3}
            return type_a if priority.get(type_a, 0) >= priority.get(type_b, 0) else type_b

        changed = True
        while changed:
            changed = False
            for nid in list(G.nodes()):
                if nid not in G or G.degree(nid) != 2 or nid in protected_attachment_nodes:
                    continue
                n1, n2 = list(G.neighbors(nid))
                p = G.nodes[nid]['pos']
                p1, p2 = G.nodes[n1]['pos'], G.nodes[n2]['pos']
                if not ((p1[0] == p[0] == p2[0]) or (p1[1] == p[1] == p2[1])):
                    continue
                et1 = G.edges[nid, n1].get("edge_type", "route")
                et2 = G.edges[nid, n2].get("edge_type", "route")
                et = dominant_edge_type(et1, et2)
                add_graph_edge(n1, n2, edge_type=et)
                G.remove_node(nid)
                nodes_data.pop(nid, None)
                wp_node_map.pop(tuple(p), None)
                changed = True
                break

        # V9：壓縮後逐一確認所有房間 attachment，而且必須維持 public-space ownership。
        # 絕不再使用「全圖最近節點」把 recovery-owned room 偷掛回 primary graph。
        safe_nodes_after = []
        for n in G.nodes():
            if G.degree(n) <= 0:
                continue
            if not point_is_safe(
                G.nodes[n]['pos'],
                clearance=min_attachment_clearance,
                mask=safe_attachment_mask,
            ):
                continue
            incident_types = {
                str(G.edges[n, nb].get("edge_type", "route"))
                for nb in G.neighbors(n)
            }
            # primary fallback 只可用 primary-side route/bridge/shortcut，semantic/recovery edge 不算。
            if incident_types and all(et.startswith("recovery_") for et in incident_types):
                continue
            safe_nodes_after.append(n)
        if not safe_nodes_after:
            safe_nodes_after = [
                n for n in G.nodes()
                if nodes_data.get(n, {}).get("type") not in {"recovery_waypoint", "recovery_gateway"}
            ]

        def _project_room_to_owned_recovery(room_id, owner_rid, center):
            owner_rid = int(owner_rid)
            owner_nodes = {
                n for n in recovery_nodes_by_rid.get(owner_rid, set())
                if n in G
            }
            if not owner_nodes or owner_rid not in integrated_recovered_ids:
                return None

            candidates = []
            for u, v, edata in list(G.edges(data=True)):
                if edata.get("edge_type") != "recovery_route":
                    continue
                if u not in owner_nodes or v not in owner_nodes:
                    continue
                p1 = tuple(G.nodes[u]['pos'])
                p2 = tuple(G.nodes[v]['pos'])
                proj = project_point_to_segment(center, p1, p2)
                d = math.hypot(proj[0] - center[0], proj[1] - center[1])
                candidates.append({
                    "point": proj,
                    "distance": float(d),
                    "fallback_score": float(d),
                    "u": u,
                    "v": v,
                    "edge_data": dict(edata),
                })

            selected, evidence, fallback_used = select_room_projection_candidate(
                room_id, center, candidates
            )
            if selected is None:
                if not owner_nodes:
                    return None
                node_candidates = [
                    {
                        "point": tuple(G.nodes[n]['pos']),
                        "distance": math.hypot(
                            G.nodes[n]['pos'][0] - center[0],
                            G.nodes[n]['pos'][1] - center[1],
                        ),
                        "node": n,
                    }
                    for n in owner_nodes
                ]
                selected, evidence, fallback_used = select_room_projection_candidate(
                    room_id, center, node_candidates
                )
                if selected is None:
                    return None
                nid = selected["node"]
                pt = tuple(selected["point"])
                suffix = "fallback_nearest" if fallback_used else "direct"
                return nid, pt, float(selected["distance"]), f"owned_recovery_nearest_node_{suffix}", evidence, fallback_used

            d = float(selected["distance"])
            u, v = selected["u"], selected["v"]
            proj, edata = tuple(selected["point"]), selected["edge_data"]
            suffix = "fallback_nearest" if fallback_used else "direct"
            if proj == tuple(G.nodes[u]['pos']):
                return u, proj, d, f"owned_recovery_edge_endpoint_{suffix}", evidence, fallback_used
            if proj == tuple(G.nodes[v]['pos']):
                return v, proj, d, f"owned_recovery_edge_endpoint_{suffix}", evidence, fallback_used

            if G.has_edge(u, v):
                G.remove_edge(u, v)
            aid = add_waypoint(proj, node_type="attachment_waypoint")
            recovery_graph_node_ids.add(aid)
            owner_nodes.add(aid)
            recovery_nodes_by_rid[owner_rid] = owner_nodes
            edge_meta = {
                k: edata.get(k) for k in ("topology_cycle", "recovery_cycle", "movement")
                if k in edata
            }
            add_graph_edge(u, aid, edge_type="recovery_route", **edge_meta)
            add_graph_edge(aid, v, edge_type="recovery_route", **edge_meta)
            return aid, proj, d, f"owned_recovery_edge_projection_{suffix}", evidence, fallback_used

        post_owner_repair_count = 0
        for rid, info in room_attachments.items():
            owner = info.get("public_owner_id")
            owner_is_recovery = owner is not None and int(owner) in set(recovered_corridor_ids)
            aid = info.get("attachment_node")

            if owner_is_recovery:
                owner_nodes_now = {
                    n for n in recovery_nodes_by_rid.get(int(owner), set()) if n in G
                }
                owner_ok = aid in owner_nodes_now
                if owner_ok and aid in G:
                    # attachment 節點必須真的屬於 owner recovery graph；gateway 不算房間抵達點。
                    incident_types = {
                        G.edges[aid, nb].get("edge_type", "route")
                        for nb in G.neighbors(aid)
                    }
                    owner_ok = bool("recovery_route" in incident_types or not incident_types)

                if not owner_ok:
                    repaired = _project_room_to_owned_recovery(rid, owner, info["room_center"])
                    if repaired is not None:
                        aid, pt, d, method, evidence, fallback_used = repaired
                        info["attachment_node"] = aid
                        info["attachment_point"] = tuple(map(int, pt))
                        info["distance_px"] = round(float(d), 2)
                        ax, ay = map(int, pt)
                        info["safe_clearance_px"] = round(float(wall_distance[ay, ax]), 2) if in_bounds(ax, ay) else None
                        info["attachment_method"] = method
                        info["public_owner_kind"] = "recovered_public"
                        store_attachment_visibility(info, evidence, fallback_used)
                        post_owner_repair_count += 1
                        continue

                    # owner recovery graph 沒建立成功時寧可標 unresolved，也不靜默掛到 primary。
                    info["attachment_node"] = None
                    info["attachment_point"] = None
                    info["distance_px"] = None
                    info["safe_clearance_px"] = None
                    info["attachment_method"] = f"owner_recovery_{int(owner)}_unavailable"
                    continue

                ax, ay = G.nodes[aid]['pos']
                info["attachment_point"] = (int(ax), int(ay))
                info["distance_px"] = round(float(math.hypot(ax - info["room_center"][0], ay - info["room_center"][1])), 2)
                info["safe_clearance_px"] = round(float(wall_distance[ay, ax]), 2)
                continue

            # primary-owned / unresolved room：重新對「實際仍存在的 primary edges」做直達投影；
            # edge 比單純最近節點有更完整的候選集合，也能避免壓縮後被迫跨越其他房間。
            if aid not in G or (aid in G and nodes_data.get(aid, {}).get("type") in {"recovery_waypoint", "recovery_gateway"}):
                center = info["room_center"]
                repaired = attach_room_to_existing_graph(rid, center, primary_only=True)
                if repaired is not None:
                    pt, aid, d, method, evidence, fallback_used = repaired
                    info["attachment_point"] = tuple(map(int, pt))
                    info["distance_px"] = round(float(d), 2)
                    info["attachment_method"] = f"post_compression_{method}"
                    store_attachment_visibility(info, evidence, fallback_used)
                else:
                    aid = None
                    info["attachment_point"] = None
                    info["attachment_method"] = "unattached_no_primary_road_graph"
            info["attachment_node"] = aid
            if aid in G:
                ax, ay = G.nodes[aid]['pos']
                info["attachment_point"] = (int(ax), int(ay))
                info["distance_px"] = round(float(math.hypot(ax - info["room_center"][0], ay - info["room_center"][1])), 2)
                info["safe_clearance_px"] = round(float(wall_distance[ay, ax]), 2)
            else:
                info["distance_px"] = None
                info["safe_clearance_px"] = None

        owner_attachment_mismatches = []
        all_recovery_nodes_now = set()
        for _rid_nodes in recovery_nodes_by_rid.values():
            all_recovery_nodes_now.update(n for n in _rid_nodes if n in G)

        for rid, info in room_attachments.items():
            owner = info.get("public_owner_id")
            aid = info.get("attachment_node")
            if owner is None or aid is None:
                continue
            if int(owner) in set(recovered_corridor_ids):
                valid_owner_nodes = {
                    n for n in recovery_nodes_by_rid.get(int(owner), set()) if n in G
                }
                ok = aid in valid_owner_nodes
            else:
                ok = aid not in all_recovery_nodes_now
            if not ok:
                owner_attachment_mismatches.append({
                    "room_id": str(rid),
                    "owner_public_id": int(owner),
                    "attachment_node": aid,
                    "attachment_method": info.get("attachment_method"),
                })
                # fail closed：錯區 attachment 比暫時未吸附更危險。
                info["attachment_node"] = None
                info["attachment_point"] = None
                info["distance_px"] = None
                info["safe_clearance_px"] = None
                info["attachment_method"] = "owner_validation_rejected_cross_region_attachment"

        with open(self.output_dir / "room_attachment_owner_validation_v9.json", "w", encoding="utf-8") as f:
            json.dump({
                "mismatch_count": len(owner_attachment_mismatches),
                "mismatches": owner_attachment_mismatches,
            }, f, ensure_ascii=False, indent=2)

        print(
            f"[房間歸屬 V9] post-compression owner repairs={post_owner_repair_count}，"
            f"final_owner_mismatches={len(owner_attachment_mismatches)}"
        )

        # 房間中心節點只保留語意，不加入道路邊。
        for rid, center in room_coords.items():
            node_id = f"R_{rid}"
            names = id_labels.get(rid, {}).get("names", [])
            attach = room_attachments.get(rid)
            attach_point = attach.get("attachment_point") if attach else None
            nodes_data[node_id] = {
                "type": "room_center",
                "name": "、".join(names) if names else f"房間_{rid}",
                "coordinates": [center[0], center[1]],
                "attachment_node": attach.get("attachment_node") if attach else None,
                "attachment_point": list(attach_point) if attach_point is not None else None,
                "arrival_rule": "導航終點為 attachment_node；抵達後即視為到達此房間"
            }

        # -------------------------------------------------
        # 13. JSON 輸出
        # -------------------------------------------------
        edges_data = []
        edge_metadata_keys = (
            "topology_cycle", "recovery_cycle", "movement",
            "semantic_transition", "semantic_only", "physical_connection_confirmed",
            "gateway_kind", "gateway_id", "recovered_id", "target_id", "navigation_notice",
        )
        for u, v, edata in G.edges(data=True):
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
            dist = round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 2)
            deg_uv = round(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 360, 1)
            deg_vu = round(math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0])) % 360, 1)
            edge_type = edata.get("edge_type", "route")
            meta = {key: edata[key] for key in edge_metadata_keys if key in edata}
            edges_data.extend([
                {"source": u, "target": v, "distance_px": dist, "direction_deg": deg_uv, "edge_type": edge_type, **meta},
                {"source": v, "target": u, "distance_px": dist, "direction_deg": deg_vu, "edge_type": edge_type, **meta}
            ])

        attachment_data = []
        for rid, info in room_attachments.items():
            attachment_point = info.get("attachment_point")
            attachment_data.append({
                "room_node": f"R_{rid}",
                "attachment_node": info.get("attachment_node"),
                "room_center": list(info["room_center"]),
                "attachment_point": list(attachment_point) if attachment_point is not None else None,
                "distance_px": info.get("distance_px"),
                "safe_clearance_px": info.get("safe_clearance_px"),
                "attachment_method": info.get("attachment_method"),
                "direct_connection": bool(info.get("direct_connection", False)),
                "single_wall_connection": bool(info.get("single_wall_connection", False)),
                "crosses_other_room": bool(info.get("crosses_other_room", False)),
                "crossed_room_ids": list(info.get("crossed_room_ids", [])),
                "wall_crossing_runs": info.get("wall_crossing_runs"),
                "nearest_projection_fallback_used": bool(info.get("nearest_projection_fallback_used", False)),
                "public_owner_id": info.get("public_owner_id"),
                "public_owner_kind": info.get("public_owner_kind"),
                "attached": bool(info.get("attachment_node") in G),
                "relation": "arrival_reference_only"
            })

        payload = {
            "map_scale": self.scale,
            "graph_connected": bool(G.number_of_nodes() > 0 and nx.is_connected(G)),
            "road_component_count": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
            "route_reduction": {
                "raw_segment_count": int(raw_segment_count),
                "unique_coordinate_segment_count": int(base_unique_count),
                "reduced_segment_count": int(reduced_segment_count),
                "removed_parallel_segments": int(max(0, base_unique_count - reduced_segment_count)),
                "parallel_threshold_used_px": int(parallel_threshold_used),
                "retention_ratio": round(float(parallel_retention_ratio), 4),
                "strategy": "topology_preserving_non_transitive"
            },
            "room_coverage": {
                "room_count": int(len(room_coords)),
                "room_center_node_count": int(len(room_coords)),
                "attached_room_count": int(sum(1 for info in room_attachments.values() if info.get("attachment_node") in G)),
                "unattached_room_count": int(sum(1 for info in room_attachments.values() if info.get("attachment_node") not in G)),
                "direct_attachment_count": int(sum(
                    1 for info in room_attachments.values()
                    if info.get("attachment_node") in G and info.get("direct_connection")
                )),
                "fallback_nearest_attachment_count": int(sum(
                    1 for info in room_attachments.values()
                    if info.get("attachment_node") in G and info.get("nearest_projection_fallback_used")
                ))
            },
            "public_space_layers": {
                "primary_public_ids": [int(v) for v in primary_corridor_ids],
                "recovered_public_ids": [int(v) for v in recovered_corridor_ids],
                "primary_topology_frozen_before_recovery": True,
                "recovery_graph_strategy": "V15_strict_HV_low_turn_rectilinear_graph_with_cycle_census",
                "recovery_graph_budget_seconds": round(float(recovery_graph_budget), 3),
                "recovery_graph_elapsed_seconds": round(float(recovery_graph_elapsed), 4),
                "recovery_graph_timed_out": bool(recovery_graph_timed_out),
                "integrated_recovered_ids": sorted(int(v) for v in integrated_recovered_ids),
                "recovery_route_edge_count": int(sum(
                    1 for _, _, e in G.edges(data=True)
                    if e.get("edge_type") == "recovery_route"
                )),
                "recovery_gateway_edge_count": int(sum(
                    1 for _, _, e in G.edges(data=True)
                    if e.get("edge_type") == "recovery_gateway"
                )),
                "recovery_room_reattach_count": int(recovery_room_reattach_count),
                "room_public_owner_report": "room_public_owner_report_v9.json",
                "room_public_owner_counts": {
                    "primary": int(primary_owned_count),
                    "recovered": int(recovered_owned_count),
                    "unresolved": int(unresolved_owned_count),
                },
                "room_attachment_owner_validation": {
                    "mismatch_count": int(len(owner_attachment_mismatches)),
                    "report_file": "room_attachment_owner_validation_v9.json",
                },
                "records": recovery_graph_records,
                "debug_file": "debug_recovery_local_graph_v8.jpg"
            },
            "nodes": nodes_data,
            "edges": edges_data,
            "room_attachments": attachment_data,
            "component_bridge_strategy": {
                "strategy_version": "V9_preserve_public_components",
                "target_component": "global_component_pair_graph_not_largest_only",
                "endpoint_policy": "any_point_on_horizontal_or_vertical_graph_edge",
                "movement": "four_direction_only_no_diagonal",
                "path_cost_priority": [
                    "routing_safety_tier",
                    "weighted_distance_with_turn_penalty",
                    "corridor_proximity",
                    "room_core_penalty"
                ],
                "new_node_policy": "split_contact_edges_and_create_turning_points",
                "map_short_side_px": int(map_short_side),
                "estimated_corridor_width_px": round(float(corridor_width_est), 2),
                "wall_collision_radius_px": int(wall_collision_radius),
                "corridor_expand_radius_px": int(corridor_expand_radius),
                "astar_scale_px_per_cell": int(astar_scale),
                "adaptive_near_radius_px": int(near_bridge_radius),
                "adaptive_far_radius_px": int(far_bridge_radius),
                "turn_penalty_px": round(float(bridge_turn_penalty_px), 2),
                "max_expansions": int(axis_bridge_max_expansions),
                "unreachable_component_policy": unresolved_policy
            },
            "component_bridges": bridge_records,
            "pruned_components": pruned_component_records,
            "unconnected_components": unconnected_records,
            "cycle_closure_strategy": {
                "search_budget_seconds": topology_search_budget,
                "search_budget_exhausted": bool(topology_search_exhausted or any(
                    topology_search_status_counts.get(key,0) for key in ('time_budget','search_budget','memory_budget'))),
                "search_status_counts": dict(topology_search_status_counts),
                "incomplete_search_is_not_proof_of_no_path": True,
                "strategy_version": "V15_topology_first_with_strict_HV_backfill",
                "movement": "strict_horizontal_vertical_segments_only",
                "backfill_turn_penalty_multiplier": round(float(v14_backfill_turn_multiplier), 3),
                "reference_source": "topology_walkable_mask_medial_skeleton",
                "reference_skeleton_node_count": int(reference_skeleton_node_count),
                "reference_cycle_rank": int(len(persistent_holes)),
                "repair_stages": [
                    "reference_skeleton_missing_branch_completion",
                    "bridge_chain_physical_bypass_test",
                    "persistent_hole_cycle_census_and_restore"
                ],
                "gap_limit_policy": "no_fixed_geometric_gap_limit",
                "estimated_corridor_width_px": round(float(corridor_width_est), 2),
                "represented_radius_px": int(represented_radius),
                "reference_support_radius_px": int(reference_support_radius),
                "persistent_hole_count": int(len(persistent_holes)),
                "reference_repairs": int(reference_repairs),
                "reference_search_engine": "full_resolution_compiled_axis_grid_with_turn_cost",
                "reference_search_cache_entries": int(len(reference_search_cache)),
                "bridge_bypass_repairs": int(bridge_repairs),
                "hole_fallback_repairs": int(hole_fallback_repairs),
                "added_cycle_repairs": int(len(topology_cycle_records)),
                "local_fusion_gap_px": int(v13_fusion_gap_px),
                "local_fusion_edges_added": int(len(topology_fusion_records)),
                "topology_component_count": int(len(topology_component_contact_report)),
                "orphan_topology_component_count": int(orphan_topology_components),
                "graph_connected_after_fusion": bool(topology_graph_connected_after_fusion),
                "promoted_topology_edges_to_route": int(topology_promoted_edge_count)
            },
            "topology_validation": {
                "persistent_loop_count": int(len(persistent_holes)),
                "represented_loop_count_before": int(sum(1 for v in hole_status_before.values() if v)),
                "represented_loop_count_after": int(sum(1 for v in hole_status_after.values() if v)),
                "initial_missing_reference_component_count": int(len(initial_missing_components)),
                "final_missing_reference_component_count": int(len(final_missing_components)),
                "initial_bridge_chain_count": int(initial_bridge_chain_count),
                "final_graph_bridge_count": int(final_bridge_count),
                "persistent_holes": persistent_holes,
                "topology_fusion": {
                    "gap_px": int(v13_fusion_gap_px),
                    "edges_added": topology_fusion_records,
                    "component_contacts": topology_component_contact_report,
                    "orphan_component_count": int(orphan_topology_components),
                    "graph_connected_after_fusion": bool(topology_graph_connected_after_fusion),
                    "promoted_topology_edges_to_route": int(topology_promoted_edge_count)
                },
                "unresolved_topology": topology_unresolved_records,
                "reference_debug_file": "debug_topology_reference_v13.jpg"
            },
            "shortcuts": shortcut_records
        }
        payload["corridor_transfers"] = build_corridor_transfers_v10(payload, res_matrix, id_labels)
        payload["performance"] = {
            "strategy_version": "FAST_V2_global_multisource_batched_mst",
            "bridge_engine": str(os.environ.get("MAP_COMPONENT_BRIDGE_ENGINE", "global_multisource")),
            "bridge_grid_max_dim": int(bridge_grid_max_dim),
            "route_preparation_seconds": round(float(component_merge_start - graph_generation_start), 4),
            "component_merge_seconds": round(float(component_merge_elapsed), 4),
            "topology_stage_seconds": round(float(topology_stage_elapsed), 4),
            "recovery_graph_seconds": round(float(recovery_graph_elapsed), 4),
            "graph_generation_before_serialization_seconds": round(
                float(time.perf_counter() - graph_generation_start), 4
            ),
        }
        # V9: connectivity must not hide a public region with no actual roads.
        road_raster = np.zeros((H, W), np.uint8)
        component_by_node = {}
        for index, component in enumerate(nx.connected_components(G)):
            for node in component:
                component_by_node[node] = index
        public_components = {int(r): set() for r in corridor_ids}
        for u, v, edata in G.edges(data=True):
            if edata.get('edge_type') == 'recovery_gateway':
                continue
            a, b = tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos'])
            cv2.line(road_raster, a, b, 255, 1)
            length=max(abs(a[0]-b[0]),abs(a[1]-b[1]))+1
            xs=np.rint(np.linspace(a[0],b[0],length)).astype(int)
            ys=np.rint(np.linspace(a[1],b[1],length)).astype(int)
            for rid in np.unique(res_matrix[ys,xs]):
                if int(rid) in public_components:
                    public_components[int(rid)].add(component_by_node[u])
        route_counts=np.bincount(res_matrix[road_raster>0],minlength=int(res_matrix.max())+1)
        coverage=[{'rid':int(r),'road_pixels':int(route_counts[int(r)]),
                   'road_components':sorted(public_components[int(r)]),
                   'has_road':bool(route_counts[int(r)]>=3)} for r in corridor_ids]
        payload['public_region_coverage'] = coverage
        payload['public_regions_without_road'] = [v['rid'] for v in coverage if not v['has_road']]
        payload['unreachable_component_policy'] = unresolved_policy
        (self.output_dir / 'public_graph_coverage_v9.json').write_text(
            json.dumps(coverage, indent=2), encoding='utf-8')

        with open(self.output_dir / "llm_navigation_graph.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

        # -------------------------------------------------
        # 14. Debug 圖
        # 綠：原始/刪減後路線；橘：component bridge；紫：V13 topology cycle repair
        # 黃圈：安全抵達點；紅點：房間中心；紅虛線：語意關聯
        # -------------------------------------------------
        debug_graph_img = cv2.cvtColor((1 - (wall_matrix > 0).astype(np.uint8)) * 255, cv2.COLOR_GRAY2BGR)
        debug_graph_img[valid_routing_mask > 0] = (245, 245, 245)

        edge_color = {
            "route": (0, 150, 0),
            "recovery_route": (255, 180, 0),
            "recovery_gateway": (0, 0, 255),
            "component_bridge": (0, 165, 255),
            "shortcut": (180, 0, 180)
        }
        edge_thickness = {
            "route": 2,
            "recovery_route": 2,
            "recovery_gateway": 3,
            "component_bridge": 3,
            "shortcut": 3,
        }
        for u, v, edata in G.edges(data=True):
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
            et = edata.get("edge_type", "route")
            cv2.line(debug_graph_img, p1, p2, edge_color.get(et, (0, 150, 0)), edge_thickness.get(et, 2))

        for nid in G.nodes():
            pt = tuple(G.nodes[nid]['pos'])
            cv2.rectangle(debug_graph_img, (pt[0] - 3, pt[1] - 3), (pt[0] + 3, pt[1] + 3), (255, 0, 0), -1)

        # 橘色圓點專門標示橋接路徑新生成的轉彎節點。
        for record in bridge_records:
            for p in record.get("new_turn_nodes", []):
                pt = tuple(map(int, p))
                cv2.circle(debug_graph_img, pt, 6, (0, 165, 255), -1)
                cv2.circle(debug_graph_img, pt, 6, (255, 255, 255), 1)

        # 紫色圓點標示 cycle-closure 新生成的轉彎節點。
        for record in shortcut_records:
            for p in record.get("new_turn_nodes", []):
                pt = tuple(map(int, p))
                cv2.circle(debug_graph_img, pt, 6, (180, 0, 180), -1)
                cv2.circle(debug_graph_img, pt, 6, (255, 255, 255), 1)

        def draw_dashed_line(img, p1, p2, color=(0, 0, 255), thickness=1, dash=8, gap=6):
            x1, y1 = p1
            x2, y2 = p2
            length = math.hypot(x2 - x1, y2 - y1)
            if length <= 1:
                return
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            pos = 0.0
            while pos < length:
                a = pos
                b = min(pos + dash, length)
                s = (int(round(x1 + ux * a)), int(round(y1 + uy * a)))
                e = (int(round(x1 + ux * b)), int(round(y1 + uy * b)))
                cv2.line(img, s, e, color, thickness)
                pos += dash + gap

        for rid, center in room_coords.items():
            # 每一個房間都先畫自己的紅色中心節點，不能因 attachment 失敗而消失。
            cv2.circle(debug_graph_img, center, 6, (0, 0, 255), -1)
            cv2.putText(
                debug_graph_img,
                f"R_{rid}",
                (center[0] + 8, center[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (50, 50, 50),
                1
            )
            info = room_attachments.get(rid)
            attach_pt = info.get("attachment_point") if info else None
            if attach_pt is None:
                # 沒有可用道路圖時，以橘色叉號標示「房間存在，但尚未吸附」。
                cv2.line(debug_graph_img, (center[0] - 5, center[1] - 5), (center[0] + 5, center[1] + 5), (0, 165, 255), 2)
                cv2.line(debug_graph_img, (center[0] - 5, center[1] + 5), (center[0] + 5, center[1] - 5), (0, 165, 255), 2)
                continue
            attach_pt = tuple(attach_pt)
            # 青虛線：不穿越其他房間的優先直達投影；紅虛線：找不到後的舊版最近點 fallback。
            relation_color = (
                (255, 170, 0)
                if info.get("direct_connection")
                else (0, 0, 255)
            )
            draw_dashed_line(debug_graph_img, center, attach_pt, relation_color, 1)
            cv2.circle(debug_graph_img, attach_pt, 5, (0, 255, 255), 2)

        safe_imwrite(str(self.output_dir / "debug_navigation_graph.jpg"), debug_graph_img)
        print(
            "[完成] 導航圖已輸出："
            f"connected={payload['graph_connected']}，"
            f"components={payload['road_component_count']}，"
            f"parallel_removed={payload['route_reduction']['removed_parallel_segments']}，"
            f"rooms={payload['room_coverage']['room_center_node_count']}/{payload['room_coverage']['room_count']}，"
            f"attached={payload['room_coverage']['attached_room_count']}，"
            f"bridges={len(bridge_records)}，topology_cycle_repairs={len(shortcut_records)}"
        )


# =========================================
# 執行入口
# =========================================


# =========================================
# Web project integration API
# =========================================
PIPELINE_SCHEMA_VERSION = "2.3.0"


class MapProcessingError(RuntimeError):
    """Raised when a map cannot produce navigation-ready artifacts."""


def _json_safe(value):
    """Convert NumPy values and tuples to ordinary JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _clean_text_list(values):
    cleaned = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _build_aliases(place_id, names, objects):
    aliases = list(names)
    rid = str(place_id)
    aliases.extend([rid, f"{rid}號", f"房間{rid}", f"{rid}號房", f"診間{rid}"])
    for obj in objects:
        label = str(obj).split("(", 1)[0].strip()
        if label:
            aliases.append(label)
    return _clean_text_list(aliases)


def _nearest_waypoint(centroid, graph_nodes):
    cx, cy = centroid
    best_id = None
    best_point = None
    best_distance = float("inf")
    for node_id, node in graph_nodes.items():
        if node.get("type") == "room_center":
            continue
        coords = node.get("coordinates")
        if not isinstance(coords, list) or len(coords) != 2:
            continue
        x, y = int(coords[0]), int(coords[1])
        distance = math.hypot(x - cx, y - cy)
        if distance < best_distance:
            best_id = str(node_id)
            best_point = [x, y]
            best_distance = distance
    return best_id, best_point, best_distance


def build_navigation_data(image_path, res_matrix, metrics_list, id_labels, graph_payload):
    """
    Build the only JSON document consumed by the web backend.

    The document intentionally separates:
    - places / llm_context: semantic information supplied to the LLM;
    - graph: deterministic routing data consumed only by the path planner;
    - map: dimensions and collision-matrix contract consumed by the web client.
    """
    image = safe_imread(image_path)
    if image is None:
        raise MapProcessingError(f"無法重新讀取地圖影像：{image_path}")
    height, width = image.shape[:2]

    graph_payload = graph_payload or {}
    graph_nodes = graph_payload.get("nodes", {})
    attachments = {
        str(item.get("room_node", "")).replace("R_", "", 1): item
        for item in graph_payload.get("room_attachments", [])
        if item.get("room_node")
    }
    metrics_by_id = {str(int(m["id"])): m for m in metrics_list}

    places = {}
    for place_id, metric in metrics_by_id.items():
        labels = id_labels.get(place_id, {})
        names = _clean_text_list(labels.get("names", []))
        objects = _clean_text_list(labels.get("objects", []))
        shapes = _clean_text_list(labels.get("shape", []))
        centroid = [int(round(metric["centroid"][0])), int(round(metric["centroid"][1]))]
        bbox = [int(v) for v in metric.get("bbox", [0, 0, 0, 0])]
        is_corridor = bool(labels.get("portal", False))
        space_type = str(labels.get("space_type") or ("public_circulation" if is_corridor else "room"))
        recovered_public_space = bool(labels.get("recovered_public_space", False))

        attachment = attachments.get(place_id, {})
        attachment_node = attachment.get("attachment_node")
        attachment_point = attachment.get("attachment_point")
        attachment_method = attachment.get("attachment_method")
        public_owner_id = attachment.get("public_owner_id", labels.get("navigation_public_owner"))
        public_owner_kind = attachment.get("public_owner_kind", labels.get("navigation_public_owner_kind"))

        # Corridor/plaza 本身仍可用 nearest waypoint。
        # 但 V9 明確禁止 recovery-owned room 在 attachment 失敗時又被 generic global fallback
        # 偷掛到 primary graph；那會再次破壞「房間只連自己的路網」契約。
        allow_global_fallback = bool(
            is_corridor
            or public_owner_kind not in {"recovered_public"}
        )
        if not attachment_node and allow_global_fallback:
            attachment_node, attachment_point, distance = _nearest_waypoint(centroid, graph_nodes)
            if attachment_node:
                attachment_method = "nearest_waypoint_for_portal" if is_corridor else "nearest_waypoint_fallback"
                attachment = {
                    **attachment,
                    "distance_px": round(float(distance), 2),
                    "attached": True,
                }

        display_name = next((name for name in names if not name.isdigit()), None)
        if not display_name:
            display_name = ("走道" if is_corridor else "房間") + f" {place_id}"

        places[place_id] = {
            "id": place_id,
            "kind": "corridor" if is_corridor else "room",
            "space_type": space_type,
            "recovered_public_space": recovered_public_space,
            "display_name": display_name,
            "names": names,
            "aliases": _build_aliases(place_id, names, objects),
            "objects": objects,
            "shape": shapes,
            "centroid": centroid,
            "bbox": bbox,
            "area_px": int(metric.get("area", 0)),
            "attachment_node": attachment_node,
            "attachment_point": attachment_point,
            "attachment_method": attachment_method,
            "public_owner_id": public_owner_id,
            "public_owner_kind": public_owner_kind,
            "attached": bool(attachment_node and attachment_node in graph_nodes),
        }

    llm_context = [
        {
            "id": place["id"],
            "type": place["kind"],
            "space_type": place.get("space_type", place["kind"]),
            "name": place["display_name"],
            "names": place["names"],
            "aliases": place["aliases"],
            "objects": place["objects"],
            "shape": place["shape"],
        }
        for place in places.values()
    ]

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generator": "0904 room-partition V3 guarded ensemble + FAST V9 owner-locked attachment + V8 rectilinear recovery",
        "map": {
            "image_width": int(width),
            "image_height": int(height),
            "scale": graph_payload.get("map_scale", "1 pixel = 0.05 meters"),
            "collision_matrix_file": "map_matrix.csv",
            "debug_graph_file": "debug_navigation_graph.jpg",
            "collision_rule": "value 1 is blocked; values greater than 1 are traversable semantic regions",
        },
        "places": places,
        "llm_context": llm_context,
        "graph": graph_payload,
        "quality": {
            "graph_connected": bool(graph_payload.get("graph_connected", False)),
            "road_component_count": int(graph_payload.get("road_component_count", 0)),
            "room_coverage": graph_payload.get("room_coverage", {}),
            "place_count": len(places),
            "attached_place_count": sum(1 for place in places.values() if place["attached"]),
        },
    }


@_pipeline_diagnostics
def process_map_pipeline(image_path, output_dir, yolo_model_path=None, k=6, save_csv=True):
    """
    Execute the 0721_4 pipeline while preserving the original web contract.

    Returns a dictionary of paths used by main.py.  Canonical V4 artifacts are
    retained, and compatibility aliases are emitted for the unchanged frontend.
    """
    import shutil

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".fast_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    pipeline_start = time.perf_counter()

    try:
        with image_path.open("rb") as image_file:
            image_size = os.fstat(image_file.fileno()).st_size
    except FileNotFoundError as exc:
        raise MapProcessingError(f"上傳影像不存在，請確認上傳已完成及路徑正確：{image_path}") from exc
    except OSError as exc:
        raise MapProcessingError(f"無法存取上傳影像：{image_path}；{exc}") from exc
    if image_size == 0:
        raise MapProcessingError(f"上傳影像是空檔（0 bytes）：{image_path}")
    if safe_imread(image_path) is None:
        raise MapProcessingError(
            f"影像解碼失敗（已使用 Unicode 路徑讀取）：{image_path}；"
            f"大小={image_size} bytes，請確認檔案未損壞且格式受 OpenCV 支援。"
        )

    yolo_detections = []
    yolo_boxes = []
    model_available = bool(yolo_model_path and Path(yolo_model_path).is_file())
    if model_available:
        with stage_timer("YOLO（單次推論/快取）", timings):
            yolo_detections, yolo_boxes = get_yolo_data(
                str(yolo_model_path), str(image_path), cache_dir=cache_dir,
                conf=float(os.environ.get("MAP_YOLO_CONF", "0.15")),
                imgsz=int(os.environ.get("MAP_YOLO_IMGSZ", "896")),
            )
    else:
        print(f"[警告] 找不到 YOLO 權重：{yolo_model_path}；本次略過圖示辨識，其餘管線仍會執行。")

    with stage_timer("OCR（共用 Reader/快取）", timings):
        ocr_results = get_ocr_data(str(image_path), cache_dir=cache_dir)

    with stage_timer("K-Means 走道分析", timings):
        color_result = analyze_colors_and_corridor(
            str(image_path), ocr_results, k=int(k), return_details=True
        )
        if color_result is None:
            raise MapProcessingError("無法建立走道與背景色彩遮罩。")
        corridor_mask, bg_mask, corridor_color_details = color_result
        if bg_mask is not None:
            safe_imwrite(str(output_dir / "debug_bg_mask.jpg"), bg_mask)

    with stage_timer("牆體提取與證據式修補", timings):
        repaired_wall_matrix = extract_walls_with_repair(
            str(image_path), output_dir, ocr_results,
            bg_mask=bg_mask, yolo_boxes=yolo_boxes,
        )
    if repaired_wall_matrix is None:
        raise MapProcessingError("牆體提取失敗，未產生可用矩陣。")

    structural_wall_matrix = repaired_wall_matrix.copy()
    if bg_mask is not None:
        height, width = repaired_wall_matrix.shape
        if bg_mask.shape != (height, width):
            bg_mask = cv2.resize(bg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        clean_bg_mask, bg_report = _build_blocked_background(
            bg_mask, safe_imread(image_path), ocr_results
        )
        with open(output_dir / "background_void_report_v4.json", "w", encoding="utf-8") as f:
            json.dump(bg_report, f, ensure_ascii=False, indent=2)
        safe_imwrite(str(output_dir / "debug_clean_bg_mask.jpg"), clean_bg_mask)
        repaired_wall_matrix[clean_bg_mask > 0] = 1

    with stage_timer("空間分割與語意標記", timings):
        segmenter = RoomSegmenter(
            output_dir,
            str(yolo_model_path) if model_available else None,
            door_ratio=float(os.environ.get("MAP_DOOR_RATIO", "0.01")),
        )
        res_matrix, metrics_list, id_labels = segmenter.process(
            str(image_path),
            wall_matrix=repaired_wall_matrix,
            corridor_mask=corridor_mask,
            ocr_data=ocr_results,
            yolo_detections=yolo_detections,
            save_csv=bool(save_csv),
            corridor_color_details=corridor_color_details,
            bg_mask=bg_mask,
            structural_wall_matrix=structural_wall_matrix,
        )

    if not metrics_list:
        raise MapProcessingError("空間分割未辨識出有效房間或走道。")
    if not any(v.get("portal") for v in id_labels.values()):
        raise MapProcessingError("未找到可信走道；請檢查 public_partition_v9.json，不能將全房間結果視為導航成功。")

    with stage_timer("導航路網、A* 橋接與捷徑", timings):
        WaypointGraphGenerator(output_dir).generate(
            wall_matrix=repaired_wall_matrix,
            res_matrix=res_matrix,
            metrics_list=metrics_list,
            id_labels=id_labels,
        )

    graph_path = output_dir / "llm_navigation_graph.json"
    if not graph_path.is_file():
        raise MapProcessingError("導航拓樸圖未成功輸出。")
    with open(graph_path, "r", encoding="utf-8") as file:
        graph_payload = json.load(file)

    # New canonical document read by both the deterministic planner and LLM adapter.
    navigation_data = build_navigation_data(
        str(image_path), res_matrix, metrics_list, id_labels, graph_payload
    )
    recovery_report = getattr(segmenter, "last_public_recovery_report", {}) or {}
    classifier_report = getattr(segmenter, "last_public_classifier_report", {}) or {}
    navigation_data.setdefault("quality", {})["recovered_public_space_count"] = int(len(recovery_report.get("accepted_ids", [])))
    navigation_data["quality"]["public_space_recovery_seconds"] = float(recovery_report.get("elapsed_seconds", 0.0) or 0.0)
    navigation_data["quality"]["public_space_recovery_timed_out"] = bool(recovery_report.get("timed_out", False))
    navigation_data["quality"]["public_classifier_mode"] = classifier_report.get("classifier_mode", "auto")
    navigation_data["quality"]["independent_regions_merged"] = bool(classifier_report.get("independent_regions_merged", False))
    navigation_data["quality"]["same_colour_regions_preserved"] = int(classifier_report.get("same_colour_regions_preserved", 0) or 0)
    navigation_data["quality"]["graph_performance"] = graph_payload.get("performance", {})
    navigation_data["quality"]["public_regions_without_road"] = graph_payload.get("public_regions_without_road", [])
    navigation_data["quality"]["fully_connected"] = bool(graph_payload.get("graph_connected", False))
    navigation_path = output_dir / "navigation_data.json"
    with open(navigation_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(navigation_data), file, ensure_ascii=False, indent=2)

    # Preserve the unchanged frontend and any old code that still requests these names.
    matrix_path = output_dir / "map_matrix.csv"
    canonical_matrix_path = output_dir / "_0721_4.csv"
    if canonical_matrix_path.is_file():
        # 內容完全相同時直接複製相容檔，不再把百萬級矩陣序列化第二次。
        shutil.copyfile(canonical_matrix_path, matrix_path)
    else:
        np.savetxt(matrix_path, res_matrix, fmt="%d", delimiter=",")
    legacy_room_path = output_dir / "room_data.json"
    with open(legacy_room_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(id_labels), file, ensure_ascii=False, indent=2)

    timings["總計"] = time.perf_counter() - pipeline_start
    profile_path = output_dir / "runtime_profile_fast.json"
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump({key: round(value, 3) for key, value in timings.items()}, file, ensure_ascii=False, indent=2)

    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "ready",
        "source_image": image_path.name,
        "model": {
            "yolo_path": str(yolo_model_path) if yolo_model_path else None,
            "yolo_loaded": model_available,
        },
        "canonical_files": {
            "navigation_data": navigation_path.name,
            "navigation_graph": graph_path.name,
            "room_semantics": "room_data_0721_4.json",
            "region_matrix": "_0721_4.csv",
            "room_partition_report": "room_partition_report_v2.json",
            "room_partition_debug": "debug_room_partition_evidence_v2.jpg",
            "room_separator_mask": "debug_room_separator_mask_v2.jpg",
            "public_classifier_report": "public_partition_v9.json",
            "map_interior_report": "map_interior_report_v2.json",
            "map_interior_debug": "debug_map_interior_mask_v2.jpg",
            "nested_room_merge_report": "nested_room_merge_report_v2.json",
            "public_space_recovery": "public_space_recovery_report_fast_v1.json",
            "public_space_recovery_debug": "debug_public_space_recovery.jpg",
            "semantic_public_gateway_report": "semantic_public_gateway_report.json",
            "semantic_public_gateway_debug": "debug_semantic_public_gateways.jpg",
            "recovery_local_graph_debug": "debug_recovery_local_graph_v8.jpg",
            "room_public_owner_report": "room_public_owner_report_v9.json",
            "room_public_owner_debug": "debug_room_public_owner_v9.jpg",
            "room_attachment_owner_validation": "room_attachment_owner_validation_v9.json",
        },
        "compatibility_files": {
            "collision_matrix": matrix_path.name,
            "legacy_room_data": legacy_room_path.name,
        },
        "timings_seconds": {key: round(value, 3) for key, value in timings.items()},
        "public_space_recovery": {
            "enabled": bool(recovery_report.get("enabled", False)),
            "accepted_ids": recovery_report.get("accepted_ids", []),
            "elapsed_seconds": recovery_report.get("elapsed_seconds", 0.0),
            "time_budget_seconds": recovery_report.get("time_budget_seconds", 0.0),
            "timed_out": bool(recovery_report.get("timed_out", False)),
        },
        "public_classifier": classifier_report,
    }
    manifest_path = output_dir / "map_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return {
        "navigation_path": str(navigation_path),
        "graph_path": str(graph_path),
        "csv_path": str(matrix_path),
        "legacy_room_path": str(legacy_room_path),
        "manifest_path": str(manifest_path),
        "debug_graph_path": str(output_dir / "debug_navigation_graph.jpg"),
        "debug_room_partition_path": str(output_dir / "debug_room_partition_evidence_v2.jpg"),
        "debug_map_interior_path": str(output_dir / "debug_map_interior_mask_v2.jpg"),
        "room_partition_report_path": str(output_dir / "room_partition_report_v2.json"),
        "nested_room_merge_report_path": str(output_dir / "nested_room_merge_report_v2.json"),
        "debug_public_space_path": str(output_dir / "debug_public_space_recovery.jpg"),
        "debug_semantic_gateway_path": str(output_dir / "debug_semantic_public_gateways.jpg"),
        "debug_recovery_graph_path": str(output_dir / "debug_recovery_local_graph_v8.jpg"),
        "debug_room_public_owner_path": str(output_dir / "debug_room_public_owner_v9.jpg"),
        "public_space_report_path": str(output_dir / "public_space_recovery_report_fast_v1.json"),
        "room_public_owner_report_path": str(output_dir / "room_public_owner_report_v9.json"),
        "room_attachment_owner_validation_path": str(output_dir / "room_attachment_owner_validation_v9.json"),
        "semantic_gateway_report_path": str(output_dir / "semantic_public_gateway_report.json"),
        "quality": navigation_data["quality"],
        "timings": timings,
    }
