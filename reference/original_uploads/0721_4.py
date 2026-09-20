import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
import easyocr
import json
import torch
import gc
import os
import math
import time
import hashlib
from contextlib import contextmanager
from collections import Counter
from ultralytics import YOLO
import networkx as nx  # 🌟 新增：處理圖論與航點連線的套件

# 強制優化 PyTorch 記憶體碎片管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================================
# FAST V1：共用快取、單次模型推論與階段計時
# =========================================
_EASYOCR_READER_CACHE = {}
CACHE_VERSION = "0721_4_fast_v1"


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
            CACHE_VERSION,
            "yolo",
            _path_signature(image_path),
            _path_signature(model_path),
            conf,
            imgsz,
        )
        cache_file = Path(cache_dir) / f"yolo_{key}.json"
        cached = _read_json_cache(cache_file)
        if cached is not None and isinstance(cached.get("detections"), list):
            detections = cached["detections"]
            boxes = [tuple(map(int, d["box"])) for d in detections]
            print(f"[快取] YOLO：載入 {len(detections)} 筆偵測。")
            return detections, boxes

    model = YOLO(model_path)
    result = model.predict(source=image_path, conf=conf, imgsz=imgsz, verbose=False)[0]
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
    return cv2.imread(str(image_path), flags)

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
    use_gpu = torch.cuda.is_available()
    if canvas_size is None:
        canvas_size = 2048 if use_gpu else 1792

    cache_file = None
    if cache_dir is not None:
        key = _cache_key(CACHE_VERSION, "ocr", _path_signature(image_path), use_gpu, canvas_size, "ch_tra,en")
        cache_file = Path(cache_dir) / f"ocr_{key}.json"
        cached = _read_json_cache(cache_file)
        if cached is not None and isinstance(cached.get("items"), list):
            print(f"[快取] OCR：載入 {len(cached['items'])} 筆文字。")
            return cached["items"]

    print("[系統] 正在執行 OCR 文字辨識 + CRAFT polygon 偵測...")
    reader_key = (("ch_tra", "en"), bool(use_gpu))
    reader = _EASYOCR_READER_CACHE.get(reader_key)
    if reader is None:
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


def analyze_colors_and_corridor(image_path, ocr_data, k=8, max_dim=1200, return_details=False):
    """
    FAST V4：保留原版 K-Means 品質，但不再把 OCR 中心點當成走道色的最終依據。

    OCR bbox 中心經常剛好落在白色字、黑色字或字元空隙，而不是店面/走道底色；
    因此舊版「哪個色群上的 OCR 中心較少，就選哪個」在這張地圖會在深紫店面色
    與淡紫走道色之間翻轉。此函式仍產生 legacy 暫定遮罩以維持相容性，但同時回傳
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

    margin = 10
    if H > margin * 2 and W > margin * 2:
        border_pixels = np.concatenate([
            labels_2d[margin, margin:W-margin],
            labels_2d[H-1-margin, margin:W-margin],
            labels_2d[margin:H-margin, margin],
            labels_2d[margin:H-margin, W-1-margin],
        ])
    else:
        border_pixels = np.concatenate([
            labels_2d[0, :], labels_2d[-1, :],
            labels_2d[:, 0], labels_2d[:, -1],
        ])

    border_counts = np.bincount(border_pixels.astype(np.int32), minlength=int(k))
    total_border_pixels = max(1, int(border_pixels.size))
    bg_ids = [
        i for i, count in enumerate(border_counts)
        if count / total_border_pixels > 0.1
    ]

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
    """回傳 0/255 skeleton；優先用 OpenCV ximgproc，沒有就用形態學 fallback。"""
    src = ((mask > 0).astype(np.uint8) * 255)
    try:
        return cv2.ximgproc.thinning(src)
    except AttributeError:
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
        return skeleton


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
    for line in lines[:max_lines, 0, :]:
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
        for line in lines[:, 0, :]:
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
    dilated_region = cv2.dilate(region_mask, np.ones((5, 5), np.uint8))
    for vx, vy, angle_penalty in candidate_dirs:
        for r in range(6, max_len, 2):
            x = int(round(pt[0] + vx * r))
            y = int(round(pt[1] + vy * r))
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            if walls[y, x] > 0 and region_mask[y, x] == 0:
                ratio = _line_region_ratio(dilated_region, pt, (x, y), thickness=1)
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


def extract_walls_with_repair(image_path, output_dir, ocr_data, bg_mask=None, yolo_boxes=None):
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
    cv2.imwrite(str(output_dir / "debug_pre_wall_prior_0721_4.jpg"), pre_wall_prior)

    print("[系統] 建立 V5 精確文字筆畫遮罩：polygon 只作 ROI，不再整塊挖掉...")
    text_poly_mask, text_stroke_mask, text_repair_seed, text_preserve_mask = _build_precise_text_stroke_masks(ocr_data, (H, W), raw_wall_binary)
    cv2.imwrite(str(output_dir / "debug_text_polygon_mask_0721_4.jpg"), text_poly_mask)
    cv2.imwrite(str(output_dir / "debug_text_precise_strokes_0721_4.jpg"), text_stroke_mask)
    cv2.imwrite(str(output_dir / "debug_text_preserved_wall_like_0721_4.jpg"), text_preserve_mask)
    cv2.imwrite(str(output_dir / "debug_text_repair_seed_0721_4.jpg"), text_repair_seed)

    print("[系統] 執行雜訊抹除 (precise OCR strokes / precise YOLO strokes / 幾何複雜度分析)...")
    noise_boxes = []
    repair_seed_mask = text_repair_seed.copy()

    # V5：只刪 OCR polygon 內實際的文字前景筆畫；牆線-like component 會保留。
    binary[text_stroke_mask > 0] = 0

    if yolo_boxes:
        yolo_roi_mask, yolo_stroke_mask, yolo_repair_seed, yolo_preserve_mask = _build_precise_box_stroke_masks(yolo_boxes, (H, W), raw_wall_binary, pad=2)
        cv2.imwrite(str(output_dir / "debug_yolo_roi_mask_0721_4.jpg"), yolo_roi_mask)
        cv2.imwrite(str(output_dir / "debug_yolo_precise_strokes_0721_4.jpg"), yolo_stroke_mask)
        cv2.imwrite(str(output_dir / "debug_yolo_preserved_wall_like_0721_4.jpg"), yolo_preserve_mask)
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
    cv2.imwrite(str(output_dir / "debug_walls_before_repair_0721_4.jpg"), final_walls)

    wall_width_map = _estimate_wall_width_map(final_walls)
    debug_repair_img = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)
    debug_nodes_img = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)

    # V5 repair region：只來自「精確刪除筆畫且靠近牆線」的小 seed，不再由整個 OCR/YOLO 框觸發。
    repair_region_mask = cv2.morphologyEx(repair_seed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imwrite(str(output_dir / "debug_repair_regions_0721_4.jpg"), repair_region_mask)

    num_regions, region_labels, region_stats, _ = cv2.connectedComponentsWithStats((repair_region_mask > 0).astype(np.uint8), connectivity=8)
    repair_report = []

    print("[系統] 以 evidence-gated two-stage repair 修補牆體...")

    def run_repair_stage(stage_name, walls_in, relaxed=False, allow_prior_restore=False):
        walls_out = walls_in.copy()
        stage_report = []
        stage_width_map = _estimate_wall_width_map(walls_out)
        stage_skeleton = _skeletonize_uint8(walls_out)
        stage_degree = _skeleton_degree(stage_skeleton)

        for ridx in range(1, num_regions):
            area = int(region_stats[ridx, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            if area > H * W * 0.08:
                stage_report.append({'stage': stage_name, 'region': ridx, 'accepted': False, 'reason': 'region_too_large', 'area': area})
                continue

            x, y, rw, rh = map(int, region_stats[ridx, :4])
            region_bbox = (x, y, rw, rh)
            region_mask = np.zeros_like(repair_region_mask)
            local_labels = region_labels[y:y+rh, x:x+rw]
            region_mask[y:y+rh, x:x+rw] = (local_labels == ridx).astype(np.uint8) * 255
            eligible, evidence = _region_wall_damage_evidence(
                region_mask, pre_wall_prior, walls_out, stage_width_map,
                stage=stage_name, region_bbox=region_bbox
            )
            if not eligible:
                stage_report.append({'stage': stage_name, 'region': ridx, 'accepted': False, 'area': area, 'reason': evidence.get('reason'), 'evidence': evidence})
                continue

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_repair_img, contours, -1, (0, 255, 255) if stage_name == 'outer' else (255, 255, 0), 1)

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
                    if _line_region_ratio(cv2.dilate(region_mask, np.ones((5, 5), np.uint8)), ep['pt'], jn['pt']) < (0.08 if relaxed else 0.14):
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
                # 只有真正接受修補後才重算 skeleton；被 evidence/topology 拒絕的區域共用同一份結果。
                stage_skeleton = _skeletonize_uint8(walls_out)
                stage_degree = _skeleton_degree(stage_skeleton)
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

    cv2.imwrite(str(output_dir / "debug_cleaned_walls_0721_4.jpg"), final_walls)
    cv2.imwrite(str(output_dir / "debug_repair_boxes_0721_4.jpg"), debug_repair_img)
    cv2.imwrite(str(output_dir / "debug_wall_skeleton_nodes_0721_4.jpg"), debug_nodes_img)
    with open(output_dir / "wall_repair_report_0721_4.json", 'w', encoding='utf-8') as f:
        json.dump(repair_report, f, ensure_ascii=False, indent=4)

    return (final_walls / 255).astype(np.uint8)

# =========================================
# 模組 C：空間分割與走道縫合 (🌟回傳值更新，輸出房間資料供後續導航網格使用)
# =========================================
class RoomSegmenter:
    def __init__(self, output_dir, yolo_model_path=None, area_ratio=1/8000, door_ratio=0.002):
        self.output_dir = Path(output_dir)
        self.area_ratio = area_ratio
        self.door_ratio = door_ratio
        self.yolo_model_path = yolo_model_path
        self._yolo_model = None

    def _fallback_yolo_detections(self, original_img_path):
        if not self.yolo_model_path:
            return []
        if self._yolo_model is None:
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

    def process(self, original_img_path, wall_matrix, corridor_mask, ocr_data, yolo_detections=None, save_csv=True, corridor_color_details=None):
        h, w = wall_matrix.shape[:2]
        min_area = int((h * w) * self.area_ratio)
        door_size = max(1, int(np.sqrt(h**2 + w**2) * self.door_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (door_size, door_size))
        closed = cv2.morphologyEx(wall_matrix, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((1 - closed).astype(np.uint8), connectivity=8)

        res_matrix = np.ones((h, w), dtype=np.int32)
        metrics_list = []
        current_id = 2
        for i in range(1, num_labels):
            lx, ly, sw, sh, area = map(int, stats[i])
            if lx <= 2 or ly <= 2 or lx + sw >= w - 2 or ly + sh >= h - 2 or area < min_area:
                continue
            local_component = labels[ly:ly+sh, lx:lx+sw] == i
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
            })
            target = res_matrix[ly:ly+sh, lx:lx+sw]
            target[local_component & (wall_matrix[ly:ly+sh, lx:lx+sw] == 0)] = current_id
            current_id += 1

        if not metrics_list:
            return res_matrix, [], {}

        # -------------------------------------------------
        # 走道辨識與碎片縫合 FAST V4 — 色彩候選由牆體拓樸裁決
        # -------------------------------------------------
        # 核心修正：
        # 1) 不再相信 OCR 中心點能代表底色；它常落在字元本身，會把深紫店面色選成走道。
        # 2) 真正走道通常是 closing 後最大的內部 free-space component。
        # 3) 選擇「最完整覆蓋最大 free-space component」的 K-Means 色群。
        # 4) 走道碎片除色彩重疊外，還必須靠近主走道且具有足夠面積/延展性。
        # 5) 不再先把大量 component 合併後，再用「只留最大兩塊」刪掉房間。
        corridor_rids = []
        main_cid = None
        corridor_debug = []
        color_debug = []
        selected_corridor_mask = corridor_mask
        selected_color_id = None

        metric_by_id = {int(m['id']): m for m in metrics_list}
        largest_metric = max(metrics_list, key=lambda m: int(m['area']))
        largest_rid = int(largest_metric['id'])
        largest_area = max(1, int(largest_metric['area']))

        # A. 使用牆體分割結果，在所有非背景 K-Means 色群中選真正走道色。
        details = corridor_color_details or {}
        labels_color = details.get('labels_2d')
        candidate_ids = [int(v) for v in details.get('candidate_ids', [])]
        if isinstance(labels_color, np.ndarray) and labels_color.shape == (h, w) and candidate_ids:
            top_metrics = sorted(metrics_list, key=lambda m: int(m['area']), reverse=True)[:3]
            top_area_sum = max(1, sum(int(m['area']) for m in top_metrics))
            best_record = None
            best_mask = None

            for color_id in candidate_ids:
                candidate_mask = _build_corridor_cluster_mask(labels_color, color_id)
                max_label = max(current_id + 1, int(res_matrix.max()) + 1)
                overlap_counts = np.bincount(
                    res_matrix[candidate_mask > 0].ravel(), minlength=max_label
                ).astype(np.int64)

                seed_overlap = float(overlap_counts[largest_rid]) / float(largest_area)
                top_weighted_overlap = float(sum(
                    int(overlap_counts[int(m['id'])]) for m in top_metrics
                )) / float(top_area_sum)

                # 若同一色群會把大量小房間判成走道，降低分數。
                false_capture_area = 0
                overlap_gt_045_count = 0
                for m in metrics_list:
                    rid = int(m['id'])
                    area = max(1, int(m['area']))
                    ratio = float(overlap_counts[rid]) / float(area)
                    if ratio > 0.45:
                        overlap_gt_045_count += 1
                        if rid != largest_rid:
                            false_capture_area += area
                false_capture_ratio = min(1.0, false_capture_area / float(largest_area))

                score = (
                    0.82 * seed_overlap
                    + 0.18 * top_weighted_overlap
                    - 0.12 * false_capture_ratio
                )
                center = details.get('centers')
                center_bgr = None
                if isinstance(center, np.ndarray) and 0 <= color_id < len(center):
                    center_bgr = [round(float(v), 2) for v in center[color_id].tolist()]
                record = {
                    "color_id": int(color_id),
                    "center_bgr": center_bgr,
                    "score": round(float(score), 6),
                    "largest_component_overlap": round(seed_overlap, 6),
                    "top3_area_weighted_overlap": round(top_weighted_overlap, 6),
                    "false_capture_area_ratio": round(false_capture_ratio, 6),
                    "components_over_0_45": int(overlap_gt_045_count),
                }
                color_debug.append(record)
                if best_record is None or score > best_record['score_raw']:
                    best_record = {**record, "score_raw": float(score)}
                    best_mask = candidate_mask

            # 必須對最大 free-space 有足夠覆蓋，否則保留 legacy 暫定遮罩。
            if best_record is not None and best_record['largest_component_overlap'] >= 0.20:
                selected_color_id = int(best_record['color_id'])
                selected_corridor_mask = best_mask
                print(
                    f"[系統] 拓樸判定走道色群={selected_color_id}，"
                    f"最大自由空間覆蓋率={best_record['largest_component_overlap']:.3f}"
                )
            else:
                print("[警告] K-Means 色群均未充分覆蓋主自由空間，改用 legacy 暫定遮罩。")

        if selected_corridor_mask is not None:
            cv2.imwrite(
                str(self.output_dir / "debug_corridor_mask_topology_v4.jpg"),
                selected_corridor_mask
            )

            max_label = max(current_id + 1, int(res_matrix.max()) + 1)
            component_counts = np.bincount(
                res_matrix.ravel(), minlength=max_label
            ).astype(np.int64)
            overlap_counts = np.bincount(
                res_matrix[selected_corridor_mask > 0].ravel(), minlength=max_label
            ).astype(np.int64)

            # B. 主走道不是「重疊率最高的小房間」，而是走道色實際覆蓋像素最多的 component。
            main_metric = max(
                metrics_list,
                key=lambda m: int(overlap_counts[int(m['id'])])
            )
            main_candidate_id = int(main_metric['id'])
            main_candidate_area = max(1, int(main_metric['area']))
            main_overlap_pixels = int(overlap_counts[main_candidate_id])
            main_overlap_ratio = main_overlap_pixels / float(main_candidate_area)

            if main_overlap_ratio >= 0.25:
                main_cid = main_candidate_id
                main_area = main_candidate_area
                main_mask = (res_matrix == main_cid).astype(np.uint8)
                distance_to_main = cv2.distanceTransform(
                    (1 - main_mask).astype(np.uint8), cv2.DIST_L2, 3
                )
                bridge_radius = float(max(18, int(round(door_size * 2.5))))

                for m in metrics_list:
                    rid = int(m['id'])
                    area = max(1, int(m['area']))
                    lx, ly, sw, sh = map(int, m.get('bbox', [0, 0, w, h]))
                    overlap_pixels = int(overlap_counts[rid])
                    overlap_ratio = overlap_pixels / float(area)
                    component_mask = (res_matrix == rid)
                    min_distance = float(distance_to_main[component_mask].min()) if np.any(component_mask) else float('inf')
                    area_ratio = area / float(main_area)
                    aspect = max(sw, sh) / float(max(1, min(sw, sh)))
                    span = max(sw / float(max(1, w)), sh / float(max(1, h)))

                    near_main = min_distance <= bridge_radius
                    structurally_significant = (
                        area_ratio >= 0.03
                        or (area_ratio >= 0.012 and aspect >= 1.6 and span >= 0.10)
                    )
                    accepted = (
                        rid == main_cid
                        or (
                            overlap_ratio > 0.45
                            and near_main
                            and structurally_significant
                        )
                    )

                    if rid == main_cid:
                        reason = "main_component_max_color_intersection"
                    elif overlap_ratio <= 0.45:
                        reason = "low_color_overlap"
                    elif not near_main:
                        reason = "remote_same_color_object"
                    elif not structurally_significant:
                        reason = "small_icon_or_room_protected"
                    else:
                        reason = "nearby_significant_corridor_fragment"

                    corridor_debug.append({
                        "rid": rid,
                        "area": area,
                        "component_pixels": int(component_counts[rid]),
                        "overlap_pixels": overlap_pixels,
                        "overlap_ratio": round(float(overlap_ratio), 6),
                        "distance_to_main_px": round(float(min_distance), 3),
                        "area_ratio_to_main": round(float(area_ratio), 6),
                        "aspect": round(float(aspect), 4),
                        "map_span": round(float(span), 6),
                        "accepted_as_corridor": bool(accepted),
                        "reason": reason,
                    })
                    if accepted:
                        corridor_rids.append(rid)

                corridor_rids = sorted(set(corridor_rids))

                # C. 僅合併通過拓樸 gate 的走道 component。
                lut = np.arange(
                    max(int(res_matrix.max()) + 1, current_id + 1),
                    dtype=np.int32
                )
                for rid in corridor_rids:
                    lut[int(rid)] = main_cid
                res_matrix = lut[res_matrix]

                # D. unknown 只在「走道色且與已接受走道相連」時補回，避免全圖灌色。
                candidate_free = (
                    (selected_corridor_mask > 0)
                    & (wall_matrix == 0)
                ).astype(np.uint8)
                cc_count, cc_labels, _, _ = cv2.connectedComponentsWithStats(
                    candidate_free, connectivity=8
                )
                accepted_seed = np.isin(
                    res_matrix,
                    np.asarray([main_cid], dtype=np.int32)
                ).astype(np.uint8)
                seed_radius = max(5, int(round(door_size * 0.5)))
                seed_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (seed_radius * 2 + 1, seed_radius * 2 + 1)
                )
                accepted_seed = cv2.dilate(accepted_seed, seed_kernel)
                keep_cc = np.zeros(cc_count, dtype=bool)
                for cc_id in range(1, cc_count):
                    keep_cc[cc_id] = bool(np.any(accepted_seed[cc_labels == cc_id] > 0))
                connected_color_fill = keep_cc[cc_labels]
                fill_mask = (
                    (res_matrix == 1)
                    & connected_color_fill
                    & (wall_matrix == 0)
                )
                res_matrix[fill_mask] = main_cid

                # 不再使用「相同 ID 只留最大兩個連通塊」；該步驟正是房間大量消失的原因。
                new_metrics = []
                for m in metrics_list:
                    rid = int(m['id'])
                    if rid == main_cid:
                        m['area'] = int(np.sum(res_matrix == main_cid))
                        new_metrics.append(m)
                    elif rid not in corridor_rids:
                        new_metrics.append(m)
                metrics_list = new_metrics
            else:
                print(
                    f"[警告] 最佳 component 的走道色覆蓋僅 {main_overlap_ratio:.3f}，"
                    "取消走道合併以保護房間。"
                )

        with open(
            self.output_dir / "corridor_topology_report_fast_v4.json",
            "w", encoding="utf-8"
        ) as f:
            json.dump({
                "mode": "topology_guided_color_selection",
                "legacy_corridor_color_id": details.get('legacy_corridor_id'),
                "selected_corridor_color_id": selected_color_id,
                "largest_free_space_id": largest_rid,
                "largest_free_space_area": largest_area,
                "main_corridor_id": main_cid,
                "corridor_ids": [int(v) for v in corridor_rids],
                "color_candidates": [
                    {k: v for k, v in rec.items() if k != 'score_raw'}
                    for rec in color_debug
                ],
                "components": corridor_debug,
            }, f, ensure_ascii=False, indent=2)

        adj_map = {int(m['id']): set() for m in metrics_list}
        dilation_kernel = np.ones((12, 12), np.uint8)
        for m in metrics_list:
            rid = int(m['id'])
            lx, ly, sw, sh = map(int, m.get('bbox', [0, 0, w, h]))
            pad = 7
            x1, y1, x2, y2 = _clip_box(lx - pad, ly - pad, lx + sw + pad, ly + sh + pad, w, h)
            local_ids = res_matrix[y1:y2, x1:x2]
            local_room = (local_ids == rid).astype(np.uint8)
            dilated = cv2.dilate(local_room, dilation_kernel)
            for n_id in np.unique(local_ids[dilated > 0]):
                n_id = int(n_id)
                if n_id > 1 and n_id != rid and n_id in adj_map:
                    adj_map[rid].add(n_id)
                    adj_map[n_id].add(rid)

        id_labels = {str(m['id']): {"names": [], "objects": [], "portal": False, "shape": []} for m in metrics_list}
        virtual_room_id = current_id
        id_labels[str(virtual_room_id)] = {"names": [], "objects": [], "portal": False, "shape": []}
        for rid_int, neighbors in adj_map.items():
            if len(neighbors) >= 5 and str(rid_int) in id_labels:
                id_labels[str(rid_int)]["portal"] = True
        if corridor_rids and str(main_cid) in id_labels:
            id_labels[str(main_cid)]["portal"] = True

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
            color = (200, 200, 200) if id_labels[str(rid)]["portal"] else random.choice(random_colors)
            vis_img[res_matrix == rid] = color
            cv2.putText(vis_img, str(rid), (int(m['centroid'][0] - 15), int(m['centroid'][1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        metric_centers = np.array([[m['centroid'][0], m['centroid'][1]] for m in metrics_list], dtype=np.float32)
        metric_ids = [int(m['id']) for m in metrics_list]
        def nearest_metric_id(x, y):
            d2 = np.sum((metric_centers - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
            return metric_ids[int(np.argmin(d2))]

        for item in ocr_data:
            tx, ty = map(int, item['center'])
            if 0 <= ty < h and 0 <= tx < w:
                target_id = int(res_matrix[ty, tx])
                if target_id == 1:
                    target_id = nearest_metric_id(tx, ty)
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
                target_id = nearest_metric_id(ox, oy)
            if str(target_id) in id_labels:
                id_labels[str(target_id)]["objects"].append(f"{label}({conf:.2f})")

        if save_csv:
            pd.DataFrame(res_matrix).to_csv(self.output_dir / "_0721_4.csv", index=False, header=False)
        cv2.imwrite(str(self.output_dir / "debug_0721_4.jpg"), vis_img)
        with open(self.output_dir / "room_data_0721_4.json", 'w', encoding='utf-8') as f:
            json.dump(id_labels, f, ensure_ascii=False, indent=4)
        print("[完成] 完美融合版 JSON 已生成。")
        return res_matrix, metrics_list, id_labels

# =========================================
# 🌟 模組 D：航點導航圖生成器 (捷運路網與十字射線正交版)
# =========================================
# 已移除：舊版 WaypointGraphGenerator（原本會被下方 V8 完全覆蓋）

        # =========================================
# 🌟 模組 D：航點導航圖生成器 V7
#    1) 房間中心只作語意節點；黃色抵達點必須位於安全走道線上
#    2) 平行路線只在「同一片無牆走道」內合併，代表線改選牆距最大的中心線
#    3) 不連通 component 使用走道遮罩 A* 接合，不再畫穿越空間的直線
#    4) 檢查明顯繞路的節點對，自動加入安全捷徑
# =========================================
class WaypointGraphGenerator:
    def __init__(self, output_dir, scale="1 pixel = 0.05 meters"):
        self.output_dir = Path(output_dir)
        self.scale = scale

    def generate(self, wall_matrix, res_matrix, metrics_list, id_labels):
        print("[系統] 正在生成 LLM 專用導航拓樸圖 V8（拓樸保護式刪減 + 全房間節點 + 安全吸附 + A* 橋接/捷徑）...")
        if res_matrix is None or not metrics_list:
            print("[警告] 缺少空間分配矩陣，無法生成導航圖。")
            return

        # -------------------------------------------------
        # 1. 導航遮罩、牆壁碰撞遮罩與牆距
        # -------------------------------------------------
        corridor_ids = [int(rid) for rid, data in id_labels.items() if data.get("portal", False)]
        pure_corridor_mask = np.zeros_like(res_matrix, dtype=np.uint8)
        for cid in corridor_ids:
            pure_corridor_mask[res_matrix == cid] = 255

        wall_uint8 = ((wall_matrix > 0).astype(np.uint8) * 255)
        wall_collision = cv2.dilate(wall_uint8, np.ones((9, 9), np.uint8), iterations=1)
        corridor_expansion = cv2.dilate(pure_corridor_mask, np.ones((25, 25), np.uint8), iterations=1)
        valid_routing_mask = cv2.bitwise_and(corridor_expansion, cv2.bitwise_not(wall_collision))
        H, W = valid_routing_mask.shape

        wall_free = cv2.bitwise_not(wall_collision)
        wall_distance = cv2.distanceTransform(wall_free, cv2.DIST_L2, 5)
        min_attachment_clearance = 2.0
        safe_attachment_mask = ((valid_routing_mask > 0) & (wall_distance >= min_attachment_clearance)).astype(np.uint8) * 255

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

        # 先去除非常接近的候選點
        corner_pts = _dedupe_points_by_radius(corner_pts, radius=12)
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

        snap_x = snap_vals([p[0] for p in corner_pts])
        snap_y = snap_vals([p[1] for p in corner_pts])
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
        min_raw_len = 18
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

        def nearest_safe_point_on_routes(rx, ry):
            best_pt, best_dist = None, float('inf')

            def consider(pt):
                nonlocal best_pt, best_dist
                x, y = int(pt[0]), int(pt[1])
                if not point_is_safe((x, y), clearance=min_attachment_clearance, mask=safe_attachment_mask):
                    return
                d = math.hypot(rx - x, ry - y)
                if d < best_dist:
                    best_dist, best_pt = d, (x, y)

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

            return best_pt, best_dist

        for rid, center in room_coords.items():
            attach_pt, attach_dist = nearest_safe_point_on_routes(*center)
            # V8：每一個房間都先建立 attachment 記錄；找不到安全投影時不再直接丟棄房間。
            info = {
                "room_center": center,
                "attachment_point": attach_pt,
                "attachment_node": None,
                "distance_px": round(float(attach_dist), 2) if attach_pt is not None else None,
                "safe_clearance_px": (
                    round(float(wall_distance[attach_pt[1], attach_pt[0]]), 2)
                    if attach_pt is not None else None
                ),
                "attachment_method": "safe_route_projection" if attach_pt is not None else "pending_graph_fallback"
            }
            room_attachments[rid] = info
            if attach_pt is not None:
                grid_nodes.add(attach_pt)

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

        def add_graph_edge(u, v, edge_type="route", mask_ratio=None):
            if u == v:
                return
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
            dist = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            attrs = {"edge_type": edge_type, "weight": dist}
            if mask_ratio is not None:
                attrs["mask_ratio"] = round(float(mask_ratio), 3)
            if G.has_edge(u, v):
                old = G.edges[u, v]
                # 保留較高語意等級的邊。
                priority = {"route": 0, "shortcut": 1, "component_bridge": 2}
                if priority.get(edge_type, 0) > priority.get(old.get("edge_type", "route"), 0):
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

        def attach_room_to_existing_graph(center):
            if G.number_of_edges() == 0:
                return None
            best = None
            # 優先使用原始 route；若沒有才容許人工邊。
            edge_priority = {"route": 0, "shortcut": 1, "component_bridge": 2}
            for u, v, edata in list(G.edges(data=True)):
                p1 = tuple(G.nodes[u]['pos'])
                p2 = tuple(G.nodes[v]['pos'])
                proj = project_point_to_segment(center, p1, p2)
                if not point_is_safe(proj, clearance=min_attachment_clearance, mask=safe_attachment_mask):
                    continue
                d = math.hypot(center[0] - proj[0], center[1] - proj[1])
                et = edata.get("edge_type", "route")
                score = d + edge_priority.get(et, 3) * 1000.0
                if best is None or score < best[0]:
                    best = (score, d, u, v, proj, dict(edata))
            if best is None:
                safe_nodes = [
                    n for n in G.nodes()
                    if point_is_safe(G.nodes[n]['pos'], clearance=min_attachment_clearance, mask=safe_attachment_mask)
                ]
                if not safe_nodes:
                    safe_nodes = list(G.nodes())
                if not safe_nodes:
                    return None
                nid = min(
                    safe_nodes,
                    key=lambda n: math.hypot(
                        G.nodes[n]['pos'][0] - center[0],
                        G.nodes[n]['pos'][1] - center[1]
                    )
                )
                pt = tuple(G.nodes[nid]['pos'])
                return pt, nid, math.hypot(pt[0] - center[0], pt[1] - center[1]), "nearest_graph_node"

            _, d, u, v, proj, edata = best
            if proj == tuple(G.nodes[u]['pos']):
                return proj, u, d, "graph_edge_endpoint"
            if proj == tuple(G.nodes[v]['pos']):
                return proj, v, d, "graph_edge_endpoint"

            old_type = edata.get("edge_type", "route")
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            aid = add_waypoint(proj, node_type="attachment_waypoint")
            add_graph_edge(u, aid, edge_type=old_type)
            add_graph_edge(aid, v, edge_type=old_type)
            return proj, aid, d, "graph_edge_projection"

        for rid, info in room_attachments.items():
            pt = info.get("attachment_point")
            nid = wp_node_map.get(tuple(pt)) if pt is not None else None
            if nid is not None and nid in G and G.degree(nid) > 0:
                info["attachment_node"] = nid
                continue

            fallback = attach_room_to_existing_graph(info["room_center"])
            if fallback is None:
                # 房間中心節點仍會保留；只有道路 attachment 暫時為空。
                info["attachment_point"] = None
                info["attachment_node"] = None
                info["attachment_method"] = "unattached_no_road_graph"
                continue
            fpt, fnid, fdist, method = fallback
            info["attachment_point"] = tuple(fpt)
            info["attachment_node"] = fnid
            info["distance_px"] = round(float(fdist), 2)
            info["safe_clearance_px"] = round(float(wall_distance[fpt[1], fpt[0]]), 2)
            info["attachment_method"] = method

        # -------------------------------------------------
        # 9. A*：所有人工連線都沿著可通行遮罩，不再直接畫兩點直線
        # -------------------------------------------------
        astar_scale = 5

        def make_astar_mask(dilate_px):
            if dilate_px <= 0:
                m = valid_routing_mask.copy()
            else:
                k = int(dilate_px) * 2 + 1
                m = cv2.dilate(valid_routing_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            return cv2.bitwise_and(m, cv2.bitwise_not(wall_collision))

        ds_w = max(1, int(math.ceil(W / astar_scale)))
        ds_h = max(1, int(math.ceil(H / astar_scale)))
        astar_modes = []
        for mode_name, route_mask, penalty in [
            ("strict", make_astar_mask(0), 0.0),
            ("relaxed_6", make_astar_mask(6), 30.0),
            ("relaxed_12", make_astar_mask(12), 80.0),
        ]:
            ds = (cv2.resize(route_mask, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
            clear_ds = cv2.distanceTransform(ds, cv2.DIST_L2, 3)
            astar_modes.append((mode_name, route_mask, penalty, ds, clear_ds))

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

        def add_polyline_to_graph(points, edge_type):
            if not points or len(points) < 2:
                return []
            node_ids = [add_waypoint(points[0])]
            for p in points[1:-1]:
                node_ids.append(add_waypoint(p, node_type="astar_waypoint"))
            node_ids.append(add_waypoint(points[-1]))
            for a, b in zip(node_ids[:-1], node_ids[1:]):
                add_graph_edge(a, b, edge_type=edge_type)
            return node_ids

        # -------------------------------------------------
        # 10. 不連通 component：以 A* 路徑連接
        # -------------------------------------------------
        bridge_records = []
        max_bridge_px = max(260.0, 0.30 * math.hypot(W, H))
        bridge_safety = 0
        while G.number_of_nodes() > 0 and not nx.is_connected(G):
            bridge_safety += 1
            if bridge_safety > 24:
                print("[警告] A* component 橋接超過安全次數，停止處理。")
                break

            comps = [list(c) for c in nx.connected_components(G)]
            candidate_pairs = []
            for i in range(len(comps)):
                nodes_a = comps[i]
                coords_a = np.array([G.nodes[n]['pos'] for n in nodes_a], dtype=np.float32)
                for j in range(i + 1, len(comps)):
                    nodes_b = comps[j]
                    coords_b = np.array([G.nodes[n]['pos'] for n in nodes_b], dtype=np.float32)
                    if coords_a.size == 0 or coords_b.size == 0:
                        continue
                    d2 = np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=2)
                    take = min(6, d2.size)
                    flat_idx = np.argpartition(d2.ravel(), take - 1)[:take]
                    for flat in flat_idx:
                        ai, bj = np.unravel_index(int(flat), d2.shape)
                        d = float(math.sqrt(float(d2[ai, bj])))
                        if d <= max_bridge_px:
                            u, v = nodes_a[ai], nodes_b[bj]
                            candidate_pairs.append((d, u, v, tuple(G.nodes[u]['pos']), tuple(G.nodes[v]['pos'])))
            candidate_pairs.sort(key=lambda z: z[0])

            best = None
            for d, u, v, pu, pv in candidate_pairs[:36]:
                result = best_astar_path(pu, pv)
                if result is None:
                    continue
                score, points, path_len, mode_name, _ = result
                # 過度繞遠的 A* 不應用來硬接 component。
                if path_len > max(d * 2.8, d + 260.0):
                    continue
                total_score = score + d * 0.05
                if best is None or total_score < best[0]:
                    best = (total_score, u, v, points, path_len, mode_name, d)

            if best is None:
                print(f"[警告] 尚有 {len(comps)} 個道路 component，A* 找不到安全可行的接合路徑。")
                break

            _, u, v, points, path_len, mode_name, euclid = best
            add_polyline_to_graph(points, edge_type="component_bridge")
            bridge_records.append({
                "source": u,
                "target": v,
                "euclidean_distance_px": round(float(euclid), 2),
                "path_distance_px": round(float(path_len), 2),
                "astar_mode": mode_name,
                "path": [list(map(int, p)) for p in points]
            })

        # -------------------------------------------------
        # 11. 捷徑檢查：目前圖上明顯繞遠，但走道遮罩中存在更短 A* 路徑時加入捷徑
        # -------------------------------------------------
        shortcut_records = []
        if G.number_of_nodes() > 1:
            candidate_nodes = list(G.nodes())
            # 優先保留端點、junction 與 attachment；普通 degree=2 點只稀疏抽樣。
            structural_nodes = [n for n in candidate_nodes if G.degree(n) != 2]
            attachment_node_candidates = []
            for info in room_attachments.values():
                n = info.get("attachment_node")
                if n in G:
                    attachment_node_candidates.append(n)
            sampled_degree2 = [n for idx, n in enumerate(sorted(candidate_nodes)) if G.degree(n) == 2 and idx % 5 == 0]
            candidate_nodes = list(dict.fromkeys(structural_nodes + attachment_node_candidates + sampled_degree2))

            shortcut_candidates = []
            max_shortcut_euclid = 300.0
            for idx, u in enumerate(candidate_nodes):
                pu = G.nodes[u]['pos']
                nearby = []
                for v in candidate_nodes[idx + 1:]:
                    if G.has_edge(u, v):
                        continue
                    pv = G.nodes[v]['pos']
                    d = math.hypot(pv[0] - pu[0], pv[1] - pu[1])
                    if 35.0 <= d <= max_shortcut_euclid:
                        nearby.append((d, v, pv))
                nearby.sort(key=lambda z: z[0])
                lengths = nx.single_source_dijkstra_path_length(G, u, weight="weight")
                for d, v, pv in nearby[:10]:
                    current = lengths.get(v)
                    if current is None:
                        continue
                    saving_lower_bound = current - d
                    if current >= max(d * 1.45, d + 65.0):
                        shortcut_candidates.append((saving_lower_bound, current, d, u, v, pu, pv))

            shortcut_candidates.sort(key=lambda z: z[0], reverse=True)
            used_endpoint_pairs = set()
            max_shortcuts = 8
            for _, old_current, euclid, u, v, pu, pv in shortcut_candidates[:36]:
                if len(shortcut_records) >= max_shortcuts:
                    break
                key = tuple(sorted((u, v)))
                if key in used_endpoint_pairs or u not in G or v not in G:
                    continue
                try:
                    current = nx.shortest_path_length(G, u, v, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                if current < max(euclid * 1.35, euclid + 50.0):
                    continue

                result = best_astar_path(pu, pv)
                if result is None:
                    continue
                _, points, path_len, mode_name, _ = result
                saving = current - path_len
                # 至少節省 60px，且新路徑不得超過原路徑的 80%。
                if saving < 60.0 or path_len > current * 0.80:
                    continue
                # 避免用極度彎曲的路徑製造新的複雜路網。
                if len(points) > 12:
                    continue

                add_polyline_to_graph(points, edge_type="shortcut")
                used_endpoint_pairs.add(key)
                shortcut_records.append({
                    "source": u,
                    "target": v,
                    "old_distance_px": round(float(current), 2),
                    "new_distance_px": round(float(path_len), 2),
                    "saved_distance_px": round(float(saving), 2),
                    "astar_mode": mode_name,
                    "path": [list(map(int, p)) for p in points]
                })

        # -------------------------------------------------
        # 12. 共線 degree=2 壓縮；attachment 節點必須保留
        # -------------------------------------------------
        protected_attachment_nodes = {
            info.get("attachment_node")
            for info in room_attachments.values()
            if info.get("attachment_node") in G
        }

        def dominant_edge_type(type_a, type_b):
            priority = {"route": 0, "shortcut": 1, "component_bridge": 2}
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

        # V8：壓縮後逐一確認所有房間 attachment；不再 pop 任何房間。
        safe_nodes_after = [
            n for n in G.nodes()
            if point_is_safe(G.nodes[n]['pos'], clearance=min_attachment_clearance, mask=safe_attachment_mask)
        ]
        if not safe_nodes_after:
            safe_nodes_after = list(G.nodes())

        for rid, info in room_attachments.items():
            aid = info.get("attachment_node")
            if aid not in G:
                center = info["room_center"]
                if safe_nodes_after:
                    aid = min(
                        safe_nodes_after,
                        key=lambda n: math.hypot(
                            G.nodes[n]['pos'][0] - center[0],
                            G.nodes[n]['pos'][1] - center[1]
                        )
                    )
                    info["attachment_point"] = tuple(G.nodes[aid]['pos'])
                    info["attachment_method"] = "post_compression_nearest_node"
                else:
                    aid = None
                    info["attachment_point"] = None
                    info["attachment_method"] = "unattached_no_road_graph"
            info["attachment_node"] = aid
            if aid in G:
                ax, ay = G.nodes[aid]['pos']
                info["attachment_point"] = (int(ax), int(ay))
                info["distance_px"] = round(float(math.hypot(ax - info["room_center"][0], ay - info["room_center"][1])), 2)
                info["safe_clearance_px"] = round(float(wall_distance[ay, ax]), 2)
            else:
                info["distance_px"] = None
                info["safe_clearance_px"] = None

        # 房間中心節點只保留語意，不加入道路邊。
        for rid, center in room_coords.items():
            node_id = f"R_{rid}"
            names = id_labels.get(rid, {}).get("names", [])
            attach = room_attachments.get(rid)
            nodes_data[node_id] = {
                "type": "room_center",
                "name": "、".join(names) if names else f"房間_{rid}",
                "coordinates": [center[0], center[1]],
                "attachment_node": attach.get("attachment_node") if attach else None,
                "attachment_point": list(attach.get("attachment_point")) if attach else None,
                "arrival_rule": "導航終點為 attachment_node；抵達後即視為到達此房間"
            }

        # -------------------------------------------------
        # 13. JSON 輸出
        # -------------------------------------------------
        edges_data = []
        for u, v, edata in G.edges(data=True):
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
            dist = round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 2)
            deg_uv = round(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 360, 1)
            deg_vu = round(math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0])) % 360, 1)
            edge_type = edata.get("edge_type", "route")
            edges_data.extend([
                {"source": u, "target": v, "distance_px": dist, "direction_deg": deg_uv, "edge_type": edge_type},
                {"source": v, "target": u, "distance_px": dist, "direction_deg": deg_vu, "edge_type": edge_type}
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
                "unattached_room_count": int(sum(1 for info in room_attachments.values() if info.get("attachment_node") not in G))
            },
            "nodes": nodes_data,
            "edges": edges_data,
            "room_attachments": attachment_data,
            "component_bridges": bridge_records,
            "shortcuts": shortcut_records
        }
        with open(self.output_dir / "llm_navigation_graph.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

        # -------------------------------------------------
        # 14. Debug 圖
        # 綠：原始/刪減後路線；橘：A* component bridge；紫：A* 捷徑
        # 黃圈：安全抵達點；紅點：房間中心；紅虛線：語意關聯
        # -------------------------------------------------
        debug_graph_img = cv2.cvtColor((1 - (wall_matrix > 0).astype(np.uint8)) * 255, cv2.COLOR_GRAY2BGR)
        debug_graph_img[valid_routing_mask > 0] = (245, 245, 245)

        edge_color = {
            "route": (0, 150, 0),
            "component_bridge": (0, 165, 255),
            "shortcut": (180, 0, 180)
        }
        edge_thickness = {"route": 2, "component_bridge": 3, "shortcut": 3}
        for u, v, edata in G.edges(data=True):
            p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
            et = edata.get("edge_type", "route")
            cv2.line(debug_graph_img, p1, p2, edge_color.get(et, (0, 150, 0)), edge_thickness.get(et, 2))

        for nid in G.nodes():
            pt = tuple(G.nodes[nid]['pos'])
            cv2.rectangle(debug_graph_img, (pt[0] - 3, pt[1] - 3), (pt[0] + 3, pt[1] + 3), (255, 0, 0), -1)

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
            draw_dashed_line(debug_graph_img, center, attach_pt, (0, 0, 255), 1)
            cv2.circle(debug_graph_img, attach_pt, 5, (0, 255, 255), 2)

        cv2.imwrite(str(self.output_dir / "debug_navigation_graph.jpg"), debug_graph_img)
        print(
            "[完成] 導航圖已輸出："
            f"connected={payload['graph_connected']}，"
            f"components={payload['road_component_count']}，"
            f"parallel_removed={payload['route_reduction']['removed_parallel_segments']}，"
            f"rooms={payload['room_coverage']['room_center_node_count']}/{payload['room_coverage']['room_count']}，"
            f"attached={payload['room_coverage']['attached_room_count']}，"
            f"bridges={len(bridge_records)}，shortcuts={len(shortcut_records)}"
        )


# =========================================
# 執行入口
# =========================================
if __name__ == "__main__":
    MODEL_PATH = 'train3/weights/best.pt'
    INPUT_MAP = 'map/demo_normal.jpg'

    output_folder = Path("map_output")
    output_folder.mkdir(parents=True, exist_ok=True)
    cache_folder = output_folder / ".fast_cache"
    cache_folder.mkdir(parents=True, exist_ok=True)
    timings = {}
    pipeline_start = time.perf_counter()

    with stage_timer("YOLO（單次推論/快取）", timings):
        yolo_detections, yolo_boxes_data = get_yolo_data(
            MODEL_PATH, INPUT_MAP, cache_dir=cache_folder, conf=0.15, imgsz=896
        )

    with stage_timer("OCR（共用 Reader/快取）", timings):
        ocr_results = get_ocr_data(INPUT_MAP, cache_dir=cache_folder)

    with stage_timer("K-Means 走道分析", timings):
        color_result = analyze_colors_and_corridor(INPUT_MAP, ocr_results, k=6, return_details=True)
        if color_result is None:
            raise RuntimeError("無法讀取輸入地圖或建立走道遮罩。")
        corridor_mask_k, bg_mask, corridor_color_details = color_result
        if bg_mask is not None:
            cv2.imwrite(str(output_folder / "debug_bg_mask_0610_1.jpg"), bg_mask)

    with stage_timer("牆體提取與兩階段修補", timings):
        repaired_wall_matrix = extract_walls_with_repair(
            INPUT_MAP, output_folder, ocr_results,
            bg_mask=bg_mask, yolo_boxes=yolo_boxes_data
        )

    if repaired_wall_matrix is not None:
        if bg_mask is not None:
            print("[系統] 正在將背景區域實體化為不可行走牆體...")
            H, W = repaired_wall_matrix.shape
            if bg_mask.shape != (H, W):
                bg_mask = cv2.resize(bg_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] > (H * W) * 0.01
            keep[0] = False
            clean_bg_mask = (keep[labels].astype(np.uint8) * 255)
            cv2.imwrite(str(output_folder / "debug_clean_bg_mask_0610_1.jpg"), clean_bg_mask)
            repaired_wall_matrix[clean_bg_mask > 0] = 1

        with stage_timer("空間分割與語意標記", timings):
            res_matrix, metrics_list, id_labels = RoomSegmenter(
                output_folder, MODEL_PATH, door_ratio=0.01
            ).process(
                INPUT_MAP,
                wall_matrix=repaired_wall_matrix,
                corridor_mask=corridor_mask_k,
                ocr_data=ocr_results,
                yolo_detections=yolo_detections,
                save_csv=os.environ.get("MAP_SAVE_CSV", "1") != "0",
                corridor_color_details=corridor_color_details,
            )

        with stage_timer("導航路網、A* 橋接與捷徑", timings):
            WaypointGraphGenerator(output_folder).generate(
                wall_matrix=repaired_wall_matrix,
                res_matrix=res_matrix,
                metrics_list=metrics_list,
                id_labels=id_labels,
            )

    total = time.perf_counter() - pipeline_start
    timings["總計"] = total
    with open(output_folder / "runtime_profile_fast.json", "w", encoding="utf-8") as f:
        json.dump({k: round(v, 3) for k, v in timings.items()}, f, ensure_ascii=False, indent=2)
    print(f"[完成] FAST V4 全流程耗時：{total:.2f} 秒")
