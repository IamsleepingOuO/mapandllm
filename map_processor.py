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
from collections import Counter
from ultralytics import YOLO

# 強制優化 PyTorch 記憶體碎片管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================
# 工具函數：支援多格式的影像讀取
# =========================================
def safe_imread(image_path, flags=cv2.IMREAD_COLOR):
    return cv2.imread(str(image_path), flags)

# =========================================
# 全域 OCR 提取模組
# =========================================
def get_ocr_data(image_path):
    print("[系統] 正在執行 OCR 文字辨識 (全域預處理)...")
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu) 
    temp_img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if temp_img is None: return []
    if len(temp_img.shape) == 3: temp_img = temp_img[:, :, 0]
        
    results = reader.readtext(temp_img, canvas_size=2560)
    ocr_data = []
    for (bbox, text, prob) in results:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        
        ocr_data.append({
            'text': text, 
            'center': (cx, cy),
            'box': (x_min, y_min, x_max, y_max)
        })
    return ocr_data

# =========================================
# 模組 A：前置光影校正與 K-Means 色彩萃取 (修正版)
# =========================================
def analyze_colors_and_corridor(image_path, ocr_data, k=8): # 🌟 修正 1：提高 K 值至 8，增強分離陰影背景與走道的能力
    img = safe_imread(image_path)
    if img is None: return None
        
    H, W = img.shape[:2]
    
    print(f"[系統] 正在執行前置光影校正 (僅保留保邊雙邊濾波)...")
    # 🌟 修正 2：移除會破壞大面積色塊的 CLAHE，直接對原圖進行雙邊濾波降噪
    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blurred = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel_close)
    
    max_dim = 1200
    scale = 1.0
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        proc_img = cv2.resize(blurred, (0, 0), fx=scale, fy=scale)
    else:
        proc_img = blurred
        
    print(f"[系統] 正在分析地圖色彩尋找走道 (K={k})...")
    Z = proc_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # 固定亂數種子，消除非確定性
    cv2.setRNGSeed(0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    labels_2d = labels.reshape(proc_img.shape[:2])
    if scale != 1.0:
        labels_2d = cv2.resize(labels_2d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        labels_2d = labels_2d.astype(np.uint8)

    # 內縮邊緣取樣法尋找背景
    margin = 10
    if H > margin * 2 and W > margin * 2:
        border_pixels = (labels_2d[margin, margin:W-margin].tolist() + 
                         labels_2d[H-1-margin, margin:W-margin].tolist() + 
                         labels_2d[margin:H-margin, margin].tolist() + 
                         labels_2d[margin:H-margin, W-1-margin].tolist())
    else:
        border_pixels = (labels_2d[0, :].tolist() + labels_2d[H-1, :].tolist() + 
                         labels_2d[:, 0].tolist() + labels_2d[:, W-1].tolist())

    # 多重背景色支援 (超過 10% 佔比即視為背景)
    border_counts = Counter(border_pixels)
    total_border_pixels = len(border_pixels)
    bg_ids = [color_id for color_id, count in border_counts.items() if (count / total_border_pixels) > 0.1]

    # 利用 OCR 輔助判斷走道
    counts = np.bincount(labels_2d.flatten(), minlength=k)
    
    # 將所有判定為背景的顏色面積歸零，徹底從走道候選名單剔除
    for b_id in bg_ids:
        counts[b_id] = 0 
    
    top2_ids = counts.argsort()[-2:][::-1]
    id_1, id_2 = top2_ids[0], top2_ids[1]

    text_count_1, text_count_2 = 0, 0
    for item in ocr_data:
        cx, cy = item['center']
        if 0 <= cx < W and 0 <= cy < H:
            label_at_text = labels_2d[cy, cx]
            if label_at_text == id_1: text_count_1 += 1
            elif label_at_text == id_2: text_count_2 += 1

    corridor_id = id_1 if text_count_1 <= text_count_2 else id_2

    corridor_mask = (labels_2d == corridor_id).astype(np.uint8) * 255
    kernel_close_corr = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    corridor_mask = cv2.morphologyEx(corridor_mask, cv2.MORPH_CLOSE, kernel_close_corr)
    corridor_mask = cv2.dilate(corridor_mask, np.ones((5,5), np.uint8), iterations=1)

    return corridor_mask

# =========================================
# 模組 B：牆體極致提取與幾何修補 (加入平行牆壁抑制 + 綠線曼哈頓拘束器)
# =========================================
def extract_walls_with_repair(image_path, output_dir, ocr_data):
    img = safe_imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    
    print("[系統] 正在提取精準牆體並進行幾何修補...")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    binary = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 6)

    for item in ocr_data:
        x_min, y_min, x_max, y_max = item['box']
        pad = 3
        cv2.rectangle(binary, (max(0, x_min - pad), max(0, y_min - pad)), 
                      (min(W, x_max + pad), min(H, y_max + pad)), 0, -1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    final_walls = np.zeros_like(binary)

    for i in range(1, n_labels):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if (w > 35 or h > 35) and area > 60:
            final_walls[labels == i] = 255

    micro_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_walls = cv2.morphologyEx(final_walls, cv2.MORPH_CLOSE, micro_kernel)

    debug_repair_img = cv2.cvtColor(final_walls, cv2.COLOR_GRAY2BGR)
    ANGLE_THRES = math.cos(math.radians(30))
    
    raw_boxes = []
    for item in ocr_data:
        x_min, y_min, x_max, y_max = item['box']
        pad = 6
        raw_boxes.append((max(0, x_min - pad), max(0, y_min - pad), min(W, x_max + pad), min(H, y_max + pad)))

    def boxes_overlap(b1, b2):
        return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

    n = len(raw_boxes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if boxes_overlap(raw_boxes[i], raw_boxes[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited, groups = [False] * n, []
    for i in range(n):
        if not visited[i]:
            comp, q = [], [i]
            visited[i] = True
            while q:
                curr = q.pop(0)
                comp.append(raw_boxes[curr])
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            groups.append(comp)

    odd_endpoints_global = []

    for group in groups:
        group_box_mask = np.zeros((H, W), dtype=np.uint8)
        for (x1, y1, x2, y2) in group:
            cv2.rectangle(group_box_mask, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(debug_repair_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        kernel_erode = np.ones((9, 9), np.uint8)
        group_inner_mask = cv2.erode(group_box_mask, kernel_erode, iterations=1)
        border_ring = cv2.bitwise_xor(group_box_mask, group_inner_mask)
        intersect = cv2.bitwise_and(final_walls, border_ring)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersect, connectivity=8)
        endpoints = []
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 3:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                local_r = 12
                lx1, ly1 = max(0, cx - local_r), max(0, cy - local_r)
                lx2, ly2 = min(W, cx + local_r), min(H, cy + local_r)
                
                local_wall = final_walls[ly1:ly2, lx1:lx2].copy()
                local_wall[group_box_mask[ly1:ly2, lx1:lx2] == 255] = 0 
                
                M = cv2.moments(local_wall)
                if M["m00"] != 0:
                    vx, vy = cx - (int(M["m10"] / M["m00"]) + lx1), cy - (int(M["m01"] / M["m00"]) + ly1)
                    norm = math.hypot(vx, vy)
                    if norm > 0:
                        endpoints.append({'pt': (cx, cy), 'dir': (vx / norm, vy / norm), 'paired': False})

        # ==========================================
        # 雙數配對 (紅線) - 包含垂直偏移量限制以防平行互連
        # ==========================================
        n_pts = len(endpoints)
        if n_pts >= 2:
            scores = []
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    pt1, d1 = endpoints[i]['pt'], endpoints[i]['dir']
                    pt2, d2 = endpoints[j]['pt'], endpoints[j]['dir']
                    
                    v12x, v12y = pt2[0] - pt1[0], pt2[1] - pt1[1]
                    dist = math.hypot(v12x, v12y)
                    if dist == 0: continue
                    v12x_n, v12y_n = v12x / dist, v12y / dist
                    
                    dot1 = d1[0] * v12x_n + d1[1] * v12y_n
                    dot2 = d2[0] * (-v12x_n) + d2[1] * (-v12y_n)
                    
                    # 計算向量外積取得垂直偏移量 (Lateral Offset)
                    offset1 = abs(d1[0] * v12y - d1[1] * v12x)
                    offset2 = abs(d2[0] * (-v12y) - d2[1] * (-v12x))
                    max_lateral_offset = max(offset1, offset2)
                    
                    # 只有在 30 度視野內，且垂直偏移量 < 20 像素時才允許配對
                    if dot1 >= ANGLE_THRES and dot2 >= ANGLE_THRES and max_lateral_offset < 20:
                        scores.append(((dot1 + dot2 - (d1[0]*d2[0] + d1[1]*d2[1])) * 50 - dist, i, j))
                    
            scores.sort(reverse=True)
            for score, i, j in scores:
                if not endpoints[i]['paired'] and not endpoints[j]['paired']:
                    endpoints[i]['paired'] = endpoints[j]['paired'] = True
                    cv2.line(final_walls, endpoints[i]['pt'], endpoints[j]['pt'], 255, 3)
                    cv2.line(debug_repair_img, endpoints[i]['pt'], endpoints[j]['pt'], (0, 0, 255), 2)
                    
        for ep in endpoints:
            if not ep['paired']: odd_endpoints_global.append((ep, group_box_mask))

    # ==========================================
    # 奇數落單端點 (綠線) - 加入曼哈頓拘束器與安全距離限制
    # ==========================================
    for ep, group_box_mask in odd_endpoints_global:
        pt, (base_vx, base_vy) = ep['pt'], ep['dir']
        curr_x, curr_y = float(pt[0]), float(pt[1])
        path, walk_vx, walk_vy = [(curr_x, curr_y)], -base_vx, -base_vy
        
        # 輔助線探索
        for step in range(30):
            cx, cy = path[-1]
            candidates = []
            for a in range(-45, 46, 15):
                rad = math.radians(a)
                c_cos, c_sin = math.cos(rad), math.sin(rad)
                nx, ny = walk_vx * c_cos - walk_vy * c_sin, walk_vx * c_sin + walk_vy * c_cos
                for r in range(3, 8):
                    sx, sy = int(cx + nx * r), int(cy + ny * r)
                    if 0 <= sx < W and 0 <= sy < H and final_walls[sy, sx] == 255 and group_box_mask[sy, sx] == 0:
                        candidates.append((sx, sy))
            
            if not candidates: break
            avg_x = sum(c[0] for c in candidates) / len(candidates)
            avg_y = sum(c[1] for c in candidates) / len(candidates)
            step_vx, step_vy = avg_x - cx, avg_y - cy
            norm = math.hypot(step_vx, step_vy)
            if norm > 0:
                step_vx, step_vy = step_vx / norm, step_vy / norm
                if walk_vx * step_vx + walk_vy * step_vy < 0.866: break
                walk_vx, walk_vy = step_vx, step_vy
            path.append((avg_x, avg_y))
        
        if len(path) > 1:
            new_vx, new_vy = pt[0] - path[-1][0], pt[1] - path[-1][1]
            norm = math.hypot(new_vx, new_vy)
            if norm > 0: base_vx, base_vy = new_vx / norm, new_vy / norm
                
        # 🌟 拘束器 1：曼哈頓角度強制對齊 (Manhattan Snapping)
        # 如果算出的方向接近絕對的上下左右 (正負 25 度內)，強制拉直
        angle = math.degrees(math.atan2(base_vy, base_vx))
        nearest_90 = round(angle / 90.0) * 90.0
        diff = abs(angle - nearest_90)
        if diff <= 25 or abs(diff - 360) <= 25:
            rad_snap = math.radians(nearest_90)
            base_vx, base_vy = math.cos(rad_snap), math.sin(rad_snap)

        hit_pt = None
        
        # 🌟 拘束器 2：嚴格限制最遠探索距離，取代無限射線
        max_ray_len = 150 # 最大容忍距離，超過代表面向廣場，安全放棄
        
        for r in range(5, max_ray_len, 2):
            found_wall = False
            # 扇形雷達吸附
            for angle_offset in range(-15, 16, 3):
                rad = math.radians(angle_offset)
                nx = base_vx * math.cos(rad) - base_vy * math.sin(rad)
                ny = base_vx * math.sin(rad) + base_vy * math.cos(rad)
                check_x, check_y = int(pt[0] + nx * r), int(pt[1] + ny * r)
                
                if check_x < 0 or check_x >= W or check_y < 0 or check_y >= H:
                    hit_pt, found_wall = (max(0, min(W-1, check_x)), max(0, min(H-1, check_y))), True
                    break
                if final_walls[check_y, check_x] == 255:
                    hit_pt, found_wall = (check_x, check_y), True
                    break
            if found_wall: break
                
        # 只有在安全距離內成功撞到實體牆壁，才會畫出綠線
        if hit_pt:
            cv2.line(final_walls, pt, hit_pt, 255, 3)
            cv2.line(debug_repair_img, pt, hit_pt, (100, 255, 100), 2)

    final_walls = cv2.morphologyEx(final_walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    
    cv2.imwrite(str(output_dir / "debug_cleaned_walls_0526_4.jpg"), final_walls)
    cv2.imwrite(str(output_dir / "debug_repair_boxes_0526_4.jpg"), debug_repair_img)
    
    return (final_walls / 255).astype(np.uint8)

# =========================================
# 模組 C：空間分割與走道縫合
# =========================================
class RoomSegmenter:
    def __init__(self, output_dir, yolo_model_path, area_ratio=1/8000, door_ratio=0.01):
        self.output_dir = output_dir
        self.area_ratio = area_ratio
        self.door_ratio = door_ratio
        self.yolo_model = YOLO(yolo_model_path)

    def process(self, original_img_path, wall_matrix, corridor_mask, ocr_data):
        h, w = wall_matrix.shape[:2]
        min_area = int((h * w) * self.area_ratio)
        door_size = int(np.sqrt(h**2 + w**2) * self.door_ratio)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (door_size, door_size))
        closed = cv2.morphologyEx(wall_matrix, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((1-closed).astype(np.uint8), connectivity=8)
        
        res_matrix = np.ones((h, w), dtype=np.int32) 
        metrics_list = []
        current_id = 2
        
        for i in range(1, num_labels):
            lx, ly, sw, sh = stats[i, :4]
            if lx <= 2 or ly <= 2 or (lx + sw) >= w - 2 or (ly + sh) >= h - 2: continue 
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                ys, xs = np.where(labels == i)
                cx, cy = centroids[i]
                metrics_list.append({
                    'id': current_id, 'area': area,
                    'max_dist': max(np.mean(np.abs(xs - cx)), np.mean(np.abs(ys - cy))),
                    'min_dist': min(np.mean(np.abs(xs - cx)), np.mean(np.abs(ys - cy))),
                    'centroid': centroids[i]
                })
                res_matrix[(labels == i) & (wall_matrix == 0)] = current_id
                current_id += 1

        if not metrics_list: return res_matrix

        corridor_rids = []
        if corridor_mask is not None:
            print("[系統] 正在基於 K-Means 色彩遮罩縫合走道碎片...")
            for m in metrics_list:
                rid = m['id']
                if np.sum((res_matrix == rid) & (corridor_mask > 0)) / m['area'] > 0.45: 
                    corridor_rids.append(rid)
        
        main_cid = None
        if corridor_rids:
            main_cid = max([m for m in metrics_list if m['id'] in corridor_rids], key=lambda x: x['area'])['id']
            for rid in corridor_rids:
                if rid != main_cid: res_matrix[res_matrix == rid] = main_cid
            res_matrix[(res_matrix == 1) & (corridor_mask > 0)] = main_cid
            
            num_c_labels, c_labels, c_stats, _ = cv2.connectedComponentsWithStats((res_matrix == main_cid).astype(np.uint8), connectivity=8)
            if num_c_labels > 1: 
                c_areas = sorted([(i, c_stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_c_labels)], key=lambda x: x[1], reverse=True)
                for i in range(2, len(c_areas)): res_matrix[c_labels == c_areas[i][0]] = 1
            
            new_metrics = []
            for m in metrics_list:
                if m['id'] == main_cid:
                    m['area'] = np.sum(res_matrix == main_cid)
                    new_metrics.append(m)
                elif m['id'] not in corridor_rids:
                    new_metrics.append(m)
            metrics_list = new_metrics

        adj_map = {m['id']: set() for m in metrics_list}
        dilation_kernel = np.ones((12, 12), np.uint8)
        for m in metrics_list:
            dilated = cv2.dilate((res_matrix == m['id']).astype(np.uint8), dilation_kernel)
            for n_id in np.unique(res_matrix[dilated == 1]):
                if n_id > 1 and n_id != m['id']:
                    adj_map[m['id']].add(int(n_id))
                    adj_map[n_id].add(int(m['id']))

        id_labels = {str(m['id']): {"names": [], "objects": [], "portal": False, "shape": []} for m in metrics_list}
        virtual_room_id = current_id
        id_labels[str(virtual_room_id)] = {"names": [], "objects": [], "portal": False, "shape": []}

        for rid_int, neighbors in adj_map.items():
            if len(neighbors) >= 5 and str(rid_int) in id_labels: id_labels[str(rid_int)]["portal"] = True
        if corridor_rids and str(main_cid) in id_labels: id_labels[str(main_cid)]["portal"] = True

        for m in sorted(metrics_list, key=lambda x: x['max_dist'], reverse=True)[:max(1, int(len(metrics_list) * 0.2))]: id_labels[str(m['id'])]["shape"].append("長寬")
        for m in sorted(metrics_list, key=lambda x: x['min_dist'])[:max(1, int(len(metrics_list) * 0.2))]: id_labels[str(m['id'])]["shape"].append("短窄")
        for m in sorted(metrics_list, key=lambda x: x['area'], reverse=True)[:max(1, int(len(metrics_list) * 0.2))]: id_labels[str(m['id'])]["shape"].append("大")
        for m in sorted(metrics_list, key=lambda x: x['area'])[:max(1, int(len(metrics_list) * 0.3))]: id_labels[str(m['id'])]["shape"].append("小")

        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        random_colors = [(255, 220, 200), (200, 255, 200), (200, 255, 255), (200, 200, 255)]
        for m in metrics_list:
            color = (200, 200, 200) if id_labels[str(m['id'])]["portal"] else random.choice(random_colors)
            vis_img[res_matrix == m['id']] = color
            cv2.putText(vis_img, str(m['id']), (int(m['centroid'][0]-15), int(m['centroid'][1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        for item in ocr_data:
            tx, ty = item['center']
            if ty < h and tx < w:
                target_id = res_matrix[ty, tx]
                if target_id == 1: target_id = min(metrics_list, key=lambda r: (tx-r['centroid'][0])**2 + (ty-r['centroid'][1])**2)['id']
                if str(target_id) in id_labels: id_labels[str(target_id)]["names"].append(item['text'])

        for obj in self.yolo_model.predict(source=original_img_path, conf=0.15, imgsz=1024, verbose=False)[0].boxes:
            ox, oy, conf, label = int((obj.xyxy[0][0] + obj.xyxy[0][2]) / 2), int((obj.xyxy[0][1] + obj.xyxy[0][3]) / 2), float(obj.conf[0]), self.yolo_model.names[int(obj.cls[0])]
            if oy >= h or ox >= w: continue
            target_id = res_matrix[oy, ox] if conf >= 0.40 else virtual_room_id
            if target_id == 1: target_id = min(metrics_list, key=lambda r: (ox-r['centroid'][0])**2 + (oy-r['centroid'][1])**2)['id']
            if str(target_id) in id_labels: id_labels[str(target_id)]["objects"].append(f"{label}({conf:.2f})")

        pd.DataFrame(res_matrix).to_csv(self.output_dir / "_0526_6.csv", index=False, header=False)
        cv2.imwrite(str(self.output_dir / "debug_0526_6.jpg"), vis_img)
        with open(self.output_dir / "room_data_0526_6.json", 'w', encoding='utf-8') as f:
            json.dump(id_labels, f, ensure_ascii=False, indent=4)
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[完成] 完美融合版 JSON 已生成。")
        return res_matrix

# =========================================
# 執行入口
# =========================================
# if __name__ == "__main__":
#     MODEL_PATH = 'train6/weights/best.pt'
#     INPUT_MAP = 'map/demo_shadow.jpg'
    
#     output_folder = Path("map_output")
#     output_folder.mkdir(parents=True, exist_ok=True)

#     # 1. 執行 OCR
#     ocr_results = get_ocr_data(INPUT_MAP)

#     # 2. 獲取走道遮罩 (前置光影校正 + 多重背景容錯 K-Means)
#     corridor_mask_k = analyze_colors_and_corridor(INPUT_MAP, ocr_results, k=6)

#     # 3. 提取精密牆體 (向量定向與扇形雷達修補)
#     repaired_wall_matrix = extract_walls_with_repair(INPUT_MAP, output_folder, ocr_results)
    
#     # 4. 進行空間分配與屬性標記
#     if repaired_wall_matrix is not None:
#         RoomSegmenter(output_folder, MODEL_PATH).process(
#             INPUT_MAP, 
#             wall_matrix=repaired_wall_matrix,
#             corridor_mask=corridor_mask_k, 
#             ocr_data=ocr_results
#         )