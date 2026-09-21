from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.pipeline import SignOCRPipeline
from navigation import app as navigation_app, UPLOAD_DIR, YOLO_MODEL_PATH, LLM_MODEL

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

SAVE_ROOT = Path(os.getenv("SAVE_ROOT", str(PROJECT_DIR / "saved"))).expanduser()
if not SAVE_ROOT.is_absolute():
    SAVE_ROOT = PROJECT_DIR / SAVE_ROOT
SAVE_ANNOTATED = os.getenv("SAVE_ANNOTATED", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
SAVE_CROPS = os.getenv("SAVE_CROPS", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
ANNOTATION_FONT = os.getenv("ANNOTATION_FONT", "").strip()

app = FastAPI(title="MapAndLLM Vision", version="2.0.0")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_pipeline: SignOCRPipeline | None = None
_pipeline_init_lock = threading.Lock()


def get_pipeline() -> SignOCRPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_init_lock:
        if _pipeline is None:
            print("[server] 第一次辨識：正在載入 Grounding DINO / PaddleOCR...", flush=True)
            from app.pipeline import SignOCRPipeline
            _pipeline = SignOCRPipeline()
            print("[server] 模型載入完成。", flush=True)
    return _pipeline


def recognize_image(image: Image.Image) -> list[dict[str, Any]]:
    detections = get_pipeline().recognize(image)
    return [asdict(item) for item in detections]


def _safe_extension(content_type: str | None, filename: str | None) -> str:
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    if content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"

    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def _make_capture_id(frame_number: int | None) -> tuple[str, str]:
    now = datetime.now().astimezone()
    day = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    frame = f"_f{frame_number:06d}" if frame_number is not None else ""
    capture_id = f"{stamp}{frame}_{uuid.uuid4().hex[:8]}"
    return day, capture_id


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_annotation_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        ANNOTATION_FONT,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _save_annotated(
    image: Image.Image,
    detections: list[dict[str, Any]],
    path: Path,
) -> None:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, annotated.width // 320)
    font_size = max(16, annotated.width // 42)
    font = _load_annotation_font(font_size)

    for index, det in enumerate(detections, start=1):
        box = det.get("box") or []
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)

        text = (det.get("text") or "").strip()
        detector_label = (det.get("detector_label") or "sign").strip()
        ocr_score = float(det.get("ocr_score") or 0.0)
        detector_score = float(det.get("detector_score") or 0.0)

        if text:
            label = f"#{index} {text} OCR {ocr_score:.0%}"
        else:
            label = f"#{index} {detector_label} DINO {detector_score:.0%}"

        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = (len(label) * max(8, font_size // 2), font_size + 4)

        pad = 4
        label_y = max(0, y1 - th - pad * 2)
        draw.rectangle(
            (x1, label_y, min(annotated.width, x1 + tw + pad * 2), label_y + th + pad * 2),
            fill="red",
        )
        try:
            draw.text((x1 + pad, label_y + pad), label, fill="white", font=font)
        except Exception:
            draw.text((x1 + pad, label_y + pad), f"#{index}", fill="white", font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(path, format="JPEG", quality=92)


def _save_crops(
    image: Image.Image,
    detections: list[dict[str, Any]],
    crop_dir: Path,
) -> list[str]:
    saved: list[str] = []
    crop_dir.mkdir(parents=True, exist_ok=True)

    for index, det in enumerate(detections, start=1):
        box = det.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(image.width, x1))
        x2 = max(0, min(image.width, x2))
        y1 = max(0, min(image.height, y1))
        y2 = max(0, min(image.height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop_path = crop_dir / f"{index:02d}.jpg"
        image.crop((x1, y1, x2, y2)).convert("RGB").save(
            crop_path, format="JPEG", quality=92
        )
        saved.append(_relative(crop_path))

    return saved


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _pipeline is not None,
        "version": "2.0.0",
        "map_model_available": YOLO_MODEL_PATH.is_file(),
        "llm_model": LLM_MODEL,
        "save_root": str(SAVE_ROOT.resolve()),
        "save_annotated": SAVE_ANNOTATED,
        "save_crops": SAVE_CROPS,
    }




@app.post("/api/recognize")
async def recognize(
    image: UploadFile = File(...),
    frame_number: int | None = Form(default=None),
    captured_at: str | None = Form(default=None),
) -> dict[str, Any]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="只接受圖片檔")

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="收到空圖片")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="圖片太大，限制 8 MB")

    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="無法解析圖片") from exc

    day, capture_id = _make_capture_id(frame_number)
    extension = _safe_extension(image.content_type, image.filename)

    capture_path = SAVE_ROOT / "captures" / day / f"{capture_id}{extension}"
    result_path = SAVE_ROOT / "results" / day / f"{capture_id}.json"
    annotated_path = SAVE_ROOT / "annotated" / day / f"{capture_id}.jpg"
    crop_dir = SAVE_ROOT / "crops" / day / capture_id

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_bytes(raw)

    width, height = pil_image.size
    started = time.perf_counter()

    base_metadata: dict[str, Any] = {
        "capture_id": capture_id,
        "server_received_at": datetime.now().astimezone().isoformat(),
        "client_captured_at": captured_at,
        "frame_number": frame_number,
        "source_filename": image.filename,
        "content_type": image.content_type,
        "image_width": width,
        "image_height": height,
        "saved": {
            "capture": _relative(capture_path),
            "result": _relative(result_path),
            "annotated": _relative(annotated_path) if SAVE_ANNOTATED else None,
            "crops": [],
        },
    }

    try:
        detections = await run_in_threadpool(recognize_image, pil_image)
    except Exception as exc:
        processing_ms = round((time.perf_counter() - started) * 1000, 1)
        error_payload = {
            **base_metadata,
            "ok": False,
            "processing_ms": processing_ms,
            "detections": [],
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        await run_in_threadpool(_save_json, result_path, error_payload)
        print(f"[server] inference error: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"推論失敗：{type(exc).__name__}: {exc}",
                "capture_id": capture_id,
                "saved_capture": _relative(capture_path),
                "saved_result": _relative(result_path),
            },
        ) from exc

    processing_ms = round((time.perf_counter() - started) * 1000, 1)

    crop_paths: list[str] = []
    if SAVE_ANNOTATED:
        await run_in_threadpool(_save_annotated, pil_image, detections, annotated_path)
    if SAVE_CROPS:
        crop_paths = await run_in_threadpool(_save_crops, pil_image, detections, crop_dir)

    payload: dict[str, Any] = {
        **base_metadata,
        "ok": True,
        "processing_ms": processing_ms,
        "detections": detections,
    }
    payload["saved"]["crops"] = crop_paths

    await run_in_threadpool(_save_json, result_path, payload)

    print(
        f"[server] saved {capture_id}: "
        f"capture={_relative(capture_path)} result={_relative(result_path)}",
        flush=True,
    )
    return payload

app.mount("/api", navigation_app, name="navigation")
