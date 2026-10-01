from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass(frozen=True)
class Detection:
    box: list[float]
    detector_score: float
    detector_label: str
    text: str
    ocr_score: float
    lines: list[dict[str, Any]]


class SignOCRPipeline:
    """Grounding DINO 定位店面招牌，再以 PaddleOCR 讀取招牌文字。"""

    def __init__(self) -> None:
        self.model_id = os.getenv(
            "DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny"
        )
        self.device = self._resolve_torch_device(os.getenv("DINO_DEVICE", "auto"))
        self.ocr_device = os.getenv("OCR_DEVICE", "cpu")
        self.ocr_det_model = os.getenv("OCR_DET_MODEL", "PP-OCRv5_server_det")
        self.ocr_rec_model = os.getenv("OCR_REC_MODEL", "PP-OCRv5_server_rec")

        self.box_threshold = float(os.getenv("DINO_BOX_THRESHOLD", "0.25"))
        self.text_threshold = float(os.getenv("DINO_TEXT_THRESHOLD", "0.22"))
        self.ocr_threshold = float(os.getenv("OCR_SCORE_THRESHOLD", "0.45"))
        self.max_detections = max(1, int(os.getenv("MAX_DETECTIONS", "3")))
        self.dino_short_edge = max(256, int(os.getenv("DINO_SHORT_EDGE", "640")))
        self.dino_long_edge = max(self.dino_short_edge, int(os.getenv("DINO_LONG_EDGE", "1067")))
        self.min_box_area_ratio = float(os.getenv("MIN_BOX_AREA_RATIO", "0.002"))
        self.crop_padding_ratio = float(os.getenv("CROP_PADDING_RATIO", "0.08"))
        self.ocr_enable_mkldnn = self._env_bool("OCR_ENABLE_MKLDNN", False)

        prompt_value = os.getenv(
            "DINO_LABELS",
            (
                "store sign|storefront sign|shop name sign|"
                "brand sign above shop entrance|wall mounted store sign"
            ),
        )
        self.labels = [item.strip() for item in prompt_value.split("|") if item.strip()]
        if not self.labels:
            raise ValueError("DINO_LABELS 至少需要一個提示詞")

        print(f"[model] Grounding DINO: {self.model_id} on {self.device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

        print(
            f"[model] PaddleOCR: lang={os.getenv('OCR_LANG', 'chinese_cht')} "
            f"device={self.ocr_device} mkldnn={self.ocr_enable_mkldnn}",
            flush=True,
        )
        # 預設關閉 oneDNN/MKLDNN：可避開部分 Paddle 3.x CPU PIR/oneDNN 問題。
        self.ocr = PaddleOCR(
            lang=os.getenv("OCR_LANG", "chinese_cht"),
            ocr_version=os.getenv("OCR_VERSION", "PP-OCRv5"),
            device=self.ocr_device,
            text_detection_model_name=self.ocr_det_model,
            text_recognition_model_name=self.ocr_rec_model,
            enable_mkldnn=self.ocr_enable_mkldnn,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.ocr_threshold,
        )

        # 同一組模型一次只處理一張圖，避免手機連續請求把 GPU/CPU 塞爆。
        self._inference_lock = threading.Lock()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _resolve_torch_device(value: str) -> str:
        if value != "auto":
            return value
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def warmup(self) -> None:
        """Run both inference engines once before a camera capture arrives."""
        from PIL import ImageDraw

        image = Image.new("RGB", (640, 384), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 130, 540, 245), fill="black")
        draw.text((180, 170), "SHOP 123", fill="white")
        with self._inference_lock:
            self._detect_signs(image)
            self._run_ocr(image.crop((100, 130, 540, 245)))

    def recognize(self, image: Image.Image) -> list[Detection]:
        detections, _ = self.recognize_with_timings(image)
        return detections

    def recognize_with_timings(self, image: Image.Image) -> tuple[list[Detection], dict[str, Any]]:
        image = image.convert("RGB")
        waiting_at = time.perf_counter()
        with self._inference_lock:
            lock_ms = (time.perf_counter() - waiting_at) * 1000
            detection_at = time.perf_counter()
            candidates, detection_timings = self._detect_signs(image, with_timings=True)
            detection_ms = (time.perf_counter() - detection_at) * 1000
            detections: list[Detection] = []
            ocr_ms: list[float] = []

            for candidate in candidates:
                ocr_at = time.perf_counter()
                padded_box = self._pad_box(candidate["box"], image.size)
                crop = image.crop(tuple(map(int, padded_box)))
                text, ocr_score, lines = self._run_ocr(crop)
                ocr_ms.append(round((time.perf_counter() - ocr_at) * 1000, 1))
                detections.append(
                    Detection(
                        box=[round(value, 2) for value in padded_box],
                        detector_score=round(candidate["score"], 4),
                        detector_label=candidate["label"],
                        text=text,
                        ocr_score=round(ocr_score, 4),
                        lines=lines,
                    )
                )

            timings = {
                "model_lock_wait_ms": round(lock_ms, 1),
                "dino_total_ms": round(detection_ms, 1),
                **detection_timings,
                "ocr_total_ms": round(sum(ocr_ms), 1),
                "ocr_per_crop_ms": ocr_ms,
                "ocr_crop_count": len(ocr_ms),
                "max_detections": self.max_detections,
                "ocr_det_model": self.ocr_det_model,
                "ocr_rec_model": self.ocr_rec_model,
            }
            return detections, timings

    def _detect_signs(
        self, image: Image.Image, *, with_timings: bool = False
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        # Transformers Grounding DINO 支援 nested list 的文字類別輸入。
        text_labels = [self.labels]
        preprocess_at = time.perf_counter()
        inputs = self.processor(
            images=image,
            text=text_labels,
            size={"shortest_edge": self.dino_short_edge, "longest_edge": self.dino_long_edge},
            return_tensors="pt",
        ).to(self.device)

        preprocess_ms = (time.perf_counter() - preprocess_at) * 1000
        inference_at = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_at) * 1000

        postprocess_at = time.perf_counter()
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        labels = results.get("text_labels")
        if labels is None:
            labels = results.get("labels", [])

        image_area = image.width * image.height
        candidates: list[dict[str, Any]] = []

        for box_tensor, score_tensor, label in zip(
            results["boxes"], results["scores"], labels
        ):
            box = [float(value) for value in box_tensor.tolist()]
            score = float(score_tensor.item())
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if image_area and area / image_area < self.min_box_area_ratio:
                continue
            candidates.append({"box": box, "score": score, "label": str(label)})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        candidates = self._nms(candidates, iou_threshold=0.55)
        selected = candidates[: self.max_detections]
        if not with_timings:
            return selected
        return selected, {
            "dino_preprocess_ms": round(preprocess_ms, 1),
            "dino_input_height": int(inputs.pixel_values.shape[-2]),
            "dino_input_width": int(inputs.pixel_values.shape[-1]),
            "dino_inference_ms": round(inference_ms, 1),
            "dino_postprocess_ms": round((time.perf_counter() - postprocess_at) * 1000, 1),
            "sign_candidates": len(candidates),
        }

    def _run_ocr(
        self, crop: Image.Image
    ) -> tuple[str, float, list[dict[str, Any]]]:
        if crop.width < 12 or crop.height < 12:
            return "", 0.0, []

        results = self.ocr.predict(np.asarray(crop))
        lines: list[dict[str, Any]] = []

        for result in results:
            payload = result.json
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                continue

            data = payload.get("res", payload)
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            boxes = data.get("rec_boxes")
            if boxes is None:
                boxes = []

            for index, text in enumerate(texts):
                cleaned = " ".join(str(text).split()).strip()
                score = float(scores[index]) if index < len(scores) else 0.0
                if not cleaned or score < self.ocr_threshold:
                    continue

                box = boxes[index] if index < len(boxes) else None
                if hasattr(box, "tolist"):
                    box = box.tolist()

                lines.append(
                    {
                        "text": cleaned,
                        "score": round(score, 4),
                        "box": box,
                    }
                )

        lines.sort(
            key=lambda item: (
                item["box"][1] if item["box"] else 0,
                item["box"][0] if item["box"] else 0,
            )
        )
        joined = " ".join(item["text"] for item in lines)
        average_score = (
            sum(item["score"] for item in lines) / len(lines) if lines else 0.0
        )
        return joined, average_score, lines

    def _pad_box(self, box: list[float], image_size: tuple[int, int]) -> list[float]:
        width, height = image_size
        x1, y1, x2, y2 = box
        padding_x = (x2 - x1) * self.crop_padding_ratio
        padding_y = (y2 - y1) * self.crop_padding_ratio
        return [
            max(0.0, x1 - padding_x),
            max(0.0, y1 - padding_y),
            min(float(width), x2 + padding_x),
            min(float(height), y2 + padding_y),
        ]

    @staticmethod
    def _nms(
        candidates: list[dict[str, Any]], iou_threshold: float
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            if all(
                SignOCRPipeline._iou(candidate["box"], item["box"])
                < iou_threshold
                for item in kept
            ):
                kept.append(candidate)
        return kept

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
