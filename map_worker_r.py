from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _best_effort_gpu_cleanup(module: Any | None) -> None:
    """程序退出前做一次整理；真正的完整回收由程序終止本身保證。"""
    try:
        cache = getattr(module, "_EASYOCR_READER_CACHE", None) if module is not None else None
        if isinstance(cache, dict):
            count = len(cache)
            cache.clear()
            print(f"[WORKER-GPU] EasyOCR Reader 快取已清除：{count}")
    except Exception as exc:
        print(f"[WORKER-GPU] 清除 EasyOCR 快取失敗：{exc}")

    gc.collect()
    try:
        # Do not load PyTorch/CUDA solely to clean up an early input failure.
        torch = sys.modules.get("torch")
        if torch is None:
            return

        if torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[WORKER-GPU] 退出前 PyTorch：allocated={allocated:.1f} MB，reserved={reserved:.1f} MB")
    except Exception as exc:
        print(f"[WORKER-GPU] PyTorch 清理略過：{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor map processing worker")
    parser.add_argument("--module", default="map_processor", help="包含 process_map_pipeline 的模組名稱")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--save-csv", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def _configure_stdio() -> None:
    # Parent should decode worker pipes as UTF-8 (encoding="utf-8").
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)


def main() -> int:
    _configure_stdio()
    args = parse_args()
    result_file = Path(args.result_file).resolve()
    module = None

    # 確保 worker 所在的專案目錄優先於其他同名套件。
    worker_dir = Path(__file__).resolve().parent
    if str(worker_dir) not in sys.path:
        sys.path.insert(0, str(worker_dir))

    try:
        print(f"[WORKER] PID={os.getpid()}，開始載入 {args.module}")
        module = importlib.import_module(args.module)
        pipeline = getattr(module, "process_map_pipeline", None)
        if not callable(pipeline):
            raise RuntimeError(f"模組 {args.module!r} 沒有可呼叫的 process_map_pipeline")

        yolo_path = Path(args.yolo_model).resolve() if args.yolo_model else None
        artifacts = pipeline(
            image_path=Path(args.image).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            yolo_model_path=yolo_path,
            k=int(args.k),
            save_csv=bool(args.save_csv),
        )
        _atomic_write_json(result_file, {"ok": True, "artifacts": artifacts, "pid": os.getpid()})
        print("[WORKER] 地圖處理成功，結果檔已寫入。")
        return 0
    except BaseException as exc:
        trace = traceback.format_exc()
        try:
            _atomic_write_json(
                result_file,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": trace,
                    "pid": os.getpid(),
                },
            )
        except Exception as write_exc:
            print(f"[WORKER] 無法寫入錯誤結果：{write_exc}", file=sys.stderr)
        print(trace, file=sys.stderr)
        return 1
    finally:
        _best_effort_gpu_cleanup(module)
        print("[WORKER] 程序即將退出；作業系統會完整回收此程序的 CUDA context。")


if __name__ == "__main__":
    raise SystemExit(main())
