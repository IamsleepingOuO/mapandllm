#!/usr/bin/env python3
"""Directly test saved OCR JSON -> map place -> stored user position."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR = ROOT / "saved/results/2026-09-21/20260921_141704_387869_f000147_33fed117.json"
DEFAULT_MAP = ROOT / "uploads/ead79a04-2b47-496a-9756-9d0142482fc3/navigation_data.json"


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing shell overrides."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def validate_result(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    if result.get("status") != "located":
        return [f"定位狀態不是 located，而是 {result.get('status')!r}"]
    if args.expect_id and str(result.get("place_id")) != args.expect_id:
        errors.append(f"place_id 預期 {args.expect_id!r}，實際 {result.get('place_id')!r}")
    if args.expect_name and str(result.get("place_name", "")).casefold() != args.expect_name.casefold():
        errors.append(f"place_name 預期 {args.expect_name!r}，實際 {result.get('place_name')!r}")
    for axis in ("x", "y"):
        expected = getattr(args, f"expect_{axis}")
        if expected is None:
            continue
        try:
            actual = float(result.get(axis))
        except (TypeError, ValueError):
            errors.append(f"{axis} 不是有效座標：{result.get(axis)!r}")
            continue
        if abs(actual - expected) > args.tolerance:
            errors.append(f"{axis} 預期 {expected}±{args.tolerance}，實際 {actual}")
    if args.require_llm and result.get("match_source") != "llm":
        errors.append(f"要求 LLM 成功判斷，但 match_source 是 {result.get('match_source')!r}")
    return errors


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="以保存的 OCR JSON 與地圖 JSON 實測相片定位。")
    value.add_argument("--ocr", type=Path, default=DEFAULT_OCR, help="OCR 結果 JSON")
    value.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP, help="navigation_data.json")
    value.add_argument("--model", help="覆蓋 LLM_MODEL")
    value.add_argument("--cpu", action="store_true", help="設定 LLM_NUM_GPU=0")
    value.add_argument("--require-llm", action="store_true", help="精確名稱備援成功仍視為失敗")
    value.add_argument("--expect-id", default="12")
    value.add_argument("--expect-name", default="PUMA")
    value.add_argument("--expect-x", type=float, default=2977.0)
    value.add_argument("--expect-y", type=float, default=433.0)
    value.add_argument("--tolerance", type=float, default=1.0)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_env(ROOT / ".env")
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.cpu:
        os.environ["LLM_NUM_GPU"] = "0"
    ocr_path, map_path = args.ocr.expanduser().resolve(), args.map_path.expanduser().resolve()
    missing = [str(path) for path in (ocr_path, map_path) if not path.is_file()]
    if missing:
        print("❌ 找不到測試檔案：\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    try:
        ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
        json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"❌ 無法讀取 JSON：{exc}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT))
    import navigation  # Settings must be overridden before this import.
    room_id = "terminal-photo-location-test"
    navigation.ROOMS[room_id] = {
        "status": "ready", "navigation_path": str(map_path), "users": {},
        "active_navigations": {}, "last_active": time.time(),
    }
    texts = [str(item.get("text", "")).strip() for item in ocr_data.get("detections", [])
             if isinstance(item, dict) and str(item.get("text", "")).strip()]
    print("=== 相片定位實測 ===")
    print(f"OCR JSON：{ocr_path}\n地圖 JSON：{map_path}")
    print(f"模型：{navigation.LLM_MODEL}（num_gpu={navigation.LLM_NUM_GPU_RAW or 'auto'}）")
    print("OCR 文字：" + (" | ".join(texts) if texts else "（沒有文字）"))
    started = time.perf_counter()
    result = navigation.locate_from_ocr(room_id, ocr_data, "terminal-test-user", "#6ee7a8")
    stored = navigation.ROOMS[room_id]["users"].get("terminal-test-user")
    print("\n定位回應：\n" + json.dumps(result, ensure_ascii=False, indent=2))
    print("\n實際寫入房間的位置：\n" + json.dumps(stored, ensure_ascii=False, indent=2))
    print(f"\n耗時：{time.perf_counter() - started:.2f} 秒")
    errors = validate_result(result, args)
    if result.get("status") == "located" and stored is None:
        errors.append("定位成功，但房間 users 沒有寫入測試使用者")
    if errors:
        print("\n❌ 測試失敗：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if result.get("match_source") != "llm":
        print(f"\n⚠️ 定位成功，但使用 {result.get('match_source')}；加 --require-llm 可強制驗證 LLM。")
    print("\n✅ 測試通過：定位正確，而且使用者座標已寫入房間。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
