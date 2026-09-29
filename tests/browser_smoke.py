"""Optional Playwright smoke test. Model and LLM responses are fixtures."""
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from unittest.mock import patch
LOCAL_BROWSER_ROOT = "/home/chh/.local/share/playwright-libs/root"
os.environ.setdefault("PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS", "1")
os.environ.setdefault("FONTCONFIG_SYSROOT", LOCAL_BROWSER_ROOT)
os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
os.environ.setdefault("BROWSER_LIBRARY_PATH", LOCAL_BROWSER_ROOT + "/usr/lib/x86_64-linux-gnu")

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright, expect
from starlette.staticfiles import StaticFiles
import uvicorn

from app import server
import navigation


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weight = root / "best.pt"
        weight.touch()
        source = root / "floor.png"
        image = Image.new("RGB", (320, 200), "#eee9df")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 130, 80), fill="#b6cbbf", outline="#555555", width=3)
        draw.rectangle((190, 20, 300, 80), fill="#c8bfd7", outline="#555555", width=3)
        draw.text((60, 45), "A", fill="black")
        draw.text((240, 45), "B", fill="black")
        image.save(source)
        for route in server.app.routes:
            if getattr(route, "path", None) == "/uploads":
                route.app = StaticFiles(directory=root)

        def process(room_id, image_path):
            folder = root / room_id
            folder.mkdir(exist_ok=True)
            np.savetxt(folder / "map_matrix.csv", np.full((200, 320), 2), fmt="%d", delimiter=",")
            (folder / "room_data.json").write_text(json.dumps({"2": {"portal": True}, "3": {"names": ["A"]}, "4": {"names": ["B"]}}))
            navigation.ROOMS[room_id].update(status="ready", image_url=f"/uploads/{image_path.name}", csv_path=str(folder / "map_matrix.csv"), json_path=str(folder / "room_data.json"), navigation_path=str(folder / "room_data.json"))

        patches = [patch.object(navigation, "camera_warmup_callback", None), patch.object(navigation, "UPLOAD_DIR", root), patch.object(navigation, "YOLO_MODEL_PATH", weight), patch.object(navigation, "process_map_background", process), patch.object(navigation, "get_user_location", return_value={"current_room_id": "3", "destination_id": "4"}), patch.object(navigation.IndoorNavigator, "generate_llm_guidance", return_value=("從 A 向右走到 B。", None, [[70, 100], [250, 100]])), patch.object(server, "SAVE_ROOT", root / "saved"), patch.object(server, "recognize_image", return_value=[{"box": [10, 10, 100, 60], "text": "A", "ocr_score": 0.99, "detector_label": "sign", "detector_score": 0.9, "lines": []}])]
        for item in patches: item.start()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        app = uvicorn.Server(uvicorn.Config(server.app, log_level="error"))
        thread = threading.Thread(target=lambda: app.run(sockets=[sock]), daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if app.started: break
                time.sleep(.05)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(env={**os.environ, "LD_LIBRARY_PATH": os.getenv("BROWSER_LIBRARY_PATH", "")}, args=["--no-sandbox", "--disable-dev-shm-usage", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
                context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, permissions=["camera"])
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{port}")
                expect(page.locator("#frameInterval")).to_have_value("1000")
                expect(page.locator("#captureWidth")).to_have_value("768")
                page.click("#navigationOnlyButton")
                page.click("#createRoomButton")
                expect(page.locator("#roomCode")).to_contain_text("房間")
                page.set_input_files("#mapInput", str(source))
                expect(page.locator("#mapStatus")).to_contain_text("地圖已就緒", timeout=20000)
                expect(page.locator("#map-image")).to_be_visible()
                page.click("#chatButton")
                page.fill("#chatInput", "我在 A，要去 B")
                page.click("#chatSendButton")
                expect(page.locator("#chatMessages")).to_contain_text("從 A 向右走到 B。")
                page.click("#mapButton")
                expect(page.locator("#path-svg polyline")).to_have_count(2)
                expect(page.locator("#path-svg polyline").nth(1)).to_have_attribute("points", "70,100 250,100")
                expect(page.locator("#navigation-guidance")).to_be_visible()
                expect(page.locator("#navigation-step-text")).to_contain_text("向右")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                page.screenshot(animations="disabled", path="/home/chh/mapandllm-v2/test-artifacts/mapandllm-v2-map.png")
                # Verify room sharing from the invitation URL in a second browser.
                other = context.new_page()
                other.goto(page.url)
                other.click("#navigationOnlyButton")
                expect(other.locator("#mapStatus")).to_contain_text("地圖已就緒", timeout=15000)
                other.close()
                page.click("#stopButton")
                page.click("#startButton")
                expect(page.locator("#status")).to_have_text("A", timeout=20000)
                page.click("#chatButton")
                page.fill("#chatInput", "去 B")
                with page.expect_request("**/api/chat") as req:
                    page.click("#chatSendButton")
                assert req.value.post_data_json["recognized_stores"] == ["A"]
                expect(page.locator("#chatMessages .pending")).to_have_count(0)
                page.uncheck("#useOcrContext")
                page.fill("#chatInput", "A 到 B")
                with page.expect_request("**/api/chat") as req:
                    page.click("#chatSendButton")
                assert req.value.post_data_json["recognized_stores"] == []
                expect(page.locator("#chatMessages .pending")).to_have_count(0)
                page.screenshot(animations="disabled", path="/home/chh/mapandllm-v2/test-artifacts/mapandllm-v2-chat.png")
                page.click("#stopButton")
                assert not errors, errors
                print("Browser smoke OK: mobile map, rooms, upload, path, camera, OCR chat context and opt-out; no JS errors")
                browser.close()
        finally:
            app.should_exit = True
            thread.join(timeout=10)
            for item in reversed(patches): item.stop()


if __name__ == "__main__":
    main()
