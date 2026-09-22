"""Contract tests without model downloads or an Ollama server."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

import navigation
from app import server


def picture():
    out = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(out, "JPEG")
    return out.getvalue()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = TestClient(server.app)
        navigation.ROOMS.clear()
        navigation.CODE_TO_UUID.clear()
        with patch.object(navigation, "camera_warmup_callback", None):
            self.room = self.client.post("/api/create_room").json()

    def tearDown(self):
        self.temp.cleanup()

    def test_room_creation_schedules_warmup_without_waiting(self):
        with patch.object(navigation, "camera_warmup_callback") as callback:
            response = self.client.post("/api/create_room")
        self.assertEqual(response.status_code, 200)
        callback.assert_called_once_with()

    def test_warmup_starts_once_and_reports_ready(self):
        class FakePipeline:
            def __init__(self):
                self.calls = 0

            def warmup(self):
                self.calls += 1

        pipeline = FakePipeline()
        with patch.object(server, "_warmup_status", "idle"), patch.object(server, "get_pipeline", return_value=pipeline), patch.object(server.threading, "Thread") as thread:
            server.request_camera_warmup()
            server.request_camera_warmup()
            thread.assert_called_once()
            self.assertTrue(thread.call_args.kwargs["daemon"])
            thread.call_args.kwargs["target"]()
            self.assertEqual(server.health()["camera_warmup_status"], "ready")
            self.assertEqual(pipeline.calls, 1)

    def test_homepage_and_health_without_models(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/health").json()["version"], "2.0.0")
        for file in ("app.js", "navigation.js", "pdr_engine.js", "style.css"):
            self.assertEqual(self.client.get("/static/" + file).status_code, 200)

    def test_join_uuid_and_case_insensitive_code(self):
        for value in (self.room["room_id"], self.room["invite_code"].lower()):
            response = self.client.post("/api/join_room", json={"code_or_id": value})
            self.assertEqual(response.json()["room_id"], self.room["room_id"])
        self.assertEqual(self.client.post("/api/join_room", json={"code_or_id": "missing"}).status_code, 404)

    def test_missing_model_is_actionable(self):
        with patch.object(navigation, "YOLO_MODEL_PATH", self.root / "missing.pt"):
            response = self.client.post("/api/upload", data={"room_id": self.room["room_id"]}, files={"file": ("map.jpg", picture(), "image/jpeg")})
        self.assertEqual(response.status_code, 503)
        self.assertIn("YOLO_MODEL_PATH", response.json()["detail"])

    def test_upload_validation_and_processing_state(self):
        weight = self.root / "best.pt"
        weight.touch()
        with patch.object(navigation, "YOLO_MODEL_PATH", weight), patch.object(navigation, "UPLOAD_DIR", self.root), patch.object(navigation, "process_map_background") as task:
            def upload(data, mime="image/jpeg"):
                return self.client.post("/api/upload", data={"room_id": self.room["room_id"]}, files={"file": ("../../map.jpg", data, mime)})
            self.assertEqual(upload(b"invalid").status_code, 400)
            self.assertEqual(upload(b"text", "text/plain").status_code, 415)
            with patch.object(navigation, "MAX_MAP_BYTES", 8):
                self.assertEqual(upload(picture()).status_code, 413)
            self.assertEqual(upload(picture()).status_code, 200)
            task.assert_called_once()
            self.assertEqual(task.call_args.args[1].parent, self.root)
            self.assertEqual(upload(picture()).status_code, 409)
            state = self.client.get("/api/room_status/" + self.room["room_id"]).json()
            self.assertEqual(state["status"], "processing")

    def test_ocr_success_saves_capture_and_result(self):
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(server, "recognize_image", return_value=[]):
            response = self.client.post("/api/recognize", files={"image": ("test.jpg", picture(), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(list(self.root.glob("captures/*/*.jpg"))), 1)
        saved = json.loads(next(self.root.glob("results/*/*.json")).read_text())
        self.assertTrue(saved["ok"])
        self.assertIn("timings", saved)
        self.assertIn("inference_total_ms", saved["timings"])
        self.assertIn("annotation_ms", saved["timings"])

    def test_ocr_timing_contract_with_model_stages(self):
        timings = {"dino_total_ms": 12.0, "ocr_total_ms": 8.0,
                   "ocr_per_crop_ms": [8.0], "ocr_crop_count": 1,
                   "max_detections": 3, "dino_input_width": 1067,
                   "dino_input_height": 600}
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(
            server, "recognize_image", return_value=([], timings)):
            response = self.client.post("/api/recognize", files={"image": ("test.jpg", picture(), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        saved = json.loads(next(self.root.glob("results/*/*.json")).read_text())
        self.assertEqual(saved["timings"]["max_detections"], 3)
        self.assertEqual(saved["timings"]["dino_input_width"], 1067)
        self.assertEqual(saved["timings"]["ocr_per_crop_ms"], [8.0])

    def test_ocr_failure_still_saves_capture(self):
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(server, "recognize_image", side_effect=RuntimeError("model unavailable")):
            response = self.client.post("/api/recognize", files={"image": ("test.jpg", picture(), "image/jpeg")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(list(self.root.glob("captures/*/*.jpg"))), 1)
        saved = json.loads(next(self.root.glob("results/*/*.json")).read_text())
        self.assertFalse(saved["ok"])

    def test_ocr_upload_limit(self):
        with patch.object(server, "MAX_UPLOAD_BYTES", 8):
            response = self.client.post("/api/recognize", files={"image": ("test.jpg", picture(), "image/jpeg")})
        self.assertEqual(response.status_code, 413)

    def prepare_map(self):
        path = self.root / "navigation_data.json"
        payload = {"map": {"scale": "1 pixel = 1 meters"}, "places": {"2": {"display_name": "Alpha", "names": ["A", "Alpha"], "aliases": [], "attachment_node": "W1"}, "3": {"display_name": "Bravo", "names": ["B", "Bravo"], "aliases": [], "attachment_node": "W4"}}, "llm_context": [{"id": "2", "names": ["A"]}, {"id": "3", "names": ["B"]}], "graph": {"nodes": {"W1": {"coordinates": [0, 0]}, "W2": {"coordinates": [10, 10]}, "W3": {"coordinates": [10, 0]}, "W4": {"coordinates": [20, 0]}}, "edges": [{"source": "W1", "target": "W2", "distance_px": 20}, {"source": "W2", "target": "W4", "distance_px": 20}, {"source": "W1", "target": "W3", "distance_px": 20.2}, {"source": "W3", "target": "W4", "distance_px": 20.2}]}}
        path.write_text(json.dumps(payload))
        navigation.ROOMS[self.room["room_id"]].update(status="ready", navigation_path=str(path), json_path=str(path))

    def test_chat_passes_ocr_context_and_returns_route(self):
        self.prepare_map()
        with patch.object(navigation, "get_user_location", return_value={"current_room_id": "2", "destination_id": "3"}) as locate, patch.object(navigation, "IndoorNavigator") as nav:
            nav.return_value.generate_llm_guidance.return_value = ("向右走", None, [[10, 20], [30, 20]])
            result = self.client.post("/api/chat", json={"room_id": self.room["room_id"], "message": "我要去 B", "recognized_stores": [" A "]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(locate.call_args.args[2], ["A"])
        self.assertEqual(result.json()["path_coords"], [[10, 20], [30, 20]])

    def test_llm_prompt_and_invalid_location(self):
        with patch.object(navigation.ollama, "generate", return_value={"response": '{"current_room_id":"999", "destination_id":"3"}'}) as generate:
            loc = navigation.get_user_location("去 B", {"2": {}, "3": {}}, ["A"])
        self.assertIsNone(loc["current_room_id"])
        self.assertIn('相機辨識文字', generate.call_args.kwargs["prompt"])

    def test_llm_unavailable_uses_explicit_local_names(self):
        self.prepare_map()
        with patch.object(navigation.ollama, "generate", side_effect=RuntimeError("offline")):
            result = self.client.post("/api/chat", json={"room_id": self.room["room_id"], "message": "Alpha 到 Bravo"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual((result.json()["resolved_start_id"], result.json()["resolved_end_id"]), ("2", "3"))

    def test_saved_ocr_locates_user_from_map_json(self):
        self.prepare_map()
        map_path = Path(navigation.ROOMS[self.room["room_id"]]["navigation_path"])
        map_data = json.loads(map_path.read_text())
        map_data["map"].update(image_width=100, image_height=100)
        map_data["places"]["2"]["attachment_point"] = [12, 23]
        map_path.write_text(json.dumps(map_data))
        capture_id = "20260922_120000_123456_f000001_abcdef12"
        result_path = self.root / "results" / "2026-09-22" / (capture_id + ".json")
        result_path.parent.mkdir(parents=True)
        result_path.write_text(json.dumps({"capture_id": capture_id, "ok": True,
            "detections": [{"text": "Alpha", "ocr_score": 0.98, "detector_score": 0.9}]}))
        body = {"room_id": self.room["room_id"], "user_id": "camera-user",
                "color": "#6ee7a8", "capture_id": capture_id}
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(
            navigation.ollama, "generate", return_value={"response": '{"place_id":"2"}'}) as generate:
            response = self.client.post("/api/locate_from_ocr", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "located")
        self.assertEqual((response.json()["x"], response.json()["y"]), (12, 23))
        self.assertIn("Alpha", generate.call_args.kwargs["prompt"])
        self.assertIn("地圖 JSON", generate.call_args.kwargs["prompt"])
        self.assertEqual(navigation.ROOMS[self.room["room_id"]]["users"]["camera-user"]["x"], 12)
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(
            navigation.ollama, "generate", return_value={"response": '{"place_id":"999"}'}):
            invalid = self.client.post("/api/locate_from_ocr", json=body)
        self.assertEqual(invalid.json()["status"], "no_match")
        self.assertEqual(navigation.ROOMS[self.room["room_id"]]["users"]["camera-user"]["x"], 12)
        with patch.object(server, "SAVE_ROOT", self.root), patch.object(
            navigation.ollama, "generate", side_effect=RuntimeError("offline")):
            offline = self.client.post("/api/locate_from_ocr", json=body)
        self.assertEqual(offline.json()["status"], "llm_unavailable")
        self.assertEqual(self.client.post("/api/locate_from_ocr", json={**body,
            "capture_id": "../secrets"}).status_code, 422)

    def test_two_user_position_limit_and_invalid_coordinates(self):
        path = "/api/update_position/" + self.room["room_id"]
        for uid in ("a", "b"):
            self.assertEqual(self.client.post(path, json={"user_id": uid, "x": 1, "y": 2, "color": "#ffffff"}).json()["status"], "ok")
        self.assertEqual(self.client.post(path, json={"user_id": "c", "x": 1, "y": 2, "color": "#ffffff"}).json()["status"], "full")
        self.assertEqual(self.client.post(path, json={"user_id": "a", "x": -1, "y": 2, "color": "#ffffff"}).status_code, 422)

    def test_topology_route_prefers_fewer_turns_within_detour_limit(self):
        self.prepare_map()
        path = navigation.ROOMS[self.room["room_id"]]["navigation_path"]
        nav = navigation.IndoorNavigator(path, self.room["room_id"])
        selected = nav.shortest_path("W1", "W4")
        self.assertEqual(selected, ["W1", "W3", "W4"])
        self.assertEqual(nav.last_path_stats["strategy"], "near_shortest_min_turns")
        self.assertLessEqual(nav.last_path_stats["selected_distance_px"], 40 * 1.03 + 5)


if __name__ == "__main__":
    unittest.main()
