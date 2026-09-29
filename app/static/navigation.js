"use strict";
// Shared only with the original PDR engine.
let currentRoomId = null;
let collisionMatrix = null;
let isMapReady = false;
let myPosition = null;
const myUserId = `user_${crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`;
const myColor = "#6ee7a8";

(() => {
  const el = id => document.getElementById(id);
  let inviteCode = "";
  let timer = null;
  let generation = 0;
  let imageUrl = null;
  let roomBusy = false;
  let pendingPath = null;
  let navigationSteps = [];
  let activeDestination = null;

  function status(message, error = false) {
    el("mapStatus").textContent = message;
    el("mapStatus").classList.toggle("error", error);
  }
  function showNavigationStep(step) {
    const panel = el("navigation-guidance");
    if (!step) { panel.hidden = true; return; }
    const index = Number.isInteger(step.active_index) ? step.active_index : step.index;
    el("navigation-step-count").textContent = `步驟 ${index + 1} / ${step.total || navigationSteps.length}`;
    el("navigation-step-text").textContent = step.instruction;
    panel.hidden = false;
  }
  function updateGuidanceForPosition(position) {
    if (!position || !navigationSteps.length) return;
    let best = null;
    navigationSteps.slice(0, -1).forEach((step, index) => {
      const [ax, ay] = step.start, [bx, by] = step.end;
      const dx = bx - ax, dy = by - ay, lengthSq = dx * dx + dy * dy;
      const progress = lengthSq ? Math.max(0, Math.min(1, ((position.x-ax)*dx + (position.y-ay)*dy) / lengthSq)) : 0;
      const px = ax + progress * dx, py = ay + progress * dy;
      const distanceSq = (position.x-px) ** 2 + (position.y-py) ** 2;
      if (!best || distanceSq < best.distanceSq) best = {distanceSq, index, progress};
    });
    let index = best ? best.index + (best.progress >= .85 ? 1 : 0) : 0;
    index = Math.min(index, navigationSteps.length - 1);
    showNavigationStep({...navigationSteps[index], active_index: index, total: navigationSteps.length});
  }
  window.updateNavigationGuidance = updateGuidanceForPosition;

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 250000);
    try {
      const response = await fetch(path, {...options, signal: controller.signal, cache: "no-store"});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `請求失敗 (${response.status})`);
      return data;
    } finally { clearTimeout(timeout); }
  }
  const post = (path, data) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});

  function resetMap() {
    isMapReady = false;
    pendingPath = null;
    navigationSteps = [];
    activeDestination = null;
    showNavigationStep(null);
    imageUrl = null;
    resetNavigationTracking();
    el("map-wrapper").hidden = true;
    el("map-image").removeAttribute("src");
    el("mapEmpty").hidden = false;
  }
  async function enterRoom(data) {
    generation += 1;
    clearTimeout(timer);
    currentRoomId = data.room_id;
    inviteCode = data.invite_code;
    resetMap();
    el("roomCode").textContent = `房間 ${inviteCode}`;
    el("mapInput").disabled = false;
    el("copyRoomButton").disabled = false;
    el("chatMessages").replaceChildren();
    appendChatMessage("assistant", "已加入房間。請上傳地圖，或等候共享地圖載入後詢問方向。");
    history.replaceState({}, "", `/?room=${encodeURIComponent(inviteCode)}`);
    await poll(generation);
  }
  async function roomAction(action) {
    if (roomBusy) return;
    roomBusy = true;
    el("createRoomButton").disabled = el("joinRoomButton").disabled = true;
    try { await enterRoom(await action()); }
    catch (error) { status(error.message, true); }
    finally {
      roomBusy = false;
      el("createRoomButton").disabled = el("joinRoomButton").disabled = false;
    }
  }
  async function poll(version) {
    if (!currentRoomId || version !== generation) return;
    try {
      const data = await api(`/api/room_status/${currentRoomId}`);
      if (version !== generation) return;
      el("mapInput").disabled = data.status === "processing";
      if (data.status === "ready") {
        isMapReady = true;
        status("地圖已就緒，可到聊天面板規劃路線");
        if (data.image_url !== imageUrl) {
          resetNavigationTracking();
          pendingPath = null;
          imageUrl = data.image_url;
          el("map-image").src = imageUrl;
          el("map-wrapper").hidden = false;
          el("mapEmpty").hidden = true;
        }
        if (!collisionMatrix) {
          const response = await fetch(`/uploads/${currentRoomId}/map_matrix.csv`, {cache: "no-store"});
          if (!response.ok) throw new Error("無法載入定位矩陣");
          const text = await response.text();
          if (version !== generation) return;
          const matrix = text.trim().split(/\r?\n/).map(row => row.split(",").map(Number));
          if (!matrix.length || !matrix[0].length || matrix.some(row => row.length !== matrix[0].length || row.some(v => !Number.isFinite(v)))) throw new Error("定位矩陣格式錯誤");
          collisionMatrix = matrix;
        }
        for (const [id, user] of Object.entries(data.users || {})) {
          if (id !== myUserId && Date.now() / 1000 - user.last_update < 30) updateDotUI(id, user.x, user.y, user.color);
        }
      } else {
        if (isMapReady || imageUrl) resetMap();
        status(data.status === "processing" ? "正在解析地圖…" : data.status === "error" ? `地圖解析失敗：${data.error || "請重試"}` : "請上傳地圖", data.status === "error");
      }
    } catch (error) {
      if (version === generation) { isMapReady = false; status(error.message, true); }
    } finally {
      if (version === generation) timer = setTimeout(() => void poll(version), 2500);
    }
  }

  el("createRoomButton").addEventListener("click", () => roomAction(() => post("/api/create_room", {})));
  const join = code => roomAction(() => post("/api/join_room", {code_or_id: code}));
  el("joinRoomButton").addEventListener("click", () => {
    const code = el("inviteCode").value.trim();
    if (code) void join(code); else status("請輸入邀請碼", true);
  });
  el("copyRoomButton").addEventListener("click", async () => {
    const url = `${location.origin}/?room=${encodeURIComponent(inviteCode)}`;
    try { await navigator.clipboard.writeText(url); status("邀請連結已複製"); }
    catch { status(`邀請連結：${url}`); }
  });
  el("mapInput").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file || !currentRoomId) return;
    if (file.size > 20 * 1024 * 1024) { status("地圖限制 20 MiB", true); event.target.value = ""; return; }
    const form = new FormData();
    form.append("file", file);
    form.append("room_id", currentRoomId);
    // Invalidate any status request for the previous map before upload.
    const version = ++generation;
    clearTimeout(timer);
    el("mapInput").disabled = true;
    status("地圖上傳中…");
    try {
      await api("/api/upload", {method: "POST", body: form});
      if (version !== generation) return;
      resetMap();
      status("正在解析地圖…");
    } catch (error) {
      if (version === generation) status(error.message, true);
    } finally {
      event.target.value = "";
      if (version === generation) {
        el("mapInput").disabled = false;
        // Leave the upload error visible before polling resumes.
        timer = setTimeout(() => void poll(version), 5000);
      }
    }
  });

  el("map-image").addEventListener("load", () => {
    if (pendingPath) drawPathOnMap(pendingPath);
  });
  el("map-image").addEventListener("error", () => status("地圖圖片載入失敗", true));
  window.navigationChat = async (message, stores) => {
    if (!currentRoomId) throw new Error("請先到地圖面板建立或加入房間");
    if (!isMapReady) throw new Error("請先上傳地圖並等候解析完成");
    const version = generation;
    const data = await post("/api/chat", {message, room_id: currentRoomId, recognized_stores: stores, user_id: myUserId});
    if (version !== generation) throw new Error("地圖或房間已變更，請重新詢問");
    if (data.path_coords?.length) {
      pendingPath = data.path_coords;
      navigationSteps = data.navigation_steps || [];
      activeDestination = data.resolved_end_id || activeDestination;
      showNavigationStep(navigationSteps.length ? {...navigationSteps[0], active_index: 0, total: navigationSteps.length} : null);
      if (el("map-image").complete && el("map-image").naturalWidth) drawPathOnMap(pendingPath);
      calculateAngleFromPath(pendingPath);
      myPosition = {x: pendingPath[0][0], y: pendingPath[0][1]};
      updateDotUI(myUserId, myPosition.x, myPosition.y, myColor);
      void syncPosition();
      calibStartPos = {...myPosition};
      calibrationState = 1;
      isTracking = false;
      hasCalibratedOffset = false;
      status("路線已繪製；可在地圖完成兩點步行校正");
    }
    return data.reply || "沒有收到回覆";
  };
  window.navigationLocateFromOcr = async captureId => {
    if (!currentRoomId || !isMapReady) return {status: "map_not_ready"};
    const roomId = currentRoomId;
    const version = generation;
    const result = await post("/api/locate_from_ocr", {
      room_id: roomId, user_id: myUserId, color: myColor, capture_id: captureId
    });
    if (version !== generation || roomId !== currentRoomId) return {status: "map_not_ready"};
    if (result.status === "located") {
      myPosition = {x: result.x, y: result.y};
      const image = el("map-image");
      if (image.complete && image.naturalWidth) updateDotUI(myUserId, result.x, result.y, myColor);
      else image.addEventListener("load", () => updateDotUI(myUserId, result.x, result.y, myColor), {once: true});
      // A camera fix advances guidance but never replaces the route destination.
      if (result.current_step) showNavigationStep(result.current_step);
      else updateGuidanceForPosition(myPosition);
      status(`相機定位：${result.place_name}${activeDestination ? "；目的地保持不變" : ""}`);
    } else if (result.status === "full") {
      status("房間已達兩人定位上限", true);
    }
    return result;
  };
  window.syncPosition = async () => {
    if (!currentRoomId || !myPosition) return;
    try {
      const data = await post(`/api/update_position/${currentRoomId}`, {user_id: myUserId, ...myPosition, color: myColor});
      if (data.status === "full") status("房間已達兩人定位上限", true);
    } catch (error) { status(error.message, true); }
  };
  // All scripts, including the PDR engine, are initialized before joining.
  window.addEventListener("load", () => {
    const code = new URLSearchParams(location.search).get("room");
    if (code) void join(code);
  });
  window.addEventListener("pagehide", () => { clearTimeout(timer); generation += 1; isTracking = false; });
  window.addEventListener("pageshow", event => { if (event.persisted && currentRoomId) void poll(generation); });
})();
