"use strict";

const MOTION_SAMPLE_WIDTH = 32;
const MOTION_SAMPLE_HEIGHT = 24;
const MOTION_THRESHOLD = 6;
const MAX_STILL_REFRESH_MS = 10000;
const JPEG_QUALITY = 0.78;

const launchScreen = document.getElementById("launchScreen");
const cameraApp = document.getElementById("cameraApp");
const video = document.getElementById("cameraVideo");
const cameraStage = document.getElementById("cameraStage");
const captureCanvas = document.getElementById("captureCanvas");
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayContext = overlayCanvas.getContext("2d");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const captureButton = document.getElementById("captureButton");
const settingsButton = document.getElementById("settingsButton");
const frameIntervalSelect = document.getElementById("frameInterval");
const captureWidthSelect = document.getElementById("captureWidth");
const motionCanvas = document.createElement("canvas");
motionCanvas.width = MOTION_SAMPLE_WIDTH;
motionCanvas.height = MOTION_SAMPLE_HEIGHT;
const motionContext = motionCanvas.getContext("2d", {willReadFrequently: true});
const autoRecognizeCheckbox = document.getElementById("autoRecognize");
const launchStatusElement = document.getElementById("launchStatus");
const statusPill = document.getElementById("statusPill");
const statusElement = document.getElementById("status");
const recognitionToast = document.getElementById("recognitionToast");
const recognitionText = document.getElementById("recognitionText");

const mapButton = document.getElementById("mapButton");
const chatButton = document.getElementById("chatButton");
const mapPanel = document.getElementById("mapPanel");
const chatPanel = document.getElementById("chatPanel");
const settingsPanel = document.getElementById("settingsPanel");

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const chatSendButton = document.getElementById("chatSendButton");

let stream = null;
let running = false;
let frameNumber = 0;
let callbackId = null;
let fallbackLastVideoTime = -1;
let requestInFlight = false;
let requestSequence = 0;
let lastResponse = null;
let lastRecognitionAt = 0;
let lastAutoCaptureAt = 0;
let lastAnalyzedSample = null;
let lastCaptureStartedAt = 0;
let locationTimer = null;
let locationRequestInFlight = false;
let chatRequestInFlight = false;

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
captureButton.addEventListener("click", () => void captureAndRecognize(true));
settingsButton.addEventListener("click", () => toggleSettingsPanel());
mapButton.addEventListener("click", () => toggleMainPanel("map"));
chatButton.addEventListener("click", () => toggleMainPanel("chat"));
chatSendButton.addEventListener("click", () => void submitChat());
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void submitChat();
  }
});
chatInput.addEventListener("input", autoResizeChatInput);

document.querySelectorAll("[data-close-panel]").forEach((button) => {
  button.addEventListener("click", () => closePanel(button.dataset.closePanel));
});

window.addEventListener("pagehide", stopCamera);
window.addEventListener("resize", redrawLastResult);
window.addEventListener("orientationchange", () => setTimeout(redrawLastResult, 150));

async function startCamera() {
  if (running) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setLaunchStatus("此瀏覽器不支援相機 API，或目前頁面不是安全環境。", true);
    return;
  }

  startButton.disabled = true;
  setLaunchStatus("正在要求相機權限……");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });

    video.srcObject = stream;
    await waitForVideoMetadata();
    await video.play();

    running = true;
    captureButton.disabled = false;
    frameNumber = 0;
    fallbackLastVideoTime = -1;
    lastAutoCaptureAt = performance.now();
    lastAnalyzedSample = null;
    lastCaptureStartedAt = 0;
    locationTimer = setInterval(() => void locateFromLatestOcr(), 5000);

    document.body.classList.add("camera-active");
    cameraApp.setAttribute("aria-hidden", "false");
    launchScreen.setAttribute("aria-hidden", "true");

    const settings = stream.getVideoTracks()[0].getSettings();
    setStatus(
      `相機 ${settings.width ?? video.videoWidth}×${settings.height ?? video.videoHeight}`,
      "ready"
    );

    scheduleNextFrame();
    setTimeout(redrawLastResult, 80);
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    setLaunchStatus(cameraErrorMessage(error), true);
  }
}

function stopCamera() {
  running = false;
  requestSequence += 1;
  lastRecognitionAt = 0;
  lastAnalyzedSample = null;
  lastCaptureStartedAt = 0;
  clearInterval(locationTimer);
  locationTimer = null;

  if (callbackId !== null) {
    if (typeof video.cancelVideoFrameCallback === "function") {
      video.cancelVideoFrameCallback(callbackId);
    } else {
      cancelAnimationFrame(callbackId);
    }
  }
  callbackId = null;

  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }

  stream = null;
  video.srcObject = null;
  lastResponse = null;
  clearDetectionOverlay();
  closeAllPanels();

  document.body.classList.remove("camera-active");
  cameraApp.setAttribute("aria-hidden", "true");
  launchScreen.setAttribute("aria-hidden", "false");
  startButton.disabled = false;
  setLaunchStatus("相機已停止");
}

function waitForVideoMetadata() {
  if (video.videoWidth > 0 && video.readyState >= 1) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("等待相機影像逾時")),
      10000
    );

    video.addEventListener("loadedmetadata", () => {
      clearTimeout(timeout);
      resolve();
    }, { once: true });
  });
}

function scheduleNextFrame() {
  if (!running) return;

  if (typeof video.requestVideoFrameCallback === "function") {
    callbackId = video.requestVideoFrameCallback(onVideoFrame);
  } else {
    callbackId = requestAnimationFrame(onAnimationFrame);
  }
}

function onVideoFrame() {
  if (!running) return;

  frameNumber += 1;
  maybeAutoCapture();

  scheduleNextFrame();
}

function onAnimationFrame() {
  if (!running) return;

  if (video.currentTime !== fallbackLastVideoTime) {
    fallbackLastVideoTime = video.currentTime;
    frameNumber += 1;

    maybeAutoCapture();
  }

  scheduleNextFrame();
}

function maybeAutoCapture() {
  if (!autoRecognizeCheckbox.checked || requestInFlight) return;
  const intervalMs = Math.max(100, Number(frameIntervalSelect.value) || 1000);
  const now = performance.now();
  if (now - lastAutoCaptureAt >= intervalMs) {
    lastAutoCaptureAt = now;
    void captureAndRecognize(false);
  }
}

async function locateFromLatestOcr() {
  if (!running || locationRequestInFlight || !lastResponse?.capture_id ||
      !document.getElementById("useOcrContext").checked ||
      Date.now() - lastRecognitionAt > 60000 ||
      typeof window.navigationLocateFromOcr !== "function") return;
  locationRequestInFlight = true;
  const captureId = lastResponse.capture_id;
  try {
    const result = await window.navigationLocateFromOcr(captureId);
    if (result?.status === "located") {
      setStatus(`已定位：${result.place_name}`, "ready");
    } else if (result?.status === "llm_unavailable") {
      setStatus("定位比對暫時無法使用", "error");
    }
  } catch (error) {
    console.error("OCR 定位失敗", error);
  } finally {
    locationRequestInFlight = false;
  }
}

function sampleCameraFrame() {
  motionContext.drawImage(video, 0, 0, MOTION_SAMPLE_WIDTH, MOTION_SAMPLE_HEIGHT);
  const rgba = motionContext.getImageData(0, 0, MOTION_SAMPLE_WIDTH, MOTION_SAMPLE_HEIGHT).data;
  const gray = new Uint8Array(MOTION_SAMPLE_WIDTH * MOTION_SAMPLE_HEIGHT);
  for (let i = 0; i < gray.length; i++) {
    const offset = i * 4;
    gray[i] = (rgba[offset] * 3 + rgba[offset + 1] * 6 + rgba[offset + 2]) / 10;
  }
  return gray;
}

function frameHasChanged(sample) {
  if (!lastAnalyzedSample || lastAnalyzedSample.length !== sample.length) return true;
  let difference = 0;
  for (let i = 0; i < sample.length; i++) {
    difference += Math.abs(sample[i] - lastAnalyzedSample[i]);
  }
  return difference / sample.length >= MOTION_THRESHOLD;
}

async function captureAndRecognize(force = false) {
  // 前一張還在推論時不排隊，避免手機畫面越跑越延遲。
  if (requestInFlight) {
    if (force) setStatus("上一張仍在辨識", "processing");
    return;
  }

  if (
    !running ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    video.videoWidth === 0
  ) {
    return;
  }

  const now = performance.now();
  let sample = null;
  try {
    sample = sampleCameraFrame();
  } catch (error) {
    console.warn("畫面變動偵測不可用，照常辨識", error);
  }
  if (!force && sample && !frameHasChanged(sample) &&
      now - lastCaptureStartedAt < MAX_STILL_REFRESH_MS) return;
  lastAnalyzedSample = sample;
  lastCaptureStartedAt = now;
  requestInFlight = true;
  captureButton.disabled = true;
  const sequence = ++requestSequence;

  try {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const configuredWidth = Number(captureWidthSelect.value) || 768;
    const width = Math.min(configuredWidth, sourceWidth);
    const height = Math.round(sourceHeight * width / sourceWidth);

    if (captureCanvas.width !== width || captureCanvas.height !== height) {
      captureCanvas.width = width;
      captureCanvas.height = height;
    }

    captureContext.drawImage(video, 0, 0, width, height);
    const blob = await canvasToBlob(captureCanvas, "image/jpeg", JPEG_QUALITY);

    setStatus(`辨識第 ${frameNumber} 幀…`, "processing");

    const formData = new FormData();
    formData.append("image", blob, `frame-${frameNumber}.jpg`);
    formData.append("frame_number", String(frameNumber));
    formData.append("captured_at", new Date().toISOString());

    const response = await fetch("/api/recognize", {
      method: "POST",
      body: formData,
      cache: "no-store"
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      const detail = typeof payload?.detail === "string"
        ? payload.detail
        : payload?.detail?.message ?? `HTTP ${response.status}`;
      throw new Error(detail);
    }

    if (sequence !== requestSequence) return;

    lastResponse = payload;
    lastRecognitionAt = Date.now();
    drawDetections(payload);
    updateRecognitionToast(payload);

    const names = payload.detections
      .map((d) => (d.text || "").trim())
      .filter(Boolean);

    if (names.length) {
      setStatus(names.slice(0, 2).join(" · "), "ready");
    } else {
      setStatus(`未讀到店名 · ${payload.processing_ms} ms`, "ready");
    }
  } catch (error) {
    console.error(error);
    setStatus(`辨識失敗：${error.message}`, "error");
  } finally {
    requestInFlight = false;
    captureButton.disabled = false;
  }
}

function drawDetections(payload) {
  const displayWidth = cameraStage.clientWidth;
  const displayHeight = cameraStage.clientHeight;
  const sourceWidth = payload.image_width;
  const sourceHeight = payload.image_height;

  if (!displayWidth || !displayHeight || !sourceWidth || !sourceHeight) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlayCanvas.width = Math.round(displayWidth * dpr);
  overlayCanvas.height = Math.round(displayHeight * dpr);
  overlayCanvas.style.width = `${displayWidth}px`;
  overlayCanvas.style.height = `${displayHeight}px`;

  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayContext.clearRect(0, 0, displayWidth, displayHeight);

  // video 使用 object-fit: cover；這裡用相同的縮放方式把辨識框對齊。
  const scale = Math.max(displayWidth / sourceWidth, displayHeight / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (displayWidth - renderedWidth) / 2;
  const offsetY = (displayHeight - renderedHeight) / 2;

  overlayContext.lineWidth = 2.5;
  overlayContext.strokeStyle = "#ff5d57";
  overlayContext.font = "600 14px system-ui, -apple-system, sans-serif";
  overlayContext.textBaseline = "middle";

  for (const detection of payload.detections) {
    const [sx1, sy1, sx2, sy2] = detection.box;
    const x1 = offsetX + sx1 * scale;
    const y1 = offsetY + sy1 * scale;
    const x2 = offsetX + sx2 * scale;
    const y2 = offsetY + sy2 * scale;

    const boxWidth = Math.max(1, x2 - x1);
    const boxHeight = Math.max(1, y2 - y1);

    overlayContext.strokeRect(x1, y1, boxWidth, boxHeight);

    const label = (detection.text || detection.detector_label || "sign").trim();
    const confidence = detection.text
      ? Math.round((detection.ocr_score || 0) * 100)
      : Math.round((detection.detector_score || 0) * 100);
    const text = confidence ? `${label} ${confidence}%` : label;

    const paddingX = 8;
    const labelHeight = 26;
    const textWidth = overlayContext.measureText(text).width;
    const labelWidth = Math.min(textWidth + paddingX * 2, displayWidth - Math.max(0, x1));
    const labelX = Math.max(0, Math.min(x1, displayWidth - labelWidth));
    const labelY = Math.max(0, y1 - labelHeight);

    overlayContext.fillStyle = "rgba(255, 93, 87, .92)";
    overlayContext.fillRect(labelX, labelY, labelWidth, labelHeight);
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(text, labelX + paddingX, labelY + labelHeight / 2 + 1);
  }
}

function redrawLastResult() {
  if (lastResponse && running) drawDetections(lastResponse);
}

function clearDetectionOverlay() {
  overlayContext.setTransform(1, 0, 0, 1, 0, 0);
  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  recognitionToast.hidden = true;
}

function updateRecognitionToast(payload) {
  const names = [...new Set(
    payload.detections
      .map((d) => (d.text || "").trim())
      .filter(Boolean)
  )];

  if (!names.length) {
    recognitionToast.hidden = true;
    return;
  }

  recognitionText.textContent = names.slice(0, 4).join(" · ");
  recognitionToast.hidden = false;
}

/* ---------- 地圖 / 聊天 / 設定 ---------- */
function toggleMainPanel(which) {
  settingsPanel.classList.remove("open");
  settingsPanel.setAttribute("aria-hidden", "true");

  const target = which === "map" ? mapPanel : chatPanel;
  const other = which === "map" ? chatPanel : mapPanel;
  const targetButton = which === "map" ? mapButton : chatButton;
  const otherButton = which === "map" ? chatButton : mapButton;
  const shouldOpen = !target.classList.contains("open");

  other.classList.remove("open");
  other.setAttribute("aria-hidden", "true");
  otherButton.setAttribute("aria-expanded", "false");

  target.classList.toggle("open", shouldOpen);
  target.setAttribute("aria-hidden", String(!shouldOpen));
  targetButton.setAttribute("aria-expanded", String(shouldOpen));

  if (which === "chat" && shouldOpen) {
    setTimeout(() => chatInput.focus(), 120);
  }
}

function closePanel(panelId) {
  const panel = document.getElementById(panelId);
  panel?.classList.remove("open");
  panel?.setAttribute("aria-hidden", "true");

  if (panelId === "mapPanel") mapButton.setAttribute("aria-expanded", "false");
  if (panelId === "chatPanel") chatButton.setAttribute("aria-expanded", "false");
}

function closeAllPanels() {
  closePanel("mapPanel");
  closePanel("chatPanel");
  settingsPanel.classList.remove("open");
  settingsPanel.setAttribute("aria-hidden", "true");
}

function toggleSettingsPanel() {
  closePanel("mapPanel");
  closePanel("chatPanel");

  const shouldOpen = !settingsPanel.classList.contains("open");
  settingsPanel.classList.toggle("open", shouldOpen);
  settingsPanel.setAttribute("aria-hidden", String(!shouldOpen));
}

/* ---------- LLM 聊天 ---------- */
async function submitChat() {
  const message = chatInput.value.trim();
  if (!message || chatRequestInFlight) return;

  appendChatMessage("user", message);
  chatInput.value = "";
  autoResizeChatInput();

  const pending = appendChatMessage("assistant", "思考中…", true);
  chatRequestInFlight = true;
  chatSendButton.disabled = true;

  try {
    const reply = await sendMessageToLLM(message);
    pending.querySelector(".chat-bubble").textContent = reply;
    pending.classList.remove("pending");
  } catch (error) {
    console.error(error);
    pending.querySelector(".chat-bubble").textContent = `聊天服務錯誤：${error.message}`;
    pending.classList.remove("pending");
  } finally {
    chatRequestInFlight = false;
    chatSendButton.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

async function sendMessageToLLM(message) {
  const stores = document.getElementById("useOcrContext").checked &&
    Date.now() - lastRecognitionAt < 60000
    ? (lastResponse?.detections ?? []).map(d => (d.text || "").trim()).filter(Boolean)
    : [];
  return window.navigationChat(message, stores.slice(0, 10));
}

function appendChatMessage(role, text, pending = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `chat-message ${role}${pending ? " pending" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;

  wrapper.append(bubble);
  chatMessages.append(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
}

function autoResizeChatInput() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 100)}px`;
}

/* ---------- 通用 ---------- */
function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Canvas 無法產生 JPEG"));
    }, type, quality);
  });
}

function cameraErrorMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
      return "相機權限遭拒，或目前網址不是 HTTPS/localhost。";
    case "NotFoundError":
      return "找不到可用相機。";
    case "NotReadableError":
      return "相機可能正在被其他 App 或分頁使用。";
    case "OverconstrainedError":
      return "裝置無法符合指定的相機參數。";
    default:
      return `無法啟動相機：${error?.message ?? String(error)}`;
  }
}

function setLaunchStatus(message, isError = false) {
  launchStatusElement.textContent = message;
  launchStatusElement.classList.toggle("error", isError);
}

function setStatus(message, state = "ready") {
  statusElement.textContent = message;
  statusPill.classList.toggle("processing", state === "processing");
  statusPill.classList.toggle("error", state === "error");
}

// 地圖與聊天可在沒有相機的桌面瀏覽器使用。
document.getElementById("navigationOnlyButton").addEventListener("click", () => {
  document.body.classList.add("camera-active");
  cameraApp.setAttribute("aria-hidden", "false");
  launchScreen.setAttribute("aria-hidden", "true");
  captureButton.disabled = true;
  setStatus("地圖與導航模式");
  toggleMainPanel("map");
});
