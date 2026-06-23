// ==========================================
// 🌐 狀態與網路通訊模組 (socket_log.js)
// ==========================================

let currentRoomId = null, currentInviteCode = null, currentServerImageUrl = null;
let pollInterval = null, isMapReady = false;
let collisionMatrix = null;

const myUserId = 'user_' + Math.random().toString(36).substr(2, 6);
const myColor = Math.random() > 0.5 ? '#ef4444' : '#3b82f6';
let myPosition = null; // 共享給 PDR 引擎更新

window.onload = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const roomCode = urlParams.get('room');
    
    if (roomCode) {
        joinRoom(roomCode);
    } else {
        setTimeout(() => {
            alert("👋 歡迎使用 AI 室內導航系統！\n\n⚠️ 請先在右下角的面板點擊「➕ 建立」房間，或「加入」已有房間，才能開始上傳地圖喔！");
        }, 300);
    }

    document.getElementById('map-input').addEventListener('click', function(event) {
        if (!currentRoomId) {
            event.preventDefault(); 
            alert("🚫 請先「建立」或「加入」房間後，才能上傳地圖！");
            
            const roomPanel = document.getElementById('room-panel');
            roomPanel.classList.remove('bg-blue-50');
            roomPanel.classList.add('bg-yellow-300', 'transition-colors', 'duration-300');
            setTimeout(() => {
                roomPanel.classList.remove('bg-yellow-300');
                roomPanel.classList.add('bg-blue-50');
            }, 600);
        }
    });
};

// --- 房間與訊息 ---
async function createRoom() {
    const res = await fetch('/create_room', { method: 'POST' });
    const data = await res.json();
    enterRoomSuccess(data.room_id, data.invite_code);
}

function joinRoomFromInput() {
    const code = document.getElementById('invite-input').value.trim();
    if (code) joinRoom(code);
}

async function joinRoom(code) {
    try {
        const res = await fetch('/join_room', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code_or_id: code })
        });
        const data = await res.json();
        if (res.ok) {
            enterRoomSuccess(data.room_id, data.invite_code);
            window.history.pushState({}, '', `/?room=${data.invite_code}`);
        } else { 
            alert(data.detail || "找不到該房間，請確認代碼是否正確！"); 
        }
    } catch (e) { 
        alert("連線失敗，請檢查伺服器狀態！"); 
    }
}

function copyInviteLink() {
    if (!currentInviteCode) return;
    const link = `${window.location.origin}/?room=${currentInviteCode}`;
    navigator.clipboard.writeText(link).then(() => {
        alert("🔗 邀請連結已複製！貼給朋友就能直接加入囉！");
    }).catch(() => {
        alert("複製失敗，請手動複製網址列。");
    });
}

function enterRoomSuccess(roomId, inviteCode) {
    currentRoomId = roomId; currentInviteCode = inviteCode;
    document.getElementById('no-room-ui').classList.add('hidden');
    document.getElementById('in-room-ui').classList.remove('hidden');
    document.getElementById('room-code-display').textContent = inviteCode;
    document.getElementById('user-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('send-btn').classList.replace('bg-gray-400', 'bg-blue-600');
    startPolling();
}

async function sendMessage() {
    const text = document.getElementById('user-input').value.trim();
    if (!text || !currentRoomId) return;
    appendMessage('你', text, true);
    document.getElementById('user-input').value = '';
    const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, room_id: currentRoomId })
    });
    const data = await res.json();
    appendMessage('Agent', data.reply, false);
    if (data.path_coords && data.path_coords.length > 0) {
        drawPathOnMap(data.path_coords); // 呼叫 pdr_engine.js 的函式
        myPosition = { x: data.path_coords[0][0], y: data.path_coords[0][1] };
        updateDotUI(myUserId, myPosition.x, myPosition.y, myColor);
        syncPosition();
        if (typeof autoSetStartPoint === 'function') {
            autoSetStartPoint(myPosition.x, myPosition.y);
        }
    }
}

function startPolling() {
    pollInterval = setInterval(async () => {
        const res = await fetch(`/room_status/${currentRoomId}?t=${Date.now()}`);
        const data = await res.json();
        if (data.status === 'ready') {
            document.getElementById('loading-overlay').classList.add('hidden');
            isMapReady = true;

            if (typeof loadCollisionMap === 'function' && !collisionMatrix) {
                loadCollisionMap(currentRoomId);
            }
        } else if (data.status === 'processing') {
            document.getElementById('loading-overlay').classList.remove('hidden');
        }
        if (data.image_url && data.image_url !== currentServerImageUrl) {
            currentServerImageUrl = data.image_url;
            document.getElementById('upload-ui').classList.add('hidden');
            document.getElementById('map-image').src = currentServerImageUrl;
            document.getElementById('map-wrapper').classList.remove('hidden');
        }
        if (data.users) {
            for (const [uid, userObj] of Object.entries(data.users)) {
                if (uid !== myUserId) updateDotUI(uid, userObj.x, userObj.y, userObj.color);
            }
        }
    }, 3000);
}

function appendMessage(sender, text, isUser) {
    const msg = document.createElement('div');
    msg.className = `flex ${isUser ? 'justify-end' : ''}`;
    msg.innerHTML = `<div class="p-3 rounded-2xl text-sm ${isUser ? 'bg-blue-600 text-white' : 'bg-white border'}">${text}</div>`;
    document.getElementById('chat-box').appendChild(msg);
    document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;
}

async function loadMap(event) {
    const file = event.target.files[0];
    document.getElementById('map-image').src = URL.createObjectURL(file);
    document.getElementById('upload-ui').classList.add('hidden');
    document.getElementById('map-wrapper').classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('room_id', currentRoomId);
    await fetch('/upload', { method: 'POST', body: formData });
}

function toggleMinimize() {
    const win = document.getElementById('chat-window');
    const body = document.getElementById('chat-body');
    const panel = document.getElementById('room-panel');
    body.classList.toggle('hidden');
    panel.classList.toggle('hidden');
    document.getElementById('minimize-icon').classList.toggle('hidden');
    document.getElementById('maximize-icon').classList.toggle('hidden');
    win.style.width = body.classList.contains('hidden') ? '200px' : '';
}

async function syncPosition() {
    if(!currentRoomId || !myPosition) return;
    fetch(`/update_position/${currentRoomId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: myUserId, x: myPosition.x, y: myPosition.y, color: myColor })
    });
}
async function loadCollisionMap(roomId) {
    try {
        const response = await fetch(`/uploads/${roomId}/map_matrix.csv`);
        const csvText = await response.text();
        
        // 過濾掉空行、空白符號，並強制轉成整數
        collisionMatrix = csvText.trim().split('\n').map(row => {
            return row.split(',')
                      .filter(val => val.trim() !== "") // 濾掉結尾多餘的逗號
                      .map(val => parseInt(val.trim(), 10)); // 確保絕對是數字
        });
        console.log(`碰撞矩陣載入完成！尺寸: ${collisionMatrix[0].length} x ${collisionMatrix.length}`);
    } catch (error) {
        console.error("無法載入碰撞矩陣", error);
    }
}