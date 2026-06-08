// ==========================================
// 🧭 定位運算與地圖渲染模組 (pdr_engine.js)
// ==========================================

let currentRawHeading = 0; // 原始陀螺儀角度
let headingOffset = 0;     // 校正偏移量
let isTracking = false, lastMag = 0, isStep = false;
let targetMapAngle = 0;
let hasCalibratedOffset = false;
let calibStartPos = null; // 紀錄第一個點的座標
let customStepPx = 0;     // 儲存算出來的專屬步長 (像素/步)
let calibrationState = 0; // 狀態機：0=尚未校正, 1=已選起點等待終點, 2=校正完成

// --- 核心定位引擎 (PDR) ---
async function requestSensorAccess() {
    try {
        if (typeof DeviceOrientationEvent.requestPermission === 'function') {
            await DeviceOrientationEvent.requestPermission();
        }
        if (typeof DeviceMotionEvent.requestPermission === 'function') {
            const motionPerm = await DeviceMotionEvent.requestPermission();
            if (motionPerm !== 'granted') {
                alert('需要加速度計權限才能算步數喔！');
                return;
            }
        }
        startSensors();
    } catch (e) {
        console.error(e);
        alert("感測器授權失敗，請確認是否為 HTTPS 連線。");
    }
}

function autoSetStartPoint(x, y) {
    calibStartPos = { x: x, y: y };
    calibrationState = 1; // 直接跳入狀態 1 (等待點擊終點)
    alert("📍 系統已定位您的起點！\n請面向前方直走一段距離並「默數步數」，然後點擊您現在地圖上的位置。");
}

function calculateAngleFromPath(pathCoords) {
    if (!pathCoords || pathCoords.length < 2) return;

    const p1 = pathCoords[0]; // 起點
    const p2 = pathCoords[1]; // 路線的下一個轉折點

    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];

    // 用 Math.atan2 算出弧度，再轉成角度
    const rad = Math.atan2(dy, dx);
    const deg = rad * (180 / Math.PI);

    // 轉換成網頁地圖的指北針系統：上=0°, 右=90°, 下=180°, 左=270°
    targetMapAngle = (deg + 90 + 360) % 360;
    
    console.log(`🧭 系統判定：初始路徑的絕對方向為 ${targetMapAngle.toFixed(1)}°`);
}

function startSensors() {
    isTracking = true;
    const btn = document.getElementById('start-tracking-btn');
    
    
    btn.textContent = "🟢 定位校正完成"; 
    btn.classList.replace('bg-green-600', 'bg-gray-400');
    btn.disabled = true;

    window.addEventListener('deviceorientation', (e) => {
        let raw = e.webkitCompassHeading || (360 - (e.alpha || 0));
        
        // 🌟 簡單的一階低通濾波（權重 0.2），讓角度不會因為手震瞬間暴跳
        if (currentRawHeading === 0) {
            currentRawHeading = raw; // 第一次直接賦值
        } else {
            currentRawHeading = currentRawHeading * 0.8 + raw * 0.2;
        }

        if (!hasCalibratedOffset) {
            headingOffset = currentRawHeading - targetMapAngle;
            hasCalibratedOffset = true;
            console.log(`🧭 方向完美鎖定！手機實體角度: ${currentRawHeading.toFixed(1)}°，地圖路徑角度: ${targetMapAngle.toFixed(1)}°`);
        }
        
        // 更新你的 UI 顯示
        const displayEl = document.getElementById('heading-display');
        if (displayEl) displayEl.innerText = `${Math.round(currentRawHeading)}°`;
    });

    window.addEventListener('devicemotion', (e) => {
        if(!myPosition || !isTracking) return;
        
        let acc = e.acceleration || e.accelerationIncludingGravity;
        if (!acc || acc.x === null) return;
        
        let mag = Math.sqrt(acc.x**2 + acc.y**2 + acc.z**2);
        let pureMovement = e.acceleration ? mag : Math.abs(mag - 9.8);
        
        if (pureMovement > 1.5 && lastMag <= 1.5 && !isStep) {
            isStep = true;
            handleStepDetected();
            setTimeout(() => { isStep = false; }, 400); 
        }
        lastMag = pureMovement;
    });
}

function handleStepDetected() {
    let relativeHeading = currentRawHeading - headingOffset;
    let rad = relativeHeading * (Math.PI / 180);
    
    // 計算出這個步伐的 X、Y 總分量
    let totalDx = Math.sin(rad) * customStepPx;
    let totalDy = -Math.cos(rad) * customStepPx;
    
    // 🌟 物理引擎升級：將這「一步」切碎，每次最多只走 2 像素來探路
    // 這樣就算牆壁只有 3 像素厚，也絕對不會發生「跨過去」的穿隧效應
    let maxSubSteps = Math.ceil(customStepPx / 2); 
    let stepDx = totalDx / maxSubSteps;
    let stepDy = totalDy / maxSubSteps;

    let walkX = myPosition.x;
    let walkY = myPosition.y;

    for (let i = 0; i < maxSubSteps; i++) {
        let testX = walkX + stepDx;
        let testY = walkY + stepDy;
        
        let canMoveX = true;
        let canMoveY = true;

        if (collisionMatrix) {
            let rows = collisionMatrix.length;
            let cols = collisionMatrix[0].length;

            // 獨立測試 X 方向會不會撞牆
            let mapX = Math.floor(testX);
            let mapY = Math.floor(walkY); // 測試 X 時，Y 先假裝不動
            if (mapY >= 0 && mapY < rows && mapX >= 0 && mapX < cols) {
                if (collisionMatrix[mapY][mapX] === 1) canMoveX = false;
            } else {
                canMoveX = false; // 超出地圖邊界也算撞牆
            }

            // 獨立測試 Y 方向會不會撞牆
            mapX = Math.floor(walkX); // 測試 Y 時，X 先假裝不動
            mapY = Math.floor(testY);
            if (mapY >= 0 && mapY < rows && mapX >= 0 && mapX < cols) {
                if (collisionMatrix[mapY][mapX] === 1) canMoveY = false;
            } else {
                canMoveY = false;
            }
        }

        // 如果走進死角 (X跟Y都撞牆)，就直接停止這一步剩下的位移
        if (!canMoveX && !canMoveY) {
            console.log("撞到死角了！");
            break; 
        }

        // 🌟 完美的沿牆滑行邏輯：哪裡沒擋住就往哪裡滑
        if (canMoveX) walkX = testX;
        if (canMoveY) walkY = testY;
    }

    // 將微步測試後的最終安全座標，正式更新給人物
    myPosition.x = walkX;
    myPosition.y = walkY;
    
    updateDotUI(myUserId, myPosition.x, myPosition.y, myColor);
    syncPosition();
}

// --- 地圖與 UI 渲染 ---
function updateDotUI(uid, naturalX, naturalY, color) {
    const wrapper = document.getElementById('map-wrapper');
    const img = document.getElementById('map-image');
    if (!img.naturalWidth) return; 

    let dot = document.getElementById('dot-' + uid);
    if (!dot) {
        dot = document.createElement('div');
        dot.id = 'dot-' + uid;
        dot.className = 'user-dot';
        dot.style.backgroundColor = color;
        wrapper.appendChild(dot);
    }
    
    dot.style.left = (naturalX / img.naturalWidth * 100) + '%';
    dot.style.top = (naturalY / img.naturalHeight * 100) + '%';
}

function drawPathOnMap(coords) {
    const svg = document.getElementById('path-svg');
    const img = document.getElementById('map-image');
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
    const pointsString = coords.map(p => `${p[0]},${p[1]}`).join(' ');
    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', pointsString);
    polyline.setAttribute('fill', 'none'); polyline.setAttribute('stroke', '#fbbf24');
    polyline.setAttribute('stroke-width', '8'); polyline.setAttribute('stroke-linecap', 'round');
    svg.appendChild(polyline);
    svg.classList.remove('opacity-0'); svg.classList.add('opacity-100');
}

// --- 地圖點擊初始化位置 ---
// --- 地圖點擊：兩點校正法 ---
document.getElementById('map-image').addEventListener('click', (e) => {
    if (!isMapReady) return;
    const rect = e.target.getBoundingClientRect();
    const scaleX = e.target.naturalWidth / rect.width;
    const scaleY = e.target.naturalHeight / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    if (calibrationState === 0) {
        // (備用) 萬一沒有問 Agent，使用者自己亂點地圖時的防呆機制
        calibStartPos = { x: clickX, y: clickY };
        myPosition = { x: clickX, y: clickY };
        updateDotUI(myUserId, myPosition.x, myPosition.y, myColor);
        calibrationState = 1;
        alert("📍 手動起點已設定！\n請直走並數步數，走到一半再點擊地圖。");

    } else if (calibrationState === 1) {
        // 【第二步】：記錄終點並計算
        let stepsInput = prompt("請輸入您從起點走到這裡，總共走了幾步？", "10");
        let steps = parseInt(stepsInput);

        if (isNaN(steps) || steps <= 0) {
            alert("❌ 請輸入有效的步數！請重新點擊地圖。");
            return;
        }

        const dx = clickX - calibStartPos.x;
        const dy = clickY - calibStartPos.y;
        customStepPx = Math.sqrt(dx*dx + dy*dy) / steps;

        const rad = Math.atan2(dy, dx);
        targetMapAngle = (rad * (180 / Math.PI) + 90 + 360) % 360;

        myPosition = { x: clickX, y: clickY };
        updateDotUI(myUserId, myPosition.x, myPosition.y, myColor);
        syncPosition();

        calibrationState = 2; // 校正完成

        alert(`✅ 校正完成！\n（步長：${customStepPx.toFixed(1)}px / 角度：${targetMapAngle.toFixed(1)}°）\n\n👉 準備好後，請「面向您剛剛行走的方向」，並點擊右下角的按鈕開始導航！`);
        
        const btn = document.getElementById('start-tracking-btn');
        btn.classList.remove('hidden');
        btn.textContent = "🧭 開始室內導航";
        btn.disabled = false;
        btn.classList.replace('bg-gray-400', 'bg-green-600');
    }
});