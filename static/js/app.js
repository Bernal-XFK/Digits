document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const canvas = document.getElementById("digitCanvas");
  const ctx = canvas.getContext("2d");
  const btnClear = document.getElementById("btnClear");
  const btnPredict = document.getElementById("btnPredict");
  const brushSize = document.getElementById("brushSize");
  const brushSizeVal = document.getElementById("brushSizeVal");
  const modelSelect = document.getElementById("modelSelect");
  const predictedDigitDisplay = document.getElementById("predictedDigitDisplay");
  const confidenceVal = document.getElementById("confidenceVal");
  const activeModelTag = document.getElementById("activeModelTag");
  const latencyBadge = document.getElementById("latencyBadge");
  const matrix8x8Grid = document.getElementById("matrix8x8Grid");
  const probabilityBarsList = document.getElementById("probabilityBarsList");
  const presetsGrid = document.getElementById("presetsGrid");

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let hasDrawn = false;
  let debounceTimer = null;

  // Initialize Canvas
  function resetCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    clearResults();
  }

  function clearResults() {
    predictedDigitDisplay.textContent = "?";
    confidenceVal.textContent = "--%";
    latencyBadge.textContent = "Latencia: -- ms";
    init8x8Grid();
    initProbabilityBars();
  }

  // 8x8 Grid Inspector init
  function init8x8Grid() {
    matrix8x8Grid.innerHTML = "";
    for (let i = 0; i < 64; i++) {
      const pixel = document.createElement("div");
      pixel.className = "matrix-pixel";
      pixel.style.backgroundColor = "#090d16";
      matrix8x8Grid.appendChild(pixel);
    }
  }

  function update8x8Grid(matrix) {
    if (!matrix || matrix.length !== 8) return;
    const pixels = matrix8x8Grid.children;
    let idx = 0;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const val = matrix[r][c]; // 0 to 16
        const norm = Math.min(Math.max(val / 16.0, 0), 1);
        // Interpolate between dark blue (#090d16) and bright blue (#3b82f6)
        const rColor = Math.round(9 + (59 - 9) * norm);
        const gColor = Math.round(13 + (130 - 13) * norm);
        const bColor = Math.round(22 + (246 - 22) * norm);
        pixels[idx].style.backgroundColor = `rgb(${rColor}, ${gColor}, ${bColor})`;
        idx++;
      }
    }
  }

  // Probabilities Bars init
  function initProbabilityBars() {
    probabilityBarsList.innerHTML = "";
    for (let i = 0; i < 10; i++) {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.id = `prob-row-${i}`;
      row.innerHTML = `
        <span class="prob-digit">${i}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="prob-bar-${i}" style="width: 0%"></div>
        </div>
        <span class="prob-pct" id="prob-pct-${i}">0.0%</span>
      `;
      probabilityBarsList.appendChild(row);
    }
  }

  function updateProbabilityBars(probs, winnerDigit) {
    if (!probs || probs.length !== 10) return;
    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const bar = document.getElementById(`prob-bar-${i}`);
      const pct = document.getElementById(`prob-pct-${i}`);
      const val = probs[i];

      bar.style.width = `${val}%`;
      pct.textContent = `${val.toFixed(1)}%`;

      if (i === winnerDigit) {
        row.classList.add("winner");
      } else {
        row.classList.remove("winner");
      }
    }
  }

  // Drawing event listeners (Mouse + Touch)
  function startDrawing(e) {
    isDrawing = true;
    hasDrawn = true;
    const { x, y } = getCanvasCoords(e);
    lastX = x;
    lastY = y;
    drawStroke(x, y);
  }

  function drawMove(e) {
    if (!isDrawing) return;
    const { x, y } = getCanvasCoords(e);
    drawStroke(x, y);
    lastX = x;
    lastY = y;
  }

  function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.beginPath();
    
    // Auto-predict on stroke release (debounce 250ms)
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (hasDrawn) performPrediction();
    }, 250);
  }

  function drawStroke(x, y) {
    ctx.lineWidth = parseInt(brushSize.value, 10);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = e.clientX;
    let clientY = e.clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  // Canvas Mouse events
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", drawMove);
  window.addEventListener("mouseup", stopDrawing);

  // Canvas Touch events (Mobile / Tablet)
  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    startDrawing(e);
  }, { passive: false });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    drawMove(e);
  }, { passive: false });

  window.addEventListener("touchend", stopDrawing);

  // Brush size control
  brushSize.addEventListener("input", (e) => {
    brushSizeVal.textContent = `${e.target.value}px`;
  });

  // Buttons
  btnClear.addEventListener("click", resetCanvas);
  btnPredict.addEventListener("click", () => {
    if (hasDrawn) performPrediction();
  });

  modelSelect.addEventListener("change", () => {
    activeModelTag.textContent = `Modelo: ${modelSelect.options[modelSelect.selectedIndex].text}`;
    if (hasDrawn) performPrediction();
  });

  // Perform Prediction via API
  async function performPrediction() {
    const base64Data = canvas.toDataURL("image/png");
    const selectedModel = modelSelect.value;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Data,
          model: selectedModel
        })
      });

      const data = await response.json();

      if (data.success) {
        predictedDigitDisplay.textContent = data.digit;
        confidenceVal.textContent = `${data.confidence}%`;
        latencyBadge.textContent = `Latencia: ${data.latency_ms} ms`;
        update8x8Grid(data.matrix_8x8);
        updateProbabilityBars(data.probabilities, data.digit);
      } else if (data.message) {
        predictedDigitDisplay.textContent = "?";
        confidenceVal.textContent = "--%";
      }
    } catch (err) {
      console.error("Error en inferencia:", err);
    }
  }

  // Presets: Generate clean sample strokes for digits 0-9
  function initPresets() {
    for (let digit = 0; digit <= 9; digit++) {
      const btn = document.createElement("button");
      btn.className = "preset-btn";
      btn.textContent = digit;
      btn.title = `Probar trazo del dígito ${digit}`;
      btn.addEventListener("click", () => drawPreset(digit));
      presetsGrid.appendChild(btn);
    }
  }

  function drawPreset(digit) {
    resetCanvas();
    hasDrawn = true;
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";

    const cx = 140;
    const cy = 140;

    switch (digit) {
      case 0:
        ctx.beginPath();
        ctx.ellipse(cx, cy, 50, 75, 0, 0, Math.PI * 2);
        ctx.stroke();
        break;
      case 1:
        ctx.beginPath();
        ctx.moveTo(cx - 20, cy - 60);
        ctx.lineTo(cx + 5, cy - 80);
        ctx.lineTo(cx + 5, cy + 80);
        ctx.stroke();
        break;
      case 2:
        ctx.beginPath();
        ctx.arc(cx, cy - 35, 40, Math.PI, 0);
        ctx.lineTo(cx - 35, cy + 70);
        ctx.lineTo(cx + 40, cy + 70);
        ctx.stroke();
        break;
      case 3:
        ctx.beginPath();
        ctx.arc(cx, cy - 35, 40, Math.PI * 1.1, Math.PI * 0.4);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy + 35, 45, -Math.PI * 0.4, Math.PI * 0.9);
        ctx.stroke();
        break;
      case 4:
        ctx.beginPath();
        ctx.moveTo(cx + 25, cy + 75);
        ctx.lineTo(cx + 25, cy - 75);
        ctx.lineTo(cx - 40, cy + 20);
        ctx.lineTo(cx + 45, cy + 20);
        ctx.stroke();
        break;
      case 5:
        ctx.beginPath();
        ctx.moveTo(cx + 35, cy - 70);
        ctx.lineTo(cx - 25, cy - 70);
        ctx.lineTo(cx - 30, cy - 10);
        ctx.arc(cx, cy + 25, 45, -Math.PI * 0.6, Math.PI * 0.8);
        ctx.stroke();
        break;
      case 6:
        ctx.beginPath();
        ctx.arc(cx, cy + 30, 42, 0, Math.PI * 2);
        ctx.moveTo(cx - 42, cy + 30);
        ctx.quadraticCurveTo(cx - 40, cy - 60, cx + 20, cy - 75);
        ctx.stroke();
        break;
      case 7:
        ctx.beginPath();
        ctx.moveTo(cx - 45, cy - 70);
        ctx.lineTo(cx + 40, cy - 70);
        ctx.lineTo(cx - 15, cy + 75);
        ctx.stroke();
        break;
      case 8:
        ctx.beginPath();
        ctx.arc(cx, cy - 38, 36, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy + 38, 44, 0, Math.PI * 2);
        ctx.stroke();
        break;
      case 9:
        ctx.beginPath();
        ctx.arc(cx, cy - 30, 42, 0, Math.PI * 2);
        ctx.moveTo(cx + 42, cy - 30);
        ctx.quadraticCurveTo(cx + 40, cy + 60, cx - 20, cy + 75);
        ctx.stroke();
        break;
    }

    performPrediction();
  }

  // Initialize
  resetCanvas();
  initPresets();
  init8x8Grid();
  initProbabilityBars();
});
