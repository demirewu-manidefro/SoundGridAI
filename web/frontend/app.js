/**
 * SoundGrid AI - Modern Landing Page & Cyber-Physical Controller
 * Manages sticky navbar scroll spy, live telemetry, dual acoustic visualizers,
 * multi-step Chart.js physics trajectories, and industrial benchmarking.
 */

// Global State
let autoLoopTimer = null;
let thermalChartInstance = null;
let acousticChartInstance = null;
let benchSafetyChartInstance = null;
let benchRewardChartInstance = null;
let stepCounter = 1;

// Sensor history for POMDP delta computation
let prevState = {
  load: 0.85,
  temp: 65.0,
  anom: 0.04
};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupNavScroll();
  setupModal();
  setupSliders();
  setupPresets();
  setupImageControls();
  checkApiHealth();
  initAcousticWaveform();
  
  // Bind buttons
  const stepBtn = document.getElementById("stepBtn");
  if (stepBtn) stepBtn.addEventListener("click", executeStep);

  const loopBtn = document.getElementById("autoLoopBtn");
  if (loopBtn) loopBtn.addEventListener("click", toggleAutoLoop);

  // Initial runs to populate all cards, charts, and tables immediately
  executeStep();
  runBenchmark();
});

// Toast Notification
function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2600);
}

// =============================================================================
// STICKY NAVBAR SCROLL SPY
// =============================================================================
function setupNavScroll() {
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-link:not(.nav-link-modal-btn)");

  window.addEventListener("scroll", () => {
    let currentId = "overview";
    const scrollPos = window.scrollY + 120;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute("id");
      }
    });

    navLinks.forEach(link => {
      link.classList.remove("active");
      if (link.getAttribute("href") === `#${currentId}`) {
        link.classList.add("active");
      }
    });
  });
}

// =============================================================================
// MODAL DIALOG CONTROLLER (METHODOLOGY & BENCHMARK POPUP)
// =============================================================================
function setupModal() {
  const modal = document.getElementById("methodologyModal");
  const openBtn = document.getElementById("openModalBtn");
  const closeBtn = document.getElementById("closeModalBtn");
  const modalRunBenchBtn = document.getElementById("modalRunBenchmarkBtn");

  if (openBtn && modal) {
    openBtn.addEventListener("click", () => {
      modal.classList.add("show");
      document.body.style.overflow = "hidden";
      if (!benchSafetyChartInstance) {
        runBenchmark();
      }
    });
  }

  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => {
      modal.classList.remove("show");
      document.body.style.overflow = "";
    });
  }

  if (modal) {
    modal.addEventListener("click", e => {
      if (e.target === modal) {
        modal.classList.remove("show");
        document.body.style.overflow = "";
      }
    });
  }

  if (modalRunBenchBtn) {
    modalRunBenchBtn.addEventListener("click", runBenchmark);
  }

  // Modal internal tabs
  const tabBtns = document.querySelectorAll(".modal-tab-btn");
  const tabContents = document.querySelectorAll(".modal-tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-modaltab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });
}

// =============================================================================
// API HEALTH CHECK
// =============================================================================
async function checkApiHealth() {
  const statusBadge = document.getElementById("connectionStatus");
  const statusText = document.getElementById("statusText");

  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "healthy" || data.ppo_model_loaded) {
      statusBadge.style.background = "#ecfdf5";
      statusBadge.style.borderColor = "#a7f3d0";
      statusBadge.style.color = "#065f46";
      statusText.textContent = "API Online (8D PPO Ready)";
    } else {
      statusBadge.style.background = "#fffbeb";
      statusBadge.style.borderColor = "#fde68a";
      statusBadge.style.color = "#92400e";
      statusText.textContent = "API Degraded (Heuristics Active)";
    }
  } catch (err) {
    statusBadge.style.background = "#fef2f2";
    statusBadge.style.borderColor = "#fecaca";
    statusBadge.style.color = "#991b1b";
    statusText.textContent = "API Disconnected";
  }
}

// =============================================================================
// SLIDERS & LIVE TELEMETRY
// =============================================================================
function setupSliders() {
  const loadInput = document.getElementById("loadInput");
  const tempInput = document.getElementById("tempInput");
  const anomInput = document.getElementById("anomInput");
  const energyInput = document.getElementById("energyInput");
  const ambientInput = document.getElementById("ambientInput");

  loadInput.addEventListener("input", e => {
    const val = (e.target.value / 100).toFixed(2);
    document.getElementById("loadValDisplay").textContent = `${e.target.value}% (${val})`;
    document.getElementById("overlayLoad").textContent = `${e.target.value}%`;
    document.getElementById("heroLoadDisplay").textContent = `${e.target.value}%`;
  });

  tempInput.addEventListener("input", e => {
    const formatted = `${Number(e.target.value).toFixed(1)}°C`;
    document.getElementById("tempValDisplay").textContent = formatted;
    document.getElementById("overlayTemp").textContent = formatted;
    document.getElementById("heroTempDisplay").textContent = formatted;
  });

  anomInput.addEventListener("input", e => {
    const val = (e.target.value / 100).toFixed(2);
    document.getElementById("anomValDisplay").textContent = val;
    document.getElementById("overlayAnom").textContent = val;
    document.getElementById("heroAnomDisplay").textContent = val;
    updateSpectrumText(parseFloat(val));
  });

  energyInput.addEventListener("input", e => {
    const val = (e.target.value / 100).toFixed(2);
    document.getElementById("energyValDisplay").textContent = val;
  });

  ambientInput.addEventListener("input", e => {
    document.getElementById("ambientValDisplay").textContent = `${Number(e.target.value).toFixed(1)}°C`;
  });
}

function updateSpectrumText(anom) {
  const text = document.getElementById("spectrumStatusText");
  const heroText = document.getElementById("heroSpectrumStatus");

  let desc = "Nominal Magnetostriction (100 Hz)";
  let col = "#10b981";

  if (anom >= 0.65) {
    desc = "⚠️ Critical Arcing Transient Shockwaves!";
    col = "#ef4444";
  } else if (anom >= 0.20) {
    desc = "Developing Partial Discharge (10-150 kHz)";
    col = "#f59e0b";
  }

  if (text) {
    text.textContent = desc;
    text.style.color = col;
  }
  if (heroText) {
    heroText.textContent = desc;
    heroText.style.color = col;
  }
}

// =============================================================================
// OPERATIONAL PRESETS
// =============================================================================
function setupPresets() {
  const chips = document.querySelectorAll(".preset-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active-preset"));
      chip.classList.add("active-preset");
      const preset = chip.getAttribute("data-preset");

      if (preset === "normal") {
        setValues(85, 62.0, 4, 15, 30.0);
      } else if (preset === "warning") {
        setValues(80, 72.0, 45, 42, 32.0);
      } else if (preset === "critical") {
        setValues(85, 75.0, 88, 85, 35.0);
      } else if (preset === "thermal") {
        setValues(95, 92.0, 8, 20, 42.0);
      }
      executeStep();
    });
  });
}

function setValues(load, temp, anom, energy, ambient) {
  document.getElementById("loadInput").value = load;
  document.getElementById("tempInput").value = temp;
  document.getElementById("anomInput").value = anom;
  document.getElementById("energyInput").value = energy;
  document.getElementById("ambientInput").value = ambient;

  document.getElementById("loadValDisplay").textContent = `${load}% (${(load / 100).toFixed(2)})`;
  document.getElementById("tempValDisplay").textContent = `${temp.toFixed(1)}°C`;
  document.getElementById("anomValDisplay").textContent = (anom / 100).toFixed(2);
  document.getElementById("energyValDisplay").textContent = (energy / 100).toFixed(2);
  document.getElementById("ambientValDisplay").textContent = `${ambient.toFixed(1)}°C`;

  document.getElementById("overlayLoad").textContent = `${load}%`;
  document.getElementById("overlayTemp").textContent = `${temp.toFixed(1)}°C`;
  document.getElementById("overlayAnom").textContent = (anom / 100).toFixed(2);

  document.getElementById("heroLoadDisplay").textContent = `${load}%`;
  document.getElementById("heroTempDisplay").textContent = `${temp.toFixed(1)}°C`;
  document.getElementById("heroAnomDisplay").textContent = (anom / 100).toFixed(2);

  updateSpectrumText(anom / 100);
}

// =============================================================================
// IMAGE UPLOAD & VIEW SWITCHING
// =============================================================================
function setupImageControls() {
  const uploadInput = document.getElementById("imageUploadInput");
  const imgElement = document.getElementById("transformerImage");
  const resetBtn = document.getElementById("resetImageBtn");
  const viewBtns = document.querySelectorAll(".image-view-btn");

  viewBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      viewBtns.forEach(b => b.classList.remove("active-view"));
      btn.classList.add("active-view");
      const src = btn.getAttribute("data-src");
      imgElement.src = src;
    });
  });

  uploadInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = evt => {
        imgElement.src = evt.target.result;
        viewBtns.forEach(b => b.classList.remove("active-view"));
        showToast("Transformer image updated!");
      };
      reader.readAsDataURL(file);
    }
  });

  resetBtn.addEventListener("click", () => {
    imgElement.src = "/images/substation_ai.png";
    viewBtns.forEach(b => b.classList.remove("active-view"));
    document.getElementById("viewBtnSubstation").classList.add("active-view");
    showToast("Substation image restored.");
  });
}

function autoSetImageForState(statusFlag, temp, anom) {
  const imgElement = document.getElementById("transformerImage");
  const viewBtns = document.querySelectorAll(".image-view-btn");

  if (statusFlag === "burnout" || temp >= 105.0) {
    imgElement.src = "/images/transformer_burnout.png";
    viewBtns.forEach(b => b.classList.remove("active-view"));
    const btn = document.getElementById("viewBtnBurnout");
    if (btn) btn.classList.add("active-view");
  } else if (imgElement.src.includes("transformer_burnout.png") && temp < 90.0 && anom < 0.3) {
    imgElement.src = "/images/substation_ai.png";
    viewBtns.forEach(b => b.classList.remove("active-view"));
    const btn = document.getElementById("viewBtnSubstation");
    if (btn) btn.classList.add("active-view");
  }
}

// =============================================================================
// SINGLE STEP EXECUTION
// =============================================================================
async function executeStep() {
  const controller = document.getElementById("activeControllerSelect").value;
  const load = parseFloat(document.getElementById("loadInput").value) / 100;
  const temp = parseFloat(document.getElementById("tempInput").value);
  const anom = parseFloat(document.getElementById("anomInput").value) / 100;
  const energy = parseFloat(document.getElementById("energyInput").value) / 100;
  const ambient = parseFloat(document.getElementById("ambientInput").value);

  const payload = {
    controller: controller,
    load: load,
    prev_load: prevState.load,
    temp: temp,
    prev_temp: prevState.temp,
    anomaly_prob: anom,
    prev_anomaly: prevState.anom,
    acoustic_energy: energy,
    ambient_temp: ambient,
    step_count: stepCounter,
    max_steps: 100
  };

  try {
    const res = await fetch("/api/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderStepResponse(data);

    prevState = { load, temp, anom };
    stepCounter++;
  } catch (err) {
    console.error("Step execution error:", err);
  }
}

function renderStepResponse(data) {
  const actionBanner = document.getElementById("actionBanner");
  const actionNameText = document.getElementById("actionNameText");
  const actionConfidenceText = document.getElementById("actionConfidenceText");
  const breakerBadge = document.getElementById("breakerBadge");
  const heroBreakerBadge = document.getElementById("heroBreakerBadge");
  const heroActionDisplay = document.getElementById("heroActionDisplay");

  // Breaker status update
  if (data.action === 4) {
    breakerBadge.className = "breaker-state-badge breaker-tripped";
    breakerBadge.textContent = "Breaker: TRIPPED (OPEN)";
    if (heroBreakerBadge) {
      heroBreakerBadge.className = "breaker-state-badge breaker-tripped";
      heroBreakerBadge.textContent = "Breaker: TRIPPED (OPEN)";
    }
  } else {
    breakerBadge.className = "breaker-state-badge breaker-closed";
    breakerBadge.textContent = "Breaker: CLOSED (ONLINE)";
    if (heroBreakerBadge) {
      heroBreakerBadge.className = "breaker-state-badge breaker-closed";
      heroBreakerBadge.textContent = "Breaker: CLOSED (ONLINE)";
    }
  }

  // Action Banner Class
  actionBanner.className = "decision-action-banner";
  if (data.action === 0) actionBanner.classList.add("banner-maintain");
  else if (data.action === 1) actionBanner.classList.add("banner-increase");
  else if (data.action === 2 || data.action === 3) actionBanner.classList.add("banner-throttle");
  else if (data.action === 4) actionBanner.classList.add("banner-trip");

  actionNameText.textContent = data.action_name;
  if (heroActionDisplay) {
    heroActionDisplay.textContent = data.action_name.split(":")[1]?.trim() || data.action_name;
    heroActionDisplay.style.color = data.action === 4 ? "#ef4444" : "#38bdf8";
  }

  // Softmax Probabilities
  if (data.action_probabilities) {
    const highestProb = Math.max(...Object.values(data.action_probabilities));
    actionConfidenceText.textContent = `Confidence: ${(highestProb * 100).toFixed(1)}%`;

    const probContainer = document.getElementById("probListContainer");
    probContainer.innerHTML = "";

    Object.entries(data.action_probabilities).forEach(([actName, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const isSelected = actName === data.action_name;

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-label-row">
          <span style="${isSelected ? 'color: var(--text-main); font-weight: 700;' : ''}">${actName}</span>
          <span style="font-family: monospace;">${pct}%</span>
        </div>
        <div class="prob-bar-container">
          <div class="prob-bar-fill" style="width: ${pct}%; background: ${isSelected ? (data.action === 4 ? '#ef4444' : '#2563eb') : '#cbd5e1'};"></div>
        </div>
      `;
      probContainer.appendChild(row);
    });
  } else {
    actionConfidenceText.textContent = "Deterministic Relay Action";
  }

  // Next State & Reward
  document.getElementById("nextTempDisplay").textContent = `${data.next_temp.toFixed(1)}°C`;
  document.getElementById("nextLoadDisplay").textContent = `${(data.next_load * 100).toFixed(0)}%`;
  
  const rewEl = document.getElementById("rewardDisplay");
  rewEl.textContent = `${data.reward >= 0 ? '+' : ''}${data.reward.toFixed(1)}`;
  rewEl.style.color = data.reward >= 0 ? '#10b981' : '#ef4444';

  autoSetImageForState(data.status_flag, data.next_temp, data.next_anomaly_prob);
}

function toggleAutoLoop() {
  const btn = document.getElementById("autoLoopBtn");
  if (autoLoopTimer) {
    clearInterval(autoLoopTimer);
    autoLoopTimer = null;
    btn.innerHTML = "<span>🔄 Auto-Loop: OFF</span>";
    btn.classList.remove("btn-primary");
    btn.classList.add("btn-secondary");
  } else {
    autoLoopTimer = setInterval(executeStep, 1000);
    btn.innerHTML = "<span>⏹ Auto-Loop: ON</span>";
    btn.classList.remove("btn-secondary");
    btn.classList.add("btn-primary");
    showToast("Continuous closed-loop control enabled");
  }
}

// =============================================================================
// TRAJECTORY SIMULATION
// =============================================================================
async function runTrajectorySimulation() {
  const controller = document.getElementById("simControllerSelect").value;
  const regime = document.getElementById("simRegimeSelect").value;

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        controller: controller,
        regime: regime,
        max_steps: 50
      })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderSimulationCharts(data);
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

function renderSimulationCharts(data) {
  const badge = document.getElementById("simOutcomeBadge");
  badge.textContent = data.outcome.replace(/_/g, " ").toUpperCase();
  badge.style.color = data.outcome_color;

  document.getElementById("simStepsText").textContent = data.steps_executed;
  document.getElementById("simMaxTempText").textContent = `${data.max_temp.toFixed(1)}°C`;
  document.getElementById("simRewardText").textContent = `${data.total_reward >= 0 ? '+' : ''}${data.total_reward.toFixed(1)}`;

  const labels = data.trajectory.map(p => `Step ${p.step}`);
  const loads = data.trajectory.map(p => Math.round(p.load * 100));
  const temps = data.trajectory.map(p => Number(p.temp.toFixed(1)));
  const anoms = data.trajectory.map(p => Number(p.anomaly_prob.toFixed(2)));
  const energies = data.trajectory.map(p => Number(p.acoustic_energy.toFixed(2)));

  // Chart 1: Thermal & Load Dynamics
  const ctxThermal = document.getElementById("thermalChart").getContext("2d");
  if (thermalChartInstance) thermalChartInstance.destroy();
  thermalChartInstance = new Chart(ctxThermal, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Core Temp (°C)",
          data: temps,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.08)",
          fill: true,
          tension: 0.2,
          yAxisID: "yTemp"
        },
        {
          label: "Delivered Load (%)",
          data: loads,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.08)",
          fill: true,
          tension: 0.2,
          yAxisID: "yLoad"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        yTemp: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Temperature (°C)" },
          suggestedMin: 50,
          suggestedMax: 110
        },
        yLoad: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Load (%)" },
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false }
        }
      }
    }
  });

  // Chart 2: Acoustic PD & Energy
  const ctxAcoustic = document.getElementById("acousticChart").getContext("2d");
  if (acousticChartInstance) acousticChartInstance.destroy();
  acousticChartInstance = new Chart(ctxAcoustic, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Acoustic Anomaly Prob",
          data: anoms,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.08)",
          fill: true,
          tension: 0.2
        },
        {
          label: "Spectral Energy",
          data: energies,
          borderColor: "#0284c7",
          backgroundColor: "rgba(2, 132, 199, 0.08)",
          fill: true,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1.0,
          title: { display: true, text: "Probability / Magnitude" }
        }
      }
    }
  });
}

// =============================================================================
// COMPARATIVE BENCHMARKING
// =============================================================================
async function runBenchmark() {
  const epCount = parseInt(document.getElementById("benchmarkEpisodesInput").value) || 5;
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 24px; color: var(--text-muted);">Running Cyber-Physical Benchmark across 4 operational regimes...</td></tr>`;

  try {
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes_per_regime: epCount })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderBenchmarkResults(data.results);
  } catch (err) {
    console.error("Benchmark error:", err);
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444;">Failed to execute benchmark. Check API connection.</td></tr>`;
  }
}

function renderBenchmarkResults(results) {
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  const names = [];
  const burnouts = [];
  const falseBlackouts = [];
  const rewards = [];

  results.forEach(m => {
    names.push(m.controller_name);
    burnouts.push(m.catastrophic_burnout_rate_pct);
    falseBlackouts.push(m.false_alarm_blackout_rate_pct);
    rewards.push(Number(m.mean_reward.toFixed(1)));

    const isProposed = m.controller_id === "soundgrid_ppo";
    const tr = document.createElement("tr");
    if (isProposed) tr.className = "proposed-row";

    tr.innerHTML = `
      <td>
        <strong>${m.controller_name}</strong>
        ${isProposed ? '<span style="margin-left: 6px; font-size: 11px; background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px;">Proposed AI</span>' : ''}
      </td>
      <td style="color: ${m.mean_reward >= 0 ? '#10b981' : '#0f172a'}; font-weight: 600;">
        ${m.mean_reward.toFixed(1)}
      </td>
      <td style="color: ${m.catastrophic_burnout_rate_pct > 0 ? '#ef4444' : '#10b981'}; font-weight: 600;">
        ${m.catastrophic_burnout_rate_pct.toFixed(1)}%
      </td>
      <td style="color: ${m.false_alarm_blackout_rate_pct > 0 ? '#f59e0b' : '#10b981'}; font-weight: 600;">
        ${m.false_alarm_blackout_rate_pct.toFixed(1)}%
      </td>
      <td>${m.safe_emergency_trip_rate_pct.toFixed(1)}%</td>
      <td>${m.average_delivered_load_pct.toFixed(1)}%</td>
      <td><span class="badge-verdict" style="background: #f1f5f9;">${m.verdict}</span></td>
    `;
    tbody.appendChild(tr);
  });

  // Chart: Safety Comparison
  const ctxSafety = document.getElementById("benchmarkSafetyChart").getContext("2d");
  if (benchSafetyChartInstance) benchSafetyChartInstance.destroy();
  benchSafetyChartInstance = new Chart(ctxSafety, {
    type: "bar",
    data: {
      labels: names.map(n => n.split("(")[0]),
      datasets: [
        {
          label: "Burnout Rate % (Lower is Better)",
          data: burnouts,
          backgroundColor: "#ef4444"
        },
        {
          label: "False Blackout Rate % (Lower is Better)",
          data: falseBlackouts,
          backgroundColor: "#f59e0b"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: "Failure Percentage (%)" } }
      }
    }
  });

  // Chart: Mean Reward
  const ctxReward = document.getElementById("benchmarkRewardChart").getContext("2d");
  if (benchRewardChartInstance) benchRewardChartInstance.destroy();
  benchRewardChartInstance = new Chart(ctxReward, {
    type: "bar",
    data: {
      labels: names.map(n => n.split("(")[0]),
      datasets: [
        {
          label: "Cumulative Reward (Higher is Better)",
          data: rewards,
          backgroundColor: "#2563eb"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: "Mean Reward" } }
      }
    }
  });
}

// =============================================================================
// ACOUSTIC WAVEFORM VISUALIZERS (Dual Hero + Monitor)
// =============================================================================
function initAcousticWaveform() {
  const canvas = document.getElementById("acousticCanvas");
  const heroCanvas = document.getElementById("heroAcousticCanvas");
  let phase = 0;

  function renderWave() {
    const anom = parseFloat(document.getElementById("anomInput").value) / 100;
    const strokeColor = anom > 0.6 ? "#ef4444" : anom > 0.25 ? "#f59e0b" : "#10b981";
    const heroStrokeColor = anom > 0.6 ? "#f87171" : anom > 0.25 ? "#fbbf24" : "#34d399";

    // Draw main monitor canvas
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = strokeColor;
      ctx.beginPath();
      const mid = canvas.height / 2;
      for (let x = 0; x < canvas.width; x++) {
        const baseHum = Math.sin((x * 0.05) + phase) * 8;
        const pdBurst = anom > 0.15 ? (Math.random() - 0.5) * anom * 32 : 0;
        const y = mid + baseHum + pdBurst;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Draw hero HUD canvas
    if (heroCanvas) {
      const ctxH = heroCanvas.getContext("2d");
      ctxH.clearRect(0, 0, heroCanvas.width, heroCanvas.height);
      ctxH.lineWidth = 1.8;
      ctxH.strokeStyle = heroStrokeColor;
      ctxH.beginPath();
      const midH = heroCanvas.height / 2;
      for (let x = 0; x < heroCanvas.width; x++) {
        const baseHum = Math.sin((x * 0.05) + phase) * 6;
        const pdBurst = anom > 0.15 ? (Math.random() - 0.5) * anom * 22 : 0;
        const y = midH + baseHum + pdBurst;
        if (x === 0) ctxH.moveTo(x, y);
        else ctxH.lineTo(x, y);
      }
      ctxH.stroke();
    }

    phase += 0.15;
    requestAnimationFrame(renderWave);
  }

  requestAnimationFrame(renderWave);
}
