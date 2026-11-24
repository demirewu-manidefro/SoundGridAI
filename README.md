# VoltEcho: Cyber-Physical RL for Power Transformer Protection

VoltEcho is an autonomous cyber-physical intelligence platform for high-voltage power transformer protection. It couples deep ultrasonic acoustic perception (10–150 kHz log-Mel spectrograms) with an 8D derivative-augmented Proximal Policy Optimization (PPO) reinforcement learning agent to prevent catastrophic transformer explosions and dielectric burnouts before thermal runaway occurs.

---

## Quickstart: How to Run This Project

### 1. Launch the Interactive Web Platform & Live Studio (Recommended)

You can launch the complete web application with a single command or by double-clicking `run_platform.bat`:

#### Option A: Double-Click Launcher (Windows)
Double-click **`run_platform.bat`** in the project folder. It will activate the virtual environment, start the server, and automatically open your web browser.

#### Option B: Terminal / PowerShell
```powershell
# 1. Activate the Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Launch the FastAPI server
python -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
```
Then open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

### 2. Run the Scientific Model Evaluation CLI

To run the rigorous 100-episode Monte Carlo evaluation benchmark comparing VoltEcho against standard IEC 60255 industrial relays and PID controllers:

```powershell
.\.venv\Scripts\python.exe evaluate_soundgrid_ppo.py
```

This tests 4 operational regimes:
- **Normal Transformer Hum** (100 Hz magnetostriction)
- **Developing Partial Discharge** (PD cavitation)
- **Severe Dielectric Arcing Flashovers** (Critical 1-step trip verification)
- **Extreme Heatwave Overload** (Ambient 45°C stress)

---

### 3. Run the Streaming Audio Inference Pipeline CLI

To execute real-time audio log-Mel feature extraction and direct PPO policy decision-making:

```powershell
.\.venv\Scripts\python.exe soundgrid_pipeline.py
```

---

### 4. Re-Train the Reinforcement Learning Agent

To train the 8D PPO agent from scratch within the physics-informed Gymnasium environment:

```powershell
.\.venv\Scripts\python.exe train_integrated_rl.py
```

The resulting weights will be saved to:
- `models/soundgrid_integrated_ppo.zip` (Stable-Baselines3 checkpoint)
- `models/soundgrid_web_model.pt` (TorchScript C++ deployment binary)

---

### 5. Open Jupyter Research Notebooks

To interactively inspect the Mel-spectrogram processing and Gymnasium training:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```
Open either:
- `notebooks/01_acoustic_anomaly_detection.ipynb`
- `notebooks/02_reinforcement_learning_control.ipynb`

---

## System Architecture

```
[Piezoelectric Hydrophones in Oil]
               │
               ▼
[PyTorch Mel-Spectrogram Engine (128 Bands)]
               │
               ▼
[VoltEchoCNN Classifier] ──► Anomaly Probability P_anom & Physical RMS Energy
               │
               ▼
[8D Cyber-Physical Gymnasium Environment]
  State: S = [P, ΔP, E, L, ΔL, T/100, ΔT/10, t/t_max] ∈ ℝ⁸
  Thermodynamics: dT/dt = (4.2 L² + 10 P) - 0.09 (T - T_ambient)
               │
               ▼
[Autonomous 8D PPO Controller]
               │
               ├──► [0] Maintain Load (Nominal Dispatch)
               ├──► [1] Increase Load (+10%)
               ├──► [2] Throttle Load (-15%)
               ├──► [3] Throttle Load (-30%)
               └──► [4] Emergency Circuit Breaker Trip (Sub-second shutdown)
```

---

## Repository Structure

```
├── run_platform.bat               # 1-click Windows launcher for Web Studio
├── requirements.txt               # Python package dependencies
├── evaluate_soundgrid_ppo.py      # Monte Carlo 100-episode benchmark script
├── soundgrid_pipeline.py          # End-to-end streaming audio inference
├── train_integrated_rl.py         # Gymnasium environment & PPO training
├── models/
│   ├── soundgrid_integrated_ppo.zip # Trained PyTorch PPO checkpoint
│   └── soundgrid_web_model.pt       # TorchScript C++ model (1.6 MB)
├── web/
│   ├── backend/
│   │   └── main.py                # FastAPI REST API & live inference routes
│   └── frontend/
│       ├── index.html             # Handcrafted INSA-style research interface
│       ├── style.css              # Modern engineering stylesheet
│       ├── app.js                 # Interactive telemetry & Chart.js logic
│       └── images/                # Substation telemetry, burnout incident, logo
└── docs/
    └── RESEARCH_PAPER_BLUEPRINT.md # Academic manuscript for IEEE Transactions
```

---

## Author & Citation

**Demirewu Manidefro**  
Cyber-Physical Smart Grid Lab  
Email: [demirewu.research@gmail.com](mailto:demirewu.research@gmail.com)  
GitHub: [https://github.com/demirewu-manidefro/SoundGridAI](https://github.com/demirewu-manidefro/SoundGridAI)
