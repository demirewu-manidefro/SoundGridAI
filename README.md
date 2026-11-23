# SoundGrid AI: Acoustic-Driven Cyber-Physical Control for Power Transformers

SoundGrid AI is an autonomous predictive maintenance and dynamic control framework for power grid transformers and high-voltage assets. It integrates deep acoustic anomaly detection with reinforcement learning (PPO) to throttle operating loads and prevent catastrophic thermal failures without triggering false blackouts.

## System Architecture

```
[Acoustic Audio Waveform]
           │
           ▼
[PyTorch Mel-Spectrogram Engine (128x94)]
           │
           ▼
[SoundGrid CNN Classifier] ──► Anomaly Risk P(anomaly)
           │
           ▼
[Cyber-Physical Gymnasium Environment]
           │  (Thermal dynamics: I^2R copper loss + dielectric arc heat)
           ▼
[Autonomous PPO Controller]
           │
           ├──► [0] Maintain Load
           ├──► [1] Increase Load (+10%)
           ├──► [2] Throttle Load (-15%)
           ├──► [3] Throttle Load (-30%)
           └──► [4] Emergency Trip (Shutdown)
```

## Key Capabilities

- **Ultra-Fast Mel-Spectrogram Processing:** Pure PyTorch implementation computing Mel-spectrograms in under 15 ms, without external audio compilation delays.
- **Acoustic Perception:** 4-stage convolutional neural network trained on industrial acoustic signatures, achieving 85.08% validation accuracy and 0.9258 ROC-AUC.
- **Physics-Informed Reinforcement Learning:** Gymnasium environment modeling transformer heating and cooling dynamics with a multi-objective reward balancing energy delivery against thermal degradation.
- **Reliability:** 0% false alarm tripping rate and 0% catastrophic thermal failures across multi-episode operational stress evaluations.
- **Edge Deployment:** Compiled to TorchScript (1.6 MB) for low-latency substation edge devices.

## Repository Structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── soundgrid_pipeline.py       # End-to-end streaming inference and control script
├── train_integrated_rl.py      # Gymnasium environment and PPO training pipeline
├── docs/
│   └── RESEARCH_PAPER_BLUEPRINT.md  # Academic paper draft and mathematical model
├── models/
│   └── soundgrid_web_model.pt       # TorchScript CNN acoustic anomaly model
└── notebooks/
    ├── 01_acoustic_anomaly_detection.ipynb   # Spectrogram extraction & CNN training
    └── 02_reinforcement_learning_control.ipynb # Integrated Gym environment & PPO agent
```

## Quickstart

### Installation

```bash
git clone https://github.com/demirewu-manidefro/SoundGridAI.git
cd SoundGridAI
pip install -r requirements.txt
```

### Run End-to-End Inference Pipeline

```bash
python soundgrid_pipeline.py
```

### Train Reinforcement Learning Agent

```bash
python train_integrated_rl.py
```

## Research & Academic Blueprint

See `docs/RESEARCH_PAPER_BLUEPRINT.md` for complete mathematical formulations, system equations, baseline benchmark setups, and target publication venues.

## Author

**Demirewu Manidefro**  
Email: demirewumanidefro@gmail.com
