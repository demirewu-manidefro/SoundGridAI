import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

from .schemas import (
    StepRequest, StepResponse,
    SimulateRequest, SimulateResponse,
    BenchmarkRequest, BenchmarkResponse
)
from .controllers import get_controller, ACTION_NAMES
from .simulator import CyberPhysicalSimulator

app = FastAPI(
    title="SoundGrid AI - Cyber-Physical Control API",
    description="FastAPI Backend for Acoustic-Driven Power Transformer Reinforcement Learning Control & Comparative Benchmarking",
    version="1.0.0"
)

# Enable CORS for future frontend connection (React, Vue, or vanilla JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Confirms system health, model presence, and runtime dependencies."""
    model_paths = [
        "models/soundgrid_integrated_ppo.zip",
        "notebooks/soundgrid_integrated_ppo.zip"
    ]
    model_found = any(os.path.exists(p) for p in model_paths)
    cnn_found = os.path.exists("models/soundgrid_web_model.pt")

    return {
        "status": "healthy" if model_found else "degraded",
        "service": "SoundGrid AI Backend API",
        "version": "1.0.0",
        "ppo_model_loaded": model_found,
        "cnn_perception_loaded": cnn_found,
        "supported_controllers": [
            "soundgrid_ppo",
            "conventional_relay",
            "pid_controller",
            "static_4d_rl"
        ]
    }


@app.get("/api/controllers")
def list_controllers() -> List[Dict[str, Any]]:
    """Lists available controllers with industrial descriptions."""
    return [
        {
            "id": "soundgrid_ppo",
            "name": "SoundGrid AI (8D PPO Agent)",
            "description": "Production acoustic-driven PPO policy with 8D derivative-augmented state space.",
            "is_proposed": True
        },
        {
            "id": "conventional_relay",
            "name": "Conventional Industrial Relay (IEC 60255)",
            "description": "Industry-standard thermal and overcurrent protection relay. Acoustically blind.",
            "is_proposed": False
        },
        {
            "id": "pid_controller",
            "name": "Classical PID Controller",
            "description": "Continuous thermal setpoint regulator (Target 72.0°C). Cannot handle sudden arcing disconnects.",
            "is_proposed": False
        },
        {
            "id": "static_4d_rl",
            "name": "Static 4D RL (Ablation Baseline)",
            "description": "Reinforcement learning policy without temporal rate-of-change derivatives (POMDP).",
            "is_proposed": False
        }
    ]


@app.get("/api/regimes")
def list_regimes() -> List[Dict[str, Any]]:
    """Lists operational transformer simulation regimes."""
    return [
        {
            "id": "normal",
            "name": "Normal Dispatch",
            "description": "Nominal transformer operation with baseline magnetostriction hum (100Hz) and cool core.",
            "color": "#10b981",
            "default_anom": 0.04,
            "default_temp": 62.0
        },
        {
            "id": "warning",
            "name": "Incipient Partial Discharge",
            "description": "Developing dielectric breakdown with intermittent discharge bursts. Requires controlled derating.",
            "color": "#f59e0b",
            "default_anom": 0.45,
            "default_temp": 72.0
        },
        {
            "id": "critical",
            "name": "Severe Arcing Hazard",
            "description": "Violent continuous high-voltage arcing discharges. Requires immediate emergency trip.",
            "color": "#ef4444",
            "default_anom": 0.88,
            "default_temp": 75.0
        },
        {
            "id": "thermal",
            "name": "Extreme Thermal Overload",
            "description": "High ambient heatwave (40°C+) coupled with peak electrical grid transmission.",
            "color": "#f97316",
            "default_anom": 0.08,
            "default_temp": 92.0
        }
    ]


@app.post("/api/step", response_model=StepResponse)
def execute_single_step(req: StepRequest):
    """Executes a single cyber-physical decision step for live interactive control."""
    try:
        return CyberPhysicalSimulator.execute_step(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate_full_episode(req: SimulateRequest):
    """Runs a full cyber-physical episode and returns complete time-series trajectories."""
    try:
        return CyberPhysicalSimulator.simulate_episode(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark", response_model=BenchmarkResponse)
def run_comparative_benchmark(req: BenchmarkRequest):
    """
    Executes head-to-head benchmarking across all 4 controllers.
    Generates side-by-side scientific comparison metrics for dissertation & publication.
    """
    try:
        return CyberPhysicalSimulator.run_benchmark(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.backend.main:app", host="127.0.0.1", port=8000, reload=True)
