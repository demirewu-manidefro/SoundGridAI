from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ControllerType(str, Enum):
    SOUNDGRID_PPO = "soundgrid_ppo"
    CONVENTIONAL_RELAY = "conventional_relay"
    PID_CONTROLLER = "pid_controller"
    STATIC_4D_RL = "static_4d_rl"

class OperationalRegime(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    THERMAL = "thermal"

class StepRequest(BaseModel):
    controller: ControllerType = Field(default=ControllerType.SOUNDGRID_PPO)
    load: float = Field(default=0.85, ge=0.0, le=1.0)
    prev_load: float = Field(default=0.85, ge=0.0, le=1.0)
    temp: float = Field(default=65.0, ge=20.0, le=140.0)
    prev_temp: float = Field(default=65.0, ge=20.0, le=140.0)
    anomaly_prob: float = Field(default=0.04, ge=0.0, le=1.0)
    prev_anomaly: float = Field(default=0.04, ge=0.0, le=1.0)
    acoustic_energy: float = Field(default=0.15, ge=0.0, le=1.0)
    ambient_temp: float = Field(default=30.0)
    step_count: int = Field(default=1, ge=0)
    max_steps: int = Field(default=100)

class StepResponse(BaseModel):
    action: int
    action_name: str
    action_probabilities: Optional[Dict[str, float]] = None
    next_load: float
    next_temp: float
    next_anomaly_prob: float
    next_acoustic_energy: float
    reward: float
    terminated: bool
    status_flag: str  # "healthy", "cooling", "derated", "emergency_trip", "burnout", "false_trip"
    details: Dict[str, Any]

class SimulateRequest(BaseModel):
    controller: ControllerType = Field(default=ControllerType.SOUNDGRID_PPO)
    regime: OperationalRegime = Field(default=OperationalRegime.NORMAL)
    max_steps: int = Field(default=100, ge=10, le=200)
    initial_load: Optional[float] = None
    initial_temp: Optional[float] = None

class TrajectoryPoint(BaseModel):
    step: int
    load: float
    temp: float
    anomaly_prob: float
    acoustic_energy: float
    action: int
    action_name: str
    reward: float
    cumulative_reward: float

class SimulateResponse(BaseModel):
    controller: str
    regime: str
    steps_executed: int
    total_reward: float
    max_temp: float
    final_load: float
    outcome: str
    outcome_color: str
    trajectory: List[TrajectoryPoint]

class BenchmarkRequest(BaseModel):
    episodes_per_regime: int = Field(default=10, ge=1, le=50)

class ControllerBenchmarkMetrics(BaseModel):
    controller_name: str
    controller_id: str
    mean_reward: float
    mean_duration: float
    catastrophic_burnout_rate_pct: float
    false_alarm_blackout_rate_pct: float
    safe_emergency_trip_rate_pct: float
    average_delivered_load_pct: float
    arcing_response_speed: str
    verdict: str

class BenchmarkResponse(BaseModel):
    episodes_tested: int
    results: List[ControllerBenchmarkMetrics]
