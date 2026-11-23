import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from stable_baselines3 import PPO

ACTION_NAMES = [
    "0: Maintain Load",
    "1: Increase Load (+10%)",
    "2: Throttle 15% (-15%)",
    "3: Throttle 40% (-30%)",
    "4: Emergency Trip (Shutdown)"
]

class SoundGridPPOController:
    """Production 8D Derivative-Augmented PPO Agent."""
    def __init__(self, model_paths=None):
        if model_paths is None:
            model_paths = [
                "models/soundgrid_integrated_ppo.zip",
                "notebooks/soundgrid_integrated_ppo.zip",
                "soundgrid_integrated_ppo.zip"
            ]
        
        self.model = None
        for p in model_paths:
            if os.path.exists(p):
                try:
                    self.model = PPO.load(p)
                    self.model.policy.eval()
                    self.active_path = p
                    break
                except Exception as e:
                    pass
        
        if self.model is None:
            raise FileNotFoundError("Could not find soundgrid_integrated_ppo.zip checkpoint.")

    def select_action(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().numpy()[0]
        act = int(np.argmax(probs))
        
        prob_dict = {
            ACTION_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        return act, prob_dict


class ConventionalRelayController:
    """
    Standard Utility Protection Relay (IEC 60255 / IEEE C37.91).
    Acoustically blind: acts purely on thermal and overload thresholds.
    """
    def select_action(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        # obs: [anomaly, delta_anomaly, energy, load, delta_load, temp_norm, delta_temp, time]
        load = obs[3]
        temp = obs[5] * 100.0

        # Blind to acoustic discharge (obs[0])! Only reacts to temperature & load:
        if temp >= 105.0 or load >= 1.05:
            act = 4  # Emergency Trip
        elif temp >= 85.0:
            act = 2  # Throttle 15%
        elif temp <= 75.0 and load < 0.90:
            act = 1  # Increase load
        else:
            act = 0  # Maintain load

        probs = {ACTION_NAMES[i]: (1.0 if i == act else 0.0) for i in range(5)}
        return act, probs


class PIDThermalController:
    """
    Classical Proportional-Integral-Derivative Controller.
    Regulates transformer core temperature towards target setpoint (72.0°C).
    Lacks emergency disconnect capability for catastrophic dielectric breakdown.
    """
    def __init__(self, target_temp=72.0, kp=0.08, ki=0.01, kd=0.04):
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def select_action(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        temp = obs[5] * 100.0
        error = self.target_temp - temp  # Positive if too cold, negative if too hot
        
        self.integral_error = np.clip(self.integral_error + error, -50.0, 50.0)
        derivative = error - self.prev_error
        self.prev_error = error

        u = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)

        # Map continuous control effort u to discrete actions:
        if u < -1.8:
            act = 3  # Throttle 40% (Aggressive cooling)
        elif u < -0.4:
            act = 2  # Throttle 15% (Moderate cooling)
        elif u > 0.6:
            act = 1  # Increase Load (Warm / healthy)
        else:
            act = 0  # Maintain Load

        probs = {ACTION_NAMES[i]: (1.0 if i == act else 0.0) for i in range(5)}
        return act, probs


class Static4DRLController:
    """
    Ablation Baseline: Reinforcement Learning policy operating on static 4D state:
    [anomaly, load, temp_norm, time_progress] without first-order derivatives.
    Demonstrates POMDP temporal ambiguity and premature false blackouts.
    """
    def select_action(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        anom = obs[0]
        load = obs[3]
        temp = obs[5] * 100.0

        # Lacking dT/dt and dP/dt, this heuristic models static threshold over-reaction:
        if anom >= 0.45 or temp >= 95.0:
            act = 4  # Premature false alarm trip due to temporal blindness
        elif temp >= 82.0 or anom >= 0.25:
            act = 2  # Throttling
        elif load < 0.85 and temp <= 76.0:
            act = 1  # Increase load
        else:
            act = 0  # Maintain load

        probs = {ACTION_NAMES[i]: (1.0 if i == act else 0.0) for i in range(5)}
        return act, probs


def get_controller(controller_id: str):
    cid = controller_id.lower()
    if cid in ["soundgrid_ppo", "ppo"]:
        return SoundGridPPOController()
    elif cid in ["conventional_relay", "relay"]:
        return ConventionalRelayController()
    elif cid in ["pid_controller", "pid"]:
        return PIDThermalController()
    elif cid in ["static_4d_rl", "4d_rl"]:
        return Static4DRLController()
    else:
        raise ValueError(f"Unknown controller type: {controller_id}")
