import numpy as np
from typing import Dict, Any, List, Tuple
from .schemas import (
    StepRequest, StepResponse, SimulateRequest, SimulateResponse,
    TrajectoryPoint, BenchmarkRequest, BenchmarkResponse, ControllerBenchmarkMetrics
)
from .controllers import get_controller, ACTION_NAMES

class CyberPhysicalSimulator:
    """Rigorous cyber-physical transformer environment simulation engine."""

    @staticmethod
    def construct_state(
        anomaly: float, prev_anomaly: float,
        energy: float, load: float, prev_load: float,
        temp: float, prev_temp: float,
        step: int, max_steps: int
    ) -> np.ndarray:
        delta_anomaly = float(np.clip(anomaly - prev_anomaly, -1.0, 1.0))
        delta_load = float(np.clip(load - prev_load, -1.0, 1.0))
        temp_norm = float(temp / 100.0)
        delta_temp = float(np.clip((temp - prev_temp) / 10.0, -1.0, 1.0))
        time_progress = float(step / max_steps)

        return np.array([
            anomaly, delta_anomaly, energy,
            load, delta_load, temp_norm, delta_temp, time_progress
        ], dtype=np.float32)

    @classmethod
    def execute_step(cls, req: StepRequest) -> StepResponse:
        ctrl = get_controller(req.controller)

        obs = cls.construct_state(
            req.anomaly_prob, req.prev_anomaly,
            req.acoustic_energy, req.load, req.prev_load,
            req.temp, req.prev_temp,
            req.step_count, req.max_steps
        )

        act, probs = ctrl.select_action(obs)

        # Apply control action
        next_load = req.load
        if act == 0:
            pass
        elif act == 1:
            next_load = min(1.0, req.load + 0.10)
        elif act == 2:
            next_load = max(0.10, req.load - 0.15)
        elif act == 3:
            next_load = max(0.10, req.load - 0.30)
        elif act == 4:
            next_load = 0.0

        # Physical thermal dynamics
        q_gen = (next_load**2 * 4.2) + (req.anomaly_prob * 10.0)
        q_diss = 0.09 * (req.temp - req.ambient_temp)
        next_temp = float(np.clip(req.temp + q_gen - q_diss, 30.0, 130.0))
        delta_t = next_temp - req.temp

        # Unified canonical reward computation
        reward = next_load * 16.0
        terminated = False
        status = "healthy"

        if req.temp <= 78.0 and req.anomaly_prob < 0.25:
            if next_load < 0.85:
                if act == 1: reward += 14.0
                elif act == 0: reward += 8.0
            elif 0.85 <= next_load <= 0.95:
                if act == 0: reward += 16.0
                elif act == 1: reward += 0.0
            else:
                if act == 0: reward += 14.0
                elif act == 1: reward -= 10.0
            if act in [2, 3]:
                reward -= 18.0

        if next_temp > 80.0:
            reward -= 2.5 * (next_temp - 80.0)

        if req.anomaly_prob >= 0.30 and req.anomaly_prob < 0.70 and next_temp < 100.0:
            status = "derated"
            if act in [2, 3]:
                if next_load > 0.35: reward += 28.0
            elif act == 0:
                if next_load <= 0.40: reward += 16.0
                else: reward -= 25.0
            elif act == 1:
                reward -= 60.0

        if next_temp > 80.0 and req.anomaly_prob < 0.30:
            status = "cooling"
            if act in [2, 3]:
                if next_load > 0.25: reward += 28.0
            elif act == 0:
                if next_load <= 0.30: reward += 16.0
                else: reward -= 25.0 + 3.0 * (next_temp - 80.0)
            elif act == 1:
                reward -= 60.0

        if act == 4:
            if req.anomaly_prob >= 0.70 or req.temp >= 100.0:
                status = "emergency_trip"
                remaining_steps = req.max_steps - req.step_count
                reward += 140.0 + remaining_steps * 8.0
            else:
                status = "false_trip"
                reward -= 180.0
            terminated = True

        if next_temp >= 118.0 and not terminated:
            status = "burnout"
            reward -= 350.0
            terminated = True

        if req.step_count >= req.max_steps:
            terminated = True

        # Acoustic evolution for next step
        next_anom = float(np.clip(req.anomaly_prob + np.random.normal(0.0, 0.01), 0.01, 1.0))
        next_energy = float(np.clip(next_anom * 0.85 + 0.10, 0.05, 1.0))

        return StepResponse(
            action=act,
            action_name=ACTION_NAMES[act],
            action_probabilities=probs,
            next_load=round(next_load, 3),
            next_temp=round(next_temp, 2),
            next_anomaly_prob=round(next_anom, 4),
            next_acoustic_energy=round(next_energy, 4),
            reward=round(reward, 2),
            terminated=terminated,
            status_flag=status,
            details={
                "q_gen": round(q_gen, 3),
                "q_diss": round(q_diss, 3),
                "delta_t": round(delta_t, 3)
            }
        )

    @classmethod
    def simulate_episode(cls, req: SimulateRequest) -> SimulateResponse:
        ctrl = get_controller(req.controller)
        if hasattr(ctrl, "reset"):
            ctrl.reset()

        # Regime initialization
        if req.regime == "normal":
            ambient = 30.0
            load = req.initial_load if req.initial_load is not None else float(np.random.uniform(0.75, 0.90))
            temp = req.initial_temp if req.initial_temp is not None else float(np.random.uniform(55.0, 68.0))
            anom = float(np.random.uniform(0.02, 0.06))
        elif req.regime == "warning":
            ambient = 30.0
            load = req.initial_load if req.initial_load is not None else float(np.random.uniform(0.80, 0.90))
            temp = req.initial_temp if req.initial_temp is not None else float(np.random.uniform(65.0, 75.0))
            anom = float(np.random.uniform(0.38, 0.52))
        elif req.regime == "critical":
            ambient = 30.0
            load = req.initial_load if req.initial_load is not None else float(np.random.uniform(0.80, 0.90))
            temp = req.initial_temp if req.initial_temp is not None else float(np.random.uniform(68.0, 78.0))
            anom = float(np.random.uniform(0.80, 0.94))
        else: # thermal
            ambient = float(np.random.uniform(38.0, 44.0))
            load = req.initial_load if req.initial_load is not None else float(np.random.uniform(0.85, 0.95))
            temp = req.initial_temp if req.initial_temp is not None else float(np.random.uniform(86.0, 96.0))
            anom = float(np.random.uniform(0.04, 0.12))

        prev_load = load
        prev_temp = temp
        prev_anom = anom
        energy = float(np.clip(anom * 0.85 + 0.10, 0.05, 1.0))

        trajectory: List[TrajectoryPoint] = []
        total_reward = 0.0
        max_temp = temp
        outcome = "normal_completion"
        outcome_color = "#10b981"  # Emerald

        for step in range(1, req.max_steps + 1):
            obs = cls.construct_state(
                anom, prev_anom, energy,
                load, prev_load, temp, prev_temp,
                step, req.max_steps
            )

            act, probs = ctrl.select_action(obs)

            prev_load = load
            prev_temp = temp
            prev_anom = anom

            # Action execution
            if act == 0: pass
            elif act == 1: load = min(1.0, load + 0.10)
            elif act == 2: load = max(0.10, load - 0.15)
            elif act == 3: load = max(0.10, load - 0.30)
            elif act == 4: load = 0.0

            q_gen = (load**2 * 4.2) + (anom * 10.0)
            q_diss = 0.09 * (temp - ambient)
            temp = float(np.clip(temp + q_gen - q_diss, 30.0, 130.0))
            max_temp = max(max_temp, temp)

            # Unified reward
            r = load * 16.0
            term = False

            if temp <= 78.0 and anom < 0.25:
                if load < 0.85: r += 14.0 if act == 1 else 8.0
                elif 0.85 <= load <= 0.95: r += 16.0 if act == 0 else 0.0
                else: r += 14.0 if act == 0 else -10.0
                if act in [2, 3]: r -= 18.0

            if temp > 80.0:
                r -= 2.5 * (temp - 80.0)

            if anom >= 0.30 and anom < 0.70 and temp < 100.0:
                if act in [2, 3] and load > 0.35: r += 28.0
                elif act == 0: r += 16.0 if load <= 0.40 else -25.0
                elif act == 1: r -= 60.0

            if temp > 80.0 and anom < 0.30:
                if act in [2, 3] and load > 0.25: r += 28.0
                elif act == 0: r += 16.0 if load <= 0.30 else -25.0 - 3.0*(temp - 80.0)
                elif act == 1: r -= 60.0

            if act == 4:
                term = True
                if anom >= 0.70 or temp >= 100.0:
                    r += 140.0 + (req.max_steps - step) * 8.0
                    outcome = "safe_emergency_trip"
                    outcome_color = "#3b82f6"  # Blue
                else:
                    r -= 180.0
                    outcome = "false_alarm_blackout"
                    outcome_color = "#f59e0b"  # Amber

            if temp >= 118.0 and not term:
                r -= 350.0
                outcome = "catastrophic_thermal_burnout"
                outcome_color = "#ef4444"  # Red
                term = True

            total_reward += r
            trajectory.append(TrajectoryPoint(
                step=step,
                load=round(load, 3),
                temp=round(temp, 2),
                anomaly_prob=round(anom, 4),
                acoustic_energy=round(energy, 4),
                action=act,
                action_name=ACTION_NAMES[act],
                reward=round(r, 2),
                cumulative_reward=round(total_reward, 2)
            ))

            if term:
                break

            # Natural drift within regime
            if req.regime == "normal":
                anom = float(np.clip(anom + np.random.normal(0.0, 0.008), 0.01, 0.08))
            elif req.regime == "warning":
                anom = float(np.clip(anom + np.random.normal(0.005, 0.015), 0.35, 0.65))
            elif req.regime == "critical":
                anom = float(np.clip(anom + np.random.normal(0.01, 0.02), 0.78, 0.98))
            elif req.regime == "thermal":
                anom = float(np.clip(anom + np.random.normal(0.0, 0.008), 0.04, 0.15))
            energy = float(np.clip(anom * 0.85 + 0.10, 0.05, 1.0))

        return SimulateResponse(
            controller=ctrl.__class__.__name__,
            regime=req.regime.value,
            steps_executed=len(trajectory),
            total_reward=round(total_reward, 2),
            max_temp=round(max_temp, 2),
            final_load=round(load, 3),
            outcome=outcome,
            outcome_color=outcome_color,
            trajectory=trajectory
        )

    @classmethod
    def run_benchmark(cls, req: BenchmarkRequest) -> BenchmarkResponse:
        controllers_to_test = [
            ("SoundGrid AI (8D PPO)", "soundgrid_ppo"),
            ("Conventional Relay (IEC 60255)", "conventional_relay"),
            ("Classical PID Controller", "pid_controller"),
            ("Static 4D RL (Ablation Baseline)", "static_4d_rl")
        ]

        regimes = ["normal", "warning", "critical", "thermal"]
        results: List[ControllerBenchmarkMetrics] = []

        total_episodes = req.episodes_per_regime * len(regimes)

        for name, cid in controllers_to_test:
            all_rewards = []
            all_durations = []
            burnouts = 0
            false_trips = 0
            safe_trips = 0
            all_loads = []
            arcing_latencies = []

            for reg in regimes:
                for ep in range(req.episodes_per_regime):
                    sim_req = SimulateRequest(
                        controller=cid,
                        regime=reg,
                        max_steps=100
                    )
                    resp = cls.simulate_episode(sim_req)
                    all_rewards.append(resp.total_reward)
                    all_durations.append(resp.steps_executed)
                    all_loads.append(resp.final_load)

                    if resp.outcome == "catastrophic_thermal_burnout":
                        burnouts += 1
                    elif resp.outcome == "false_alarm_blackout":
                        false_trips += 1
                    elif resp.outcome == "safe_emergency_trip":
                        safe_trips += 1
                        if reg == "critical":
                            arcing_latencies.append(resp.steps_executed)

            avg_arcing_speed = f"{np.mean(arcing_latencies):.1f} steps" if arcing_latencies else "N/A"
            
            # Formulate verdict based on safety
            burnout_pct = (burnouts / total_episodes) * 100.0
            false_pct = (false_trips / total_episodes) * 100.0
            if burnout_pct == 0.0 and false_pct <= 5.0:
                verdict = "✅ Optimal Publication-Grade"
            elif burnout_pct > 20.0:
                verdict = "❌ Unacceptable Fire/Burnout Hazard"
            elif false_pct > 20.0:
                verdict = "⚠️ High False-Alarm Grid Instability"
            else:
                verdict = "⚠️ Sub-optimal Trade-off"

            results.append(ControllerBenchmarkMetrics(
                controller_name=name,
                controller_id=cid,
                mean_reward=round(float(np.mean(all_rewards)), 2),
                mean_duration=round(float(np.mean(all_durations)), 1),
                catastrophic_burnout_rate_pct=round(burnout_pct, 1),
                false_alarm_blackout_rate_pct=round(false_pct, 1),
                safe_emergency_trip_rate_pct=round((safe_trips / total_episodes) * 100.0, 1),
                average_delivered_load_pct=round(float(np.mean(all_loads)) * 100.0, 1),
                arcing_response_speed=avg_arcing_speed,
                verdict=verdict
            ))

        return BenchmarkResponse(
            episodes_tested=total_episodes,
            results=results
        )
