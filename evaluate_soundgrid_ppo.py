"""
SoundGrid AI - Comprehensive Scientific Evaluation Engine
==========================================================
Evaluates the trained PPO model located at:
'notebooks/soundgrid_integrated_ppo.zip'

Conducts:
1. Architectural & weight tensor inspection.
2. 100-Episode Monte Carlo evaluation across 4 distinct operational regimes:
   - Regime I:   Normal Grid Operation (25 episodes)
   - Regime II:  Developing Partial Discharges (25 episodes)
   - Regime III: Severe Electrical Arcing (25 episodes)
   - Regime IV:  Thermal Stress & Heat Saturation (25 episodes)
3. Exact calculation of:
   - Mean, std, min, max rewards
   - Mean episode duration / step counts
   - False Alarm Blackout Trips (Trip with Anom < 0.70 and Temp < 100°C)
   - Justified Safety Trips (Trip with Anom >= 0.70 or Temp >= 100°C)
   - Catastrophic Overheating Failures (Temp >= 118°C)
   - Complete Action Selection Breakdown (Counts and %) for Actions 0, 1, 2, 3, 4
4. Execution and stress-testing of 3 critical operational edge cases:
   - High-load normal dispatch
   - Rapidly rising thermal gradient
   - Severe electrical arcing bursts
"""

import os
import sys
import json
import numpy as np
import torch
from stable_baselines3 import PPO

MODEL_PATH = "notebooks/soundgrid_integrated_ppo.zip"
TOTAL_EPISODES = 100
EPISODES_PER_CONDITION = 25

class EvaluationSoundGridEnv:
    """Rigorous evaluation environment reproducing the cyber-physical dynamics."""
    ACTION_NAMES = [
        "0: Maintain Load",
        "1: Increase Load (+10%)",
        "2: Throttle 15% (-15%)",
        "3: Throttle 40% (-30%)",
        "4: Emergency Trip (Shutdown)"
    ]

    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.ambient_temp = 30.0
        self.load = 0.75
        self.prev_load = 0.75
        self.temp = 62.0
        self.prev_temp = 62.0
        self.anomaly_prob = 0.05
        self.prev_anomaly = 0.05
        self.acoustic_energy = 0.15
        self.step_count = 0

    def reset(self, load=None, temp=None, anomaly=None, ambient_temp=30.0):
        self.step_count = 0
        self.ambient_temp = ambient_temp
        self.load = float(load if load is not None else np.random.uniform(0.65, 0.85))
        self.prev_load = self.load
        self.temp = float(temp if temp is not None else np.random.uniform(55.0, 72.0))
        self.prev_temp = self.temp
        self.anomaly_prob = float(anomaly if anomaly is not None else 0.05)
        self.prev_anomaly = self.anomaly_prob
        self.acoustic_energy = float(np.clip(self.anomaly_prob * 0.8 + np.random.uniform(0.05, 0.15), 0.05, 1.0))
        return self._get_state()

    def _get_state(self):
        delta_anomaly = float(np.clip(self.anomaly_prob - self.prev_anomaly, -1.0, 1.0))
        delta_load = float(np.clip(self.load - self.prev_load, -1.0, 1.0))
        temp_norm = float(self.temp / 100.0)
        delta_temp = float(np.clip((self.temp - self.prev_temp) / 10.0, -1.0, 1.0))
        time_progress = float(self.step_count / self.max_steps)

        return np.array([
            self.anomaly_prob,
            delta_anomaly,
            self.acoustic_energy,
            self.load,
            delta_load,
            temp_norm,
            delta_temp,
            time_progress
        ], dtype=np.float32)

    def step(self, action, next_anomaly, next_energy=None):
        self.step_count += 1
        self.prev_load = self.load
        self.prev_temp = self.temp
        self.prev_anomaly = self.anomaly_prob

        # Update acoustic anomaly and energy for current step
        self.anomaly_prob = float(next_anomaly)
        if next_energy is not None:
            self.acoustic_energy = float(next_energy)
        else:
            self.acoustic_energy = float(np.clip(self.anomaly_prob * 0.85 + 0.10, 0.05, 1.0))

        # Apply control action
        if action == 0:
            pass # Maintain
        elif action == 1:
            self.load = min(1.0, self.load + 0.10)
        elif action == 2:
            self.load = max(0.10, self.load - 0.15)
        elif action == 3:
            self.load = max(0.10, self.load - 0.30)
        elif action == 4:
            self.load = 0.0 # Emergency Trip

        # Non-linear thermal physics
        q_gen = (self.load**2 * 4.2) + (self.anomaly_prob * 10.0)
        q_diss = 0.09 * (self.temp - self.ambient_temp)
        self.temp += q_gen - q_diss
        self.temp = float(np.clip(self.temp, 30.0, 130.0))

        # Reward computation
        reward = self.load * 12.0
        terminated = False

        # Safe dispatch
        if self.temp <= 80.0 and self.anomaly_prob < 0.35:
            reward += 12.0
            if action == 1:
                reward += 10.0
            elif action == 0:
                reward += 4.0

        # Unnecessary throttling penalty
        if self.temp < 74.0 and self.anomaly_prob < 0.20 and action in [2, 3]:
            reward -= 8.0

        # Thermal gradient warning penalty
        if self.temp > 82.0:
            reward -= 2.5 * (self.temp - 82.0)

        # Proactive cooling incentive
        if self.temp > 84.0 or self.anomaly_prob > 0.45:
            if action in [2, 3]:
                reward += 15.0
            elif action == 1:
                reward -= 50.0

        # Emergency trip penalty vs reward
        if action == 4:
            if self.anomaly_prob >= 0.72 or self.temp >= 102.0:
                reward += 60.0
            else:
                reward -= 120.0
            terminated = True

        # Catastrophic thermal breakdown
        if self.temp >= 118.0 and not terminated:
            reward -= 200.0
            terminated = True

        if self.step_count >= self.max_steps:
            terminated = True

        return self._get_state(), reward, terminated, {
            "load": self.load,
            "temp": self.temp,
            "anomaly_prob": self.anomaly_prob,
            "energy": self.acoustic_energy
        }


def run_full_evaluation(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 95)
    print("🔬 SOUNDGRID AI: RIGOROUS SCIENTIFIC EVALUATION OF PPO AGENT")
    print(f"📁 Target Model Checkpoint: {MODEL_PATH}")
    print("=" * 95)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH)
    policy = model.policy
    policy.eval()

    # --- TASK 2: ARCHITECTURAL & WEIGHT TENSOR INSPECTION ---
    print("\n" + "=" * 95)
    print("📐 TASK 2: POLICY NETWORK ARCHITECTURE & HEAD INSPECTION")
    print("=" * 95)

    print(f"Observation Space: {model.observation_space}")
    print(f"Action Space:      {model.action_space}")
    print(f"Policy Class:      {policy.__class__.__name__}")

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total Parameters:  {total_params:,}")

    # Hidden layers
    w0 = policy.mlp_extractor.policy_net[0].weight.detach()
    b0 = policy.mlp_extractor.policy_net[0].bias.detach()
    w2 = policy.mlp_extractor.policy_net[2].weight.detach()
    b2 = policy.mlp_extractor.policy_net[2].bias.detach()

    # Action & Value heads
    wa = policy.action_net.weight.detach()
    ba = policy.action_net.bias.detach()
    wv = policy.value_net.weight.detach()
    bv = policy.value_net.bias.detach()

    print(f"\nPolicy Backbone:")
    print(f"  Input Layer (0):     Weight {tuple(w0.shape)} | Mean={w0.mean():.4f}, Std={w0.std():.4f}, L2 Norm={torch.norm(w0):.4f}")
    print(f"  Input Bias (0):       Bias   {tuple(b0.shape)} | Mean={b0.mean():.4f}, Std={b0.std():.4f}")
    print(f"  Hidden Layer (2):    Weight {tuple(w2.shape)} | Mean={w2.mean():.4f}, Std={w2.std():.4f}, L2 Norm={torch.norm(w2):.4f}")
    print(f"  Hidden Bias (2):      Bias   {tuple(b2.shape)} | Mean={b2.mean():.4f}, Std={b2.std():.4f}")

    print(f"\nAction Head (Discrete 5 Actions):")
    print(f"  Action Net Weight:   Weight {tuple(wa.shape)} | Mean={wa.mean():.4f}, Std={wa.std():.4f}, L2 Norm={torch.norm(wa):.4f}")
    print(f"  Action Net Bias:     Bias   {tuple(ba.shape)} | Values: {ba.numpy().round(5).tolist()}")
    for idx, act_name in enumerate(EvaluationSoundGridEnv.ACTION_NAMES):
        print(f"    [{idx}] {act_name:<30} -> Bias = {ba[idx]:+.5f} | Weight Norm = {torch.norm(wa[idx]):.5f}")

    print(f"\nValue Head (Critic 1 Output):")
    print(f"  Value Net Weight:    Weight {tuple(wv.shape)} | Mean={wv.mean():.4f}, Std={wv.std():.4f}, L2 Norm={torch.norm(wv):.4f}")
    print(f"  Value Net Bias:      Bias   {tuple(bv.shape)} | Value = {bv.item():+.5f}")

    # Feature input sensitivities
    feature_names = [
        "0: anomaly_prob", "1: delta_anomaly", "2: acoustic_energy",
        "3: load", "4: delta_load", "5: temp_norm", "6: delta_temp", "7: time_progress"
    ]
    mean_abs_w0 = torch.mean(torch.abs(w0), dim=0).numpy()
    print(f"\nFirst-Layer Input Feature Sensitivities (Mean Absolute Weights):")
    for fname, maw in zip(feature_names, mean_abs_w0):
        bar = "█" * int(maw * 100)
        print(f"  {fname:<20}: {maw:.5f}  {bar}")

    # --- TASK 3 & 4: 100-EPISODE MONTE CARLO EVALUATION ---
    print("\n" + "=" * 95)
    print("🎲 TASK 3 & 4: 100-EPISODE MONTE CARLO EVALUATION ACROSS DIVERSE CONDITIONS")
    print("=" * 95)

    regimes = ["Normal", "Developing Discharges", "Severe Arcing", "Thermal Stress"]
    env = EvaluationSoundGridEnv(max_steps=100)

    all_episode_rewards = []
    all_episode_durations = []
    all_actions = []

    total_false_trips = 0
    total_safe_trips = 0
    total_overheat_failures = 0

    per_regime_data = {r: {
        "rewards": [],
        "durations": [],
        "actions": [],
        "false_trips": 0,
        "safe_trips": 0,
        "overheats": 0,
        "max_temps": [],
        "final_loads": []
    } for r in regimes}

    for regime in regimes:
        print(f"Running 25 Monte Carlo episodes for regime: {regime}...")
        for ep in range(EPISODES_PER_CONDITION):
            # Scenario parameters based on regime
            if regime == "Normal":
                init_ambient = 30.0
                init_load = np.random.uniform(0.65, 0.85)
                init_temp = np.random.uniform(55.0, 70.0)
                init_anom = np.random.uniform(0.01, 0.08)
            elif regime == "Developing Discharges":
                init_ambient = 30.0
                init_load = np.random.uniform(0.65, 0.85)
                init_temp = np.random.uniform(60.0, 74.0)
                init_anom = np.random.uniform(0.35, 0.50)
            elif regime == "Severe Arcing":
                init_ambient = 30.0
                init_load = np.random.uniform(0.70, 0.85)
                init_temp = np.random.uniform(65.0, 78.0)
                init_anom = np.random.uniform(0.75, 0.90)
            elif regime == "Thermal Stress":
                init_ambient = np.random.uniform(38.0, 44.0) # extreme ambient
                init_load = np.random.uniform(0.85, 0.95)    # heavy grid demand
                init_temp = np.random.uniform(85.0, 96.0)    # already hot
                init_anom = np.random.uniform(0.08, 0.25)

            obs = env.reset(load=init_load, temp=init_temp, anomaly=init_anom, ambient_temp=init_ambient)
            ep_reward = 0.0
            ep_steps = 0
            ep_actions = []
            ep_max_temp = env.temp
            done = False

            while not done:
                # Agent action
                act, _ = model.predict(obs, deterministic=True)
                act = int(act)
                ep_actions.append(act)
                all_actions.append(act)

                # Prior state observation for trip assessment
                prior_anom = obs[0]
                prior_temp = obs[5] * 100.0

                if act == 4:
                    if prior_anom >= 0.70 or prior_temp >= 100.0:
                        total_safe_trips += 1
                        per_regime_data[regime]["safe_trips"] += 1
                    else:
                        total_false_trips += 1
                        per_regime_data[regime]["false_trips"] += 1

                # Generate next state disturbance for regime
                if regime == "Normal":
                    next_anom = np.clip(prior_anom + np.random.normal(0.0, 0.015), 0.01, 0.12)
                    next_energy = np.clip(next_anom * 0.8 + np.random.uniform(0.05, 0.15), 0.05, 0.25)
                elif regime == "Developing Discharges":
                    next_anom = np.clip(prior_anom + np.random.normal(0.01, 0.03), 0.32, 0.65)
                    next_energy = np.clip(next_anom * 0.9 + np.random.uniform(0.1, 0.2), 0.30, 0.65)
                elif regime == "Severe Arcing":
                    next_anom = np.clip(prior_anom + np.random.normal(0.01, 0.02), 0.78, 0.99)
                    next_energy = np.clip(next_anom * 0.95 + np.random.uniform(0.05, 0.1), 0.70, 1.0)
                elif regime == "Thermal Stress":
                    next_anom = np.clip(prior_anom + np.random.normal(0.005, 0.02), 0.05, 0.45)
                    next_energy = np.clip(next_anom * 0.8 + 0.1, 0.10, 0.45)

                obs, rew, done, info = env.step(act, next_anom, next_energy)

                if info["temp"] > ep_max_temp:
                    ep_max_temp = info["temp"]

                if info["temp"] >= 118.0 and not (act == 4):
                    total_overheat_failures += 1
                    per_regime_data[regime]["overheats"] += 1

                ep_reward += rew
                ep_steps += 1

            all_episode_rewards.append(ep_reward)
            all_episode_durations.append(ep_steps)

            per_regime_data[regime]["rewards"].append(ep_reward)
            per_regime_data[regime]["durations"].append(ep_steps)
            per_regime_data[regime]["actions"].extend(ep_actions)
            per_regime_data[regime]["max_temps"].append(ep_max_temp)
            per_regime_data[regime]["final_loads"].append(env.load)

    # Calculate overall metrics
    mean_rew = float(np.mean(all_episode_rewards))
    std_rew = float(np.std(all_episode_rewards))
    min_rew = float(np.min(all_episode_rewards))
    max_rew = float(np.max(all_episode_rewards))

    mean_dur = float(np.mean(all_episode_durations))
    std_dur = float(np.std(all_episode_durations))
    min_dur = int(np.min(all_episode_durations))
    max_dur = int(np.max(all_episode_durations))

    total_steps_executed = len(all_actions)
    action_counts = [all_actions.count(i) for i in range(5)]
    action_percentages = [(c / total_steps_executed) * 100 for c in action_counts]

    print("\n" + "=" * 95)
    print("📈 AGGREGATE 100-EPISODE SCIENTIFIC PERFORMANCE METRICS")
    print("=" * 95)
    print(f"Total Evaluated Episodes:          {TOTAL_EPISODES}")
    print(f"Total Decision Steps Executed:     {total_steps_executed:,}")
    print(f"Mean Episode Reward:               {mean_rew:8.2f} +/- {std_rew:.2f}")
    print(f"Episode Reward Range:              [{min_rew:.2f}, {max_rew:.2f}]")
    print(f"Mean Episode Duration (Steps):     {mean_dur:8.2f} +/- {std_dur:.2f}")
    print(f"Episode Duration Range:            [{min_dur}, {max_dur}]")
    print(f"False Alarm Blackout Trips:        {total_false_trips:5d} ({total_false_trips/TOTAL_EPISODES*100:5.1f}%) [Criteria: Anom < 0.70 & Temp < 100°C]")
    print(f"Justified Safety Trips:            {total_safe_trips:5d} ({total_safe_trips/TOTAL_EPISODES*100:5.1f}%) [Criteria: Anom >= 0.70 | Temp >= 100°C]")
    print(f"Catastrophic Overheating Failures: {total_overheat_failures:5d} ({total_overheat_failures/TOTAL_EPISODES*100:5.1f}%) [Criteria: Temp >= 118°C]")

    print("\nComplete Action Selection Breakdown:")
    for a_idx, a_name in enumerate(EvaluationSoundGridEnv.ACTION_NAMES):
        cnt = action_counts[a_idx]
        pct = action_percentages[a_idx]
        bar = "█" * int(pct / 2)
        print(f"  {a_name:<32}: {cnt:5d} ({pct:6.2f}%)  {bar}")

    # Per-regime metrics
    print("\n" + "-" * 95)
    print("📊 PER-REGIME OPERATIONAL PERFORMANCE (25 EPISODES PER REGIME)")
    print("-" * 95)
    print(f"{'Regime':<24} | {'Mean Reward':<16} | {'Mean Steps':<10} | {'Safe Trip':<9} | {'False Trip':<10} | {'Overheat':<8} | {'Dominant Action':<18}")
    print("-" * 105)

    regime_results = {}
    for r in regimes:
        rd = per_regime_data[r]
        r_acts = rd["actions"]
        r_tot_a = len(r_acts)
        r_act_cnt = [r_acts.count(i) for i in range(5)]
        r_act_pct = [(c / r_tot_a * 100) if r_tot_a > 0 else 0 for c in r_act_cnt]
        dom_act_idx = int(np.argmax(r_act_cnt))
        dom_act_str = f"A{dom_act_idx} ({r_act_pct[dom_act_idx]:.1f}%)"

        regime_results[r] = {
            "mean_reward": float(np.mean(rd["rewards"])),
            "std_reward": float(np.std(rd["rewards"])),
            "min_reward": float(np.min(rd["rewards"])),
            "max_reward": float(np.max(rd["rewards"])),
            "mean_duration": float(np.mean(rd["durations"])),
            "safe_trips": rd["safe_trips"],
            "false_trips": rd["false_trips"],
            "overheats": rd["overheats"],
            "action_counts": r_act_cnt,
            "action_percentages": r_act_pct,
            "mean_max_temp": float(np.mean(rd["max_temps"])),
            "mean_final_load": float(np.mean(rd["final_loads"]))
        }

        m_rew_str = f"{regime_results[r]['mean_reward']:.1f} +/- {regime_results[r]['std_reward']:.1f}"
        m_step_str = f"{regime_results[r]['mean_duration']:.1f}"
        print(f"{r:<24} | {m_rew_str:<16} | {m_step_str:<10} | {rd['safe_trips']:^9} | {rd['false_trips']:^10} | {rd['overheats']:^8} | {dom_act_str:<18}")

    # --- TASK 5: CRITICAL OPERATIONAL EDGE CASES ---
    print("\n" + "=" * 95)
    print("🔬 TASK 5: CRITICAL OPERATIONAL EDGE CASES RIGOROUS TESTING")
    print("=" * 95)

    # Edge Case 1: Normal Operation at High Load
    print("\n[Edge Case 1]: Normal Operation at High Load (Load=0.90, Temp=62.0°C, Anom=0.04)")
    print("Hypothesis: In the absence of anomalies and cool temps, the agent should maintain or safely dispatch load.")
    ec1_data = []
    ec1_env = EvaluationSoundGridEnv(max_steps=10)
    ec1_obs = ec1_env.reset(load=0.90, temp=62.0, anomaly=0.04, ambient_temp=30.0)

    for step in range(1, 11):
        act, _ = model.predict(ec1_obs, deterministic=True)
        act = int(act)
        obs_t = torch.tensor(ec1_obs).unsqueeze(0)
        with torch.no_grad():
            dist = policy.get_distribution(obs_t)
            probs = dist.distribution.probs.numpy()[0]

        ec1_data.append({
            "step": step,
            "load": ec1_env.load,
            "temp": ec1_env.temp,
            "anomaly": ec1_env.anomaly_prob,
            "action": act,
            "action_name": EvaluationSoundGridEnv.ACTION_NAMES[act],
            "probs": probs.round(4).tolist()
        })

        next_anom = np.clip(ec1_env.anomaly_prob + np.random.normal(0.0, 0.005), 0.02, 0.08)
        ec1_obs, _, _, _ = ec1_env.step(act, next_anom)

    for item in ec1_data:
        p_str = f"Maint={item['probs'][0]*100:4.1f}%, Inc={item['probs'][1]*100:4.1f}%, Thr15={item['probs'][2]*100:4.1f}%, Thr40={item['probs'][3]*100:4.1f}%, Trip={item['probs'][4]*100:4.1f}%"
        print(f"  Step {item['step']:02d}: Load={item['load']:.2f}, Temp={item['temp']:.1f}°C, Anom={item['anomaly']:.2f} -> Selected: [{item['action']}] {item['action_name']:<22} | Distribution: {p_str}")

    # Edge Case 2: Rapidly Rising Thermal Gradient
    print("\n[Edge Case 2]: Rapidly Rising Thermal Gradient (Temp ramps 72°C -> 112°C, dT = +4.5°C/step, Anom=0.20)")
    print("Hypothesis: As core temperature escalates towards thermal threshold, does it throttle proactively or trip before burnout?")
    ec2_data = []
    ec2_temp = 72.0
    ec2_load = 0.85
    ec2_anom = 0.20
    prev_l = ec2_load
    prev_t = 67.5
    prev_a = ec2_anom
    ec2_tripped = False
    ec2_trip_step = None

    for step in range(1, 11):
        delta_anom = ec2_anom - prev_a
        delta_load = ec2_load - prev_l
        temp_norm = ec2_temp / 100.0
        delta_temp = (ec2_temp - prev_t) / 10.0
        time_prog = step / 100.0
        energy = 0.25
        obs_ec2 = np.array([ec2_anom, delta_anom, energy, ec2_load, delta_load, temp_norm, delta_temp, time_prog], dtype=np.float32)

        act, _ = model.predict(obs_ec2, deterministic=True)
        act = int(act)
        obs_t = torch.tensor(obs_ec2).unsqueeze(0)
        with torch.no_grad():
            dist = policy.get_distribution(obs_t)
            probs = dist.distribution.probs.numpy()[0]

        ec2_data.append({
            "step": step,
            "load": ec2_load,
            "temp": ec2_temp,
            "delta_temp": delta_temp * 10.0,
            "action": act,
            "action_name": EvaluationSoundGridEnv.ACTION_NAMES[act],
            "probs": probs.round(4).tolist()
        })

        p_str = f"Maint={probs[0]*100:4.1f}%, Inc={probs[1]*100:4.1f}%, Thr15={probs[2]*100:4.1f}%, Thr40={probs[3]*100:4.1f}%, Trip={probs[4]*100:4.1f}%"
        print(f"  Step {step:02d}: Load={ec2_load:.2f}, Temp={ec2_temp:5.1f}°C, dT={delta_temp*10:+4.1f}°C/step -> [{act}] {EvaluationSoundGridEnv.ACTION_NAMES[act]:<22} | Distribution: {p_str}")

        if act == 4:
            ec2_tripped = True
            ec2_trip_step = step
            print(f"  ==> 🛑 SAFETY TRIP TRIGGERED AT STEP {step} (Temp={ec2_temp:.1f}°C)")
            break

        prev_l = ec2_load
        prev_t = ec2_temp
        prev_a = ec2_anom
        if act == 2: ec2_load = max(0.10, ec2_load - 0.15)
        elif act == 3: ec2_load = max(0.10, ec2_load - 0.30)
        ec2_temp += 4.5 # Forced severe thermal ramp

    # Edge Case 3: Severe Electrical Arcing Bursts
    print("\n[Edge Case 3]: Severe Electrical Arcing Bursts (Acoustic anomaly jumps suddenly from 0.05 to 0.88 - 0.96)")
    print("Hypothesis: In the presence of acute arcing discharges, does the model execute an emergency trip reliably and immediately?")
    ec3_data = []
    ec3_temp = 74.0
    ec3_load = 0.80
    arcing_sequence = [
        (0.05, 0.15, "Normal Hum"),
        (0.06, 0.18, "Normal Hum"),
        (0.85, 0.90, "⚡ Severe Arcing Burst 1"),
        (0.92, 0.95, "⚡ Severe Arcing Burst 2"),
        (0.96, 0.98, "⚡ Continuous Critical Arcing")
    ]
    prev_l = ec3_load
    prev_t = ec3_temp
    prev_a = 0.05
    ec3_tripped = False
    ec3_trip_step = None

    for step, (anom_val, energy_val, desc) in enumerate(arcing_sequence, start=1):
        delta_anom = anom_val - prev_a
        delta_load = ec3_load - prev_l
        temp_norm = ec3_temp / 100.0
        delta_temp = (ec3_temp - prev_t) / 10.0
        time_prog = step / 100.0
        obs_ec3 = np.array([anom_val, delta_anom, energy_val, ec3_load, delta_load, temp_norm, delta_temp, time_prog], dtype=np.float32)

        act, _ = model.predict(obs_ec3, deterministic=True)
        act = int(act)
        obs_t = torch.tensor(obs_ec3).unsqueeze(0)
        with torch.no_grad():
            dist = policy.get_distribution(obs_t)
            probs = dist.distribution.probs.numpy()[0]

        ec3_data.append({
            "step": step,
            "description": desc,
            "load": ec3_load,
            "temp": ec3_temp,
            "anomaly": anom_val,
            "energy": energy_val,
            "action": act,
            "action_name": EvaluationSoundGridEnv.ACTION_NAMES[act],
            "probs": probs.round(4).tolist()
        })

        p_str = f"Maint={probs[0]*100:4.1f}%, Inc={probs[1]*100:4.1f}%, Thr15={probs[2]*100:4.1f}%, Thr40={probs[3]*100:4.1f}%, Trip={probs[4]*100:4.1f}%"
        print(f"  Step {step:02d} [{desc:<26}]: Anom={anom_val:4.2f}, Energy={energy_val:4.2f}, Temp={ec3_temp:5.1f}°C -> [{act}] {EvaluationSoundGridEnv.ACTION_NAMES[act]:<22} | Distribution: {p_str}")

        if act == 4:
            ec3_tripped = True
            ec3_trip_step = step
            print(f"  ==> 🛑 EMERGENCY TRIP ACTIVATED AT STEP {step} (Condition: {desc})")
            break

        prev_l = ec3_load
        prev_t = ec3_temp
        prev_a = anom_val
        if act == 2: ec3_load = max(0.10, ec3_load - 0.15)
        elif act == 3: ec3_load = max(0.10, ec3_load - 0.30)
        ec3_temp += (ec3_load**2 * 4.2) + (anom_val * 10.0) - 0.09 * (ec3_temp - 30.0)

    # Save complete structured dataset
    output_dict = {
        "model_path": MODEL_PATH,
        "evaluation_episodes": TOTAL_EPISODES,
        "overall_metrics": {
            "mean_reward": mean_rew,
            "std_reward": std_rew,
            "min_reward": min_rew,
            "max_reward": max_rew,
            "mean_duration": mean_dur,
            "std_duration": std_dur,
            "min_duration": min_dur,
            "max_duration": max_dur,
            "total_steps": total_steps_executed,
            "false_alarm_blackout_trips": total_false_trips,
            "false_alarm_rate_pct": total_false_trips / TOTAL_EPISODES * 100,
            "justified_safety_trips": total_safe_trips,
            "justified_safety_rate_pct": total_safe_trips / TOTAL_EPISODES * 100,
            "catastrophic_overheating_failures": total_overheat_failures,
            "catastrophic_overheat_rate_pct": total_overheat_failures / TOTAL_EPISODES * 100,
            "action_counts": action_counts,
            "action_percentages": action_percentages
        },
        "per_regime_results": regime_results,
        "edge_cases": {
            "edge_case_1_normal_high_load": ec1_data,
            "edge_case_2_thermal_gradient": ec2_data,
            "edge_case_3_severe_arcing": ec3_data
        }
    }

    output_path = "evaluation_report_data.json"
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print("\n" + "=" * 95)
    print(f"💾 COMPREHENSIVE EVALUATION DATA SAVED TO: {output_path}")
    print("=" * 95)
    return output_dict

if __name__ == "__main__":
    run_full_evaluation()
