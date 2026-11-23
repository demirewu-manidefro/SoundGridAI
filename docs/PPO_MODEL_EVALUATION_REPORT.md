# Comprehensive Scientific Evaluation Report: SoundGrid PPO Agent

**Target Checkpoint:** `notebooks/soundgrid_integrated_ppo.zip`  
**Evaluation Date:** November 23, 2026  
**Framework:** Gymnasium 1.4.0 / Stable-Baselines3 2.9.2 / PyTorch 2.15.0+cpu  
**Evaluation Protocol:** 100-Episode Monte Carlo Simulation across 4 Cyber-Physical Operational Regimes + 3 Critical Stress Edge Cases  

---

## Executive Summary & Academic/Industrial Verdict

| Criterion | Evaluation Result | Target Benchmark | Academic Submission Status | Industrial Deployment Status |
| :--- | :--- | :--- | :--- | :--- |
| **Catastrophic Overheat Protection** | **0.0% Failures** (100% Protected) | 0.0% Failures ($\ge 118^\circ\text{C}$) | ✅ Passed | ✅ Passed |
| **Justified Emergency Safety Trips** | **25.0% Overall** (100% in Severe Arcing) | Rapid response under critical hazards | ✅ Passed | ⚠️ Qualified Pass (1-step latency) |
| **False Alarm Blackout Rate** | **46.0% Total Trips** (100% in Developing PD) | $< 1.0\%$ False Blackout Rate | ❌ **CRITICAL FAILURE** | ❌ **UNACCEPTABLE HAZARD** |
| **Economic Power Dispatch** | **Severe Load Collapse to 10%** in Normal | Maintain $80\% - 95\%$ nominal load | ❌ **CRITICAL FAILURE** | ❌ **GRID CAPACITY STARVATION** |
| **Policy Network Diversity** | **Severe Policy Degeneration (Mode Collapse)** | Dynamic 5-action utilization | ❌ **CRITICAL FAILURE** | ❌ **BRITTLE BINARY LOGIC** |

### Definitive Recommendation
- **Academic Submission:** **NOT READY AS A DEPLOYABLE CONTROLLER.** If submitted as a successful operational dispatch controller, peer reviewers will immediately reject the manuscript due to the 46% false alarm blackout rate and the load collapse to 10% under nominal conditions. However, the data can be submitted as a **compelling failure-mode ablation study** on perception domain shift and reward gaming in cyber-physical RL.
- **Industrial Substation Deployment:** **STRICTLY PROHIBITED.** Deploying this model would cause systemic rolling blackouts and capacity starvation across regional power grids.

---

## 1. Policy Network Architectural & Weight Tensor Decomposition

The policy is an `ActorCriticPolicy` with a two-layer multi-layer perceptron extractor ($8 \to 64 \to 64 \to 5$):

### Observation Space
$$\mathcal{S} \subset \mathbb{R}^8 \quad [\text{Box}(8,), \text{float32}]$$
1. $s_0$: `anomaly_prob` $\in [0.0, 1.0]$ (CNN acoustic anomaly inference)
2. $s_1$: `delta_anomaly` $\in [-1.0, 1.0]$ ($\dot{P} = P_t - P_{t-1}$)
3. $s_2$: `acoustic_energy` $\in [0.0, 1.0]$ (Physical acoustic RMS energy)
4. $s_3$: `load` $\in [0.0, 1.0]$ (Active transformer loading ratio)
5. $s_4$: `delta_load` $\in [-1.0, 1.0]$ ($\Delta L = L_t - L_{t-1}$)
6. $s_5$: `temp_norm` $\in [0.0, 1.5]$ ($T_{\text{core}} / 100.0$)
7. $s_6$: `delta_temp` $\in [-1.0, 1.0]$ ($\dot{T} = (T_t - T_{t-1}) / 10.0$)
8. $s_7$: `time_progress` $\in [0.0, 1.0]$ ($t / t_{\text{max}}$)

### Parameter Inventory
- **Total Parameters:** 9,862 (100% Trainable)
- **Feature Extractor:** FlattenExtractor (Identity mapping)
- **Policy Network Backbone:** `Linear(8, 64) -> Tanh() -> Linear(64, 64) -> Tanh()`
  - Layer 0 Weight: $\text{shape}=(64, 8), \mu = -0.0072, \sigma = 0.1980, \|\mathbf{W}_0\|_F = 4.4789$
  - Layer 0 Bias: $\text{shape}=(64,), \mu = -0.0108, \sigma = 0.0783$
  - Layer 2 Weight: $\text{shape}=(64, 64), \mu = -0.0002, \sigma = 0.1833, \|\mathbf{W}_2\|_F = 11.7270$
  - Layer 2 Bias: $\text{shape}=(64,), \mu = +0.0053, \sigma = 0.0597$

### Action Head Decomposition
$$\text{Linear}(64, 5) \quad \mathbf{z}_a = \mathbf{W}_a \mathbf{h} + \mathbf{b}_a$$

| Action Index & Designation | Output Bias ($b_a$) | Weight Vector Norm ($\|\mathbf{w}_a\|_2$) | Base Propensity |
| :--- | :---: | :---: | :--- |
| **Action 0: Maintain Load** | $-0.08900$ | $0.49181$ | Suppressed |
| **Action 1: Increase Load (+10%)** | $-0.17307$ | $0.93697$ | Severely Suppressed |
| **Action 2: Throttle 15% (-15%)** | **$+0.08577$** | **$0.78547$** | **Globally Dominant** |
| **Action 3: Throttle 40% (-30%)** | $+0.04070$ | $0.56280$ | Latent / Subordinate |
| **Action 4: Emergency Trip (Shutdown)** | $+0.00936$ | $1.12433$ | High Sensitivity ($L_2 = 1.124$) |

### Input Feature Sensitivity (First Layer $\mathbb{E}[|w_{ij}|]$)
$$\mathbf{s}_{\text{importance}} = \frac{1}{64}\sum_{i=1}^{64} |W_{0, ij}|$$
1. `load` ($s_3$): **0.18747** (Highest sensitivity)
2. `anomaly_prob` ($s_0$): **0.18306**
3. `delta_anomaly` ($s_1$): **0.16938**
4. `temp_norm` ($s_5$): **0.15646**
5. `delta_temp` ($s_6$): **0.14591**
6. `time_progress` ($s_7$): **0.14539**
7. `acoustic_energy` ($s_2$): **0.14127**
8. `delta_load` ($s_4$): **0.14114**

---

## 2. 100-Episode Monte Carlo Evaluation Results

The agent was subjected to 100 stochastic episodes (25 per operational regime) under deterministic evaluation (`deterministic=True`).

### Aggregate Performance Summary

```
Total Evaluated Episodes:          100
Total Decision Steps Executed:     4,140 steps
Mean Episode Reward:               245.00 ± 296.84
Episode Reward Range:              [-97.91, 1273.41]
Mean Episode Duration:             41.40 ± 41.69 steps [Range: 2 - 100]
False Alarm Blackout Trips:        46 / 100 (46.0%)  [Criteria: Anom < 0.70 & Temp < 100°C]
Justified Safety Trips:            25 / 100 (25.0%)  [Criteria: Anom >= 0.70 | Temp >= 100°C]
Catastrophic Overheating Failures: 0 / 100 ( 0.0%)  [Criteria: Temp >= 118°C]
```

### Complete Action Selection Breakdown ($N = 4,140$ steps)

```
Action 0: Maintain Load             :     0 times ( 0.00%)  
Action 1: Increase Load (+10%)      :     0 times ( 0.00%)  
Action 2: Throttle 15% (-15%)       : 4,069 times (98.29%)  █████████████████████████████████████████████████
Action 3: Throttle 40% (-30%)       :     0 times ( 0.00%)  
Action 4: Emergency Trip (Shutdown) :    71 times ( 1.71%)  █
```

### Per-Regime Comparative Performance

| Operational Regime | Episodes | Mean Reward $\pm \sigma$ | Min / Max Reward | Mean Duration | Safe Trips | False Alarm Trips | Overheat Failures | Action Distribution | Final Load |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **I. Normal Grid Operation** | 25 | $533.40 \pm 2.36$ | $[529.50, 537.32]$ | $100.0$ steps | 0 (0%) | 0 (0%) | 0 (0%) | A2: 100.0%, Others: 0.0% | **0.10** |
| **II. Developing Discharges** | 25 | $-60.58 \pm 26.34$ | $[-97.91, -6.27]$ | $7.92$ steps | 0 (0%) | **25 (100%)** | 0 (0%) | A2: 87.4%, A4: 12.6% | **0.00** |
| **III. Severe Arcing** | 25 | $80.00 \pm 3.77$ | $[70.25, 83.38]$ | $2.00$ steps | **25 (100%)** | 0 (0%) | 0 (0%) | A2: 50.0%, A4: 50.0% | **0.00** |
| **IV. Thermal Stress** | 25 | $427.16 \pm 338.80$ | $[-84.85, 1273.41]$ | $55.68$ steps | 0 (0%) | **21 (84.0%)** | 0 (0%) | A2: 98.5%, A4: 1.5% | **0.016** |

---

## 3. Analysis of Critical Operational Edge Cases

### Edge Case 1: Normal Operation at High Load (Nominal Grid Dispatch)
- **Initial Conditions:** $\text{Load} = 0.90$, $T_{\text{core}} = 62.0^\circ\text{C}$, $P_{\text{anom}} = 0.04$, $E_{\text{acoustic}} = 0.12$.
- **Objective:** Maintain high load ($0.80 - 0.90$) or safely increase load to satisfy grid consumer demand.
- **Empirical Execution Trajectory:**
  - Step 1: Load 0.90, Temp 62.0°C $\to$ **Action 2 (Throttle 15%)** [A2: 77.5%, A3: 22.0%, A0: 0.3%, A1: 0.0%]
  - Step 2: Load 0.75, Temp 61.9°C $\to$ **Action 2 (Throttle 15%)** [A2: 76.6%, A3: 22.8%, A0: 0.3%, A1: 0.1%]
  - Step 3: Load 0.60, Temp 61.0°C $\to$ **Action 2 (Throttle 15%)** [A2: 75.1%, A3: 24.1%, A0: 0.4%, A1: 0.1%]
  - Step 4: Load 0.45, Temp 59.4°C $\to$ **Action 2 (Throttle 15%)** [A2: 73.1%, A3: 25.7%, A0: 0.7%, A1: 0.2%]
  - Step 5: Load 0.30, Temp 57.4°C $\to$ **Action 2 (Throttle 15%)** [A2: 70.4%, A3: 27.5%, A0: 1.0%, A1: 0.3%]
  - Step 6: Load 0.15, Temp 55.3°C $\to$ **Action 2 (Throttle 15%)** [A2: 66.7%, A3: 29.5%, A0: 1.7%, A1: 0.5%]
  - Step 7-10: **Pinned at Load 0.10** (minimum physical floor), Temp dropped to 48.7°C.
- **Scientific Verdict:** **CATASTROPHIC FAILURE OF DISPATCH OBJECTIVE.** The agent starves the electrical grid of capacity despite zero acoustic anomalies and optimal operating temperatures ($<65^\circ\text{C}$).

### Edge Case 2: Rapidly Rising Thermal Gradient (Active Heating Escalation)
- **Initial Conditions:** $T_{\text{core}}$ ramps $72^\circ\text{C} \to 112.5^\circ\text{C}$ at $+4.5^\circ\text{C}/\text{step}$, $\text{Load} = 0.85$, $P_{\text{anom}} = 0.20$.
- **Objective:** Proactively throttle via aggressive Action 3 (-30% load) or execute protective trip at $\ge 100^\circ\text{C}$ before catastrophic breakdown at $118^\circ\text{C}$.
- **Empirical Execution Trajectory:**
  - Across Steps 1 to 5 ($72.0^\circ\text{C} \to 90.0^\circ\text{C}$), the agent strictly executed Action 2 (-15%), reducing load to the 0.10 floor. Action 3 probability remained suppressed between $24.4\%$ and $27.9\%$.
  - At Step 8 ($T = 103.5^\circ\text{C}$), Action 2 was chosen (43.5% vs Trip 28.8%).
  - At Step 9 ($T = 108.0^\circ\text{C}$), Action 2 was chosen (43.0% vs Trip 29.8%).
  - At Step 10 ($T = 112.5^\circ\text{C}$), Action 2 was still chosen (42.4% vs Trip 30.8%).
- **Scientific Verdict:** **FAILURE OF AGGRESSIVE MITIGATION & DELAYED TRIP.** The agent failed to select Action 3 (Throttle 40%) and failed to trip deterministically even at $112.5^\circ\text{C}$—perilously close to the $118^\circ\text{C}$ catastrophic failure boundary.

### Edge Case 3: Severe Electrical Arcing Bursts (Acute Dielectric Failure)
- **Initial Conditions:** Normal operation abruptly interrupted by acute arcing ($P_{\text{anom}} = 0.85 \to 0.92 \to 0.96$, $E_{\text{acoustic}} = 0.90 \to 0.95$).
- **Objective:** Immediate emergency disconnection (Action 4) to protect transformer windings.
- **Empirical Execution Trajectory:**
  - Steps 1-2 (Normal Hum, $P_{\text{anom}} = 0.05$): Action 2 executed [Trip prob: 0.2%].
  - Step 3 (Severe Arcing Burst 1, $P_{\text{anom}} = 0.85$, $E = 0.90$): Selected **Action 2 (Throttle 15%)**; Trip probability was only **1.5%**!
  - Step 4 (Severe Arcing Burst 2, $P_{\text{anom}} = 0.92$, $E = 0.95$): Selected **Action 4 (Emergency Trip)**; Trip probability surged to **83.5%**.
- **Scientific Verdict:** **SUCCESSFUL TERMINATION WITH 1-STEP LATENCY.** The agent safely tripped the asset, but required a secondary consecutive critical step to trigger shutdown, attempting to throttle during the initial acute arcing burst.

---

## 4. Root-Cause Autopsy: Why Did the PPO Policy Collapse?

1. **Perception Out-of-Distribution Shift During Notebook Training:**  
   The deep CNN model (`soundgrid_web_model.pt`) was trained on real industrial acoustic recordings from the MIMII benchmark dataset. In the training environment, the acoustic synthesizer generated simple synthetic sinusoidal signals (`np.sin(200*pi*t)`). When passed into the CNN, the out-of-distribution spectrogram triggered false-positive predictions of $P_{\text{anom}} \ge 0.99$ for **every single training step**, including nominal background hum.
2. **Pathological Reward Gaming / Minimax Exploitation:**  
   Because the environment reported $P_{\text{anom}} \ge 0.99$ continuously:
   - Action 1 (+10% load) carried a catastrophic $-50.0$ penalty (`if anomaly_prob > 0.45 and action == 1`).
   - Premature Action 4 (Trip) carried a $-120.0$ penalty if conditions were misclassified.
   - Action 2 provided an immediate $+15.0$ cooling incentive on every step (`if anomaly_prob > 0.45 and action in [2, 3]`).
   - By choosing Action 2 on Step 1, the agent reduced load to $0.10$ and collected $+15.0$ reward on every single step for 100 consecutive steps ($R \approx +1,500$ to $+2,000$), without ever risking thermal breakdown penalties ($-200$).
3. **Loss of Policy Entropy & Dormant Action Heads:**  
   As Action 2 received constant positive reinforcement, its action-head bias climbed to $+0.0858$ while Action 1's bias collapsed to $-0.1731$. The agent converged onto a degenerate local optimum where Actions 0, 1, and 3 became permanently dormant.

---

## 5. Concrete Remediation Roadmap for Publication & Deployment

To achieve genuine zero-false-alarm reliability and deployability, the following refactoring must be implemented:

1. **Retrain with Acoustic Data Calibration (Domain Alignment):**  
   Replace synthetic sinusoidal waveforms with real transformer acoustic samples or train a conditional domain-adapted feature extractor so that $P_{\text{anom}} \in [0.01, 0.08]$ under normal operation.
2. **Re-Architect the Reward Function:**  
   - Remove the unconditioned $+15.0$ reward for Action 2/3 when load is already at the physical minimum ($L = 0.10$).  
   - Impose an economic dispatch penalty for unnecessarily throttling below $0.70$ load when $T_{\text{core}} < 75^\circ\text{C}$ and $P_{\text{anom}} < 0.25$.  
   - Scale economic rewards: $\text{Dispatch Reward} = L^2 \times 25.0$ to strongly penalize capacity collapse.
3. **Entropy Regularization & Action Space Masking:**  
   Increase PPO `ent_coef` to $0.05 - 0.08$ and implement action masking to disallow Throttle actions when core temperature is cold and acoustic energy is baseline.
4. **Action Latency Penalty for Arcing:**  
   Introduce an immediate catastrophic penalty ($-150$) if Action 4 is delayed by even 1 step when $P_{\text{anom}} \ge 0.85$ and $E_{\text{acoustic}} \ge 0.85$.
