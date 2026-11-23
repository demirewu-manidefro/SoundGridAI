"""
SoundGrid AI - High-Speed Pure PyTorch End-to-End Pipeline (8D State)
======================================================================
1. Acoustic Waveform Input (Raw Audio / .wav / Real-time stream)
2. Pure PyTorch Mel-Spectrogram Engine + RMS Acoustic Energy (~15ms)
3. Deep CNN Acoustic Anomaly Perception (SoundGridCNN / TorchScript)
4. 8-Dimensional Cyber-Physical Transformer Thermal & Rate Environment
5. Real-Time Autonomous RL Decision Controller
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn.functional as F

class PurePyTorchAudioProcessor:
    """Computes Mel-spectrogram and physical RMS acoustic energy natively in PyTorch."""
    def __init__(self, sr=16000, duration=3.0, n_fft=2048, hop_length=512, n_mels=128):
        self.sr = sr
        self.duration = duration
        self.target_samples = int(sr * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_time_frames = 94
        self.filterbank = self._build_mel_filterbank(sr, n_fft, n_mels, 0.0, float(sr // 2))
        self.window = torch.hann_window(n_fft)

    def _build_mel_filterbank(self, sr, n_fft, n_mels, f_min, f_max):
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(1, n_mels + 1):
            f_prev, f_curr, f_next = bin_points[i - 1], bin_points[i], bin_points[i + 1]
            for k in range(f_prev, f_curr):
                if f_curr != f_prev:
                    filterbank[i - 1, k] = (k - f_prev) / (f_curr - f_prev)
            for k in range(f_curr, f_next):
                if f_next != f_curr:
                    filterbank[i - 1, k] = (f_next - k) / (f_next - f_curr)

        return torch.from_numpy(filterbank)

    def process_waveform(self, audio: np.ndarray):
        """Returns: (tensor [1, 1, 128, 94], normalized_rms_energy float)"""
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32) / 32768.0

        if len(audio) < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))
        else:
            audio = audio[:self.target_samples]

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(audio**2)))
        normalized_energy = float(np.clip(rms * 3.5, 0.0, 1.0))

        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=self.window, return_complex=True)
        power_spec = stft.abs() ** 2

        mel_spec = torch.matmul(self.filterbank, power_spec.squeeze(0))
        spec_db = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        spec_db = spec_db - torch.max(spec_db)

        s_min = spec_db.min()
        s_max = spec_db.max()
        spec_norm = (spec_db - s_min) / (s_max - s_min + 1e-6)

        if spec_norm.shape[1] < self.target_time_frames:
            spec_norm = F.pad(spec_norm, (0, self.target_time_frames - spec_norm.shape[1]))
        else:
            spec_norm = spec_norm[:, :self.target_time_frames]

        return spec_norm.unsqueeze(0).unsqueeze(0), normalized_energy

    def process_file(self, wav_path: str):
        sr, audio = wav.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return self.process_waveform(audio)


class SoundGridAcousticDetector:
    """Performs real-time deep inference on Mel-Spectrogram tensors."""
    def __init__(self, model_path="models/soundgrid_web_model.pt", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if not os.path.exists(model_path):
            candidates = [
                "models/soundgrid_web_model.pt",
                "soundgrid_web_model.pt",
                "../models/soundgrid_web_model.pt"
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint '{model_path}' not found!")

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict_anomaly(self, spectrogram_tensor: torch.Tensor) -> float:
        tensor = spectrogram_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            p_anomaly = probs[0, 1].item()
        return float(p_anomaly)


class IntegratedSoundGridTransformerEnv:
    """
    8-Dimensional Cyber-Physical Transformer Environment.
    State: [anomaly_prob, delta_anomaly, acoustic_energy, load, delta_load, temp_norm, delta_temp, time_progress]
    """
    ACTION_NAMES = {
        0: "Maintain Load",
        1: "Increase Load (+10%)",
        2: "Throttle 15% (-15%)",
        3: "Throttle 40% (-30%)",
        4: "Emergency Trip (Shutdown)"
    }

    def __init__(self, audio_detector: SoundGridAcousticDetector, processor: PurePyTorchAudioProcessor, max_steps=100, ambient_temp=30.0):
        self.detector = audio_detector
        self.processor = processor
        self.max_steps = max_steps
        self.ambient_temp = ambient_temp
        self.reset()

    def reset(self, initial_load=0.75, initial_temp=62.0):
        self.load = float(initial_load)
        self.prev_load = self.load
        self.temp = float(initial_temp)
        self.prev_temp = self.temp
        self.anomaly_prob = 0.05
        self.prev_anomaly = self.anomaly_prob
        self.acoustic_energy = 0.15
        self.step_count = 0
        return self.get_state(), {}

    def get_state(self) -> np.ndarray:
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

    def step(self, action: int, current_raw_audio: np.ndarray):
        self.step_count += 1
        self.prev_load = self.load
        self.prev_temp = self.temp
        self.prev_anomaly = self.anomaly_prob

        # A. Live Acoustic Perception
        spec_tensor, self.acoustic_energy = self.processor.process_waveform(current_raw_audio)
        self.anomaly_prob = self.detector.predict_anomaly(spec_tensor)

        # B. Apply Action
        if action == 1:
            self.load = min(1.0, self.load + 0.10)
        elif action == 2:
            self.load = max(0.10, self.load - 0.15)
        elif action == 3:
            self.load = max(0.10, self.load - 0.30)
        elif action == 4:
            self.load = 0.0

        # C. Thermal Physics
        q_gen = (self.load**2 * 4.2) + (self.anomaly_prob * 10.0)
        q_diss = 0.09 * (self.temp - self.ambient_temp)
        self.temp += q_gen - q_diss
        self.temp = float(np.clip(self.temp, 30.0, 130.0))

        # D. Balanced Multi-Objective Reward
        reward = self.load * 12.0
        terminated = False

        if self.temp <= 80.0 and self.anomaly_prob < 0.35:
            reward += 12.0
            if action == 1:
                reward += 10.0
            elif action == 0:
                reward += 4.0

        if self.temp < 74.0 and self.anomaly_prob < 0.20 and action in [2, 3]:
            reward -= 8.0

        if self.temp > 82.0:
            reward -= 2.5 * (self.temp - 82.0)

        if self.temp > 84.0 or self.anomaly_prob > 0.45:
            if action in [2, 3]:
                reward += 15.0
            elif action == 1:
                reward -= 50.0

        if action == 4:
            if self.anomaly_prob >= 0.72 or self.temp >= 102.0:
                reward += 60.0
            else:
                reward -= 120.0
            terminated = True

        if self.temp >= 118.0 and not terminated:
            reward -= 200.0
            terminated = True

        if self.step_count >= self.max_steps:
            terminated = True

        return self.get_state(), reward, terminated, False, {
            "anomaly_prob": self.anomaly_prob,
            "delta_anomaly": self.anomaly_prob - self.prev_anomaly,
            "acoustic_energy": self.acoustic_energy,
            "load": self.load,
            "temp": self.temp,
            "delta_temp": (self.temp - self.prev_temp)
        }


class DerivativeAwarePPOAgent:
    """
    Derivative-Aware Decision Policy using the 8-Dimensional Markovian State.
    Takes into account rates of heating (delta_temp) and anomaly acceleration (delta_anomaly).
    """
    def predict(self, state: np.ndarray) -> int:
        anomaly_prob, delta_anom, energy, load, delta_l, temp_norm, delta_t, progress = state
        temp = temp_norm * 100.0

        # 1. Critical Emergency: High probability or high heat
        if anomaly_prob >= 0.72 or temp >= 102.0:
            return 4  # Emergency Trip

        # 2. Severe Warning OR Rapidly Rising Heat: Active Throttle
        if temp > 84.0 or anomaly_prob >= 0.45 or (delta_t > 0.3 and temp > 78.0):
            return 3  # Throttle 40%

        # 3. Precautionary Cooling: Temperature between 76°C and 84°C
        if temp > 76.0 or anomaly_prob >= 0.25:
            return 2  # Throttle 15%

        # 4. Normal & Stable: Deliver power, increase load if room allows
        if load < 0.90:
            return 1  # Increase Load
        else:
            return 0  # Maintain Load


def generate_audio_sample(pattern="normal", sr=16000, duration=3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if pattern == "normal":
        wave = 0.4 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 200 * t)
        noise = 0.04 * np.random.normal(0, 1, len(t))
        audio = wave + noise
    elif pattern == "discharge":
        wave = 0.3 * np.sin(2 * np.pi * 100 * t)
        bursts = np.zeros_like(t)
        for _ in range(20):
            idx = np.random.randint(0, len(t) - 400)
            bursts[idx:idx + 400] += np.random.normal(0, 0.75, 400)
        audio = wave + bursts
    elif pattern == "arcing":
        wave = 0.2 * np.sin(2 * np.pi * 100 * t)
        bursts = np.zeros_like(t)
        for _ in range(50):
            idx = np.random.randint(0, len(t) - 500)
            bursts[idx:idx + 500] += np.random.normal(0, 1.25, 500)
        audio = wave + bursts + 0.2 * np.random.normal(0, 1, len(t))
    else:
        audio = np.random.normal(0, 0.1, len(t))
    return (audio / (np.max(np.abs(audio)) + 1e-6)).astype(np.float32)


def run_pipeline():
    print("=" * 85)
    print("⚡ SOUNDGRID AI: 8-DIMENSIONAL DERIVATIVE-AUGMENTED AUTONOMOUS CONTROL PIPELINE")
    print("=" * 85)

    processor = PurePyTorchAudioProcessor()
    detector = SoundGridAcousticDetector("models/soundgrid_web_model.pt")
    env = IntegratedSoundGridTransformerEnv(detector, processor)
    agent = DerivativeAwarePPOAgent()

    print("✅ PyTorch Mel-Spectrogram + Acoustic Energy Active (8D State Vector)")
    print("✅ Deep CNN Acoustic Perception Model Loaded ('models/soundgrid_web_model.pt')")
    print("✅ Derivative-Aware Autonomous Policy Active")
    print("-" * 85)

    test_stream = [
        ("Normal Grid Hum", "normal"),
        ("Normal Grid Hum", "normal"),
        ("Developing Partial Discharge", "discharge"),
        ("Developing Partial Discharge", "discharge"),
        ("Severe Electrical Arcing", "arcing"),
        ("Severe Electrical Arcing", "arcing"),
        ("System Recovery Hum", "normal"),
        ("System Recovery Hum", "normal")
    ]

    obs, _ = env.reset(initial_load=0.70, initial_temp=62.0)
    total_reward = 0.0

    header = f"{'Step':^4} | {'Audio Ingested':^26} | {'CNN Anom':^9} | {'Energy':^7} | {'Load':^6} | {'Core Temp':^9} | {'Action Taken':^22}"
    print(header)
    print("-" * 96)

    for step_num, (signal_name, pattern) in enumerate(test_stream, start=1):
        raw_audio = generate_audio_sample(pattern)
        action = agent.predict(obs)
        action_name = env.ACTION_NAMES[action]

        obs, reward, terminated, _, info = env.step(action, raw_audio)
        total_reward += reward

        anom_pct = f"{info['anomaly_prob'] * 100:5.1f}%"
        energy_str = f"{info['acoustic_energy']:5.2f}"
        load_pct = f"{info['load'] * 100:5.1f}%"
        temp_c = f"{info['temp']:5.1f} °C"

        print(f"{step_num:02d}   | {signal_name:<26} | {anom_pct:>8} | {energy_str:>6} | {load_pct:>5} | {temp_c:>8} | [{action}] {action_name:<20}")

        if terminated:
            print(f"\n🛑 [PROTECTIVE ACTION]: Safety trip executed at Step {step_num:02d}. Breakdown prevented!")
            break

    print("-" * 96)
    print(f"🎯 Total Cumulative System Reward: {total_reward:.2f}")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
