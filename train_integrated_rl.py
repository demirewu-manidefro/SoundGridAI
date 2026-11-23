"""
SoundGrid AI - Fully Coupled PPO Training Script
================================================
Drop this code into Google Colab or Kaggle to retrain the PPO agent
with live audio signals passing through SoundGridCNN in every step!
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing gymnasium and stable_baselines3 (installed in Colab/Kaggle)
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Notice: Run '!pip install gymnasium stable-baselines3' when training on Colab/Kaggle.")

# 1. Native PyTorch Mel-Spectrogram Feature Extractor
class FastMelExtractor:
    def __init__(self, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_time_frames = 94
        self.window = torch.hann_window(n_fft)
        self.filterbank = self._get_fb(sr, n_fft, n_mels)

    def _get_fb(self, sr, n_fft, n_mels):
        mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + (sr // 2) / 700.0)
        m_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(1, n_mels + 1):
            for k in range(bins[i - 1], bins[i]):
                fb[i - 1, k] = (k - bins[i - 1]) / (bins[i] - bins[i - 1] + 1e-6)
            for k in range(bins[i], bins[i + 1]):
                fb[i - 1, k] = (bins[i + 1] - k) / (bins[i + 1] - bins[i] + 1e-6)
        return torch.from_numpy(fb)

    def extract(self, audio_np: np.ndarray) -> torch.Tensor:
        target_len = int(self.sr * 3.0)
        if len(audio_np) < target_len:
            audio_np = np.pad(audio_np, (0, target_len - len(audio_np)))
        else:
            audio_np = audio_np[:target_len]

        x = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=self.window, return_complex=True)
        power = stft.abs() ** 2
        mel = torch.matmul(self.filterbank, power.squeeze(0))
        spec_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        spec_db = spec_db - torch.max(spec_db)
        norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)
        if norm.shape[1] < self.target_time_frames:
            norm = F.pad(norm, (0, self.target_time_frames - norm.shape[1]))
        else:
            norm = norm[:, :self.target_time_frames]
        return norm.unsqueeze(0).unsqueeze(0)


# 2. Fully-Integrated Gymnasium Environment
if SB3_AVAILABLE:
    class EndToEndSoundGridEnv(gym.Env):
        def __init__(self, cnn_model=None, audio_dataset_paths=None):
            super().__init__()
            self.cnn_model = cnn_model
            self.extractor = FastMelExtractor()
            self.audio_dataset_paths = audio_dataset_paths or []
            
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.5], dtype=np.float32),
                dtype=np.float32
            )
            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            self.load = float(np.random.uniform(0.65, 0.85))
            self.temp = float(np.random.uniform(55.0, 75.0))
            self.anomaly_prob = 0.05
            return np.array([self.anomaly_prob, self.load, self.temp / 100.0], dtype=np.float32), {}

        def _generate_synthetic_audio(self, condition="normal"):
            t = np.linspace(0, 3.0, 48000, endpoint=False)
            if condition == "normal":
                return 0.5 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.random.normal(0, 1, len(t))
            elif condition == "warning":
                wave = 0.3 * np.sin(2 * np.pi * 100 * t)
                bursts = np.zeros_like(t)
                for _ in range(15):
                    idx = np.random.randint(0, len(t) - 400)
                    bursts[idx:idx + 400] += np.random.normal(0, 0.7, 400)
                return wave + bursts
            else:
                bursts = np.zeros_like(t)
                for _ in range(45):
                    idx = np.random.randint(0, len(t) - 500)
                    bursts[idx:idx + 500] += np.random.normal(0, 1.1, 500)
                return bursts + 0.2 * np.random.normal(0, 1, len(t))

        def step(self, action):
            self.step_count += 1

            # A. Dynamic Audio Condition Selection
            # 70% normal, 20% developing warning, 10% severe arcing
            r = np.random.rand()
            if r < 0.70:
                audio = self._generate_synthetic_audio("normal")
            elif r < 0.90:
                audio = self._generate_synthetic_audio("warning")
            else:
                audio = self._generate_synthetic_audio("critical")

            # B. Real-Time Deep CNN Feature Inference
            if self.cnn_model is not None:
                spec_tensor = self.extractor.extract(audio)
                with torch.no_grad():
                    logits = self.cnn_model(spec_tensor)
                    probs = F.softmax(logits, dim=1)
                    self.anomaly_prob = float(probs[0, 1].item())
            else:
                self.anomaly_prob = 0.05 if r < 0.70 else (0.45 if r < 0.90 else 0.85)

            # C. Load Control Action
            if action == 1:
                self.load = min(1.0, self.load + 0.10)
            elif action == 2:
                self.load = max(0.10, self.load - 0.15)
            elif action == 3:
                self.load = max(0.10, self.load - 0.30)
            elif action == 4:
                self.load = 0.0

            # D. Thermal Dynamics
            ambient = 30.0
            q_gen = (self.load**2 * 4.2) + (self.anomaly_prob * 10.0)
            q_diss = 0.09 * (self.temp - ambient)
            self.temp = float(np.clip(self.temp + q_gen - q_diss, 30.0, 130.0))

            # E. Reward Function
            reward = self.load * 8.0
            terminated = False

            if self.temp <= 80.0 and self.anomaly_prob < 0.40:
                reward += 10.0
                if action == 1:
                    reward += 5.0

            if self.temp > 85.0:
                reward -= 2.0 * (self.temp - 85.0)

            if self.temp > 85.0 or self.anomaly_prob > 0.50:
                if action in [2, 3]:
                    reward += 12.0
                elif action == 1:
                    reward -= 40.0

            if action == 4:
                if self.anomaly_prob >= 0.75 or self.temp >= 105.0:
                    reward += 50.0
                else:
                    reward -= 70.0
                terminated = True

            if self.temp >= 118.0 and not terminated:
                reward -= 150.0
                terminated = True

            if self.step_count >= 100:
                terminated = True

            next_state = np.array([self.anomaly_prob, self.load, self.temp / 100.0], dtype=np.float32)
            return next_state, reward, terminated, False, {}

print("SoundGrid EndToEnd environment script prepared.")
