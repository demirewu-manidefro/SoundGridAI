import time
import json
import urllib.request
import threading
import sys
from pathlib import Path
import uvicorn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.backend.main import app

class ServerThread(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8000):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True

def send_get(path):
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url) as resp:
        return resp.status, json.loads(resp.read().decode())

def send_get_raw(path):
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url) as resp:
        return resp.status, resp.read().decode("utf-8", errors="ignore")

def send_post(path, data):
    url = f"http://127.0.0.1:8000{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def test_live_api():
    print("=" * 70)
    print("🚀 STARTING LIVE FASTAPI SERVER ON http://127.0.0.1:8000")
    print("=" * 70)

    server = ServerThread()
    server.start()
    time.sleep(1.5)  # Allow server to bind and start

    try:
        # 1. Health Endpoint
        print("\n[1] GET /api/health")
        status, data = send_get("/api/health")
        print(f"Status: {status} | Service: {data['service']} | PPO Model Loaded: {data['ppo_model_loaded']}")
        assert status == 200

        # 2. Controllers List
        print("\n[2] GET /api/controllers")
        status, data = send_get("/api/controllers")
        print(f"Status: {status} | Registered Controllers: {len(data)}")
        for c in data:
            print(f"  - [{c['id']}] {c['name']}")
        assert status == 200

        # 3. Regimes List
        print("\n[3] GET /api/regimes")
        status, data = send_get("/api/regimes")
        print(f"Status: {status} | Registered Regimes: {len(data)}")
        for reg in data:
            print(f"  - [{reg['id']}] {reg['name']} (Color: {reg['color']})")
        assert status == 200

        # 4. Single-Step Decision
        print("\n[4] POST /api/step (SoundGrid 8D PPO Control)")
        step_payload = {
            "controller": "soundgrid_ppo",
            "load": 0.85, "prev_load": 0.85,
            "temp": 65.0, "prev_temp": 65.0,
            "anomaly_prob": 0.04, "prev_anomaly": 0.04,
            "acoustic_energy": 0.15, "ambient_temp": 30.0,
            "step_count": 1, "max_steps": 100
        }
        status, data = send_post("/api/step", step_payload)
        print(f"Status: {status} | Decision: {data['action_name']}")
        print(f"Probabilities: {data['action_probabilities']}")
        print(f"Next State: Load={data['next_load']} | Temp={data['next_temp']}°C | Reward={data['reward']}")
        assert status == 200

        # 5. Full Episode Simulation (Severe Arcing)
        print("\n[5] POST /api/simulate (Regime: Severe Arcing)")
        sim_payload = {
            "controller": "soundgrid_ppo",
            "regime": "critical",
            "max_steps": 50
        }
        status, data = send_post("/api/simulate", sim_payload)
        print(f"Status: {status} | Outcome: {data['outcome']} ({data['outcome_color']})")
        print(f"Shutdown Executed at Step: {data['steps_executed']} | Total Reward: {data['total_reward']}")
        print(f"Generated Trajectory Points: {len(data['trajectory'])}")
        assert status == 200

        # 6. Head-to-Head Comparative Benchmark
        print("\n[6] POST /api/benchmark (Testing all 4 controllers)")
        bench_payload = {"episodes_per_regime": 3}
        status, data = send_post("/api/benchmark", bench_payload)
        print(f"Status: {status} | Total Episodes Tested: {data['episodes_tested']}")
        print("-" * 85)
        print(f"{'Controller Name':<35} | {'Mean Rew':>8} | {'Burnout':>7} | {'False Trip':>10} | Verdict")
        print("-" * 85)
        for m in data["results"]:
            print(f"{m['controller_name']:<35} | {m['mean_reward']:>8.1f} | {m['catastrophic_burnout_rate_pct']:>6.1f}% | {m['false_alarm_blackout_rate_pct']:>9.1f}% | {m['verdict']}")
        print("-" * 85)
        assert status == 200

        # 7. Frontend Static UI Files
        print("\n[7] GET / (Frontend Web Application)")
        status, html = send_get_raw("/")
        print(f"Status: {status} | HTML Size: {len(html)} bytes | Contains Title: {'SoundGrid AI' in html}")
        assert status == 200
        assert "SoundGrid AI" in html

        status_css, css = send_get_raw("/style.css")
        print(f"GET /style.css -> Status: {status_css} | Size: {len(css)} bytes")
        assert status_css == 200

        status_js, js = send_get_raw("/app.js")
        print(f"GET /app.js -> Status: {status_js} | Size: {len(js)} bytes")
        assert status_js == 200

        # Verify image serving
        with urllib.request.urlopen("http://127.0.0.1:8000/images/substation_ai.png") as r:
            assert r.status == 200
            print(f"GET /images/substation_ai.png -> Status: 200 | Size: {len(r.read())} bytes")

        with urllib.request.urlopen("http://127.0.0.1:8000/images/transformer_burnout.png") as r:
            assert r.status == 200
            print(f"GET /images/transformer_burnout.png -> Status: 200 | Size: {len(r.read())} bytes")

        print("\n" + "=" * 70)
        print("🎉 ALL FASTAPI REST ENDPOINTS & FRONTEND UI TESTED LIVE AND PASSED 100%!")
        print("=" * 70)

    finally:
        server.stop()

if __name__ == "__main__":
    test_live_api()
