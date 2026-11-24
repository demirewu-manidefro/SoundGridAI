# VoltEcho AI - Docker Deployment Guide 🐳

This guide explains how to build, test, and deploy the entire **VoltEcho AI** power transformer protection platform using Docker and Docker Compose.

---

## 🚀 Quick Start (Local Docker Run)

### 1. Build and Run in Detached Mode
To build the container and start the platform in the background, run:
```bash
docker-compose up --build -d
```
*(Or if using the Docker CLI v2: `docker compose up --build -d`)*

### 2. Verify Container Health
Check the container status and health:
```bash
docker-compose ps
```
You should see:
```text
NAME                     COMMAND                  SERVICE       STATUS
voltecho-ai-platform     "uvicorn web.backen…"   voltecho-ai   running (healthy)
```

### 3. Open the Application
Navigate in your browser to:
* **Landing Page & Live AI Studio:** [http://localhost:8000/](http://localhost:8000/)
* **Scientific Methodology & Algorithms (Full Page):** [http://localhost:8000/methodology.html](http://localhost:8000/methodology.html)
* **Backend API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **System Health Endpoint:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 4. View Real-Time Server Logs
```bash
docker-compose logs -f voltecho-ai
```

### 5. Stop the Platform
```bash
docker-compose down
```

---

## ☁️ Cloud Deployment Options

### Option A: Render (Zero-Config One-Click)
1. Push your repository to GitHub.
2. Sign in to [render.com](https://render.com) and click **New + > Web Service**.
3. Select your repository `demirewu-manidefro/SoundGridAI`.
4. Choose **Docker** as the Runtime environment.
5. Render will automatically detect the `Dockerfile`, build it, and launch your public URL (e.g. `https://soundgrid-ai.onrender.com`).

### Option B: Railway
1. Sign in to [railway.app](https://railway.app).
2. Click **New Project > Deploy from GitHub Repo**.
3. Select this repository.
4. Railway will automatically build the `Dockerfile` and assign an HTTPS domain.

### Option C: Cloud VPS (AWS EC2, DigitalOcean, Hetzner, Linode)
1. SSH into your Ubuntu/Debian server:
   ```bash
   ssh ubuntu@your-server-ip
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/demirewu-manidefro/SoundGridAI.git
   cd SoundGridAI
   ```
3. Start with Docker Compose:
   ```bash
   sudo docker-compose up -d --build
   ```
4. Configure Nginx with Let's Encrypt SSL for your custom domain.

---

## 🛡️ Production Architectural Highlights
- **Base Image:** `python:3.12-slim` for minimal vulnerability footprint and small image size.
- **PyTorch CPU Wheels:** Avoids bulky CUDA GPU binaries (~800MB saved), ensuring fast builds and reliable execution on any standard CPU server.
- **Persistent Storage:** `contacts_inquiries.json` is mounted via Docker volume, so academic reviewer messages and contact requests are never lost on container restarts.
- **Built-in Healthcheck:** Periodically queries `/api/health` every 30s to verify FastAPI and PPO model checkpoint readiness.
