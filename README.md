<div align="center">

# 🚗 Real-Time Vehicle Detection & Tracking Platform

### *Production-Grade AI-Powered Traffic Intelligence System*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/vehicle-tracking/ci.yml?branch=main&style=flat-square&label=CI)](../../actions)
[![Model mAP50](https://img.shields.io/badge/mAP50-98.8%25-brightgreen?style=flat-square)]()
[![Model mAP50-95](https://img.shields.io/badge/mAP50--95-86.9%25-brightgreen?style=flat-square)]()
[![Precision](https://img.shields.io/badge/Precision-99.1%25-brightgreen?style=flat-square)]()
[![Recall](https://img.shields.io/badge/Recall-96.5%25-brightgreen?style=flat-square)]()

**Detect · Track · Count · Analyze · Deploy**

An end-to-end platform that trains a custom YOLOv8 model on vehicle datasets, performs real-time detection and multi-object tracking with ByteTrack, exposes 15+ REST API endpoints via FastAPI, delivers a modern React dashboard with live analytics, and deploys via Docker with full CI/CD, monitoring, and MLOps pipelines.

[🔗 Live Demo](#-live-demo) · [📖 Documentation](#-table-of-contents) · [🚀 Quick Start](#-quick-start) · [🐛 Report Bug](../../issues) · [✨ Request Feature](../../issues)

</div>

---

## 📸 Screenshots

<div align="center">

| Dashboard | Image Detection |
|:---------:|:---------------:|
| *Real-time stats, class distribution, activity feed* | *Drag-drop upload, bounding box visualization* |

| Video Tracking | Live Camera |
|:--------------:|:-----------:|
| *ByteTrack multi-object tracking, vehicle trails* | *Real-time webcam detection via WebSocket* |

| Analytics | API Documentation |
|:---------:|:-----------------:|
| *Recharts timeline, pie charts, performance metrics* | *Interactive Swagger UI with 15+ endpoints* |

</div>

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Quick Start](#-quick-start)
- [Installation Guide](#-installation-guide)
- [Dataset Documentation](#-dataset-documentation)
- [Training Pipeline](#-training-pipeline)
- [Inference Pipeline](#-inference-pipeline)
- [ByteTrack Integration](#-bytetrack-integration)
- [Backend Documentation](#-backend-documentation)
- [API Reference](#-api-reference)
- [Frontend Documentation](#-frontend-documentation)
- [Database Design](#-database-design)
- [Monitoring & Observability](#-monitoring--observability)
- [Security](#-security)
- [Docker Setup](#-docker-setup)
- [Deployment Guide](#-deployment-guide)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Environment Variables](#-environment-variables)
- [Testing](#-testing)
- [Benchmarking](#-benchmarking)
- [Roadmap](#-roadmap)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)

---

## 🎯 Executive Summary

### What It Does

This platform provides a **complete, production-ready pipeline** for vehicle detection and traffic intelligence — from raw CVAT-annotated data to a deployed, monitored web application accessible from any device.

### Real-World Business Use Cases

| Domain | Application |
|--------|-------------|
| **Smart Cities** | Traffic flow optimization, congestion detection, signal timing |
| **Parking Management** | Occupancy monitoring, vehicle counting, duration tracking |
| **Highway Monitoring** | Speed estimation, vehicle classification, incident detection |
| **Toll Systems** | Vehicle counting per lane, class-based toll computation |
| **Security** | Perimeter surveillance, vehicle tracking across cameras |
| **Logistics** | Fleet tracking, loading dock monitoring, yard management |

### Key Technical Innovations

- **Custom-trained YOLOv8n** achieving **98.8% mAP50** on vehicle detection with only a 5.9MB model
- **Browser-to-server WebSocket pipeline** for real-time webcam detection without server-side camera access
- **Line-crossing counter** with cross-product direction detection for bidirectional traffic counting
- **Temporal-decay heatmaps** for both offline analysis and live streaming traffic density visualization
- **DVC-versioned datasets** with reproducible pipelines from raw CVAT XML to trained model

---

## 🔗 Live Demo

| Resource | URL |
|----------|-----|
| 🌐 Production App | `https://your-deployment-url.com` |
| 📡 API Endpoint | `https://your-api-url.com/docs` |
| 📊 Grafana Dashboard | `https://your-monitoring-url.com:3001` |
| 🧪 MLflow UI | `https://your-mlflow-url.com:5000` |
| 🎬 Demo Video | *Coming soon* |

---

## 📊 Model Performance

<div align="center">

| Metric | Value | Description |
|:------:|:-----:|:------------|
| **mAP50** | **98.8%** | Mean Average Precision at IoU 0.50 — primary object detection metric |
| **mAP50-95** | **86.9%** | Mean AP averaged across IoU thresholds 0.50→0.95 (COCO standard) |
| **Precision** | **99.1%** | Fraction of correct detections among all detections (low false positives) |
| **Recall** | **96.5%** | Fraction of actual objects detected (low false negatives) |
| **Model Size** | **5.9 MB** | Compact YOLOv8n architecture — deployable on edge devices |
| **Architecture** | **YOLOv8n** | 3.01M parameters, 8.2 GFLOPs — nano variant for speed |
| **Classes** | **2** | `car`, `minivan` — fine-tuned on custom vehicle dataset |
| **Training** | **25 epochs** | AdamW optimizer, pretrained COCO weights, early stopping |

</div>

### Metric Explanations

- **mAP50** — The gold standard for object detection. A value of 98.8% means the model correctly localizes and classifies nearly every vehicle at a 50% IoU threshold.
- **mAP50-95** — A stricter metric averaging across IoU thresholds from 50% to 95%. Our 86.9% indicates excellent bounding box precision even at tight overlaps.
- **Precision** — With 99.1%, virtually every detection the model makes is correct, minimizing false alarms in production.
- **Recall** — At 96.5%, the model catches the vast majority of vehicles in every frame.

---

## ✨ Features

### Core Detection & Tracking

| Feature | Description | Module |
|---------|-------------|--------|
| 🎯 **Vehicle Detection** | YOLOv8n-based detection with configurable confidence/IoU thresholds | `src/inference/detector.py` |
| 🏷️ **Vehicle Classification** | 2-class classification: cars and minivans with per-class statistics | `src/inference/detector.py` |
| 🔗 **Multi-Object Tracking** | ByteTrack algorithm for persistent vehicle IDs across frames | `src/inference/tracker.py` |
| 📷 **Image Processing** | Single-image detection with annotated output and result storage | `src/inference/pipeline.py` |
| 🎬 **Video Processing** | Full video tracking with annotated output, summary statistics | `src/inference/pipeline.py` |
| 📹 **Live Camera** | Real-time webcam detection via WebSocket (browser → server → browser) | `backend/app/routers/websocket.py` |
| 📡 **RTSP Streams** | Process IP camera feeds via RTSP URLs | `src/inference/pipeline.py` |

### Analytics Engine

| Feature | Description | Module |
|---------|-------------|--------|
| 🔢 **Vehicle Counting** | Line-crossing counter with bidirectional detection (up/down) | `src/analytics/counter.py` |
| 🏎️ **Speed Estimation** | Pixel-to-real-world speed estimation with smoothing window | `src/analytics/speed_estimator.py` |
| 🗺️ **Traffic Heatmaps** | Spatial density maps with gaussian spread and temporal decay | `src/analytics/heatmap.py` |
| 📊 **Detection Timeline** | Historical trends by day/week/month with class breakdowns | `backend/app/routers/analytics.py` |

### Platform

| Feature | Description | Module |
|---------|-------------|--------|
| 🌐 **REST API** | 15+ endpoints with Swagger docs, pagination, file upload | `backend/app/routers/` |
| 🔐 **JWT Authentication** | Register, login, token refresh, role-based access | `backend/app/routers/auth.py` |
| 📱 **React Dashboard** | 5-page SPA with dark theme, glassmorphism, live stats | `frontend/src/` |
| 🧪 **MLflow Tracking** | Experiment logging, metric comparison, model versioning | `src/training/train.py` |
| 📦 **DVC Pipelines** | Reproducible data pipeline from CVAT XML → trained model | `dvc.yaml` |
| 🐳 **Docker** | 8-service compose (API, frontend, Postgres, Redis, Nginx, MLflow, Prometheus, Grafana) | `deployment/` |
| 🔄 **CI/CD** | GitHub Actions for lint → test → build → push → deploy | `.github/workflows/` |
| 📈 **Monitoring** | Prometheus metrics, Grafana dashboards, health checks | `monitoring/` |
| 🛡️ **Model Export** | ONNX, TorchScript, OpenVINO export for production deployment | `src/training/export.py` |

---

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw Dataset<br/>CVAT XML + Images] --> B[Data Pipeline]
        B --> C[Processed Dataset<br/>YOLO Format]
    end

    subgraph "Training Layer"
        C --> D[YOLOv8 Training]
        D --> E[MLflow<br/>Experiment Tracking]
        D --> F[Model Registry<br/>best.pt / ONNX]
    end

    subgraph "Inference Layer"
        F --> G[Vehicle Detector]
        F --> H[Vehicle Tracker<br/>ByteTrack]
        G --> I[Detection Pipeline]
        H --> I
        I --> J[Analytics Engine]
    end

    subgraph "API Layer"
        I --> K[FastAPI Backend]
        J --> K
        K --> L[REST Endpoints]
        K --> M[WebSocket<br/>Real-time]
        K --> N[SQLite/PostgreSQL]
    end

    subgraph "Frontend Layer"
        L --> O[React Dashboard]
        M --> O
        O --> P[Dashboard]
        O --> Q[Detection UI]
        O --> R[Live Camera]
        O --> S[Analytics Charts]
    end

    subgraph "DevOps Layer"
        T[Docker Compose] --> K
        T --> O
        T --> N
        U[GitHub Actions] --> T
        V[Prometheus] --> K
        W[Grafana] --> V
    end
```

### Training Pipeline

```mermaid
graph LR
    A[CVAT XML] -->|cvat_to_yolo.py| B[YOLO Labels]
    B -->|validator.py| C[Validated Data]
    C -->|dataset_splitter.py| D[Train/Val/Test Split]
    D -->|analyzer.py| E[Dataset Stats]
    D -->|augmentation.py| F[Augmented Data]
    D -->|train.py| G[YOLOv8 Training]
    G -->|MLflow| H[Metrics & Artifacts]
    G --> I[best.pt]
    I -->|export.py| J[ONNX / TorchScript]
```

### Inference Pipeline

```mermaid
graph LR
    A[Input] --> B{Source Type}
    B -->|Image| C[VehicleDetector]
    B -->|Video| D[VehicleTracker]
    B -->|Webcam| E[WebSocket Handler]
    B -->|RTSP| D

    C --> F[DetectionResult]
    D --> G[TrackingSummary]
    E --> F

    F --> H[Annotated Output]
    F --> I[Database Record]
    G --> H
    G --> I

    F --> J[Analytics Engine]
    J --> K[Counter]
    J --> L[Speed Estimator]
    J --> M[Heatmap Generator]
```

### Backend Architecture

```mermaid
graph TB
    subgraph "FastAPI Application"
        A[main.py<br/>Lifespan + CORS]

        subgraph "Routers"
            B[auth.py<br/>JWT Auth]
            C[predict.py<br/>Detection]
            D[upload.py<br/>Files]
            E[analytics.py<br/>Stats]
            F[models.py<br/>Registry]
            G[health.py<br/>Status]
            H[websocket.py<br/>Real-time]
        end

        subgraph "Core"
            I[config.py<br/>Settings]
            J[database.py<br/>SQLAlchemy]
            K[schemas/<br/>Pydantic]
            L[models/<br/>ORM]
        end

        subgraph "Inference"
            M[DetectionPipeline<br/>Singleton]
        end
    end

    A --> B & C & D & E & F & G & H
    C & D --> M
    H --> M
    C & D & E --> J
    J --> L
    B & C --> K
```

### CI/CD Workflow

```mermaid
graph LR
    A[Push/PR] --> B[Ruff Lint]
    B --> C[Pytest + Coverage]
    C --> D[Frontend Build]
    D --> E[Docker Build]
    E --> F{Branch?}
    F -->|main| G[Push to GHCR]
    G --> H[Deploy to Render]
    F -->|develop| I[Artifacts Only]
```

---

## 🛠️ Tech Stack

### AI / Machine Learning

| Technology | Version | Purpose | Why Chosen |
|-----------|---------|---------|------------|
| **YOLOv8** | 8.4+ | Object detection | State-of-the-art accuracy, built-in tracking, easy export |
| **PyTorch** | 2.0+ | Deep learning | Industry standard, dynamic graphs, CUDA support |
| **ByteTrack** | — | Multi-object tracking | Zero-shot tracking, no ReID overhead, high accuracy |
| **Albumentations** | 2.0+ | Data augmentation | Fastest augmentation library, bbox-safe transforms |
| **MLflow** | 2.8+ | Experiment tracking | Open-source, model registry, artifact storage |
| **DVC** | 3.30+ | Data versioning | Git-like workflow for datasets, reproducible pipelines |
| **ONNX** | — | Model export | Cross-platform, TensorRT compatible, 2-3x speedup |

### Backend

| Technology | Version | Purpose | Why Chosen |
|-----------|---------|---------|------------|
| **FastAPI** | 0.104+ | REST API framework | Async, auto-docs, type-safe, highest performance |
| **Uvicorn** | 0.24+ | ASGI server | Production-grade, HTTP/2, WebSocket support |
| **SQLAlchemy** | 2.0+ | ORM | Mature, async support, migration-ready |
| **PostgreSQL** | 15 | Production database | ACID, JSON support, scalable |
| **SQLite** | — | Development database | Zero-config, file-based, perfect for dev |
| **Redis** | 7+ | Caching & rate limiting | In-memory speed, pub/sub for real-time |
| **Pydantic** | 2.0+ | Data validation | FastAPI native, runtime type checking |

### Frontend

| Technology | Version | Purpose | Why Chosen |
|-----------|---------|---------|------------|
| **React** | 18.2 | UI framework | Component-based, hooks, massive ecosystem |
| **Vite** | 5.0+ | Build tool | 10-100x faster than Webpack, HMR |
| **React Router** | 6.20+ | Client routing | Standard SPA routing, nested routes |
| **Recharts** | 2.10+ | Data visualization | React-native charts, composable, responsive |
| **Axios** | 1.6+ | HTTP client | Interceptors, JWT auto-attach, upload progress |

### Infrastructure & DevOps

| Technology | Purpose |
|-----------|---------|
| **Docker** | Containerization (multi-stage builds) |
| **Docker Compose** | 8-service orchestration |
| **Nginx** | Reverse proxy, rate limiting, SSL termination |
| **GitHub Actions** | CI/CD (lint → test → build → deploy) |
| **Prometheus** | Metrics collection |
| **Grafana** | Metrics visualization & alerting |

### Security

| Technology | Purpose |
|-----------|---------|
| **python-jose** | JWT token creation/validation |
| **passlib + bcrypt** | Password hashing (60+ rounds) |
| **CORS Middleware** | Cross-origin request control |
| **Nginx Rate Limiting** | 100 req/min per IP |

---

## 📁 Folder Structure

```
vehicle-tracking-platform/
│
├── 📂 backend/                    # FastAPI REST API
│   ├── app/
│   │   ├── main.py                # App entry, lifespan, CORS, router registration
│   │   ├── config.py              # Pydantic settings (env-based configuration)
│   │   ├── database.py            # SQLAlchemy engine, session factory, Base
│   │   ├── models/
│   │   │   └── detection.py       # ORM: User, DetectionRecord, VideoRecord, ModelRecord, Analytics
│   │   ├── schemas/
│   │   │   └── __init__.py        # Pydantic: request/response models for all endpoints
│   │   ├── routers/
│   │   │   ├── auth.py            # POST /register, /login, /refresh, GET /me
│   │   │   ├── predict.py         # POST /image, /video, GET /{id}, GET /
│   │   │   ├── upload.py          # POST /upload/image, /upload/video
│   │   │   ├── analytics.py       # GET /summary, /timeline, /classes, /recent
│   │   │   ├── models.py          # GET /, /active, POST /activate/{id}, GET /info
│   │   │   ├── health.py          # GET /health, /health/ready
│   │   │   └── websocket.py       # WS /api/v1/ws/detect (real-time camera)
│   │   ├── services/              # Business logic layer
│   │   ├── repositories/          # Database access layer
│   │   ├── middleware/             # Custom middleware
│   │   └── utils/                 # Backend utilities
│   └── Dockerfile                 # Multi-stage: python-slim builder → runtime
│
├── 📂 frontend/                   # React SPA
│   ├── src/
│   │   ├── App.jsx                # Sidebar navigation, routing, page transitions
│   │   ├── main.jsx               # React DOM entry point
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Stat cards, class distribution, recent activity
│   │   │   ├── ImageDetection.jsx # Drag-drop upload, detection results table
│   │   │   ├── VideoTracking.jsx  # Video upload, tracking summary, tracks table
│   │   │   ├── LiveCamera.jsx     # WebSocket webcam detection with canvas overlay
│   │   │   └── Analytics.jsx      # Recharts bar/pie/line charts, performance metrics
│   │   ├── services/
│   │   │   └── api.js             # Axios client, JWT interceptors, all API methods
│   │   └── styles/
│   │       └── index.css          # Premium dark design system (glassmorphism, gradients)
│   ├── index.html                 # HTML entry with Inter font, SEO meta tags
│   ├── vite.config.js             # Vite config with API proxy to backend
│   ├── package.json               # Dependencies: React, Vite, Recharts, Axios
│   └── Dockerfile                 # Multi-stage: Node build → Nginx serve
│
├── 📂 src/                        # Core ML modules
│   ├── data/
│   │   ├── cvat_to_yolo.py        # CVAT XML → YOLO format converter
│   │   ├── validator.py           # Label validation (ranges, classes, duplicates)
│   │   ├── dataset_splitter.py    # Stratified 70/20/10 train/val/test split
│   │   ├── analyzer.py            # Dataset EDA: class stats, distributions, plots
│   │   └── augmentation.py        # Albumentations pipeline (weather, lighting, occlusion)
│   ├── inference/
│   │   ├── detector.py            # VehicleDetector: YOLOv8 wrapper, Detection, DetectionResult
│   │   ├── tracker.py             # VehicleTracker: ByteTrack, TrackInfo, TrackingSummary
│   │   └── pipeline.py            # DetectionPipeline: unified image/video/stream/bytes
│   ├── training/
│   │   ├── train.py               # Full training pipeline with MLflow + early stopping
│   │   └── export.py              # Model export: ONNX, TorchScript, OpenVINO
│   ├── analytics/
│   │   ├── counter.py             # Line-crossing vehicle counter (cross-product direction)
│   │   ├── speed_estimator.py     # Pixel→real-world speed with smoothing
│   │   └── heatmap.py             # Traffic density heatmap with decay + hotspots
│   └── utils/
│       └── visualization.py       # Drawing: boxes, trails, counting lines, overlays
│
├── 📂 configs/
│   ├── training_config.yaml       # All training hyperparameters
│   └── (dataset.yaml in datasets/processed/)
│
├── 📂 datasets/
│   ├── raw/                       # Original CVAT annotations + images
│   └── processed/                 # YOLO-format labels + train/val/test splits
│
├── 📂 tests/
│   ├── conftest.py                # Shared fixtures: temp datasets, PNGs, API client
│   ├── unit/
│   │   ├── test_data_pipeline.py  # Validator, splitter, analyzer tests
│   │   ├── test_inference.py      # Detection, DetectionResult, TrackInfo tests
│   │   └── test_analytics.py      # Counter, speed estimator, heatmap tests
│   └── api/
│       └── test_api.py            # Health, auth, analytics, upload, predict tests
│
├── 📂 deployment/
│   ├── docker-compose.yml         # 8 services: backend, frontend, postgres, redis, mlflow, nginx, prometheus, grafana
│   └── nginx/
│       ├── nginx.conf             # Reverse proxy, rate limiting, WebSocket, gzip
│       └── frontend.conf          # SPA routing, asset caching
│
├── 📂 monitoring/
│   └── prometheus/
│       └── prometheus.yml         # Scrape config: backend (10s), self, node-exporter
│
├── 📂 .github/workflows/
│   ├── ci.yml                     # Lint → Test → Docker Build (push/PR)
│   └── cd.yml                     # Build → Push GHCR → Deploy Render (main only)
│
├── 📂 scripts/
│   ├── prepare_data.py            # One-command data pipeline orchestrator
│   └── run_training.py            # Standalone training launcher
│
├── dvc.yaml                       # DVC pipeline: prepare → validate → split → analyze → train
├── requirements.txt               # 30+ Python dependencies (categorized)
├── pyproject.toml                 # Pytest, Ruff, Black, Coverage configuration
├── .env.example                   # Environment variables template
├── .gitignore                     # Git exclusions
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ ([python.org](https://python.org))
- Node.js 18+ ([nodejs.org](https://nodejs.org))
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/vehicle-tracking-platform.git
cd vehicle-tracking-platform

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

This runs the full pipeline: CVAT convert → validate → split (70/20/10) → analyze → augment.

### 3. Train Model

```bash
python scripts/run_training.py
```

Trains YOLOv8n for 25 epochs. Monitor live progress (losses, mAP, precision, recall per epoch). Outputs `models/vehicle_detector.pt`.

### 4. Start Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

API available at **http://localhost:8000/docs**

### 5. Start Frontend

```bash
cd frontend
npm install
node node_modules\vite\bin\vite.js --port 3000
```

Dashboard available at **http://localhost:3000**

### 6. Detect!

1. Open **http://localhost:3000**
2. Navigate to **Image Detection** → drag & drop any car image → click **Run Detection**
3. Navigate to **Live Camera** → click **Start Detection** for real-time webcam detection

---

## 📦 Installation Guide

### Windows

```powershell
# Python (from python.org)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Node.js (from nodejs.org)
cd frontend
npm install

# GPU support (optional — NVIDIA CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install

# GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Docker (Recommended for Production)

```bash
cd deployment
docker-compose up -d

# Services available:
# http://localhost       → Frontend (Nginx)
# http://localhost:8000  → Backend API
# http://localhost:5000  → MLflow
# http://localhost:9090  → Prometheus
# http://localhost:3001  → Grafana (admin/admin)
```

---

## 📊 Dataset Documentation

### Source

**Cars Tracking & Object Detection Dataset** — Aerial vehicle footage with CVAT annotations.

### Annotation Format

| Stage | Format | Details |
|-------|--------|---------|
| **Raw** | CVAT XML | Track-based annotations with frame interpolation |
| **Processed** | YOLO TXT | `class_id center_x center_y width height` (normalized) |

### Class Distribution

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `car` | Standard passenger vehicles |
| 1 | `minivan` | Larger passenger vehicles, vans |

### Data Pipeline

```
Raw CVAT XML ──→ cvat_to_yolo.py ──→ YOLO Labels
                                          │
                              validator.py │ (format, range, duplicate checks)
                                          │
                         dataset_splitter.py │ (stratified 70/20/10)
                                          │
                              ┌──────────┼──────────┐
                           Train (210)  Val (60)  Test (31)
```

### Augmentation Strategy

Implemented via `albumentations` with bbox-safe transforms:

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| Random Brightness/Contrast | ±20% | Lighting variation |
| Hue/Saturation shift | ±15° / ±30% | Color robustness |
| Gaussian blur | kernel 3-7 | Camera blur simulation |
| Random rain/fog | OpenCV-based | Weather robustness |
| Horizontal flip | 50% | Directional invariance |
| Random crop + resize | 80-100% | Scale invariance |

### Dataset Versioning

```bash
# Initialize DVC
dvc init
dvc add datasets/raw

# Reproduce pipeline
dvc repro
```

---

## 🏋️ Training Pipeline

### Training Workflow

```mermaid
graph TB
    A[Load Config<br/>training_config.yaml] --> B[Load Pretrained<br/>yolov8n.pt]
    B --> C[Setup MLflow<br/>Experiment Tracking]
    C --> D[Train on Dataset]
    D --> E{Early Stopping?}
    E -->|No improvement<br/>10 epochs| F[Stop Training]
    E -->|Improving| D
    F --> G[Save best.pt]
    G --> H[Export to ONNX]
    H --> I[Log to MLflow]
    I --> J[Copy to models/]
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `yolov8n.pt` | Nano variant — fastest, smallest |
| Epochs | 25 (CPU) / 100 (GPU) | Training iterations |
| Batch size | 8 (CPU) / 16 (GPU) | Samples per gradient step |
| Image size | 640×640 | Standard YOLO input |
| Optimizer | AdamW (auto) | Adaptive learning rate |
| Learning rate | 0.001667 | Auto-tuned by Ultralytics |
| Weight decay | 0.0005 | L2 regularization |
| Patience | 10 | Early stopping epochs |
| Mosaic | 1.0 | 4-image mosaic augmentation |
| Mixup | 0.1 | Image blending augmentation |

### Experiment Tracking

All runs are logged to **MLflow**:

```bash
# View experiment history
mlflow server --backend-store-uri runs/mlflow
# Open http://127.0.0.1:5000
```

Tracked metrics: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/mAP50`, `val/mAP50-95`, `val/precision`, `val/recall`.

---

## 🔍 Inference Pipeline

### Pipeline Flow

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1. Ingest | `DetectionPipeline` | Image bytes / video path / webcam ID | OpenCV frame(s) |
| 2. Detect | `VehicleDetector` | BGR numpy array | `DetectionResult` (list of `Detection`) |
| 3. Track | `VehicleTracker` | BGR numpy array | `DetectionResult` + track IDs |
| 4. Annotate | `_draw_detections()` | Frame + detections | Annotated frame |
| 5. Analyze | Analytics engine | Detection stream | Counts, speeds, heatmaps |
| 6. Store | Database | Results | `DetectionRecord` / `VideoRecord` |

### Detection Result Schema

```python
Detection(
    x1, y1, x2, y2,   # Bounding box coordinates
    confidence,         # Detection confidence (0-1)
    class_id,           # 0=car, 1=minivan
    class_name,         # "car" or "minivan"
    track_id,           # Persistent ID (tracking only)
)
```

---

## 🔗 ByteTrack Integration

### How ByteTrack Works

ByteTrack is a **simple, fast, and strong multi-object tracker** that associates detections across frames without requiring a separate re-identification model.

```
Frame N detections ──→ High-confidence matching (IoU) ──→ Matched tracks
                                                              │
         Frame N+1 ──→ Low-confidence matching ──→ Recovered tracks
                                                              │
                        Unmatched detections ──→ New tracks
                        Unmatched tracks ──→ Lost (buffer) ──→ Removed
```

### Key Properties

| Property | Value | Benefit |
|----------|-------|---------|
| **ID persistence** | Track IDs survive occlusion | Accurate counting |
| **Two-stage matching** | High + low confidence | Fewer ID switches |
| **No ReID model** | IoU-only association | Zero overhead |
| **Travel distance** | Euclidean path tracking | Speed estimation input |

---

## 🌐 Backend Documentation

### Architecture Pattern

```
Router (endpoint) → Service (logic) → Repository (database) → ORM Model
                         ↓
                   DetectionPipeline (ML inference)
```

### Middleware Stack

| Layer | Purpose |
|-------|---------|
| **CORS** | Allow frontend origin (`localhost:3000`, `localhost:5173`) |
| **Static Files** | Serve `/uploads/` and `/results/` directories |
| **Lifespan** | Database init, model preloading, directory creation |

### Dependency Injection

```python
# Database session (auto-close)
db: Session = Depends(get_db)

# Pipeline singleton (lazy-loaded)
pipeline = get_pipeline()
```

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/auth/register` | Create new user | ❌ |
| `POST` | `/api/v1/auth/login` | Get JWT tokens | ❌ |
| `POST` | `/api/v1/auth/refresh` | Refresh access token | 🔒 |
| `GET` | `/api/v1/auth/me` | Get current user profile | 🔒 |

### Detection & Tracking

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/predict/image` | Upload image → detect vehicles | ❌ |
| `POST` | `/api/v1/predict/video` | Upload video → track vehicles | ❌ |
| `GET` | `/api/v1/predict/{id}` | Get detection result by ID | ❌ |
| `GET` | `/api/v1/predict/` | List all detections (paginated) | ❌ |
| `WS` | `/api/v1/ws/detect` | Real-time webcam detection | ❌ |

### File Upload

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/upload/image` | Upload image file | ❌ |
| `POST` | `/api/v1/upload/video` | Upload video file | ❌ |

### Analytics

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/v1/analytics/summary` | Platform-wide statistics | ❌ |
| `GET` | `/api/v1/analytics/timeline?days=30` | Detection trends | ❌ |
| `GET` | `/api/v1/analytics/classes` | Class distribution | ❌ |
| `GET` | `/api/v1/analytics/recent?limit=10` | Recent detections | ❌ |

### Model Management

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/v1/models/` | List registered models | ❌ |
| `GET` | `/api/v1/models/active` | Get active model | ❌ |
| `POST` | `/api/v1/models/activate/{id}` | Set active model | ❌ |
| `GET` | `/api/v1/models/info` | Model metadata + metrics | ❌ |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Basic health check |
| `GET` | `/api/v1/health/ready` | Readiness (model loaded, DB connected) |

### Example: Image Detection

```bash
curl -X POST "http://localhost:8000/api/v1/predict/image" \
  -F "file=@car_photo.jpg" \
  -F "confidence=0.25"
```

**Response:**

```json
{
  "id": 1,
  "source_type": "image",
  "count": 3,
  "class_counts": {"car": 2, "minivan": 1},
  "inference_time_ms": 45.2,
  "fps": 22.1,
  "result_url": "/results/images/abc123_result.jpg",
  "detections": [
    {
      "bbox": [120, 200, 380, 450],
      "class_name": "car",
      "confidence": 0.95,
      "width": 260,
      "height": 250
    }
  ]
}
```

---

## 🎨 Frontend Documentation

### Page Architecture

| Page | Route | Components | Data Source |
|------|-------|------------|------------|
| **Dashboard** | `/` | StatCard, ClassDistribution, RecentActivity | `GET /analytics/summary` + `GET /analytics/recent` |
| **Image Detection** | `/detect` | UploadZone, ImagePreview, ResultsTable | `POST /predict/image` |
| **Video Tracking** | `/track` | VideoUpload, TrackingSummary, TracksTable | `POST /predict/video` |
| **Live Camera** | `/live` | WebcamCapture, CanvasOverlay, LiveStats | `WS /api/v1/ws/detect` |
| **Analytics** | `/analytics` | BarChart, PieChart, PerformanceGrid | `GET /analytics/*` |

### Design System

The CSS uses CSS custom properties for a premium dark theme:

| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#0a0e17` | Page background |
| `--bg-secondary` | `#111827` | Card backgrounds |
| `--bg-glass` | `rgba(255,255,255,0.03)` | Glassmorphism effect |
| `--gradient-primary` | `#00d2ff → #7928ca` | Primary accent gradient |
| `--gradient-secondary` | `#7928ca → #f5576c` | Secondary gradient |

### WebSocket Camera Flow

```
Browser getUserMedia → Canvas capture (JPEG 70%) → WebSocket send
                                                         ↓
                                               Backend decode → YOLOv8 detect
                                                         ↓
                                               JSON response (detections)
                                                         ↓
                                          Canvas overlay draw (boxes + labels)
```

---

## 🗄️ Database Design

### Entity Relationship Diagram

```mermaid
erDiagram
    USERS ||--o{ DETECTIONS : creates
    USERS ||--o{ VIDEOS : creates
    USERS {
        int id PK
        string email UK
        string hashed_password
        string full_name
        string role
        bool is_active
        datetime created_at
    }
    DETECTIONS {
        int id PK
        int user_id FK
        string source_type
        string source_filename
        string result_filename
        int total_objects
        int car_count
        int minivan_count
        float avg_confidence
        float inference_time_ms
        json detections_json
        string status
        datetime created_at
    }
    VIDEOS {
        int id PK
        int user_id FK
        string filename
        string result_filename
        int frame_count
        int unique_vehicles
        float avg_fps
        json tracking_summary
        string status
        datetime created_at
    }
    MODELS {
        int id PK
        string name
        string version
        string file_path
        float map50
        float map50_95
        bool is_active
        json class_names
        datetime created_at
    }
    ANALYTICS {
        int id PK
        datetime date
        int total_detections
        int car_count
        int minivan_count
        float avg_confidence
        int total_images_processed
        int total_videos_processed
    }
```

---

## 📈 Monitoring & Observability

### Stack

```
Application → Prometheus (scrape every 10s) → Grafana (dashboards + alerts)
```

### Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total API requests by method/status |
| `http_request_duration_seconds` | Histogram | Request latency distribution |
| `model_inference_seconds` | Histogram | ML inference time |
| `active_websocket_connections` | Gauge | Current live camera sessions |
| `detections_total` | Counter | Total detections by class |

### Health Checks

| Service | Check | Interval |
|---------|-------|----------|
| Backend | `GET /api/v1/health` | 30s |
| Frontend | HTTP 200 on `/` | 30s |
| PostgreSQL | `pg_isready` | 10s |
| Redis | `redis-cli ping` | 10s |

---

## 🔐 Security

| Mechanism | Implementation | Details |
|-----------|---------------|---------|
| **Password Hashing** | `passlib[bcrypt]` | bcrypt with auto-salting |
| **JWT Tokens** | `python-jose` | Access (30min) + Refresh (7 days) |
| **CORS** | FastAPI middleware | Configurable allowed origins |
| **Rate Limiting** | Nginx `limit_req_zone` | 100 req/min per IP |
| **Security Headers** | Nginx | X-Frame-Options, X-XSS-Protection, CSP |
| **File Validation** | MIME type check | Only allowed image/video types |
| **Upload Limit** | 100MB | Configurable via `MAX_UPLOAD_SIZE` |
| **SQL Injection** | SQLAlchemy ORM | Parameterized queries |

---

## 🐳 Docker Setup

### Service Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Nginx (port 80)                │
│              Reverse Proxy + Rate Limiting           │
├────────────────────┬────────────────────────────────┤
│                    │                                │
│  Frontend (React)  │     Backend (FastAPI)           │
│    Nginx:80        │       Uvicorn:8000             │
│                    │            │                    │
│                    │     ┌──────┴──────┐            │
│                    │     │             │            │
│                    │  PostgreSQL    Redis           │
│                    │   :5432        :6379           │
│                    │                                │
├────────────────────┴────────────────────────────────┤
│       MLflow:5000  │  Prometheus:9090  │ Grafana:3001│
└────────────────────┴──────────────────┴─────────────┘
```

### Commands

```bash
# Start all services
cd deployment && docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

---

## ☁️ Deployment Guide

### Render (Free Tier)

1. Push code to GitHub
2. Connect repository to [render.com](https://render.com)
3. Create **Web Service** for backend (Docker, port 8000)
4. Create **Static Site** for frontend (`npm run build` → `dist/`)
5. Add environment variables from `.env.example`

### AWS (ECS/EC2)

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker build -t vehicle-api -f backend/Dockerfile .
docker tag vehicle-api:latest $ECR_URI/vehicle-api:latest
docker push $ECR_URI/vehicle-api:latest
```

### Kubernetes

```bash
kubectl apply -f deployment/k8s/
kubectl get pods -w
```

---

## 🔄 CI/CD Pipeline

### CI Pipeline (`.github/workflows/ci.yml`)

| Step | Tool | Trigger |
|------|------|---------|
| Lint | Ruff | Push to main/develop, PRs |
| Test | Pytest + PostgreSQL service | Push to main/develop, PRs |
| Build | Docker Buildx | After tests pass |

### CD Pipeline (`.github/workflows/cd.yml`)

| Step | Tool | Trigger |
|------|------|---------|
| Build & Push | Docker → GHCR | Push to main |
| Deploy Backend | Render API | After push |
| Deploy Frontend | Render API | After push |

---

## ⚙️ Environment Variables

```env
# App
DEBUG=true

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=your-secret-key-at-least-32-chars-long-change-me
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DATABASE_URL=sqlite:///./vehicle_tracking.db        # Dev
# DATABASE_URL=postgresql://user:pass@host:5432/db   # Prod

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Model
MODEL_PATH=models/vehicle_detector.pt
DETECTION_CONFIDENCE=0.25
DETECTION_IOU=0.45
DETECTION_IMG_SIZE=640
PRELOAD_MODEL=false

# File Storage
UPLOAD_DIR=uploads
RESULTS_DIR=results
MAX_UPLOAD_SIZE=104857600

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
```

---

## 🧪 Testing

### Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov=backend --cov-report=html

# Run specific category
pytest tests/unit/ -v            # Unit tests
pytest tests/api/ -v             # API integration tests
```

### Test Coverage

| Module | Tests | Covers |
|--------|-------|--------|
| `test_data_pipeline.py` | 6 | Validator, splitter, analyzer |
| `test_inference.py` | 5 | Detection, DetectionResult, TrackInfo, TrackingSummary |
| `test_analytics.py` | 7 | VehicleCounter, SpeedEstimator, TrafficHeatmap |
| `test_api.py` | 11 | Health, auth, analytics, upload, predict endpoints |

---

## ⚡ Benchmarking

### Inference Performance (CPU — Intel i5-8250U)

| Input | Resolution | Inference | FPS |
|-------|-----------|-----------|-----|
| Image | 1920×1080 | ~120ms | 8.3 |
| Image | 640×640 | ~45ms | 22.2 |
| Video frame | 1920×1080 | ~150ms | 6.7 |
| Webcam frame | 1280×720 | ~80ms | 12.5 |

### With GPU (NVIDIA RTX 3060)

| Input | Resolution | Inference | FPS |
|-------|-----------|-----------|-----|
| Image | 1920×1080 | ~8ms | 125 |
| Image | 640×640 | ~5ms | 200 |
| Video frame | 1920×1080 | ~10ms | 100 |

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Custom YOLOv8 training pipeline
- [x] ByteTrack multi-object tracking
- [x] FastAPI backend with 15+ endpoints
- [x] React dashboard with 5 pages
- [x] Real-time webcam detection (WebSocket)
- [x] Analytics engine (counting, speed, heatmaps)
- [x] Docker Compose (8 services)
- [x] CI/CD (GitHub Actions)
- [x] Full test suite

### 🔜 Short-term

- [ ] WebSocket tracking (not just detection)
- [ ] License plate recognition (OCR)
- [ ] Multi-camera support
- [ ] Export detection reports (PDF/CSV)
- [ ] Dark/Light theme toggle

### 🔮 Long-term

- [ ] Edge deployment (Jetson Nano / Raspberry Pi)
- [ ] Multi-class expansion (truck, bus, motorcycle)
- [ ] Anomaly detection (wrong-way driving, stopped vehicles)
- [ ] 3D tracking with depth estimation
- [ ] Federated learning across camera nodes

---

## 🔧 Troubleshooting

<details>
<summary><b>CUDA / GPU not detected</b></summary>

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
</details>

<details>
<summary><b>Training crashes with OOM</b></summary>

Edit `configs/training_config.yaml`:
```yaml
batch: 4        # Reduce batch size
imgsz: 416      # Reduce image size
workers: 0      # For Windows
```
</details>

<details>
<summary><b>Frontend: `npm run dev` fails with '&' in path</b></summary>

The `&` character in `Car Tracking & Object Detection Dataset` confuses Windows `cmd.exe`. Use:
```bash
node node_modules\vite\bin\vite.js --port 3000
```
</details>

<details>
<summary><b>Backend: Model not found</b></summary>

Ensure the model exists at the configured path:
```bash
# Check
ls models/vehicle_detector.pt

# Or train a new one
python scripts/run_training.py
```
</details>

<details>
<summary><b>MLflow file store error</b></summary>

Set environment variable before running:
```bash
set MLFLOW_ALLOW_FILE_STORE=true   # Windows
export MLFLOW_ALLOW_FILE_STORE=true # Linux/Mac
```
</details>

<details>
<summary><b>WebSocket connection refused</b></summary>

1. Ensure backend is running on port 8000
2. Check CORS settings include your frontend origin
3. Verify the WebSocket URL matches: `ws://localhost:8000/api/v1/ws/detect`
</details>

<details>
<summary><b>Docker Compose port conflicts</b></summary>

Check which ports are in use:
```bash
netstat -an | findstr "8000 3000 5432 6379"
```
Change conflicting ports in `docker-compose.yml`.
</details>

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** with conventional commits: `git commit -m "feat: add speed estimation"`
4. **Push** to your fork: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Branch Naming

| Prefix | Purpose |
|--------|---------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation |
| `refactor/` | Code refactoring |
| `test/` | Test additions |

### Commit Conventions

```
feat: add vehicle speed estimation
fix: correct ByteTrack ID assignment
docs: update API reference
refactor: simplify detection pipeline
test: add analytics unit tests
```

### Code Quality

```bash
ruff check src/ backend/    # Lint
black src/ backend/          # Format
pytest tests/ -v             # Test
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies.
```

---

## 👨‍💻 Authors

<div align="center">

**Built with ❤️ for the Computer Vision & MLOps community**

</div>

---

<div align="center">

### ⭐ Star this repo if you found it useful!

[![GitHub Stars](https://img.shields.io/github/stars/your-username/vehicle-tracking-platform?style=social)](../../stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/your-username/vehicle-tracking-platform?style=social)](../../forks)

</div>
#   R e a l - T i m e - V e h i c l e - D e t e c t i o n - T r a c k i n g - P l a t f o r m  
 