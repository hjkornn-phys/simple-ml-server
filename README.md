# ML Server (FastAPI + LightGBM)

Simple ML inference API server built with FastAPI, served by Uvicorn, and backed by a LightGBM model. On first run, it trains a tiny demo model and saves it to models/model.txt.

## Requirements
- Python 3.13+
- uv (modern Python package/venv manager)

Install uv if needed:

```bash
pip install uv
```

## Setup
1) Create a virtual environment (optional, recommended)

```bash
uv venv -p 3.13 .venv
source .venv/bin/activate
```

2) Sync dependencies from pyproject.toml (modern way)

```bash
uv sync
```

- uv sync reads pyproject.toml (and uv.lock if present) and installs the exact set of dependencies for this project.
- Note: uv pip sync is for syncing against requirements files; when using a pyproject-based project, prefer uv sync.

## Running
Use the provided script (auto-detects .venv):

```bash
./run.sh
```

Or run Uvicorn directly:

```bash
uv run uvicorn ml_server.app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment variables:
- MODEL_PATH: path to load/save the LightGBM model (default: models/model.txt)
- HOST, PORT: server bind host/port (defaults: 0.0.0.0 and 8000 in run.sh)

## Containerization (Docker & Compose)
This repo includes a multi-stage Dockerfile and docker-compose.yaml for a slim, secure image.

- Build the image (first ensure Docker Desktop/Engine is installed and running):

```bash
docker compose build
```

- Prepare host directories for bind mounts (so external tools can write training data and you can persist models/logs):

```bash
mkdir -p training_data_directory models logs
```

- Start the service:

```bash
docker compose up -d
```

- Check logs and health:

```bash
docker compose logs -f
curl http://localhost:8000/health
```

- Stop the service:

```bash
docker compose down
```

Volumes (host:container):
- ./training_data_directory:/app/data
- ./models:/app/models
- ./logs:/app/logs

Windows note:
- With Docker Desktop on Windows, these relative paths bind-mount from your project directory. Ensure the folders exist before starting. External Windows programs can write CSVs into training_data_directory and the app will see them at /app/data.

Environment (container defaults):
- HOST=0.0.0.0, PORT=8000
- MODEL_PATH=/app/models/model.txt
- LOG_LEVEL=INFO (set to DEBUG to increase verbosity)
- Optionally set TRAIN_DATA_PATH (e.g., /app/data/train.csv) to control the training input path used by the /train endpoint.

Security & runtime details:
- Runs as a non-root user inside the container.
- Uses tini as PID 1 for proper signal handling.
- Git is available inside the image.

## Logging
The app writes logs to both console and files:
- logs/app.log: rotates daily at local midnight; keeps 365 backups.
- logs/error.log: non-rotating; captures ERROR and above from all loggers.

To adjust verbosity, set LOG_LEVEL (e.g., DEBUG, INFO, WARNING).

## API Usage
Base URL: http://localhost:8000

- Health check

```bash
curl -X GET http://localhost:8000/health
```
Expected response:

```json
{"status":"ok"}
```

- Predict
The demo model is trained on 6 features. Send a 2D array of floats under features.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "features": [
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
          [1, 2, 3, 4, 5, 6]
        ]
      }'
```
Example response:

```json
{"predictions":[0.42,0.97]}
```

## Project Structure
- pyproject.toml: project metadata and dependencies
- run.sh: helper script to start the server
- ml_server/app/main.py: FastAPI application (routes)
- ml_server/app/model_service.py: LightGBM model loading/training/prediction
- models/: directory where the model file (model.txt) is saved

## Notes
- On first startup, if models/model.txt does not exist, a small synthetic model is trained and saved automatically.
- To use a custom trained model, set MODEL_PATH to point to your model file before starting the server.
