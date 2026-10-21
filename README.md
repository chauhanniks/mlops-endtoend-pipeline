# End-to-End MLOps Hands-on Project: CI, CT, and CD

A small, runnable customer-churn classification project for learning:
- **CI (Continuous Integration):** lint, tests, dependency audit, Docker build.
- **CT (Continuous Training):** reproduce the DVC pipeline and evaluate a candidate model when training code/data changes or manually.
- **Data and experiment tracking:** DVC versions the prepared dataset and pipeline outputs; MLflow records training parameters, metrics, and model artifacts.
- **CD (Continuous Delivery/Deployment):** build a versioned inference image, deploy to staging, smoke-test, and optionally promote to production.

This starter runs locally with Python + Docker + GitHub Actions. It deliberately does **not** deploy to a real cloud account by default, so you can learn safely without cloud costs. The workflows build artifacts and explain where to add a real deployment target.

## Architecture

```text
Pull request / push
        |
        v
       CI: lint + tests + dependency audit + Docker build
        |
        +-----------------------------+
                                      |
Training code/data change or manual dispatch
        |
        v
CT: train -> evaluate -> quality gate -> candidate model artifact
        |
        v
CD: build inference image -> staging deployment -> smoke test
        |
        v
Optional production promotion (manual approval/environment)
```

CI, CT, and CD are separate workflows because an application-code change should not always trigger an expensive model retraining, and training a candidate should not automatically make it production.

## Requirements

- Python 3.11+
- Git
- Docker (optional for local container steps)
- GitHub account for Actions workflow practice

## 1. Run locally

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

ruff check .
pytest -q
dvc repro
python -m src.evaluate --metrics artifacts/metrics.json
```

The DVC `prepare-data` stage exports scikit-learn's small public breast-cancer classification dataset to `data/breast_cancer.csv`; the `train` stage consumes this file. The generated CSV is DVC-managed rather than committed directly to Git. It is for demonstration only, not a medical product.

A successful training run writes:
- `artifacts/model.joblib` — serialized candidate model
- `artifacts/metrics.json` — evaluation metrics and model metadata

The evaluation command fails if the configured F1 threshold is not met.

## 2. Version data and track experiments

DVC stores data and pipeline outputs in its local cache by default. After `dvc repro`, commit `dvc.yaml`, `dvc.lock` (created by DVC), and `params.yaml` to record the pipeline definition, data version, and parameters. To share cached data between machines, configure a DVC remote and push the cache:

```bash
dvc remote add -d storage <remote-storage-url>
dvc push
```

Keep remote credentials out of Git; use your storage provider's credential mechanism. `dvc pull` restores data and outputs from the configured remote, while `dvc repro` regenerates stale stages.

Each training run is logged to the MLflow experiment `mlops-demo` in the local `mlruns/` directory by default. To view runs:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Port 5000 must be free before starting MLflow. An administrator can identify the listener with:

```bash
sudo ss -ltnp 'sport = :5000'
```

If that confirms Datadog owns the port and disabling its monitoring is intended, stop the agent:

```bash
sudo systemctl stop datadog-agent
```

Otherwise, reconfigure the service that owns the port. This project does not manage system services. Set `MLFLOW_TRACKING_URI` to use a remote tracking server and `MLFLOW_EXPERIMENT_NAME` to choose another experiment. Training logs the model parameters, dataset SHA-256, evaluation metrics, serialized model, and feature/metric files.

## 3. Run the inference API

After training:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

The prediction API loads `artifacts/model.joblib` at startup. It expects the numeric feature names used by scikit-learn's breast-cancer dataset. This is a teaching example; replace the schema with your business model's feature contract.

## 4. Run with Docker

Train first, then build the image:

```bash
docker build -t mlops-demo:local .
docker run --rm -p 8000:8000 mlops-demo:local
```

The Docker image includes the trained artifact. For production, prefer a separately versioned model artifact or model registry and a controlled model-loading process. Never load untrusted pickle/joblib artifacts.

## 5. GitHub Actions setup

Push this repository to GitHub. Workflows are in `.github/workflows/`:
- `ci.yml`: PR/push CI checks.
- `ct.yml`: manual or training-source-triggered training and evaluation.
- `cd.yml`: manual deployment workflow demonstrating staging/production gates and Docker image publication as a workflow artifact.

The CD workflow is a **safe template**, not a cloud deployment. Add your target-specific deploy steps after setting up a staging environment. For production, configure a protected GitHub Environment with required reviewers, use OIDC/short-lived credentials for cloud deployments, pin third-party actions to reviewed full commit SHAs, scan images, and implement rollback.

## 6. What to learn / exercise

1. Break a unit test and verify CI fails.
2. Change the `MIN_F1` repository variable (or edit `src/evaluate.py`) to a threshold that the model cannot meet and verify CT fails.
3. Change a training parameter and run CT.
4. Run the API and send an invalid request; verify input validation rejects it.
5. Build the Docker image and test `/health`.
6. Add a staging deployment step for ECS, Kubernetes, or SageMaker.
7. Configure a shared DVC remote and compare dataset versions.
8. Add data drift monitoring and a retraining trigger.
9. Add image scanning (Trivy), SBOM generation, and secret scanning.
10. Add a protected production environment and a rollback procedure.

## Security and production notes

- GitHub Actions workflows use minimal `contents: read` permissions by default.
- Do not commit credentials, production data, or model artifacts.
- Pin actions to reviewed commit SHAs for production.
- Use OIDC instead of long-lived cloud keys.
- Add an explicit model/data license and privacy review before using real datasets.
- Test the full pipeline with representative data before relying on its quality gate.
- The built-in demo dataset and default F1 threshold are pedagogical examples, not production acceptance criteria.
# mlops-endtoend-pipeline
mlops end to end pipeline
