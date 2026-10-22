# End-to-End MLOps Hands-on Project: CI, CT, and CD

A small, runnable customer-churn classification project for learning:
- **CI (Continuous Integration):** lint, tests, dependency audit, Docker build.
- **CT (Continuous Training):** reproduce the DVC pipeline and evaluate a candidate model when training code/data changes or manually.
- **Data and model versioning:** DVC versions pipeline outputs, and every MLflow training run gets an immutable, run-ID-based model snapshot that can be selected at API startup.
- **Monitoring:** Prometheus scrapes API request, latency, and prediction metrics; Grafana provisions a dashboard automatically.
- **CD (Continuous Delivery/Deployment):** build a versioned inference image, deploy to staging, smoke-test, and optionally promote to production.

This starter runs locally with Python + Docker Compose + GitHub Actions. It includes a single-EC2 Docker Compose deployment path, but does not create AWS resources or deploy to a cloud account automatically.

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
- Docker and the Docker Compose plugin
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
- `artifacts/model_version.json` — pointer to the latest immutable model version
- `artifacts/versions/<mlflow-run-id>/` — model, feature schema, and metrics snapshot for that run

The evaluation command fails if the configured F1 threshold is not met.

## 2. Version data and track experiments

DVC stores data and pipeline outputs in its local cache by default. After `dvc repro`, commit `dvc.yaml`, `dvc.lock` (created by DVC), and `params.yaml` to record the pipeline definition, data version, and parameters. To share cached data between machines, configure a DVC remote and push the cache:

```bash
dvc remote add -d storage <remote-storage-url>
dvc push
```

Keep remote credentials out of Git; use your storage provider's credential mechanism. `dvc pull` restores data and outputs from the configured remote, while `dvc repro` regenerates stale stages.

Each training run is logged to the MLflow experiment `mlops-demo` in a local SQLite database (`mlflow.db`) with model artifacts in `mlartifacts/` by default. The MLflow run ID is also used as the immutable model version, and its snapshot is included in the DVC pipeline output. Use the DVC remote configured above to share model versions with EC2. On another machine, use `dvc pull` to restore the pipeline outputs. To see the current version ID:

```bash
python -c "import json; print(json.load(open('artifacts/model_version.json'))['model_version'])"
```

To serve a particular version, set `MODEL_VERSION` to that exact run ID; `latest` selects the version in `model_version.json`. Keep DVC remote credentials out of Git and grant the EC2 instance access using an IAM role where possible.

To view MLflow runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
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

If you have runs in the older `mlruns/` file store, back it up and migrate it before logging new runs to SQLite:

```bash
mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db
```

## 3. Run the inference API

After training:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

The prediction API loads the latest versioned snapshot at startup. Set `MODEL_VERSION` to an exact MLflow run ID to pin or roll back the serving model. It expects the numeric feature names used by scikit-learn's breast-cancer dataset. This is a teaching example; replace the schema with your business model's feature contract. The `/metrics` endpoint exposes Prometheus metrics and `/health` reports the loaded model version.

## 4. Run the API and monitoring locally

Train first and prepare the local environment file:

```bash
cp .env.example .env
# Set GRAFANA_ADMIN_PASSWORD in .env to a unique password.
docker compose up --build -d
```

The API is available at `http://localhost:8000`; Grafana is at `http://localhost:3000` (user `admin`), and Prometheus is at `http://localhost:9090`. Grafana is preconfigured with the Prometheus data source and the MLOps API dashboard. Verify API metrics at `http://localhost:8000/metrics`.

Set `MODEL_VERSION` in `.env` to a version ID to pin or roll back the model, then recreate the API:

```bash
docker compose up -d --build api
```

To stop the stack, run `docker compose down`. Prometheus and Grafana data persist in Docker volumes. The Docker image includes the versioned model snapshots available at build time. Never load untrusted pickle/joblib artifacts.

## 5. Deploy on AWS EC2

This is a single-host learning deployment, not a highly available production architecture.

1. Launch an Ubuntu EC2 instance with enough memory for Docker and the model. Install Docker Engine, the Docker Compose plugin, and `curl`; configure the EC2 user to run Docker without `sudo`.
2. In the EC2 security group, allow SSH only from your trusted self-hosted GitHub runner or other restricted deployment source. Do not open SSH to `0.0.0.0/0`. Allow TCP `8000` only from trusted API clients. Do **not** expose Prometheus `9090` or Grafana `3000` publicly. Add a TLS reverse proxy and domain for production use.
3. Prepare the persistent deployment directory and environment file on EC2:

   ```bash
   mkdir -p "$HOME/mlops-demo"
   cp .env.example "$HOME/mlops-demo/.env"
   chmod 600 "$HOME/mlops-demo/.env"
   # Edit $HOME/mlops-demo/.env and set a unique GRAFANA_ADMIN_PASSWORD.
   ```

   Copy `.env.example` from this repository if needed. The deployment workflow preserves this file and updates only its `IMAGE_TAG`.
4. In GitHub, create environments named `staging` and `production` under **Settings → Environments**. Add these secrets to each environment:

   - `EC2_HOST` — EC2 public DNS name or reachable IP.
   - `EC2_USER` — SSH user, for example `ubuntu`.
   - `EC2_SSH_PRIVATE_KEY` — private key authorized for that EC2 user.
   - `EC2_SSH_KNOWN_HOSTS` — verified SSH host-key line(s) for the instance.

   Obtain and verify the EC2 host key through a trusted channel before saving it; the workflow uses strict host-key checking. Protect `production` with required reviewers in its environment settings to require manual approval before deployment. A self-hosted runner inside your VPC avoids exposing SSH to GitHub-hosted runner address ranges.
5. In **Actions → CD - Deploy to EC2 → Run workflow**, select `staging` or `production`. The workflow trains and quality-gates a model, builds and smoke-tests the image, then pauses for any configured environment approval. After approval it transfers the image and monitoring configuration over SSH, starts the Compose stack, and checks that the API reports a loaded model. If a deployment health check fails and a prior image is present, it attempts to restore the prior image.

The API is reachable on port `8000`. To access Grafana securely from your workstation, create an SSH tunnel:

```bash
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 ubuntu@<EC2-public-IP>
```

Then open `http://localhost:3000` for Grafana or `http://localhost:9090` for Prometheus.

The workflow deploys the image built from the selected commit, including that run's versioned model. To roll back, run the workflow using the previous release commit; the workflow updates `IMAGE_TAG` and redeploys that image. Monitor disk usage and back up the EC2 volumes; this setup does not include automated cloud provisioning, HTTPS termination, alert routing, or multi-instance scaling.

## 6. GitHub Actions setup

Push this repository to GitHub. Workflows are in `.github/workflows/`:
- `ci.yml`: PR/push CI checks.
- `ct.yml`: manual or training-source-triggered training and evaluation.
- `cd.yml`: manual quality-gated build, protected-environment approval, and SSH deployment to the selected EC2 target.

The CD workflow deploys through SSH and uses no AWS access keys. Protect production with required GitHub Environment reviewers, pin third-party actions to reviewed full commit SHAs, scan images, and test rollback procedures.

## 7. What to learn / exercise

1. Break a unit test and verify CI fails.
2. Change the `MIN_F1` repository variable (or edit `src/evaluate.py`) to a threshold that the model cannot meet and verify CT fails.
3. Change a training parameter and run CT.
4. Run the API and send an invalid request; verify input validation rejects it.
5. Build the Docker Compose stack and inspect API metrics in Grafana.
6. Deploy the Compose stack to a small EC2 instance.
7. Configure a shared DVC remote and compare dataset versions.
8. Add data drift monitoring and a retraining trigger.
9. Add image scanning (Trivy), SBOM generation, and secret scanning.
10. Add a protected production environment and a rollback procedure.

## Security and production notes

- GitHub Actions workflows use minimal `contents: read` permissions by default.
- Do not commit credentials, production data, or model artifacts.
- Pin actions to reviewed commit SHAs for production.
- Use OIDC instead of long-lived cloud keys.
- CI currently ignores `PYSEC-2026-2447` because DVC depends on diskcache 5.6.3 and no fixed release is available; remove the exception once DVC adopts a patched version.
- Add an explicit model/data license and privacy review before using real datasets.
- Test the full pipeline with representative data before relying on its quality gate.
- The built-in demo dataset and default F1 threshold are pedagogical examples, not production acceptance criteria.
# mlops-endtoend-pipeline
mlops end to end pipeline
