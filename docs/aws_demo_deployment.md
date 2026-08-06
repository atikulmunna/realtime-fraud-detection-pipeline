# AWS Demo Deployment

How to put the full stack on a single EC2 instance so an evaluator can exercise it live, and how to keep the cost inside a small fixed budget.

## Why a single instance

The stack is twelve long-running containers. Kafka and Zookeeper need roughly 1.5 GB, the Flink JobManager and TaskManager roughly 2.5 GB, and the remaining services roughly 1.5 GB. That is an 8 GB host. On 4 GB the Flink TaskManager is killed under load.

EKS is the wrong shape here. The control plane alone costs about $73 per month before any worker node, which exceeds a $40 project budget on its own, and the repository carries no Kubernetes manifests.

Managed alternatives were considered and rejected for this workload. MSK starts near $100 per month, Kinesis Data Analytics for Flink bills per streaming unit-hour, and RDS plus a managed registry would each add more than the entire budget. A single instance running Compose reproduces the documented topology exactly and costs the least.

## Cost model

Prices below are `us-east-1` on-demand and were accurate at the time of writing. Confirm current rates in the AWS pricing console before launching.

| Item | Rate | Notes |
| --- | --- | --- |
| `t3.large` (2 vCPU, 8 GB) | $0.0832/hour | Billed only while running |
| `gp3` root volume, 30 GB | $0.08/GB-month | Billed while the volume exists, running or stopped |
| Public IPv4 address | $0.005/hour | Charged even when attached |
| Data transfer out | First 100 GB/month free | A demo stays well inside this |

The decisive variable is uptime, not instance size.

| Uptime pattern | Approximate cost |
| --- | --- |
| Two evaluation sessions, 4 hours each | Under $3 |
| One week continuously | About $17 |
| One month continuously | About $67 |

Run the instance only around evaluation windows. A stopped instance costs only its EBS volume, roughly $2.40 per month. Stopping and starting is the entire cost strategy: leaving it running for a month is the only way to breach the budget.

Set a billing alert before launching: AWS Billing, Budgets, then a monthly cost budget at $40 with an alert at 50 percent.

## Running on demand

The cheapest pattern is to keep the instance stopped and start it only when someone asks for a demo. A stopped instance costs its EBS volume and nothing else, because the auto-assigned public IP is released on stop.

| Demo frequency | Monthly cost |
| --- | --- |
| Idle all month | $2.40 |
| One 2-hour demo | $2.58 |
| Four 2-hour demos | $3.11 |
| Two 3-hour demos a week | $4.52 |

Two practical notes.

**Allow 5 to 8 minutes of cold start.** Instance boot is under a minute, but Kafka, Postgres, MLflow, and both Flink processes have to pass healthchecks, the streaming job has to be resubmitted, and the dashboard needs a couple of minutes of traffic before the rate panels are meaningful. Start it before announcing it is ready. Nothing is rebuilt or re-seeded: the Docker layer cache and every named volume live on the EBS volume and survive a stop.

**The public IP changes on every start.** `DEMO_BASE_URL` feeds Grafana's `root_url` and Prometheus's `external-url`, and a stale value breaks Grafana's login redirect. `start-demo.sh` detects the current address from instance metadata and refreshes it automatically, so restarting is still one command:

```bash
scripts/deploy/start-demo.sh
```

Credentials are never rotated by that refresh, so anything already given to an evaluator keeps working. To refresh the address by itself:

```bash
scripts/deploy/make-secrets.sh --base-url-only
```

Do not use `--force` for this. It regenerates every credential, which invalidates the Grafana password and API key an evaluator may already hold.

An Elastic IP removes the address change but bills $0.005/hour whether or not the instance runs, which is about $3.65 per month of otherwise idle time. For an occasional-demo pattern that more than doubles the standing cost and is rarely worth it. If you do attach one, or you front the instance with a domain name, set `SKIP_URL_REFRESH=1` so the start script leaves `DEMO_BASE_URL` alone.

## Launch

### 1. Create the instance

- AMI: Ubuntu Server 24.04 LTS, x86_64. Do not choose an ARM AMI. The Confluent images and the PyFlink wheels are unreliable on `aarch64`.
- Type: `t3.large`.
- Storage: 30 GB `gp3`.
- Key pair: create or select one, and keep the `.pem` file.
- User data: paste the contents of `scripts/deploy/bootstrap-ec2.sh`.

### 2. Security group

This is the control that keeps the stack private. Compose merges port lists and cannot unpublish the base file's host ports, so Postgres on 5432, Kafka on 9092, and the service UIs are all bound on the host. Only the security group stops them being reachable.

Docker writes its own iptables rules that bypass a host firewall such as `ufw`, so do not rely on one. The security group is enforced outside the instance and is what actually holds.

| Type | Port | Source |
| --- | --- | --- |
| SSH | 22 | Your IP only |
| HTTP | 80 | Evaluator's IP, or `0.0.0.0/0` if unknown |

Open nothing else. Everything the evaluator needs is proxied through port 80.

### 3. Upload the model artifacts

`models/` is gitignored, so the clone on the host has none. The Flink job requires all three and will not start without them. They total under 2 MB.

```powershell
scp -i <key.pem> `
  models/isolation_forest_v1.joblib `
  models/autoencoder_v1.joblib `
  models/sgd_classifier_v1.joblib `
  ubuntu@<public-ip>:~/realtime-fraud-detection-pipeline/models/
```

### 4. Generate credentials and start

```bash
ssh -i <key.pem> ubuntu@<public-ip>
cd realtime-fraud-detection-pipeline
scripts/deploy/make-secrets.sh
scripts/deploy/start-demo.sh
```

`make-secrets.sh` prints the Grafana password, the proxy basic-auth password, and the feedback API key exactly once. Record them then. Only the bcrypt hash is stored on disk.

`start-demo.sh` runs a preflight that loads all three artifacts inside the Flink image before starting anything. Read the section below if it fails.

## The numpy ceiling

Model artifacts are joblib pickles written by training and read by both the app image and the Flink image. Those pickles do not survive a numpy major-version boundary. An `MLPRegressor` embeds an MT19937 BitGenerator, and ndarrays reference `numpy._core`, neither of which numpy 1 can read from a numpy 2 pickle. The failure looks like this:

```
isolation_forest_v1  ModuleNotFoundError: No module named 'numpy._core'
autoencoder_v1       ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module
```

It surfaces as a Flink job that dies seconds after submission, which reads like a Kafka problem.

`apache-flink==1.19.1` pins `numpy<1.25`, so that ceiling propagates to the entire project. Every environment is therefore aligned on the same versions:

| Environment | Python | numpy | scikit-learn |
| --- | --- | --- | --- |
| Training and tests | 3.11 | 1.24.4 | 1.7.2 |
| App image | 3.11 | 1.24.4 | 1.7.2 |
| Flink image | 3.10 | 1.24.4 | 1.7.2 |

Python differs between images because the `flink:1.19.1` base is Ubuntu 22.04, whose `python3` is 3.10. That is fine: pickles are portable across Python minor versions when the library versions match. It does constrain scikit-learn, since 1.8 and newer require Python 3.11, which is why the pin sits on the 1.7 line.

Two guards keep this from drifting back:

- `tests/unit/test_dependency_alignment.py` fails if `requirements.txt` and `infra/flink/requirements.txt` disagree on numpy, scikit-learn, scipy, or joblib, or if numpy crosses the PyFlink ceiling.
- `start-demo.sh` loads all three artifacts inside the built Flink image with warnings promoted to errors, before starting anything.

If you retrain, do it in the project environment (`uv run python -m src.models.train_*`). Training anywhere with numpy 2 produces artifacts the streaming job cannot read.

Raising numpy past 1.25 requires upgrading Flink first, which also means a new base image and a matching `flink-sql-connector-kafka` jar.

## What the evaluator sees

Give them the public address and the credentials. The landing page at `/` explains the system and links to each surface.

| Surface | Path | Authentication |
| --- | --- | --- |
| Landing page | `/` | None |
| Grafana dashboard | `/grafana/` | Grafana login |
| Feedback API console | `/api/docs` | `X-API-Key` on write endpoints |
| Prometheus | `/prometheus/graph` | Proxy basic auth |
| MLflow registry | `/mlflow/` | Proxy basic auth |
| Flink REST | `/flink/overview` | Proxy basic auth |

The Flink web console is not proxied. Its bundled UI assumes it is mounted at the root and renders broken behind a stripped path prefix, so only the REST endpoints are exposed. Reach the full console through a tunnel:

```powershell
ssh -i <key.pem> -L 8081:localhost:8081 ubuntu@<public-ip>
```

Then open `http://localhost:8081`.

A traffic generator publishes synthetic transactions to `raw-events` continuously, so the dashboard carries live data without anyone driving it. Tune it with `DEMO_TRAFFIC_RATE` and `DEMO_FRAUD_RATIO` in `infra/.env`, or run a burst by hand:

```bash
docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml \
  run --rm traffic-generator \
  python -m src.demo.traffic_generator \
    --bootstrap-servers kafka:29092 --rate 40 --duration 60 --fraud-ratio 0.2
```

## Shutting down

```bash
scripts/deploy/stop-demo.sh          # stop containers, keep volumes
```

Then stop the instance from the EC2 console or with `aws ec2 stop-instances --instance-ids <id>`. **Stopping the containers does not stop the billing. Only stopping the instance does.**

Restarting is `scripts/deploy/start-demo.sh` on its own. It refreshes the public address before bringing the stack up.

`scripts/deploy/stop-demo.sh --destroy` also removes the volumes. That discards Kafka offsets, feedback rows, MLflow versions, and Flink checkpoints, and cannot be undone.

## Security posture

This is a time-boxed demo configuration, not an internet-grade deployment. Stated plainly:

- Traffic is plain HTTP. Credentials cross the network in the clear. Do not reuse any password used elsewhere.
- Basic auth in front of Prometheus, MLflow, and the Flink REST path is a gate, not an identity system.
- There is no rate limiting, no audit retention, and no secret rotation.
- All data is synthetic. No real transaction data is present, and none should be added.

`docs/contracts_and_guarantees.md` describes what a genuine production posture requires. Take the instance down when the evaluation ends.

### HTTPS with a domain

Let's Encrypt will not issue for `*.compute.amazonaws.com`, so a bare EC2 hostname cannot get a certificate. With any domain you control, including a free dynamic DNS name, point an A record at the instance and replace the `:80` site header in `infra/caddy/Caddyfile` with the hostname:

```
demo.example.com {
```

Remove the `auto_https off` line from the global block, open 443 in the security group alongside 80, update `DEMO_BASE_URL` in `infra/.env` to the `https://` form, and restart. Caddy provisions and renews the certificate automatically.
