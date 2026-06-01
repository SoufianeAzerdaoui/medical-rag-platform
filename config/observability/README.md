# Observability Stack Assets

These files are the runtime-sidecar configs for the Grafana stack:

- `alloy.river`: ships backend log files to Loki.
- `prometheus.yml`: scrapes the backend `/metrics` endpoint.
- `loki.yml`: local single-node Loki config for self-hosted setups.

The backend writes structured application logs to `LOGS_DIR/backend.log` by default.
Use Alloy rather than Promtail for new deployments, since Promtail is in long-term support and past its end-of-life window as of 2026.
For a container deployment, mount the log volume read-only into Alloy at `/var/log/medical-rag`.

## Compose

The root `docker-compose.yml` exposes these assets behind the `observability` profile.

Start the full stack with:

```bash
docker compose --profile observability up --build
```

That brings up Grafana on `127.0.0.1:3001`, Prometheus on `127.0.0.1:9090`, Loki on `127.0.0.1:3100`, and Alloy on `127.0.0.1:12345`.
