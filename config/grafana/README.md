# Grafana Provisioning Assets

This directory is structured for production-style Grafana provisioning.

## Files

- `medical_rag_llm_observability_dashboard.json`: dashboard JSON file.
- `provisioning/datasources/medical-rag-datasources.yml`: Loki and Prometheus datasources.
- `provisioning/dashboards/medical-rag-dashboards.yml`: dashboard file provider.
- `provisioning/alerting/medical_rag_llm_alerts.yml`: Grafana-managed alert rules.

## Mounting

Mount the directory tree in two places:

- `config/grafana/provisioning` -> `/etc/grafana/provisioning`
- `config/grafana` -> `/var/lib/grafana/dashboards`

The dashboard provider reads JSON dashboards from the dashboard folder while Grafana reads provisioning YAML from `/etc/grafana/provisioning`.
The root `docker-compose.yml` wires these mounts automatically behind the `observability` profile.

## Logging

The backend writes structured application logs to `LOGS_DIR/backend.log`. Use the Alloy config in `config/observability/alloy.river` to tail that file and ship the logs to Loki.
