# medical-rag-platform
Development of an intelligent system for extracting, structuring, and querying medical reports using RAG architectures.

## Local stack

Run the application stack:

```bash
docker compose up --build
```

Run the full observability stack, including Grafana, Loki, Prometheus, and Alloy:

```bash
docker compose --profile observability up --build
```

Useful local endpoints:

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- Loki: http://localhost:3100
- Alloy: http://localhost:12345

Grafana admin credentials can be overridden with `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in `.env`.
