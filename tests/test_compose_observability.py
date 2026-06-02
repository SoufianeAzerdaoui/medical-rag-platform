from __future__ import annotations

import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


class TestComposeObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        cls.services = dict(cls.compose.get("services") or {})

    def test_ollama_service_is_present(self) -> None:
        self.assertIn("ollama", self.services)
        ollama = self.services["ollama"]
        self.assertEqual(ollama.get("image"), "${OLLAMA_IMAGE:-ollama/ollama}")
        self.assertIn("127.0.0.1:11434:11434", list(ollama.get("ports") or []))
        self.assertIn("medical-rag-ollama-data:/root/.ollama", list(ollama.get("volumes") or []))

    def test_observability_services_are_profiled(self) -> None:
        expected = {"grafana", "loki", "prometheus", "alloy"}
        self.assertTrue(expected.issubset(self.services))

        for service_name in expected:
            profiles = list(self.services[service_name].get("profiles") or [])
            self.assertIn("observability", profiles)

    def test_observability_images_are_pinned(self) -> None:
        self.assertEqual(self.services["grafana"].get("image"), "grafana/grafana:13.0.1-security-01")
        self.assertEqual(self.services["loki"].get("image"), "grafana/loki:3.7.2")
        self.assertEqual(self.services["prometheus"].get("image"), "prom/prometheus:v3.12.0")
        self.assertEqual(self.services["alloy"].get("image"), "grafana/alloy:v1.16.1")

    def test_grafana_is_wired_to_dashboard_assets(self) -> None:
        grafana = self.services["grafana"]
        volumes = list(grafana.get("volumes") or [])
        self.assertIn("./config/grafana:/var/lib/grafana/dashboards:ro", volumes)
        self.assertIn("./config/grafana/provisioning:/etc/grafana/provisioning:ro", volumes)
        self.assertIn("medical-rag-grafana-data:/var/lib/grafana", volumes)

        environment = dict(grafana.get("environment") or {})
        self.assertEqual(environment.get("GF_USERS_ALLOW_SIGN_UP"), "false")
        self.assertEqual(environment.get("GF_AUTH_ANONYMOUS_ENABLED"), "false")
        self.assertEqual(environment.get("GF_SECURITY_ADMIN_USER"), "${GRAFANA_ADMIN_USER:-admin}")

    def test_backend_logs_volume_is_shared_with_alloy(self) -> None:
        backend = self.services["backend"]
        backend_healthcheck = dict(backend.get("healthcheck") or {})
        self.assertEqual(
            backend_healthcheck.get("test"),
            ["CMD", "curl", "-fsS", "http://localhost:8000/health"],
        )

        backend_volumes = list(self.services["backend"].get("volumes") or [])
        alloy_volumes = list(self.services["alloy"].get("volumes") or [])
        backend_environment = dict(backend.get("environment") or {})
        frontend_depends_on = dict(self.services["frontend"].get("depends_on") or {})

        self.assertIn("medical-rag-logs:/app/logs", backend_volumes)
        self.assertIn("medical-rag-logs:/var/log/medical-rag:ro", alloy_volumes)
        self.assertEqual(backend_environment.get("MEDICAL_RAG_OLLAMA_URL"), "${MEDICAL_RAG_OLLAMA_URL:-http://ollama:11434}")
        self.assertEqual(backend_environment.get("MEDICAL_RAG_OLLAMA_MODEL"), "${MEDICAL_RAG_OLLAMA_MODEL:-llama3.2:latest}")
        self.assertEqual(backend_environment.get("MEDICAL_RAG_GEMINI_MODEL"), "${MEDICAL_RAG_GEMINI_MODEL:-gemini-2.5-flash}")
        self.assertEqual(dict(backend.get("depends_on") or {}).get("ollama", {}).get("condition"), "service_healthy")
        self.assertEqual(frontend_depends_on.get("backend", {}).get("condition"), "service_healthy")

    def test_observability_ports_are_bound_locally(self) -> None:
        grafana_ports = list(self.services["grafana"].get("ports") or [])
        loki_ports = list(self.services["loki"].get("ports") or [])
        prometheus_ports = list(self.services["prometheus"].get("ports") or [])
        alloy_ports = list(self.services["alloy"].get("ports") or [])

        self.assertIn("127.0.0.1:3001:3000", grafana_ports)
        self.assertIn("127.0.0.1:3100:3100", loki_ports)
        self.assertIn("127.0.0.1:9090:9090", prometheus_ports)
        self.assertIn("127.0.0.1:12345:12345", alloy_ports)


if __name__ == "__main__":
    unittest.main()
