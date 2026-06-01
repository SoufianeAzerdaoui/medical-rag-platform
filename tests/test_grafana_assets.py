from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAFANA_DIR = PROJECT_ROOT / "config" / "grafana"
OBSERVABILITY_DIR = PROJECT_ROOT / "config" / "observability"


class TestGrafanaAssets(unittest.TestCase):
    def test_dashboard_is_provisionable_and_datasource_aware(self) -> None:
        dashboard = json.loads((GRAFANA_DIR / "medical_rag_llm_observability_dashboard.json").read_text(encoding="utf-8"))
        self.assertEqual(dashboard.get("uid"), "medical-rag-llm-observability")
        variables = [entry.get("name") for entry in dashboard.get("templating", {}).get("list", [])]
        self.assertIn("app_label", variables)
        self.assertIn("prom_job", variables)

        panels = list(dashboard.get("panels") or [])
        self.assertGreaterEqual(len(panels), 17)

        datasource_types = {str(panel.get("datasource", {}).get("type") or "") for panel in panels}
        self.assertIn("loki", datasource_types)
        self.assertIn("prometheus", datasource_types)

        panel_titles = {str(panel.get("title") or "") for panel in panels}
        self.assertIn("LLM Attempt Rate", panel_titles)
        self.assertIn("Backend Metrics Up", panel_titles)
        self.assertIn("Ingestion Pipeline Success vs Failure", panel_titles)

    def test_datasource_provisioning_contains_loki_and_prometheus(self) -> None:
        payload = yaml.safe_load(
            (GRAFANA_DIR / "provisioning" / "datasources" / "medical-rag-datasources.yml").read_text(encoding="utf-8")
        )
        datasources = list(payload.get("datasources") or [])
        names = {str(item.get("name") or "") for item in datasources}
        uids = {str(item.get("uid") or "") for item in datasources}
        self.assertIn("Loki", names)
        self.assertIn("Prometheus", names)
        self.assertIn("loki", uids)
        self.assertIn("prometheus", uids)

    def test_alerting_provisioning_covers_core_signals(self) -> None:
        payload = yaml.safe_load(
            (GRAFANA_DIR / "provisioning" / "alerting" / "medical_rag_llm_alerts.yml").read_text(encoding="utf-8")
        )
        groups = list(payload.get("groups") or [])
        self.assertGreaterEqual(len(groups), 2)

        rules: list[dict[str, object]] = []
        for group in groups:
            rules.extend(list(group.get("rules") or []))

        titles = {str(rule.get("title") or "") for rule in rules}
        expected = {
            "Medical RAG Contract Violation Detected",
            "Medical RAG Hard Gate Errors Detected",
            "Medical RAG Safety Signals Detected",
            "Medical RAG Fallback After LLM Elevated",
            "Medical RAG LLM Accept Rate Dropped",
            "Medical RAG Response Time Elevated",
            "Medical RAG Backend Metrics Scrape Down",
            "Medical RAG Ingestion Queue Depth High",
        }
        self.assertTrue(expected.issubset(titles))

    def test_observability_configs_parse(self) -> None:
        prometheus = yaml.safe_load((OBSERVABILITY_DIR / "prometheus.yml").read_text(encoding="utf-8"))
        loki = yaml.safe_load((OBSERVABILITY_DIR / "loki.yml").read_text(encoding="utf-8"))
        self.assertIn("scrape_configs", prometheus)
        self.assertIn("schema_config", loki)
        alloy_text = (OBSERVABILITY_DIR / "alloy.river").read_text(encoding="utf-8")
        self.assertIn("loki.source.file", alloy_text)
        self.assertIn("loki.write", alloy_text)
        self.assertIn("backend.log", alloy_text)


if __name__ == "__main__":
    unittest.main()
