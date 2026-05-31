from __future__ import annotations

import json
import sys

from backend.services import p0_readiness_service


def main() -> int:
    report = p0_readiness_service.run_p0_readiness_check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    status = str(report.get("overall_status") or "fail").lower()
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
