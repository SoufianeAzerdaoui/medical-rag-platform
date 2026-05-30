from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from backend import config


def _scan_path(path: Path) -> tuple[bool, str]:
    cmd = [config.ANTIVIRUS_CLAMSCAN_CMD, "--no-summary", str(path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(config.ANTIVIRUS_TIMEOUT_SECONDS)),
        )
    except FileNotFoundError:
        if config.ANTIVIRUS_REQUIRED:
            raise RuntimeError("ClamAV non installé (clamscan introuvable).")
        return True, "antivirus_unavailable"
    except subprocess.TimeoutExpired:
        if config.ANTIVIRUS_REQUIRED:
            raise RuntimeError("Antivirus timeout.")
        return True, "antivirus_timeout"
    except Exception as exc:
        if config.ANTIVIRUS_REQUIRED:
            raise RuntimeError(f"Erreur antivirus: {exc}") from exc
        return True, "antivirus_error"

    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode == 0:
        return True, output
    if proc.returncode == 1:
        return False, output
    if config.ANTIVIRUS_REQUIRED:
        raise RuntimeError(f"ClamAV indisponible (code={proc.returncode}).")
    return True, output


def scan_file_or_raise(path: Path) -> None:
    ok, detail = _scan_path(path)
    if ok:
        return
    raise RuntimeError(f"Fichier rejeté par antivirus: {path.name}. {detail}".strip())


def scan_bytes_or_raise(raw: bytes, filename: str = "upload.pdf") -> None:
    suffix = Path(str(filename or "upload.pdf")).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        scan_file_or_raise(Path(tmp.name))

