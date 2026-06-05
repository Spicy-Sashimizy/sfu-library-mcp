"""Reader for the model-versions registry (Phase N GUI tracking).

Backs the design's Admin "Model Versions" table (activate / rollback / changelog).
The registry itself is a hand-maintained JSON file (models/model_registry.json); this
module just loads and flattens it for the analytics endpoint. It also reflects the
live presence of the trained LambdaMART weights so the table never claims a model is
deployable when its file is missing.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "models" / "model_registry.json"


def _load() -> dict:
    try:
        return json.loads(_REGISTRY_PATH.read_text())
    except Exception as e:
        logger.debug("Model registry unavailable (%s)", e)
        return {"components": {}}


def list_versions() -> list[dict]:
    """Flatten the registry to one row per (component, version) for the table."""
    data = _load()
    rows: list[dict] = []
    for component, versions in (data.get("components") or {}).items():
        for v in versions:
            row = {"component": component, **v}
            # Reflect on-disk reality: a model file that isn't present can't be active.
            path = v.get("path")
            if path:
                row["available"] = (_REPO_ROOT / path).exists()
            rows.append(row)
    return rows


def registry() -> dict:
    """Return the full registry dict (with `updated` timestamp)."""
    return _load()
