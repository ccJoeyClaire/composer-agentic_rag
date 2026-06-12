from __future__ import annotations

import os
import sys
from pathlib import Path

from eval.paths import REPO_ROOT


def setup_eval_env() -> None:
    """Load .env and apply eval-safe defaults (DashScope batch size, etc.)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    base = (
        os.environ.get("EMBEDDING_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or ""
    ).lower()
    if "dashscope" in base or "aliyuncs.com" in base:
        os.environ["EMBEDDING_BATCH_SIZE"] = "10"
    else:
        os.environ.setdefault("EMBEDDING_BATCH_SIZE", "10")
