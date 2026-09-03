"""
audit_log.py

Every time the agent makes a routing decision or calls a model,
it logs a structured entry here. This is the "not a black box"
evidence judges specifically asked for.

Kept dead simple on purpose: a JSON-lines file. Good enough for a
hackathon demo; swap for a real DB later if needed.
"""

import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "audit_trail.jsonl")


def log_step(step_name: str, details: dict):
    """
    step_name: e.g. "router_decision", "specialist_model_call", "gemini_call"
    details: any structured info worth showing a judge
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step_name,
        "details": details,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_trail(limit: int = 50):
    """Return the most recent audit entries (for displaying in the UI)."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(line) for line in lines]
