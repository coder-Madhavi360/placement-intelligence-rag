"""
feedback/storage.py
Stores user feedback in a JSON file.
"""

import json
from pathlib import Path

FEEDBACK_FILE = Path("feedback_data.json")


def save_feedback(data: dict):
    """
    Append a feedback record to feedback_data.json
    """

    records = []

    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    records.append(data)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_feedback():
    """
    Load all feedback records.
    """

    if not FEEDBACK_FILE.exists():
        return []

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []