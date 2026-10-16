import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)
