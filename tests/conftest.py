"""Make the repo root importable as `src` and provide a small CPU config fixture."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vistacfusion.utils.config import merge_configs  # noqa: E402


@pytest.fixture
def cfg():
    """Merged model+train+data config, shrunk for fast CPU tests (mock encoder)."""
    c = merge_configs(
        os.path.join(ROOT, "configs", "model.yaml"),
        os.path.join(ROOT, "configs", "train.yaml"),
        os.path.join(ROOT, "configs", "data.yaml"),
    )
    c.encoder.checkpoint = None          # -> MockEncoder, no DINOv3 weights needed
    c.dataset = "synthetic"
    return c
