"""Activate the generic PDK for tests in gsim.common that build components.

Activation happens before every test (autouse fixture) rather than only at
collection time: Palace tests (e.g. the IHP mesh-regression suite) switch the
active PDK during execution and never restore it, so module-level activation
alone lets the IHP PDK leak into later ``gsim.common`` tests under the random
test ordering used by ``pytest-randomly``.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest


@pytest.fixture(autouse=True)
def _generic_pdk_active() -> None:
    """Ensure the generic PDK is active for each test."""
    gf.gpdk.PDK.activate()
