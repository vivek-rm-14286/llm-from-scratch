"""pytest tests --impl solution|starter   (default: starter, so plain pytest is the to-do list)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_addoption(parser):
    parser.addoption("--impl", default="starter", choices=["solution", "starter"],
                     help="which implementation of the learner-written modules to test")


@pytest.fixture(scope="session")
def impl(request):
    from oncall.impl import load_impl
    return load_impl(request.config.getoption("--impl"))


@pytest.fixture
def fleet():
    from oncall.simulator import Fleet
    return Fleet(seed=0, faults={"query-api": "memory_leak"})


@pytest.fixture
def toolbox(fleet):
    from oncall.tools import build_toolbox
    return build_toolbox(fleet, "internal")


@pytest.fixture
def alert_task():
    from oncall.scenarios import BY_ID, make_task
    return make_task(BY_ID["N01"])
