"""pytest configuration: --impl picks which package the tests exercise."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--impl", default="starter", choices=["solution", "starter"],
                     help="package under test: the reference solution or your starter code")


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs a model download and trained artifacts (deselect with -m 'not model')")


@pytest.fixture(scope="session")
def impl(request):
    name = request.config.getoption("--impl")

    def load(module):
        return importlib.import_module(f"{name}.{module}")
    return load
