"""Selects which package the tests grade.

    pytest                    # grades starter/  (the default)
    pytest --impl solution    # grades solution/
    NN_IMPL=solution pytest   # same, through an environment variable
"""
import importlib
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--impl", default=os.environ.get("NN_IMPL", "starter"),
                     help="package to test: starter (default) or solution")


def pytest_report_header(config):
    return f"implementation under test: {config.getoption('--impl')}/"


class Impl:
    """`impl.tensor` imports `<package>.tensor` on first use."""

    def __init__(self, package):
        self.package = package

    def __getattr__(self, name):
        module = importlib.import_module(f"{self.package}.{name}")
        setattr(self, name, module)
        return module


@pytest.fixture(scope="session")
def impl(request):
    return Impl(request.config.getoption("--impl"))
