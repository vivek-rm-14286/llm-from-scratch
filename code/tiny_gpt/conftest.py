"""Chooses which implementation the tests grade.

    pytest                      # grades starter/ (your work)
    pytest --impl solution      # grades the reference implementation
    TINY_GPT_IMPL=solution pytest
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_addoption(parser):
    parser.addoption("--impl", default=os.environ.get("TINY_GPT_IMPL", "starter"),
                     help="directory under code/tiny_gpt to grade: starter (default) or solution")


def pytest_configure(config):
    impl = config.getoption("--impl")
    path = impl if os.path.isabs(impl) else os.path.join(HERE, impl)
    if not os.path.isdir(path):
        raise pytest.UsageError(f"--impl {impl}: no such directory {path}")
    sys.path.insert(0, path)            # tests do `import bpe`, `import model`, ...


def pytest_report_header(config):
    return f"tiny_gpt implementation under test: {config.getoption('--impl')}"
