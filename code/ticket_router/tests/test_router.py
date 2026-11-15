"""The interface contract that other projects rely on."""
import warnings

import numpy as np
import pytest


@pytest.fixture(scope="module")
def R(impl):
    return impl("router")


@pytest.fixture(scope="module")
def queues(impl):
    return impl("data").QUEUES


def check_contract(result, queues):
    assert set(result) == {"queue", "probs", "confidence", "abstain"}
    assert result["queue"] in queues
    assert list(result["probs"]) == list(queues)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in result["probs"].values())
    assert sum(result["probs"].values()) == pytest.approx(1.0, abs=1e-6)
    assert isinstance(result["confidence"], float)
    assert result["confidence"] == pytest.approx(max(result["probs"].values()))
    assert result["probs"][result["queue"]] == pytest.approx(result["confidence"])
    assert isinstance(result["abstain"], bool)


def test_fallback_when_no_artifact(R, queues, tmp_path):
    with pytest.warns(UserWarning, match="keyword"):
        router = R.load_router(tmp_path)
    assert "keyword" in router.name
    check_contract(router.route("We were charged twice and need a refund for the invoice."), queues)


def test_fallback_routes_obvious_messages_and_abstains_on_noise(R, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        router = R.load_router(tmp_path)
    billing = router.route("We were charged twice, please refund the invoice and fix our billing.")
    assert billing["queue"] == "billing" and not billing["abstain"]
    access = router.route("SSO login fails and my password reset never arrives, I am locked out.")
    assert access["queue"] == "access"
    for text in ("asdf jkl qwer", "", "What is a good recipe for banana bread?"):
        assert router.route(text)["abstain"] is True


def test_decision_rule_uses_temperature_and_threshold(R, queues):
    logits = np.zeros((1, len(queues)))
    logits[0, 2] = 3.0
    sharp = R.Router("fake", lambda texts: np.repeat(logits, len(texts), 0), temperature=1.0, threshold=0.7)
    soft = R.Router("fake", lambda texts: np.repeat(logits, len(texts), 0), temperature=3.0, threshold=0.7)
    a, b = sharp.route("x"), soft.route("x")
    check_contract(a, queues)
    check_contract(b, queues)
    p = np.exp(3.0) / (np.exp(3.0) + len(queues) - 1)
    assert a["queue"] == b["queue"] == queues[2]
    assert a["confidence"] == pytest.approx(p) and not a["abstain"]
    assert b["confidence"] < a["confidence"] and b["abstain"]
    assert len(sharp.route_batch(["x", "y", "z"])) == 3


@pytest.mark.model
def test_trained_router_contract(R, queues):
    """Needs `python -m solution.calibrate` to have written out/router.json."""
    if not (R.DEFAULT_OUT / "router.json").exists():
        pytest.skip("no trained artifact under out/")
    router = R.load_router()
    assert "keyword" not in router.name
    result = router.route("Our card statement shows a duplicate payment to Nimbus for October.")
    check_contract(result, queues)
    assert result["queue"] == "billing"
    assert router.route("asdf jkl qwer zxcv")["confidence"] < result["confidence"]


@pytest.mark.model
def test_real_decision_lm_is_order_invariant(impl):
    """Downloads HuggingFaceTB/SmolLM2-135M (about 270 MB) on first use."""
    import torch
    dlm = impl("decision_lm")
    model = dlm.DecisionLM().eval()
    torch.nn.init.normal_(model.scorer.weight, std=0.05)
    options = dlm.queue_options()
    context = dlm.make_context("Data points are rejected with NM-4012 after we upgraded our plan.")
    with torch.no_grad():
        base = model([context], [options])[0]
        rev = model([context], [options[::-1]])[0]
    assert torch.allclose(base, rev.flip(0), atol=1e-3)
