"""The generator: deterministic, correctly labeled, and with every kind of difficulty present."""
import pytest

SMALL = dict(n_train=400, n_cal=150, n_test=300, n_ood=50)


@pytest.fixture(scope="module")
def data(impl):
    return impl("data")


@pytest.fixture(scope="module")
def splits(data):
    return data.generate(seed=0, **SMALL)


def test_same_seed_same_rows(data, splits):
    assert data.generate(seed=0, **SMALL) == splits


def test_different_seed_different_rows(data, splits):
    other = data.generate(seed=1, **SMALL)
    assert [r["text"] for r in other["train"]] != [r["text"] for r in splits["train"]]


def test_sizes_and_fields(data, splits):
    assert {k: len(v) for k, v in splits.items()} == {"train": 400, "cal": 150, "test": 300, "ood": 50}
    for name in ("train", "cal", "test"):
        for row in splits[name]:
            assert set(row) == {"id", "text", "label", "clean_label", "kind", "secondary", "noisy"}
            assert row["label"] in data.QUEUES and row["clean_label"] in data.QUEUES
            assert row["text"].strip()
    assert len({r["id"] for s in splits.values() for r in s}) == 900


def test_every_queue_and_kind_present(data, splits):
    assert {r["clean_label"] for r in splits["train"]} == set(data.QUEUES)
    assert {r["kind"] for r in splits["train"]} == {"plain", "two_queue", "short"}
    assert {r["kind"] for r in splits["test"]} == {"plain", "paraphrase", "two_queue", "short"}


def test_two_queue_rows_name_a_different_secondary(splits):
    rows = [r for r in splits["test"] if r["kind"] == "two_queue"]
    assert rows
    assert all(r["secondary"] and r["secondary"] != r["clean_label"] for r in rows)


def test_label_noise_only_in_train_and_near_requested_rate(data):
    s = data.generate(seed=0, n_train=2000, n_cal=100, n_test=100, n_ood=10, noise=0.05)
    noisy = [r for r in s["train"] if r["noisy"]]
    assert 0.03 <= len(noisy) / 2000 <= 0.07
    assert all(r["label"] != r["clean_label"] for r in noisy)
    assert all(r["label"] == r["clean_label"] for r in s["train"] if not r["noisy"])
    assert not any(r["noisy"] for r in s["cal"] + s["test"])


def test_zero_noise_means_clean_labels(data):
    s = data.generate(seed=0, noise=0.0, **SMALL)
    assert all(r["label"] == r["clean_label"] for r in s["train"])


def test_no_long_message_leaks_from_train_to_eval(splits):
    train = {r["text"] for r in splits["train"]}
    for name in ("cal", "test"):
        assert not [r for r in splits[name] if r["kind"] != "short" and r["text"] in train]


def test_ood_slice_is_labeled_ood(data, splits):
    assert all(r["label"] == "ood" and r["kind"] == "ood" for r in splits["ood"])
    assert not {r["text"] for r in splits["ood"]} & {r["text"] for r in splits["train"]}
