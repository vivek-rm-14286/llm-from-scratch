import pytest

Q = "What is the capital of France?"
W20 = {"kind": "max_words", "n": 20}
ONE = {"kind": "one_sentence"}
L3 = {"kind": "numbered_list", "n": 3}


def test_good_answer_passes_everything(impl):
    j = impl.judge.judge(Q, "The capital of France is Paris.", W20, True)
    assert j["passed"] and j["score"] == 6
    assert set(j["checks"]) == set(impl.judge.CHECKS) and all(j["checks"].values())


@pytest.mark.parametrize("response, constraint, finished, failed", [
    ("The capital of France is Paris.", W20, False, {"finished"}),                 # hit the token limit
    ("Paris.", W20, True, {"not_empty", "on_topic"}),                              # under 3 words
    ("The capital of France is Paris.\nuser\nThanks!", W20, True, {"clean"}),      # invented a user turn
    ("The capital of France is Paris.<|im_start|>", W20, True, {"clean"}),         # leaked a marker
    ("The capital of France is Paris. The capital of France is Paris.", W20, True, {"no_repeat"}),
    ("I like green apples a lot.", W20, True, {"on_topic"}),                       # no shared content word
    ("The capital of France is Paris, a large city on the Seine that has been the seat of "
     "government for many centuries now.", W20, True, {"constraint"}),             # 23 words
    ("The capital is Paris. France is in Europe.", ONE, True, {"constraint"}),     # two sentences
    ("1. Paris is the capital\n2. France is in Europe", L3, True, {"constraint"}), # two items
    ("Here is a list about France:\n1. a\n2. b\n3. c", L3, True, {"constraint"}),  # extra prose line
    ("", W20, True, {"not_empty", "on_topic", "constraint"}),
])
def test_single_failures(impl, response, constraint, finished, failed):
    checks = impl.judge.check_response(Q, response, constraint, finished)
    assert {k for k, ok in checks.items() if not ok} == failed


def test_constraints_that_pass(impl):
    ok = lambda r, c: impl.judge.judge(Q, r, c, True)["passed"]
    assert ok("The capital of France is Paris!", ONE)
    assert ok("1. Paris is the capital of France\n2. Lyon is large\n3. Nice is sunny", L3)
    assert ok("1) Paris, capital of France\n2) Lyon\n3) Nice", L3)


def test_has_repeated_ngram(impl):
    f = impl.judge.has_repeated_ngram
    assert f("the cat sat on the cat sat on a mat", 4)             # "the cat sat on" twice
    assert not f("the cat sat on the mat and the cat slept", 4)
    assert f("No no no no no", 4)                                  # case-insensitive
    assert not f("a b c", 4)


def test_content_words_ignores_short_and_stopwords(impl):
    assert impl.judge.content_words("What is the capital of France?") == {"capital", "france"}


def test_on_topic_passes_when_question_has_no_content_words(impl):
    assert impl.judge.check_response("2+2=?", "It is four, of course.", W20, True)["on_topic"]


def test_pick_pair(impl):
    pick = impl.judge.pick_pair
    assert pick([3, 6, 2, 6]) == (1, 2)          # first best, first worst
    assert pick([4, 4, 4, 4]) is None            # no gap, no pair
    assert pick([5, 6], min_gap=2) is None
    assert pick([2, 6, 2], min_gap=2) == (1, 0)
    assert pick([]) is None
