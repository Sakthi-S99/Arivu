from query.ask import _normalize_question


def test_empty_and_whitespace_only_left_unchanged():
    assert _normalize_question("") == ""
    assert _normalize_question("   ") == "   "


def test_question_mark_left_unchanged():
    assert _normalize_question("What is a delinquency plan?") == "What is a delinquency plan?"
    assert _normalize_question("?") == "?"


def test_bare_keyword_is_rewritten_into_a_question():
    out = _normalize_question("Delinquency")
    assert out == 'What does the documentation say about "Delinquency"?'


def test_long_topic_phrase_without_question_mark_is_rewritten():
    # Regression test: an earlier word-count cutoff (<=4 words) missed this
    # exact 5-word phrase, which hit the same false-refusal bug as a bare
    # single-word query. Length must not matter — only phrasing does.
    phrase = "delinquency workflow configuration steps overview"
    out = _normalize_question(phrase)
    assert out == f'What does the documentation say about "{phrase}"?'


def test_question_starter_words_left_unchanged_even_without_question_mark():
    for starter in ("What", "how", "Explain", "list", "Describe", "Can", "does"):
        text = f"{starter} the delinquency workflow works"
        assert _normalize_question(text) == text


def test_imperative_phrase_left_unchanged():
    text = "explain the delinquency workflow"
    assert _normalize_question(text) == text
