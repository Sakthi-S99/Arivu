from query.expander import expand_query


def test_no_acronym_present_returns_query_unchanged():
    assert expand_query("hello there friend") == "hello there friend"


def test_expands_known_builtin_acronym():
    out = expand_query("What is an API?")
    assert out == "What is an API? (Application Programming Interface)"


def test_matches_are_case_insensitive():
    out = expand_query("Explain SDK setup")
    assert "Software Development Kit" in out


def test_substring_is_not_treated_as_a_whole_word_match():
    # "apitude" contains "api" as a substring but is a different word —
    # expansion must key off whole tokens, not substrings.
    out = expand_query("describe your apitude here")
    assert out == "describe your apitude here"


def test_multiple_acronyms_each_expanded_once():
    out = expand_query("sdk and api together")
    assert "Software Development Kit" in out
    assert "Application Programming Interface" in out


def test_expansion_not_duplicated_if_already_spelled_out():
    text = "Explain Single Sign-On (sso) setup"
    out = expand_query(text)
    # "single sign-on" already present verbatim (case-insensitive) — no
    # redundant "(Single Sign-On)" suffix should be appended.
    assert out == text
