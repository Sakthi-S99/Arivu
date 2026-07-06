from config.settings import MAX_CHUNK_CHARS, MAX_WORD_CHARS
from ingestion.ingest import chunk_text, _split_long_words


def words(n, prefix="word"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_single_short_line_returns_one_chunk():
    chunks = chunk_text("hello world foo bar", size=10, overlap=2)
    assert chunks == ["hello world foo bar"]


def test_never_splits_a_line_across_chunks():
    # Each line has more words than fits alongside the next line in one
    # `size`-word chunk, so a naive word-count window (the old behavior)
    # would clip a line in half. Line-aware chunking must keep every full
    # line intact within a single chunk instead.
    line_a = words(6, "alpha")
    line_b = words(6, "beta")
    line_c = words(6, "gamma")
    text = f"{line_a}\n{line_b}\n{line_c}"

    chunks = chunk_text(text, size=8, overlap=2)

    for line in (line_a, line_b, line_c):
        assert any(line in c for c in chunks), f"line was split across chunks: {line!r}"


def test_respects_hard_chapter_boundary():
    chapter1_body = words(5, "onewordA")
    chapter2_body = words(5, "twowordB")
    text = f"chapter 1\n{chapter1_body}\nchapter 2\n{chapter2_body}"

    # Deliberately generous size so a naive fixed-size window WOULD merge
    # both chapters into one chunk if chapter boundaries weren't respected.
    chunks = chunk_text(text, size=50, overlap=0)

    for c in chunks:
        assert not ("chapter 1" in c.lower() and "chapter 2" in c.lower()), (
            f"chunk mixes two chapters: {c!r}"
        )


def test_chapter_heading_stays_with_its_own_content():
    text = f"chapter 7\n{words(5, 'stepA')}"
    chunks = chunk_text(text, size=50, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].lower().startswith("chapter 7")


def test_overlap_carries_trailing_words_into_next_chunk():
    line_a = words(6, "alpha")
    line_b = words(6, "beta")
    text = f"{line_a}\n{line_b}"

    # size=6 forces a split after line_a; overlap=6 should carry line_a
    # forward into the start of the next chunk for context continuity.
    chunks = chunk_text(text, size=6, overlap=6)

    assert len(chunks) == 2
    assert chunks[0] == line_a
    assert chunks[1].startswith(line_a)
    assert line_b in chunks[1]


def test_oversized_single_line_falls_back_to_word_window():
    # A single line longer than `size` (e.g. pathological table extraction
    # with no internal newlines) must still get split, not silently kept
    # as one giant chunk or dropped.
    line = words(20, "tok")
    chunks = chunk_text(line, size=5, overlap=1)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 5


def test_split_long_words_breaks_monster_token():
    monster = "x" * (MAX_WORD_CHARS * 2 + 50)
    out = _split_long_words([monster])
    assert len(out) > 1
    assert all(len(w) <= MAX_WORD_CHARS for w in out)


def test_chunk_char_ceiling_is_enforced():
    # Enough words to blow past MAX_CHUNK_CHARS if packed into one chunk
    # with a generously large `size`.
    huge_line = " ".join("longishword" for _ in range(MAX_CHUNK_CHARS // 5))
    chunks = chunk_text(huge_line, size=100_000, overlap=0)
    for c in chunks:
        assert len(c) <= MAX_CHUNK_CHARS


def test_char_ceiling_overflow_is_not_silently_dropped():
    # Regression test: an earlier version of _pack_to_char_ceiling trimmed
    # overflow past MAX_CHUNK_CHARS and discarded it instead of carrying it
    # into another chunk — 900 words in produced only 333 words out. Every
    # input word must survive across the returned chunks.
    n = 900
    huge_line = " ".join(f"longishword{i}" for i in range(n))
    chunks = chunk_text(huge_line, size=100_000, overlap=0)
    total_words_out = sum(len(c.split()) for c in chunks)
    assert total_words_out == n
