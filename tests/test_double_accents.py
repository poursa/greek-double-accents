from double_accents import Entry, State, StateMsg, fix_entry


def make_test(
    word: str,
    word_idx: int,
    line_str: str,
    state: State,
    msg: str,
    spacy_checks: dict[str, dict[str, str]] | None = None,
) -> None:
    entry = Entry(word, word_idx, line_str.split())
    received = fix_entry(entry)
    expected = StateMsg(state, msg)
    # (Optional) Test that the spaCy analysis is sound.
    if not spacy_checks is None:
        si = entry.semantic_info
        assert not si is None
        for sc_word, sc_constraint in spacy_checks.items():
            assert sc_word in si
            word_si = si[sc_word]
            for grammar_type, value in sc_constraint.items():
                assert word_si[grammar_type] == value
    assert received == expected


def test_incorrect() -> None:
    make_test(
        word="πρωτεύουσα",
        word_idx=1,
        line_str="η πρωτεύουσα του.",
        state=State.INCORRECT,
        msg="2PUNCT",
    )


def test_false_trisyllable() -> None:
    make_test(
        word="μάτια",
        word_idx=1,
        line_str="τα μάτια του.",
        state=State.CORRECT,
        msg="1~3SYL",
    )


def test_verb_ambiguous() -> None:
    # Ο άνθρωπος μου είπε / Ο άνθρωπός μου είπε
    make_test(
        word="άνθρωπος",
        word_idx=1,
        line_str="O άνθρωπος μου είπε",
        state=State.AMBIGUOUS,
        msg="1NOUN 3VERB",
        spacy_checks={
            "άνθρωπος": {"pos": "NOUN"},
            "είπε": {"pos": "VERB"},
        },
    )
