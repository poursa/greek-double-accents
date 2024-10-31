from double_accents import Entry, State, StateMsg, fix_entry


def make_test(
    word: str,
    word_idx: int,
    line_str: str,
    state: State,
    msg: str,
) -> None:
    received = fix_entry(Entry(word, word_idx, line_str.split()))
    expected = StateMsg(state, msg)
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
