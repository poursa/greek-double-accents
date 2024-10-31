"""
TODO:
- Clean
- Time it
- Make some tests

Does spacy syllabify?
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import spacy
import spacy.cli

# For ancient greek but seems to work fine for modern
# pip install greek-accentuation==1.2.0
from greek_accentuation.syllabify import syllabify

model_name = "el_core_news_sm"
try:
    nlp = spacy.load(model_name)
    # print(nlp.path)
except OSError:
    print(f"Model '{model_name}' not found. Downloading...")
    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)

_PUNCT = re.escape(",.!?-;:\n«»\"'")
PUNCT = re.compile(rf"[{_PUNCT}]")
VOWEL_ACCENTED = re.compile(r"[έόίύάήώ]")

# These words are parsed as trisyllables by syllabify
# but they actually have only two syllables.
FALSE_TRISYL = {"χέρια", "μάτια", "πόδια", "λόγια", "δίκιο", "δίκια", "σπίτια"}

# Grammar
# http://ebooks.edu.gr/ebooks/v/html/8547/2009/Grammatiki_E-ST-Dimotikou_html-apli/index_C8a.html
PRON_ACC_SING = {
    "με",
    # "σε", # Example?
    "τον",
    "την",
    "το",
}

PRON_ACC_PLUR = {
    "μας",
    "σας",
    "τους",
    "τις",
    "τα",
}

PRON_ACC = {*PRON_ACC_SING, *PRON_ACC_PLUR}

PRON_GEN_SING = {
    "μου",
    "σου",
    "του",
    "της",
    # "τη", # Never happens in double-accents?
    "του",
}

PRON_GEN_PLUR = {
    "μας",
    "σας",
    "τους",
    "τις",
    "τα",
}

PRON_GEN = {*PRON_GEN_SING, *PRON_GEN_PLUR}

PRON = {*PRON_ACC, *PRON_GEN}


def split_punctuation(word: str) -> tuple[str, str | None]:
    """Splits a word into its base form and any trailing punctuation."""
    if mtch := PUNCT.search(word):
        return word[: mtch.start()], word[mtch.start() :]
    return word, None


class State(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"


@dataclass
class StateMsg:
    state: State
    msg: str


@dataclass
class Entry:
    word: str
    word_idx: int
    line: list[str]
    line_number: int = 0
    entry_id: int = 0
    # Otherwise mutable default error etc.
    statemsg: StateMsg = field(default_factory=lambda: StateMsg(State.AMBIGUOUS, "TODO"))
    semantic_info: dict[str, dict[str, str]] | None = None
    # words?

    @property
    def line_ctx(self) -> str:
        return " ".join(
            self.line[max(0, self.word_idx - 1) : min(len(self.line), self.word_idx + 5)]
        )

    def add_semantic_info(self) -> None:
        assert self.word_idx < len(self.line) - 2, "Faulty sentence with no final punctuation."

        words = [split_punctuation(w)[0] for w in self.line[self.word_idx : self.word_idx + 3]]
        self.words = words
        assert len(words) == 3

        doc = nlp(" ".join(self.line))
        semantic_info = {}

        # TODO: Keep the tokens for the whole sentence to debug

        for token in doc:
            word, _ = split_punctuation(token.text)
            if words.count(word) > 1:
                raise ValueError("The word appears twice in the sentence")
            if word not in words:
                continue
            semantic_info[word] = {}
            semantic_info[word]["pos"] = token.pos_
            if word == words[1]:
                # Why does spacy thinks that σου is an ADV/ADJ/NOUN?
                if word != "σου":
                    assert token.pos_ in (
                        "DET",
                        "PRON",
                        "ADP",  # με
                    ), f"Unexpected pos {token.pos_} from {token.text}"
            semantic_info[word]["case"] = token.morph.get("Case", ["X"])[0]

        assert len(semantic_info) == 3, f"{semantic_info}\n{words}\n{self.word} || {self.line}"

        self.semantic_info = semantic_info

    def detailed_str(self) -> str:
        hstart = "\033[1m"
        hend = "\033[0m"
        line_ctx = self.line_ctx
        hctx = line_ctx.replace(self.word, f"{hstart}{self.word}{hend}")
        state_colors = {
            State.CORRECT: "\033[32m",  # Green
            State.INCORRECT: "\033[31m",  # Red
            State.PENDING: "\033[33m",  # Yellow
            State.AMBIGUOUS: "\033[34m",  # Blue
        }

        color = state_colors.get(self.statemsg.state, "\033[0m")
        return f"{color}{str(self.statemsg.state)[6:]:<9} [{self.statemsg.msg:<12}]{hend} {hctx}"

    def __str__(self) -> str:
        # return f"{self.state:<15} {self.get_line_ctx}"
        return self.detailed_str()


def find_candidates(text: str) -> None:
    lines = text.splitlines()

    n_candidates_total = 0
    record = {State.CORRECT: 0, State.INCORRECT: 0, State.PENDING: 0, State.AMBIGUOUS: 0}

    for n_par, paragraph in enumerate(lines, start=1):
        par_lines = re.findall(r"[^.!?;:]+[.!?;:]?", paragraph)
        for n_line, line in enumerate(par_lines, start=1):
            if line := line.strip():
                # print(f"[{n_par}:{n_line}] Line:", line, "\n", paragraph)
                states = fix_lines(line, n_par, n_candidates_total)
                n_candidates_total += len(states)
                for state in states:
                    record[state] += 1

        # TODO: remove
        if n_candidates_total >= 7000:
            break

    print(f"\nFound {n_candidates_total} candidates.")
    for state, cnt in record.items():
        print(f"{str(state)[6:]:<9} {cnt}")


def fix_lines(line: str, n_line: int, n_candidates_total: int) -> list[State]:
    """TODO: return (fixed_sentence, n_errors)"""
    words = line.split()
    n_words = len(words)
    cnt = 0
    states = []

    for idx, word in enumerate(words):
        # TODO: export this into a function

        # Word is at the end
        if idx == n_words - 1:
            continue

        # Punctuation automatically makes this word correct
        word, wpunct = split_punctuation(word)
        if wpunct:
            continue

        # Need at least three syllables, with the antepenult accented...
        syllables = syllabify(word)
        if len(syllables) < 3 or not VOWEL_ACCENTED.search(syllables[-3]):
            continue
        # ...and the last one unaccented (otherwise it is not an error)
        if VOWEL_ACCENTED.search(syllables[-1]):
            continue

        # From here on, it is tricky
        entry = Entry(word, idx, words, n_line, n_candidates_total + cnt)
        cnt += 1

        statemsg = fix_entry(entry)
        entry.statemsg = statemsg

        # Tested to correctly work: ignore them
        to_ignore = ("2~3SYL", "2~PRON")
        if statemsg.msg in to_ignore:
            continue

        # At this point you are an error / undecidable
        print(entry)

        # Quick print semantic info if PENDING
        if entry.statemsg.state == State.PENDING:
            wi = entry
            assert wi.semantic_info
            try:
                w1, w2, w3 = wi.words[:3]
                si1 = wi.semantic_info[w1]
                si2 = wi.semantic_info[w2]
                si3 = wi.semantic_info[w3]
            except Exception as e:
                print(wi.line)
                print(wi.word)
                print(wi.semantic_info)
                raise e
            pos1 = si1["pos"]
            pos2 = si2["pos"]
            pos3 = si3["pos"]
            print(pos1, pos2, pos3, " || ", si1["case"], si2["case"], si3["case"])

        states.append(entry.statemsg.state)

    return states


def fix_entry(entry: Entry) -> StateMsg:
    # Verify that the word is not banned (False trisyllables)
    if entry.word in FALSE_TRISYL:
        return StateMsg(State.CORRECT, "1~3SYL")

    # Next word (we assume it exists), must be a pronoun
    detpron, punct = split_punctuation(entry.line[entry.word_idx + 1])
    if detpron not in PRON:
        return StateMsg(State.CORRECT, "2~PRON")

    if punct:
        # This is a mistake and it is fixable
        return StateMsg(State.INCORRECT, "2PUNCT")
    else:
        # Semantic analysis
        entry.add_semantic_info()
        return semantic_analysis(entry)


def semantic_analysis(wi: Entry) -> StateMsg:
    """Return True if correct, False if incorrect or undecidable."""

    assert wi.semantic_info
    try:
        w1, w2, w3 = wi.words[:3]
        si1 = wi.semantic_info[w1]
        si2 = wi.semantic_info[w2]
        si3 = wi.semantic_info[w3]
    except Exception as e:
        print(wi.line)
        print(wi.word)
        print(wi.semantic_info)
        raise e
    pos1 = si1["pos"]
    pos2 = si2["pos"]
    pos3 = si3["pos"]

    if "X" in (pos1 + pos3):
        # Ambiguous: incomplete information
        return StateMsg(State.AMBIGUOUS, "NO INFO")

    same_case12 = si1["case"] == si2["case"]
    same_case13 = si1["case"] == si3["case"]
    same_case23 = si2["case"] == si3["case"]

    match pos1:
        case "VERB":
            return StateMsg(State.PENDING, "1VERB")
        case "PROPN":
            # High chance of being correct
            if pos3 == "VERB":
                # CEx: Ο Άνγελός μου είπε...
                return StateMsg(State.AMBIGUOUS, "1PROPN 3VERB")
            else:
                return StateMsg(State.CORRECT, "1PROPN")
        case "NOUN":
            # The pronoun must be genitive
            if w2 not in PRON_GEN:
                return StateMsg(State.CORRECT, f"1NOUN 2{w2}~GEN")

            match pos3:
                case "NOUN":
                    if same_case23:
                        return StateMsg(State.CORRECT, "13NOUN 23SC")
                case "VERB":
                    # CEx: Το άνθρωπο της έδωσε / Το άνθρωπο τής έδωσε.
                    return StateMsg(State.AMBIGUOUS, "1NOUN 3VERB")

            return StateMsg(State.PENDING, "1NOUN")
        case "ADV":
            return StateMsg(State.CORRECT, "1ADV")

    return StateMsg(State.AMBIGUOUS, "TODO")


def main() -> None:
    filepath = Path("book.txt")
    with filepath.open("r") as file:
        text = file.read().strip()
        find_candidates(text)


if __name__ == "__main__":
    main()
