from backend.config import load_settings
from generator.text_utils import (
    content_hash,
    dedup_key,
    dominant_script,
    has_control_chars,
    normalize_text,
    word_jaccard,
)

SCRIPTS = {k: v.ranges for k, v in load_settings().languages.scripts.items()}


def test_normalize_collapses_whitespace_keeps_case():
    assert normalize_text("  How   can\tI\n do  X? ") == "How can I do X?"


def test_normalize_nfc():
    decomposed = "é"  # e + combining acute
    assert normalize_text(decomposed) == "é"


def test_dedup_key_ignores_case_punct_space():
    assert dedup_key("How can I use poetry?") == dedup_key("how  can i use POETRY")
    assert content_hash("A b, c!") == content_hash("a B c")


def test_dedup_key_keeps_different_words_apart():
    assert dedup_key("How can I use poetry") != dedup_key("How can I use drama")


def test_dominant_script_latin():
    assert dominant_script("Who wrote De architectura?", SCRIPTS) == ("Latn", 1.0)


def test_dominant_script_devanagari_with_marks():
    script, conf = dominant_script("डी आर्किटेक्चर ग्रंथ के लेखक कौन थे?", SCRIPTS)
    assert script == "Deva" and conf == 1.0


def test_dominant_script_mixed_reports_share():
    script, conf = dominant_script("मैं tutorials देख रहा हूँ", SCRIPTS)
    assert script == "Deva" and 0.5 < conf < 1.0


def test_dominant_script_no_letters():
    assert dominant_script("123 ?!", SCRIPTS) == (None, 0.0)


def test_control_chars_detected_but_zwj_allowed():
    assert has_control_chars("abc\x07")
    assert not has_control_chars("क्‍ष line\nbreak")


def test_word_jaccard():
    assert word_jaccard("a b c", "a b c") == 1.0
    assert word_jaccard("a b", "c d") == 0.0
    assert word_jaccard("How can I use poetry", "How can I use drama") == 4 / 6
