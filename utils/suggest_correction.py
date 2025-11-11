import difflib


def suggest_correction(word: str, valid_words: list[str]) -> str | None:
    """
    Предлагает наиболее вероятное исправление слова (для английских слов).
    """
    matches = difflib.get_close_matches(word, valid_words, n=1, cutoff=0.6)
    return matches[0] if matches else None


def suggest_correction_ru(word: str, valid_words: list[str]) -> str | None:
    """
    Предлагает наиболее вероятное исправление слова (для русских слов).
    """
    matches = difflib.get_close_matches(word, valid_words, n=1, cutoff=0.6)
    return matches[0] if matches else None
