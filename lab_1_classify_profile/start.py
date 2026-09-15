"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code

from main import tokenize, remove_stop_words

def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    result = None
    assert result, "Detection result is None"

    de_tokens = tokenize(de_text)
    unknown_tokens = tokenize(unknown_text)
    en_tokens = tokenize(en_text)

    if de_tokens is None or unknown_tokens is None or en_tokens is None:
        return None

    de_filtered = remove_stop_words(de_tokens, stopwords)
    unknown_filtered = remove_stop_words(unknown_tokens, stopwords)
    en_filtered = remove_stop_words(en_tokens, stopwords)

    if de_filtered is None or unknown_filtered is None or en_filtered is None:
        return None


if __name__ == "__main__":
    main()
