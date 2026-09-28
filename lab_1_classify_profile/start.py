"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code

from lab_1_classify_profile.main import (
    calculate_frequencies,
    check_profile,
    create_language_profile,
    detect_language_by_mse,
    detect_language_by_top_n,
    get_top_n_words,
    remove_stop_words,
    tokenize,
)


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

    # Practical assignment 1-4

    de_tokens = tokenize(de_text)
    assert de_tokens is not None

    de_filt = remove_stop_words(de_tokens, stopwords)
    assert de_filt is not None

    de_freq = calculate_frequencies(de_filt)
    assert de_freq is not None

    top_7_words = get_top_n_words(de_freq, 7)
    assert top_7_words is not None

    print(f'Top 7 words in de_text: {top_7_words}')

    # Practical assignment 1-7

    de_profile = create_language_profile('de', de_text, stopwords)
    assert de_profile is not None

    unknown_profile = create_language_profile('unknown', unknown_text, stopwords)
    assert unknown_profile is not None

    en_profile = create_language_profile('en', en_text, stopwords)
    assert en_profile is not None

    assert check_profile(de_profile), 'Error: De profile invalid'
    assert check_profile(unknown_profile), 'Error: Unknown profile invalid'
    assert check_profile(en_profile), 'Error: En profile invalid'

    detected_lang_top_n = detect_language_by_top_n(unknown_profile, de_profile, en_profile, 15)
    assert detected_lang_top_n is not None

    print(f'Detected language by top-15 words: {detected_lang_top_n}')

    # Practical assignment 1-10

    detected_lang_mse = detect_language_by_mse(unknown_profile, de_profile, en_profile)
    assert detected_lang_mse is not None

    print(f'Detected language by mse: {detected_lang_mse}')

    result = detected_lang_mse
    assert result, 'Detection result is None'


if __name__ == "__main__":
    main()
