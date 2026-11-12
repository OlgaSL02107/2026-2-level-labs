"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
    WordProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    print('Mark 4 task')

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)
    if encoded_text is None:
        return

    decoded_text = text_processor.decode(encoded_text)
    if decoded_text is None:
        return

    print('Encoded text:')
    print(encoded_text[-100:])
    print('\nDecoded text:')
    print(decoded_text[-100:])

    print('\nNGram LM check task')

    model = NGramLanguageModel(encoded_text, 7)
    model_built = model.build()

    print('Model built:', model_built)

    print('\nMark 6 task')

    greedy_generator = GreedyTextGenerator(model, text_processor)
    generated_text = greedy_generator.run('Vernon', 51)
    print('Generated text:')
    print(generated_text)

    print('\nMark 8 task')
    generator = BeamSearchTextGenerator(
        language_model=model,
        text_processor=text_processor,
        beam_width=3,
    )
    generated = generator.run('Vernon', 56)
    print('Generated text:')
    print(generated)

    print('\nMark 10 task 1')
    reader = NGramLanguageModelReader(
        "./lab_3_generate_by_ngrams/assets/en_own.json", '_'
    )
    models = []
    for ngram_size in (2, 3, 4):
        loaded_model = reader.load(ngram_size)
        if loaded_model is not None:
            models.append(loaded_model)

    backoff_generator = BackOffGenerator(
        tuple(models),
        reader.get_text_processor(),
    )
    generated_text = backoff_generator.run('Vernon', 100)

    result = generated_text
    assert result

    print(generated_text)

    print('\nMark 10 task 2')
    prompt = 'Harry looked at'

    word_processor = WordProcessor('<eos>')
    encoded_text_2 = word_processor.encode(text)
    if encoded_text_2 is None:
        return

    word_models = {}
    for ngram_size in (2, 3, 4):
        word_model = NGramLanguageModel(encoded_text_2, ngram_size)
        word_model.build()
        word_models[ngram_size] = word_model

    greedy_continue = GreedyTextGenerator(word_models[4], word_processor)
    print('\nGreedy algorithm continue:')
    print(greedy_continue.run(prompt, 12))

    beam_continue = BeamSearchTextGenerator(
        language_model=word_models[4],
        text_processor=word_processor,
        beam_width=3,
    )
    print('\nBeam Search algorithm continue:')
    print(beam_continue.run(prompt, 12))

    backoff_continue = BackOffGenerator(
        tuple(word_models.values()),
        word_processor,
    )
    print('\nBackOff algorithm continue:')
    print(backoff_continue.run(prompt, 12))

    print('\nComparison to greedy generated text at the level of letters:')
    print(greedy_generator.run(prompt, 60))


if __name__ == "__main__":
    main()
