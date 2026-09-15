"""Google Translate helper functions using deep-translator."""

import time

from deep_translator import GoogleTranslator


def translate_dutch_text(
    text,
    source_language_code="nl",
    target_language_code="en",
    region_name=None,
    client=None,
):
    """Translate Dutch text to English using Google Translate."""

    if text is None:
        return None

    text = str(text).strip()

    if text == "":
        return None

    try:
        return GoogleTranslator(
            source=source_language_code,
            target=target_language_code,
        ).translate(text)

    except Exception as exc:
        print(f"Translation failed for {text}: {exc}")
        return None


def add_translated_column(
    df,
    source_column,
    translated_column,
    translate_function=translate_dutch_text,
    sleep_seconds=1,
    region_name=None,
):
    """Translate unique values from one column and map translations back."""

    unique_values = (
        df[source_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    translation_dict = {}

    for index, value in enumerate(unique_values, start=1):
        translated = translate_function(value)
        translation_dict[value] = translated

        if sleep_seconds:
            time.sleep(sleep_seconds)

        if index % 50 == 0:
            print(f"translated {index}/{len(unique_values)} unique values")

    df[translated_column] = (
        df[source_column]
        .astype(str)
        .map(translation_dict)
    )

    return df, translation_dict
