"""AWS Translate helper functions."""

import time

import boto3
from botocore.exceptions import ClientError


def get_aws_translate_client(region_name="eu-west-1"):
    """Create an AWS Translate client."""

    return boto3.client(
        service_name="translate",
        region_name=region_name,
    )


def translate_dutch_text(
    text,
    source_language_code="nl",
    target_language_code="en",
    region_name="eu-west-1",
    client=None,
):
    """Translate Dutch text to English using AWS Translate."""

    if text is None:
        return None

    text = str(text).strip()

    if text == "":
        return None

    if client is None:
        client = get_aws_translate_client(region_name=region_name)

    try:
        response = client.translate_text(
            Text=text,
            SourceLanguageCode=source_language_code,
            TargetLanguageCode=target_language_code,
        )
        return response["TranslatedText"]

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")

        if error_code == "SubscriptionRequiredException":
            raise RuntimeError(
                "AWS Translate is not active for this AWS account/key yet. "
                "Finish AWS billing/account activation, then rerun this cell."
            ) from exc

        print(f"Translation failed for {text}: {exc}")
        return None

    except Exception as exc:
        print(f"Translation failed for {text}: {exc}")
        return None


def add_translated_column(
    df,
    source_column,
    translated_column,
    translate_function=translate_dutch_text,
    sleep_seconds=0.05,
    region_name="eu-west-1",
):
    """Translate unique values from one column and map translations back."""

    unique_values = (
        df[source_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    client = get_aws_translate_client(region_name=region_name)
    translation_dict = {}

    for index, value in enumerate(unique_values, start=1):
        translated = translate_function(
            value,
            region_name=region_name,
            client=client,
        )

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
