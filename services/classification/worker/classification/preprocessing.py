import pandas as pd
import regex as re

from common.database.classification_result import ErrorType
from common.settings import settings


def remove_handles_iban_urls(x: str) -> str:
    # Remove social media handles (e.g., @username)
    x = re.sub(r"@\S+", "", x)

    # Remove IBAN numbers (might need adjustment based on use case)
    x = re.sub(r"[A-Z]{2}[0-9]{2}(?:\s?[0-9]{4}){4}(?:\s?[0-9]{1,2})?", "", x)

    # Remove URLs, including different cases and short links
    x = re.sub(
        r"(https?:\/\/|ftp:\/\/|www\.|WWW\.|t\.me\/|T\.ME\/|bit\.ly\/|tinyurl\.com\/|goo\.gl\/|t\.co\/)[\S]+",
        "",
        x,
        flags=re.IGNORECASE,
    )

    return x


def get_length(x: str) -> int:  # TODO: replace by word count instead of character count
    alpha = sum(i.isalpha() for i in x)
    number = sum(i.isnumeric() for i in x)
    return alpha + number


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(
        remove_handles_iban_urls
    )  # TODO: handle errors when calling apply
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    df["text_length"] = df["text"].apply(get_length)
    df.loc[df["text_length"] == 0, "error"] = ErrorType.EMPTY_TEXT
    df.loc[df["text_length"] < settings.classification_min_char_length, "error"] = (
        ErrorType.TEXT_TOO_SHORT
    )

    return df
