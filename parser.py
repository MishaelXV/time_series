import os
import pandas as pd


def load_key_rate_from_cbr(
    start_date: str,
    end_date: str
):

    url = (
        "https://www.cbr.ru/hd_base/keyrate/"
        f"?UniDbQuery.Posted=True"
        f"&UniDbQuery.From={start_date}"
        f"&UniDbQuery.To={end_date}"
    )

    tables = pd.read_html(
        url,
        decimal=","
    )

    key_rate = tables[0]

    key_rate.columns = [
        "Date",
        "KeyRate"
    ]

    key_rate["Date"] = pd.to_datetime(
        key_rate["Date"],
        dayfirst=True
    )

    key_rate["KeyRate"] = (
        key_rate["KeyRate"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    key_rate = key_rate.sort_values("Date").reset_index(drop=True)

    return key_rate


def save_key_rate_to_excel(
    start_date: str,
    end_date: str,
    file_name: str = "KEY_RATE.xlsx",
    folder: str = "data"
):

    key_rate = load_key_rate_from_cbr(
        start_date=start_date,
        end_date=end_date
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    file_path = os.path.join(
        folder,
        file_name
    )

    key_rate.to_excel(
        file_path,
        index=False
    )

    return key_rate