import os
import pandas as pd


def load_data(
    file_name: str,
    sheet_name: str = "Data"
):

    project_path = os.getcwd()

    file_path = os.path.join(
        project_path,
        "Data",
        file_name
    )

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name
    )

    return df