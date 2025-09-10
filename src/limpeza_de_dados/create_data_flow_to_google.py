# Create methods to read and write from a google spreadsheet.
import os

import gspread
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe, set_with_dataframe

RUN_IN_STREAMLIT: bool = False
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_drive_info(
    run_in_streamlit: bool = RUN_IN_STREAMLIT, scopes: list[str] = SCOPES
):
    if run_in_streamlit:
        service_account_info = {
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": (
                st.secrets["private_key"].replace("\\n", "\n")
                if st.secrets["private_key"]
                else None
            ),
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"],
            "universe_domain": st.secrets["universe_domain"],
        }

        if not service_account_info["private_key"]:
            raise ValueError("Missing or improperly formatted private key")

        return {
            "creds": service_account.Credentials.from_service_account_info(
                service_account_info, scopes=scopes
            ),
            "original_form_spreadsheet_url": st.secrets[
                "original_form_spreadsheet_url"
            ],
            "final_form_spreadsheet_url": st.secrets["final_form_spreadsheet_url"],
            "geographies_url": st.secrets["geographies_url"],
        }

    if not run_in_streamlit:
        load_dotenv()
        service_account_file = os.getenv("SERVICE_ACCOUNT_PATH")

        return {
            "creds": service_account.Credentials.from_service_account_file(
                service_account_file, scopes=scopes
            ),
            "original_form_spreadsheet_url": os.getenv("ORIGINAL_FORM_SPREADSHEET_URL"),
            "final_form_spreadsheet_url": os.getenv("FINAL_FORM_SPREADSHEET_URL"),
            "geographies_url": os.getenv("GEOGRAPHIES_URL"),
        }


def read_spreadsheet_as_df(url: str, gc: gspread.client.Client) -> pd.DataFrame:
    sh: gspread.spreadsheet.Spreadsheet = gc.open_by_url(url)
    wh: gspread.worksheet.Worksheet = sh.sheet1
    return get_as_dataframe(wh, header=0, evaluate_formulas=False)


def append_row_to_spreadsheet(url: str, gc: gspread.client.Client, row_to_append: list):
    sh: gspread.spreadsheet.Spreadsheet = gc.open_by_url(url)
    wh: gspread.worksheet.Worksheet = sh.sheet1
    wh.append_row(row_to_append, value_input_option="USER_ENTERED")


def write_df_as_spreadsheet(df: pd.DataFrame, url: str, gc: gspread.client.Client):
    sh: gspread.spreadsheet.Spreadsheet = gc.open_by_url(url)
    wh: gspread.worksheet.Worksheet = sh.sheet1
    set_with_dataframe(wh, df)
