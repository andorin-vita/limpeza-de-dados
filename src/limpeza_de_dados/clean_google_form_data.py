"""
Ferramenta de limpeza de dados adquiridos pelo Google Forms
------------------------------------------------------

Este script limpa os dados adquiridos pelo formulário Google Forms de forma
a torná-los compatíveis com os dados já validados.

Deve ser integrado na ferramenta de validação.

O input deve ser a Google Sheet ligada ao Google Forms e o output uma versão intermediária da que
será usada para a ferramenta de validação.

Principais funcionalidades:
---------------------------
1. Lista de colunas provenientes do Google Form e seus pares na tabela já validada
2. Conversão dos nomes das colunas
3. Separação das coordenadas da estrutura de nidificação em colunas Latitude e Longitude

Estrutura do código:
--------------------
- Importações de bibliotecas.
- Lista de colunas provenientes do Google Form e seus pares na tabela já validada
- Definições de funções unitárias (cada função só faz uma tarefa).
- Função main() com a sequência das funções a executar.
"""

import pandas as pd
from geopy.geocoders import Nominatim

from limpeza_de_dados.utils import split_coordinates

GEOLOCATOR: Nominatim = Nominatim(user_agent="my-agent")
CONVERSION: dict[str, str] = {
    "Código": "Código",
    "Nº de registo": "Nº de registo",
    "ID da colónia": "ID da colónia",
    "Espécie": "Espécie",
    "Coordenadas": "Coordenadas",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Distrito": "Distrito",
    "Concelho": "Concelho",
    "Freguesia": "Freguesia",
    "Estrutura de nidificação": "Estrutura de nidificação",
    "Número de ninhos": "Nº ninhos ocupados",
    "Altura média (em andares) dos ninhos?": "Altura (andares)",
    "Data da observação?": "Data",
    "Nome": "Nome",
    "Endereço de email": "Email",
    "Comentários": "Comentários",
    "Dados em Falta": "Dados em Falta",
    "Carimbo de data/hora": "Timestamp",
    "Edifício ou estrutura em uso ou abandonada?": "Estado da estrutura",
    "Onde estão os ninhos?": "Local de nidificação",
    "Carregue um vídeo, fotografia ou áudio": "Media",
}


def convert_column_names(
    df: pd.DataFrame, convert_dict: dict[str, str] = CONVERSION
) -> pd.DataFrame:
    df = df.rename(columns=convert_dict)
    return df


def add_detailed_location(row: pd.Series, geolocator=GEOLOCATOR):
    row["Freguesia"] = None
    row["Concelho"] = None
    row["Distrito"] = None

    freguesia_keys = [
        "city_district",
        "village",
        "suburb",
        "neighborhood",
        "neighbourhood",
        "town",
        "borough",
    ]
    concelho_keys = ["city", "municipality", "town"]
    distrito_keys = ["county", "state", "region"]
    try:
        address = geolocator.reverse((row["Latitude"], row["Longitude"])).raw["address"]
        row["Freguesia"] = next(
            (address[key] for key in freguesia_keys if key in address), None
        )
        row["Concelho"] = next(
            (address[key] for key in concelho_keys if key in address), None
        )
        row["Distrito"] = next(
            (address[key] for key in distrito_keys if key in address), None
        )
    except Exception:
        pass
    return row


def convert_datatypes(df: pd.DataFrame):
    if not df.empty:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["Nº ninhos ocupados"] = df["Nº ninhos ocupados"].astype("Int32")
        df["Altura (andares)"] = df["Altura (andares)"].astype("Int32")

    return df


def sort_df(
    df: pd.DataFrame, cols_to_sort: list[str] = ["Data"], ascending: bool = False
):
    if not df.empty:
        df = df.sort_values(by=cols_to_sort, ascending=ascending)
    return df


def add_code_col(df: pd.DataFrame, code_col: str = "Código"):
    df[code_col] = ""
    return df


def add_missing_data_col(df: pd.DataFrame, missing_data_col: str = "Dados em Falta"):
    df[missing_data_col] = False
    return df


def full_clean_data(df_raw: pd.DataFrame):
    df = split_coordinates(df_raw)
    df = convert_datatypes(df)
    df = add_code_col(df)
    df = add_missing_data_col(df)
    df = sort_df(df)

    return df
