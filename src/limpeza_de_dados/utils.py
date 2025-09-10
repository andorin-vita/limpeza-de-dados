from copy import deepcopy

import numpy as np
import pandas as pd


def save_clean_data(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def split_coordinates(
    df: pd.DataFrame,
    original_col: str = "Coordenadas",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    if not df.empty:
        latitudes = []
        longitudes = []
        for _, row in df.iterrows():
            try:
                lat, lon = row[original_col].split(",")
            except Exception:
                lat, lon = (np.nan, np.nan)
            latitudes.append(lat)
            longitudes.append(lon)
        df[lat_col] = pd.to_numeric(latitudes, errors="coerce")
        df[lon_col] = pd.to_numeric(longitudes, errors="coerce")
    return df


def assign_nest_id(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    register_number_col: str = "Nº de registo",
    new_id_col: str = "ID da colónia",
) -> pd.DataFrame:
    df[new_id_col] = (
        df.groupby([lat_col, lon_col])[register_number_col]
        .transform("min")
        .astype("Int64")
    )
    return df


def count_values_by_label(
    df: pd.DataFrame,
    label_col: str,
    filter_data: dict[str, list[str]],
    nest_id_col: str = "ID da colónia",
    drop_duplicated_colony_id: bool = True,
    normalize: bool = True,
    multiplier: int = 100,
) -> pd.Series:

    new_df = deepcopy(df)
    if drop_duplicated_colony_id:
        new_df = new_df.drop_duplicates(subset=[nest_id_col], keep="first")

    mask: pd.Series = pd.Series(True, index=new_df.index)
    for key, values in filter_data.items():
        mask &= new_df[key].isin(values)

    filtered_df: pd.Series = new_df[mask]
    return filtered_df[label_col].value_counts(normalize=normalize) * multiplier


def filter_small_values(series: pd.Series, min_value: float):
    return series[series >= min_value]


def find_new_entries(
    df_raw: pd.DataFrame, df_analysis: pd.DataFrame, unique_col: str = "Timestamp"
):
    return df_raw[~df_raw[unique_col].isin(df_analysis[unique_col])]


def replace_values_in_col(
    df: pd.DataFrame,
    col: str = "Estrutura de nidificação",
    replacement: tuple[str, str] = (
        "Edifício de apoio (arrecadação, garagem, etc.)",
        "Ed. de apoio",
    ),
) -> pd.DataFrame:
    df[col] = df[col].replace({replacement[0]: replacement[1]})
    return df


def trim_empty_spaces(df: pd.DataFrame, cols_to_trim: list[str] | None = None):
    if not cols_to_trim:
        cols_to_trim = df.columns
    df[cols_to_trim] = df[cols_to_trim].apply(
        lambda x: x.str.strip() if x.dtype == "object" else x
    )
    return df
