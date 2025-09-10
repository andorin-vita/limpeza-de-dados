from copy import deepcopy

import gspread
import leafmap.foliumap as leafmap
import pandas as pd
import streamlit as st
import yaml

from limpeza_de_dados.create_data_flow_to_google import (
    get_drive_info,
    read_spreadsheet_as_df,
)

DESCRIPTION_ABOVE_MAP: str = """ ### Registos de ninhos de andorinhas e andorinhões em Portugal

Pode visualizar todos os registos de andorinhas e andorinhões da Andorin até ao momento

**Obrigado pelo seu contributo!**
"""

DRIVE_INFO: dict[str, str] = get_drive_info()
GC: gspread.client.Client = gspread.authorize(DRIVE_INFO["creds"])

COLS_TO_USE: list[str] = [
    "Espécie",
    "ID da colónia",
    "Latitude",
    "Longitude",
    "Distrito",
    "Concelho",
    "Freguesia",
    "Estrutura de nidificação",
    "Nº ninhos ocupados",
    "Altura (andares)",
    "Estado da estrutura",
    "Local de nidificação",
    "Data",
]

REGIONS_YML: dict[str, str] = {
    "Distrito": "yamls/distritos.yml",
    "Concelho": "yamls/concelhos.yml",
    "Freguesia": "yamls/freguesias.yml",
}


def load_yaml_data(yaml_path: str):
    with open(yaml_path, "r") as file:
        content: list[str] = yaml.safe_load(file)

    if not content:
        content = []
    return content


def get_region_options(yaml_path: str, df: pd.DataFrame, col: str) -> list[str]:
    from_yaml: list[str] = load_yaml_data(yaml_path)
    from_df: list[str] = df[col].dropna().unique()

    return sorted(list(set(from_df) | set(from_yaml)), key=str.casefold)


@st.cache_data
def load_all_region_options(
    df: pd.DataFrame, col_yaml: dict[str, str] = REGIONS_YML
) -> dict[str, str]:
    col_options: dict[str, str] = {}
    for col, yaml_path in col_yaml.items():
        col_options[col] = get_region_options(yaml_path=yaml_path, df=df, col=col)
    return col_options


# === Lê os dados de um CSV ===
@st.cache_data
def load_data_local(file_path):
    return pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")


@st.cache_data
def load_data(
    url: str = DRIVE_INFO["final_form_spreadsheet_url"],
    gc: gspread.client.Client = GC,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    n_nests_col: str = "Nº ninhos ocupados",
    height_col: str = "Altura (andares)",
    date_col: str = "Data",
    colony_id_col: str = "ID da colónia",
    cols_to_use: list[str] = COLS_TO_USE,
):
    df = read_spreadsheet_as_df(url=url, gc=gc)
    df = df[cols_to_use]
    df = df.dropna(subset=[lat_col, lon_col])
    df = df.sort_values(by=[colony_id_col, date_col], ascending=True)
    df = df.drop_duplicates(subset=[colony_id_col], keep="last")
    df[n_nests_col] = df[n_nests_col].astype(pd.Int64Dtype(), errors="ignore")
    df[height_col] = df[height_col].astype(pd.Int64Dtype(), errors="ignore")
    df[colony_id_col] = df[colony_id_col].astype(pd.Int64Dtype(), errors="ignore")
    df[date_col] = pd.to_datetime(df[date_col])
    df["Year"] = df[date_col].dt.year.astype(pd.Int64Dtype(), errors="ignore")
    return df


def create_cluster_map(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
):
    m = leafmap.Map(
        location=[38.736946, -9.142685],
        zoom=6,
        draw_control=False,
        measure_control=False,
        fullscreen_control=False,
        attribution_control=False,
    )

    m.add_points_from_xy(
        df,
        x=lon_col,
        y=lat_col,
        color_column="Espécie",
        layer_name="Points",
        icon_names=["square"],
        spin=False,
        add_legend=True,
        max_cluster_radius=st.session_state["cluster_radius"],
    )
    streamlit_map = m.to_streamlit(height=600, width=1200)
    return streamlit_map


def create_map_sidebar(
    df: pd.DataFrame,
    region_options: dict[str, str],
    species_col: str = "Espécie",
    nest_structure_col: str = "Estrutura de nidificação",
    n_nests_col: str = "Nº ninhos ocupados",
    districts_col: str = "Distrito",
    concelho_col: str = "Concelho",
    freguesia_col: str = "Freguesia",
    year_col: str = "Year",
):

    with st.sidebar:
        with st.form("my_form"):
            species: list[str] = sorted(
                df[species_col].dropna().unique(), key=str.casefold
            )
            nest_structure: list[str] = sorted(
                df[nest_structure_col].dropna().unique(), key=str.casefold
            )
            districts: list[str] = region_options[districts_col]
            n_nests: list[str] = sorted(df[n_nests_col].dropna().unique())
            concelho: list[str] = region_options[concelho_col]
            freguesia: list[str] = region_options[freguesia_col]
            year = sorted(df[year_col].dropna().unique())

            st.title("Filtros")

            species_selected: str = st.multiselect(label="Espécie", options=species)

            n_nests_selected: str = st.slider(
                label="Ninhos",
                min_value=int(n_nests[0]),
                max_value=int(n_nests[-1]),
                value=(int(n_nests[0]), int(n_nests[-1])),
            )

            structure_selected: list[str] = st.multiselect(
                label="Estrutura",
                options=nest_structure,
            )

            districts_selected: list[str] = st.multiselect(
                label="Distrito", options=districts
            )
            concelhos_selected: list[str] = st.multiselect(
                label="Concelho", options=concelho
            )
            freguesia_selected: list[str] = st.multiselect(
                label="Freguesia", options=freguesia
            )

            years_selected: int = st.slider(
                label="Ano de Registo",
                min_value=int(year[0]),
                max_value=int(year[-1]),
                value=(int(year[0]), int(year[-1])),
            )

            cluster_radius: int = st.slider(
                label="Raio",
                min_value=1,
                max_value=200,
                value=25,
            )

            submit_button: bool = st.form_submit_button("Submit", type="primary")

            if submit_button:
                filtered_df: pd.DataFrame = deepcopy(df)
                if n_nests_selected:
                    filtered_df = filtered_df[
                        (filtered_df[n_nests_col] >= n_nests_selected[0])
                        & (filtered_df[n_nests_col] <= n_nests_selected[1])
                    ]
                if structure_selected:
                    filtered_df = filtered_df[nest_structure_col].isin(
                        structure_selected
                    )
                if districts_selected:
                    filtered_df = filtered_df[
                        filtered_df[districts_col].isin(districts_selected)
                    ]
                if species_selected:
                    filtered_df = filtered_df[
                        filtered_df[species_col].isin(species_selected)
                    ]
                if concelhos_selected:
                    filtered_df = filtered_df[
                        filtered_df[concelho_col].isin(concelhos_selected)
                    ]
                if freguesia_selected:
                    filtered_df = filtered_df[
                        filtered_df[freguesia_col].isin(freguesia_selected)
                    ]
                if years_selected:
                    filtered_df = filtered_df[filtered_df["Year"].isin(years_selected)]

                if cluster_radius:
                    st.session_state["cluster_radius"] = cluster_radius

                st.session_state.reload_map = True  # Set the flag to reload the map
                return filtered_df
            else:
                return df


if __name__ == "__main__":
    # === Indica aqui o caminho do teu ficheiro CSV ===
    df: pd.DataFrame = load_data()

    # Inicializações
    if "reload_map" not in st.session_state:
        st.session_state["reload_map"] = True
    if "cluster_radius" not in st.session_state:
        st.session_state["cluster_radius"] = 25

    # Listar regiões
    region_options: dict[str, str] = load_all_region_options(df)

    # Lógica principal
    filtered_df = create_map_sidebar(df=df, region_options=region_options)

    # Só cria o mapa se reload_map flag for True
    if filtered_df.empty:
        st.warning("Nenhum ponto para mostrar no mapa!")
    else:
        st.write(DESCRIPTION_ABOVE_MAP)
        create_cluster_map(df=filtered_df)
        st.session_state.reload_map = False
