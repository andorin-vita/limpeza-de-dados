from copy import deepcopy

import gspread
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from unidecode import unidecode

from limpeza_de_dados.create_data_flow_to_google import (
    get_drive_info,
    read_spreadsheet_as_df,
)

DESCRIPTION_ABOVE_MAP: str = """"""

MIN_RADIUS_PIXELS: int = 2
MAX_RADIUS_PIXELS: int = 10
ZOOM_RADIUS_PIXELS: int = 500
HEIGHT: int = 600
WIDTH: int = 900
VERTICAL_LEGEND: int = 190
HORIZONTAL_LEGEND: int = 650

# Campos para os filtros e/ou mapa
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

# Campos para tooltip
COLS_TOOLTIP: list[str] = [
    "Espécie",
    "ID da colónia",
    "Coordenadas",
    "Distrito",
    "Concelho",
    "Freguesia",
    "Estrutura de nidificação",
    "Nº ninhos ocupados",
    "Altura (andares)",
    "Estado da estrutura",
    "Local de nidificação",
]


def create_cmap_manual():
    return {
        "Andorinhão-pálido": {"rgb": (200, 0, 0)},
        "Andorinha-das-barreiras": {"rgb": (255, 87, 0)},
        "Andorinhão-preto": {"rgb": (255, 235, 59)},
        "Andorinha-dáurica": {"rgb": (124, 179, 66)},
        "Andorinhão-cafre": {"rgb": (135, 206, 250)},
        "Andorinha-dos-beirais": {"rgb": (171, 71, 188)},
        "Andorinhão-real": {"rgb": (183, 109, 84)},
        "Andorinha-das-chaminés": {"rgb": (66, 66, 66)},
        "Andorinha-das-rochas": {"rgb": (25, 118, 210)},
        "Andorinhão-da-serra": {"rgb": (255, 192, 203)},
        # Adicionar abaixo a nova espécie de andorinha
        # "Andorinha-xxx": {"rgb": (179, 136, 102)}
    }

# Não mexer
DRIVE_INFO: dict[str, str] = get_drive_info()
GC: gspread.client.Client = gspread.authorize(DRIVE_INFO["creds"])


def get_region_options_from_geographies(
    geographies_df: pd.DataFrame, df: pd.DataFrame, col: str
) -> list[str]:
    """Get region options from both the main dataframe and the geographies Google Sheet"""
    from_geographies: list[str] = geographies_df[col].dropna().unique().tolist()
    return from_geographies


def create_cmap(df: pd.DataFrame, color_col: str):
    unique_species = df[color_col].unique()
    colors = cm.tab20(np.linspace(0, 1, len(unique_species)))
    colors = (colors[:, :3] * 255).round().astype(int).tolist()

    # para fazez color map manual (comentar a linha abaixo)
    # color_map = {'Andorinha-dos-beirais': [148,1,1],
    # 'Andorinha-das-chaminés': [148,1,1]}

    color_map = {species: color for species, color in zip(unique_species, colors)}
    return color_map


@st.cache_data
def load_all_region_options(
    df: pd.DataFrame, geographies_df: pd.DataFrame
) -> dict[str, list[str]]:
    """Load region options for Distrito, Concelho, and Freguesia from Google Sheets and main dataframe"""
    region_options = {
        "Distrito": get_region_options_from_geographies(
            geographies_df=geographies_df, df=df, col="Distrito"
        ),
        "Concelho": get_region_options_from_geographies(
            geographies_df=geographies_df, df=df, col="Concelho"
        ),
        "Freguesia": get_region_options_from_geographies(
            geographies_df=geographies_df, df=df, col="Freguesia"
        ),
    }

    return region_options


# === Lê os dados de um CSV ===
@st.cache_data
def load_data_local(file_path):
    return pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")


@st.cache_data
def load_geographies_data(
    url: str = DRIVE_INFO["geographies_url"], _gc: gspread.client.Client = GC
) -> pd.DataFrame:
    """Load geographical data from Google Sheets with columns: dicofre, districto, concelho, freguesia"""
    return read_spreadsheet_as_df(url=url, gc=_gc)


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
    species_col: str = "Espécie",
    remove_if_not_identified: bool = True,
):
    df = read_spreadsheet_as_df(url=url, gc=gc)
    df = df[cols_to_use]
    if df.empty:
        return df
    df = df.dropna(subset=[lat_col, lon_col])
    df = df.sort_values(by=[colony_id_col, date_col], ascending=True)
    df = df.drop_duplicates(subset=[colony_id_col], keep="last")
    df[n_nests_col] = df[n_nests_col].astype(pd.Int64Dtype(), errors="ignore")
    df[height_col] = df[height_col].astype(pd.Int64Dtype(), errors="ignore")
    df[colony_id_col] = df[colony_id_col].astype(pd.Int64Dtype(), errors="ignore")
    df[date_col] = pd.to_datetime(df[date_col])
    df["Year"] = df[date_col].dt.year.astype(pd.Int64Dtype(), errors="ignore")
    if remove_if_not_identified:
        df = df[~df[species_col].str.contains("não ide", case=False, na=True)]
    # substituir espaços em branco por espaço em estrutura de nidificação
    df["Estrutura de nidificação"] = (
        df["Estrutura de nidificação"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return df


def create_point_map(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    color_col: str = "color",
    legend_col: str = "Espécie",
    center_lat: float = 39.69484,
    center_lon: float = -8.13031,
    zoom: int = 6,
):

    # ---- SET VIEW STATE ----
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    # ---- DECK LAYER ----
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[{}, {}]".format(lon_col, lat_col),
        get_fill_color=color_col,
        get_radius=ZOOM_RADIUS_PIXELS,
        pickable=True,
        radius_min_pixels=MIN_RADIUS_PIXELS,
        radius_max_pixels=MAX_RADIUS_PIXELS,
    )

    # Render the map and legend side by side
    st.pydeck_chart(
        pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={
                "html": "<div style='font-size:12px; line-height:1.4;'>"
                + "<br/>".join(
                    [
                        f"{col}: {{{col}}}"
                        for col in COLS_TOOLTIP
                        if col not in [color_col, "Data"]
                    ]
                )
                + "</div>"
            },
        ),
        height=HEIGHT,
        width=WIDTH,
        use_container_width=True,
    )


def get_cascading_options(
    geographies_df: pd.DataFrame,
    districts_selected: list[str] = None,
    concelhos_selected: list[str] = None,
) -> dict[str, list[str]]:
    """Get cascading filter options based on current selections"""

    # Always show all districts
    all_districts = sorted(geographies_df["Distrito"].dropna().unique(), key=unidecode)

    # Filter concelhos based on selected districts
    if districts_selected:
        available_concelhos = (
            geographies_df[geographies_df["Distrito"].isin(districts_selected)][
                "Concelho"
            ]
            .dropna()
            .unique()
        )
    else:
        available_concelhos = geographies_df["Concelho"].dropna().unique()

    available_concelhos = sorted(available_concelhos, key=unidecode)

    # Filter freguesias based on selected concelhos (and districts if selected)
    if concelhos_selected:
        filtered_geo = geographies_df[
            geographies_df["Concelho"].isin(concelhos_selected)
        ]
        if districts_selected:
            filtered_geo = filtered_geo[
                filtered_geo["Distrito"].isin(districts_selected)
            ]
        available_freguesias = filtered_geo["Freguesia"].dropna().unique()
    elif districts_selected:
        available_freguesias = (
            geographies_df[geographies_df["Distrito"].isin(districts_selected)][
                "Freguesia"
            ]
            .dropna()
            .unique()
        )
    else:
        available_freguesias = geographies_df["Freguesia"].dropna().unique()

    available_freguesias = sorted(available_freguesias, key=unidecode)

    return {
        "Distrito": all_districts,
        "Concelho": available_concelhos,
        "Freguesia": available_freguesias,
    }


def create_map_sidebar(
    df: pd.DataFrame,
    geographies_df: pd.DataFrame,
    species_cmap: dict[str, dict[str, str]],
    species_col: str = "Espécie",
    nest_structure_col: str = "Estrutura de nidificação",
    n_nests_col: str = "Nº ninhos ocupados",
    districts_col: str = "Distrito",
    concelho_col: str = "Concelho",
    freguesia_col: str = "Freguesia",
    year_col: str = "Year",
):

    with st.sidebar:
        species: list[str] = sorted(df[species_col].dropna().unique(), key=unidecode)
        nest_structure: list[str] = sorted(
            df[nest_structure_col].dropna().unique(), key=unidecode
        )
        n_nests: list[str] = sorted(df[n_nests_col].dropna().unique())
        year = sorted(df[year_col].dropna().unique())

        st.title("Filtros")

        # Display color legend
        st.write("**Legenda de Cores:**")
        for species, data in species_cmap.items():
            rgb = data["rgb"]
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {hex_color}; border: 1px solid #ccc; margin-right: 10px;"></div>{species}</div>',
                unsafe_allow_html=True,
            )

        species_selected: str = st.multiselect(
            label="Espécie", options=species_cmap.keys(), key="species-filter"
        )

        n_nests_selected: tuple = st.slider(
            label="Ninhos",
            min_value=int(n_nests[0]),
            max_value=int(n_nests[-1]),
            value=(int(n_nests[0]), int(n_nests[-1])),
            key="nests_filter",
        )

        structure_selected: list[str] = st.multiselect(
            label="Estrutura", options=nest_structure, key="structure_filter"
        )

        # Geographic filters with cascading logic
        districts_selected: list[str] = st.multiselect(
            label="Distrito",
            options=sorted(geographies_df["Distrito"].dropna().unique(), key=unidecode),
            key="distrito_filter",
        )

        # Get dynamic options based on distrito selection
        cascading_options = get_cascading_options(
            geographies_df,
            districts_selected=districts_selected if districts_selected else None,
        )

        concelhos_selected: list[str] = st.multiselect(
            label="Concelho",
            options=cascading_options["Concelho"],
            key="concelho_filter",
        )

        # Update cascading options again based on concelho selection
        cascading_options = get_cascading_options(
            geographies_df,
            districts_selected=districts_selected if districts_selected else None,
            concelhos_selected=concelhos_selected if concelhos_selected else None,
        )

        freguesia_selected: list[str] = st.multiselect(
            label="Freguesia",
            options=cascading_options["Freguesia"],
            key="freguesia_filter",
        )

        years_selected: tuple = st.slider(
            label="Ano de Registo",
            min_value=int(year[0]),
            max_value=int(year[-1]),
            value=(int(year[0]), int(year[-1])),
            key="year_filter",
        )

        # Apply filters immediately (no submit button needed)
        filtered_df: pd.DataFrame = deepcopy(df)

        if species_selected:
            filtered_df = filtered_df[filtered_df[species_col].isin(species_selected)]

        if n_nests_selected != (int(n_nests[0]), int(n_nests[-1])):
            filtered_df = filtered_df[
                (filtered_df[n_nests_col] >= n_nests_selected[0])
                & (filtered_df[n_nests_col] <= n_nests_selected[1])
            ]

        if structure_selected:
            filtered_df = filtered_df[
                filtered_df[nest_structure_col].isin(structure_selected)
            ]

        if districts_selected:
            filtered_df = filtered_df[
                filtered_df[districts_col].isin(districts_selected)
            ]

        if concelhos_selected:
            filtered_df = filtered_df[
                filtered_df[concelho_col].isin(concelhos_selected)
            ]

        if freguesia_selected:
            filtered_df = filtered_df[
                filtered_df[freguesia_col].isin(freguesia_selected)
            ]

        if years_selected != (int(year[0]), int(year[-1])):
            filtered_df = filtered_df[
                (filtered_df["Year"] >= years_selected[0])
                & (filtered_df["Year"] <= years_selected[1])
            ]

        return filtered_df


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # === Indica aqui o caminho do teu ficheiro CSV ===
    df: pd.DataFrame = load_data()

    if df.empty:
        spreadsheet_url = DRIVE_INFO["final_form_spreadsheet_url"]
        st.warning(f"No data in the spreadsheet: {spreadsheet_url}")
        st.stop()

    # Load geographical data from Google Sheets
    geographies_df: pd.DataFrame = load_geographies_data()

    # Inicializações
    if "reload_map" not in st.session_state:
        st.session_state["reload_map"] = True

    # Listar regiões
    region_options = load_all_region_options(df, geographies_df)

    # Adicionar cores
    color_map = create_cmap_manual()
    df["color"] = [color_map[key]["rgb"] for key in df["Espécie"]]

    # Lógica principal
    filtered_df = create_map_sidebar(
        df=df, geographies_df=geographies_df, species_cmap=color_map
    )

    # Só cria o mapa se reload_map flag for True
    if filtered_df.empty:
        st.warning("Nenhum ponto para mostrar no mapa!")
    else:
        st.write(DESCRIPTION_ABOVE_MAP)
        create_point_map(df=filtered_df)
        st.session_state.reload_map = False
