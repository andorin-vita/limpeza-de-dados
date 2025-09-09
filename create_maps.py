import pandas as pd
import streamlit as st
from copy import deepcopy
import gspread
import yaml
import pydeck as pdk
import matplotlib.cm as cm
import numpy as np
from unidecode import unidecode


from limpeza_de_dados.create_data_flow_to_google import (
    read_spreadsheet_as_df, get_drive_info)


DESCRIPTION_ABOVE_MAP: str = """"""

MIN_RADIUS_PIXELS: int = 2
MAX_RADIUS_PIXELS: int = 10
ZOOM_RADIUS_PIXELS: int = 500
HEIGHT: int = 600
WIDTH: int = 900
VERTICAL_LEGEND: int = 350
HORIZONTAL_LEGEND: int =650

# Campos para os filtros e/ou mapa
COLS_TO_USE: list[str] = ['Espécie',
                          'ID da colónia',
                          'Latitude',
                          'Longitude',
                           'Distrito',
                           'Concelho',
                           'Freguesia',
                           'Estrutura de nidificação',
                           'Nº ninhos ocupados',
                           'Altura (andares)',
                           'Estado da estrutura',
                           'Local de nidificação',
                           'Data'
]

# Campos para tooltip
COLS_TOOLTIP: list[str] = ['Espécie',
                          'ID da colónia',
                           'Distrito',
                           'Concelho',
                           'Freguesia',
                           'Estrutura de nidificação',
                           'Nº ninhos ocupados',
                           'Altura (andares)',
                           'Estado da estrutura',
                           'Local de nidificação',
]

# Não mexer
DRIVE_INFO: dict[str, str] = get_drive_info()
GC: gspread.client.Client = gspread.authorize(DRIVE_INFO['creds'])

# Não mexer
REGIONS_YML: dict[str, str] = {
    'Distrito': 'yamls/distritos.yml',
    'Concelho': 'yamls/concelhos.yml',
    'Freguesia': 'yamls/freguesias.yml',
}

def load_yaml_data(yaml_path: str):
    with open(yaml_path, 'r') as file:
        content: list[str] = yaml.safe_load(file)

    if not content:
        content = []
    return content

def get_region_options(yaml_path: str, df: pd.DataFrame, col: str) -> list[str]:
    from_yaml: list[str] = load_yaml_data(yaml_path)
    from_df: list[str] = df[col].dropna().unique()

    return sorted(list(set(from_df) | set(from_yaml)), key=unidecode)

def create_cmap(df: pd.DataFrame,
                           color_col: str):
    unique_species = df[color_col].unique()
    colors = cm.tab20(np.linspace(0, 1, len(unique_species)))
    colors = (colors[:, :3] * 255).round().astype(int).tolist()

    # para fazez color map manual (comentar a linha abaixo)
    #color_map = {'Andorinha-dos-beirais': [148,1,1], 
    # 'Andorinha-das-chaminés': [148,1,1]}

    color_map = {species: color for species, color in zip(unique_species, colors)}
    return color_map

def create_cmap_manual():
    return {
    "Andorinhão-pálido":   {"emoji": "🟥", "rgb": (244, 67, 54)},
    "Andorinha-das-barreiras": {"emoji": "🟧", "rgb": (255, 152, 0)},
    "Andorinhão-preto": {"emoji": "🟨", "rgb": (255, 204, 50)},
    "Andorinha-dáurica": {"emoji": "🟩", "rgb": (124, 179, 66)},
    "Andorinhão-cafre": {"emoji": "🟦", "rgb": (25, 118, 210)},
    "Andorinha-dos-beirais": {"emoji": "🟪", "rgb": (171, 71, 188)},
    "Andorinhão-real": {"emoji": "🟫", "rgb": (183, 109, 84)},
    "Andorinha-das-chaminés": {"emoji": "⬛", "rgb": (66, 66, 66)},
    "Andorinha-das-rochas": {"emoji": "⬜", "rgb": (225, 225, 225)},
    "Andorinhão-da-serra": {"emoji": "🏻", "rgb": (250, 220, 184)},  
    # Adicionar abaixo a nova espécie de andorinha
    #"Andorinha-xxx": {"emoji": "🏽", "rgb": (179, 136, 102)}
    }


@st.cache_data
def load_all_region_options(df: pd.DataFrame, col_yaml: dict[str, str]=REGIONS_YML)->dict[str, str]:
    col_options: dict[str, str] = {}
    for col, yaml_path in col_yaml.items():
        col_options[col] = get_region_options(yaml_path=yaml_path, df=df, col=col)
    return col_options

# === Lê os dados de um CSV ===
@st.cache_data
def load_data_local(file_path):
    return pd.read_csv(file_path,  encoding='utf-8', encoding_errors='replace')

@st.cache_data
def load_data(url: str = DRIVE_INFO['final_form_spreadsheet_url'],
              gc: gspread.client.Client = GC,
              lat_col: str = 'Latitude',
              lon_col: str = 'Longitude',
              n_nests_col: str = 'Nº ninhos ocupados',
              height_col: str = 'Altura (andares)',
              date_col: str = 'Data',
              colony_id_col: str = 'ID da colónia',
              cols_to_use: list[str] = COLS_TO_USE,
              species_col: str = 'Espécie',
              remove_if_not_identified: bool = True
              ):
    df = read_spreadsheet_as_df(url=url, gc=gc)
    df = df[cols_to_use]
    df = df.dropna(subset=[lat_col, lon_col])
    df = df.sort_values(by=[colony_id_col, date_col], ascending=True)
    df = df.drop_duplicates(subset=[colony_id_col], keep='last')
    df[n_nests_col] = df[n_nests_col].astype(pd.Int64Dtype(), errors='ignore')
    df[height_col] = df[height_col].astype(pd.Int64Dtype(), errors='ignore')
    df[colony_id_col] = df[colony_id_col].astype(pd.Int64Dtype(), errors='ignore')
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year'] = df[date_col].dt.year.astype(pd.Int64Dtype(), errors='ignore')
    if remove_if_not_identified:
        df = df[~df[species_col].str.contains('não ide', case=False, na=True)]
    return df

def create_point_map(df: pd.DataFrame,
                     lat_col: str = 'Latitude',
                     lon_col: str = 'Longitude',
                     color_col: str = 'color',
                     legend_col: str= 'Espécie',
                     center_lat: float = 39.69484,
                     center_lon: float = -8.13031,
                     zoom: int = 6
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
        get_position='[{}, {}]'.format(lon_col, lat_col),
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
                + "<br/>".join([f"{col}: {{{col}}}" for col in COLS_TOOLTIP if col not in [color_col, 'Data']])
                + "</div>"
            },
        ),
    height=HEIGHT,
    width=WIDTH,
    use_container_width=True,
    )




def create_map_sidebar(df: pd.DataFrame,
                       region_options: dict[str, str],
                       species_cmap: dict[str, dict[str, str]],
                       species_col: str = 'Espécie',
                       nest_structure_col: str = 'Estrutura de nidificação',
                       n_nests_col: str = 'Nº ninhos ocupados',
                       districts_col: str = 'Distrito',
                       concelho_col: str = 'Concelho',
                       freguesia_col: str = 'Freguesia',
                       year_col: str = 'Year'):
    with st.sidebar:
        with st.form("my_form"):
            species: list[str] = sorted(df[species_col].dropna().unique(), key=unidecode)
            nest_structure: list[str] = sorted(df[nest_structure_col].dropna().unique(), key=unidecode)
            districts: list[str] = region_options[districts_col]
            n_nests: list[str] = sorted(df[n_nests_col].dropna().unique())
            concelho: list[str] = region_options[concelho_col]
            freguesia: list[str] = region_options[freguesia_col]
            year = sorted(df[year_col].dropna().unique())

            st.title("Filtros")

            fmt = {key: f"{value['emoji']} {key}" for key, value in species_cmap.items()}
            species_selected: str = st.pills(label="Espécie",
                                             options=species_cmap.keys(),
                                             format_func=lambda x: fmt[x],
                                             selection_mode='multi',
                                             key='species-filter'
                                                 )

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
                label="Distrito",
                options=districts
            )
            concelhos_selected: list[str] = st.multiselect(
                label="Concelho",
                options=concelho
            )
            freguesia_selected: list[str] = st.multiselect(
                label="Freguesia",
                options=freguesia
            )

            years_selected: int = st.slider(
                label="Ano de Registo",
                min_value=int(year[0]),
                max_value=int(year[-1]),
                value=(int(year[0]), int(year[-1])),
            )

            submit_button: bool = st.form_submit_button("Submit", type="primary")

            if submit_button:
                filtered_df: pd.DataFrame = deepcopy(df)
                if n_nests_selected:
                    filtered_df = filtered_df[
                        (filtered_df[n_nests_col] >= n_nests_selected[0]) &
                        (filtered_df[n_nests_col] <= n_nests_selected[1])
                    ]
                if structure_selected:
                    filtered_df = filtered_df[nest_structure_col].isin(structure_selected)
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
                    filtered_df = filtered_df[filtered_df['Year'].isin(years_selected)]
                
                st.session_state.reload_map = True  # Set the flag to reload the map
                return filtered_df
            else:
                return df

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
        .st-key-species-filter button div[data-testid="stMarkdownContainer"] p {
            font-size: 11px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # === Indica aqui o caminho do teu ficheiro CSV ===
    df: pd.DataFrame = load_data()

    # Inicializações
    if 'reload_map' not in st.session_state:
        st.session_state['reload_map'] = True

    # Listar regiões
    region_options: dict[str, str] = load_all_region_options(df)

    # Adicionar cores
    color_map = create_cmap_manual()
    df['color'] = [color_map[key]["rgb"] for key in df['Espécie']]

    # Lógica principal
    filtered_df = create_map_sidebar(df=df, region_options=region_options, species_cmap=color_map)

    # Só cria o mapa se reload_map flag for True
    #if filtered_df.empty:
    #    st.warning("Nenhum ponto para mostrar no mapa!")
    if st.session_state.reload_map:
        st.write(DESCRIPTION_ABOVE_MAP)
        create_point_map(df=filtered_df)
        st.session_state.reload_map = False