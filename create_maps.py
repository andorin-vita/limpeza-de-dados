import pandas as pd
import streamlit as st
from copy import deepcopy
import gspread
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
VERTICAL_LEGEND: int = 190
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
                          'Coordenadas',
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



def get_region_options_from_geographies(geographies_df: pd.DataFrame, df: pd.DataFrame, col: str) -> list[str]:
    """Get region options from both the main dataframe and the geographies Google Sheet"""
    from_geographies: list[str] = geographies_df[col].dropna().unique().tolist()
    return from_geographies

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

@st.cache_data
def load_all_region_options(df: pd.DataFrame, geographies_df: pd.DataFrame) -> dict[str, list[str]]:
    """Load region options for Distrito, Concelho, and Freguesia from Google Sheets and main dataframe"""
    region_options = {
        'Distrito': get_region_options_from_geographies(geographies_df=geographies_df, df=df, col='Distrito'),
        'Concelho': get_region_options_from_geographies(geographies_df=geographies_df, df=df, col='Concelho'),
        'Freguesia': get_region_options_from_geographies(geographies_df=geographies_df, df=df, col='Freguesia')
    }
    
    return region_options

# === Lê os dados de um CSV ===
@st.cache_data
def load_data_local(file_path):
    return pd.read_csv(file_path,  encoding='utf-8', encoding_errors='replace')

@st.cache_data
def load_geographies_data(url: str = DRIVE_INFO['geographies_url'],
                         _gc: gspread.client.Client = GC) -> pd.DataFrame:
    """Load geographical data from Google Sheets with columns: dicofre, districto, concelho, freguesia"""
    return read_spreadsheet_as_df(url=url, gc=_gc)

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
    # substituir espaços em branco por espaço em estrutura de nidificação
    df["Estrutura de nidificação"] = df["Estrutura de nidificação"].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df


def create_point_map(df: pd.DataFrame,
                     color_map: dict[str, list[int]],
                     lat_col: str = 'Latitude',
                     lon_col: str = 'Longitude',
                     color_col: str = 'color',
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
    col1, col2 = st.columns([4, 1])
    with col1:
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
    use_container_width=False,
)
    with col2:
        for species, color in color_map.items():
            st.markdown(
                f'<span style="color: rgb{tuple(color)}; font-size: 11px;">■ {species}</span>',
                unsafe_allow_html=True
            )

def create_point_map_abs_pos(df: pd.DataFrame,
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

    df_colors: pd.DataFrame = df.drop_duplicates(legend_col)
    # Build HTML for overlay legend
    legend_html = f"""<div style='position:absolute; top:{VERTICAL_LEGEND}px; 
    left:{HORIZONTAL_LEGEND}px; background:white; 
    padding:8px; z-index:999; border-radius:4px; box-shadow:0 2px 6px rgba(0,0,0,0.2);'>"""
    for _, row in df_colors.iterrows():
        color_str = f"rgb({row[color_col][0]},{row[color_col][1]},{row[color_col][2]})"
        legend_html += f"<div style='display:flex; align-items:center; gap:6px; margin-bottom:2px;'>"
        legend_html += f"<div style='width:16px; height:16px; background:{color_str};'></div>"
        legend_html += f"<span>{row[legend_col]}</span></div>"
    legend_html += "</div>"

    # Use st.markdown with unsafe_allow_html to float the legend
    st.markdown(legend_html, unsafe_allow_html=True)

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
    use_container_width=False,
    )


def get_cascading_options(geographies_df: pd.DataFrame, 
                         districts_selected: list[str] = None,
                         concelhos_selected: list[str] = None) -> dict[str, list[str]]:
    """Get cascading filter options based on current selections"""
    
    # Always show all districts
    all_districts = sorted(geographies_df['Distrito'].dropna().unique(), key=unidecode)
    
    # Filter concelhos based on selected districts
    if districts_selected:
        available_concelhos = geographies_df[
            geographies_df['Distrito'].isin(districts_selected)
        ]['Concelho'].dropna().unique()
    else:
        available_concelhos = geographies_df['Concelho'].dropna().unique()
    
    available_concelhos = sorted(available_concelhos, key=unidecode)
    
    # Filter freguesias based on selected concelhos (and districts if selected)
    if concelhos_selected:
        filtered_geo = geographies_df[geographies_df['Concelho'].isin(concelhos_selected)]
        if districts_selected:
            filtered_geo = filtered_geo[filtered_geo['Distrito'].isin(districts_selected)]
        available_freguesias = filtered_geo['Freguesia'].dropna().unique()
    elif districts_selected:
        available_freguesias = geographies_df[
            geographies_df['Distrito'].isin(districts_selected)
        ]['Freguesia'].dropna().unique()
    else:
        available_freguesias = geographies_df['Freguesia'].dropna().unique()
    
    available_freguesias = sorted(available_freguesias, key=unidecode)
    
    return {
        'Distrito': all_districts,
        'Concelho': available_concelhos,
        'Freguesia': available_freguesias
    }

def create_map_sidebar(df: pd.DataFrame,
                       geographies_df: pd.DataFrame,
                       species_col: str = 'Espécie',
                       nest_structure_col: str = 'Estrutura de nidificação',
                       n_nests_col: str = 'Nº ninhos ocupados',
                       districts_col: str = 'Distrito',
                       concelho_col: str = 'Concelho',
                       freguesia_col: str = 'Freguesia',
                       year_col: str = 'Year'):
    
    with st.sidebar:
        species: list[str] = sorted(df[species_col].dropna().unique(), key=unidecode)
        nest_structure: list[str] = sorted(df[nest_structure_col].dropna().unique(), key=unidecode)
        n_nests: list[str] = sorted(df[n_nests_col].dropna().unique())
        year = sorted(df[year_col].dropna().unique())

        st.title("Filtros")

        species_selected: list[str] = st.multiselect(
            label="Espécie",
            options=species,
            key="species_filter"
        )

        n_nests_selected: tuple = st.slider(
            label="Ninhos",
            min_value=int(n_nests[0]),
            max_value=int(n_nests[-1]),
            value=(int(n_nests[0]), int(n_nests[-1])),
            key="nests_filter"
        )

        structure_selected: list[str] = st.multiselect(
            label="Estrutura",
            options=nest_structure,
            key="structure_filter"
        )

        # Geographic filters with cascading logic
        districts_selected: list[str] = st.multiselect(
            label="Distrito",
            options=sorted(geographies_df['Distrito'].dropna().unique(), key=unidecode),
            key="distrito_filter"
        )
        
        # Get dynamic options based on distrito selection
        cascading_options = get_cascading_options(
            geographies_df, 
            districts_selected=districts_selected if districts_selected else None
        )
        
        concelhos_selected: list[str] = st.multiselect(
            label="Concelho",
            options=cascading_options['Concelho'],
            key="concelho_filter"
        )
        
        # Update cascading options again based on concelho selection
        cascading_options = get_cascading_options(
            geographies_df, 
            districts_selected=districts_selected if districts_selected else None,
            concelhos_selected=concelhos_selected if concelhos_selected else None
        )
        
        freguesia_selected: list[str] = st.multiselect(
            label="Freguesia",
            options=cascading_options['Freguesia'],
            key="freguesia_filter"
        )

        years_selected: tuple = st.slider(
            label="Ano de Registo",
            min_value=int(year[0]),
            max_value=int(year[-1]),
            value=(int(year[0]), int(year[-1])),
            key="year_filter"
        )

        # Apply filters immediately (no submit button needed)
        filtered_df: pd.DataFrame = deepcopy(df)
        
        if species_selected:
            filtered_df = filtered_df[filtered_df[species_col].isin(species_selected)]
        
        if n_nests_selected != (int(n_nests[0]), int(n_nests[-1])):
            filtered_df = filtered_df[
                (filtered_df[n_nests_col] >= n_nests_selected[0]) &
                (filtered_df[n_nests_col] <= n_nests_selected[1])
            ]
        
        if structure_selected:
            filtered_df = filtered_df[filtered_df[nest_structure_col].isin(structure_selected)]
        
        if districts_selected:
            filtered_df = filtered_df[filtered_df[districts_col].isin(districts_selected)]
        
        if concelhos_selected:
            filtered_df = filtered_df[filtered_df[concelho_col].isin(concelhos_selected)]
        
        if freguesia_selected:
            filtered_df = filtered_df[filtered_df[freguesia_col].isin(freguesia_selected)]
        
        if years_selected != (int(year[0]), int(year[-1])):
            filtered_df = filtered_df[
                (filtered_df['Year'] >= years_selected[0]) &
                (filtered_df['Year'] <= years_selected[1])
            ]
        
        return filtered_df

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # === Indica aqui o caminho do teu ficheiro CSV ===
    df: pd.DataFrame = load_data()

    # Load geographical data from Google Sheets
    geographies_df: pd.DataFrame = load_geographies_data()

    # Inicializações
    if 'reload_map' not in st.session_state:
        st.session_state['reload_map'] = True

    # Listar regiões
    region_options = load_all_region_options(df, geographies_df)

    # Adicionar cores
    color_map = create_cmap(df=df, color_col="Espécie")
    df['color'] = df['Espécie'].map(color_map)

    # Lógica principal
    filtered_df = create_map_sidebar(df=df, geographies_df=geographies_df)

    # Só cria o mapa se reload_map flag for True
    if filtered_df.empty:
        st.warning("Nenhum ponto para mostrar no mapa!")
    else:
        st.write(DESCRIPTION_ABOVE_MAP)
        create_point_map_abs_pos(df=filtered_df)
        st.session_state.reload_map = False