import streamlit as st
import pandas as pd
import numpy as np

from limpeza_de_dados.create_sidebar import (
    get_submission_to_validate, create_filter_options,
    create_filter_parameters, apply_all_filters,
    show_selected_row_as_table, show_results)

from limpeza_de_dados.create_map import create_full_map
from limpeza_de_dados.create_data_flow_to_google import (
    get_drive_info, read_spreadsheet_as_df, append_row_to_spreadsheet
)
import gspread



from limpeza_de_dados.utils import find_new_entries
from limpeza_de_dados.clean_google_form_data import full_clean_data, CONVERSION

DRIVE_INFO: dict[str, str] = get_drive_info()
GC: gspread.client.Client = gspread.authorize(DRIVE_INFO['creds'])

@st.cache_data
def load_data_local(csv_path="data/form_submissions.csv"):
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Carregadas {len(df)} submissões de '{csv_path}'")
        return df
    except FileNotFoundError:
        st.error(f"Ficheiro CSV não encontrado em: {csv_path}")
        return pd.DataFrame()  # Return empty DataFrame as fallback


@st.cache_data
def load_data_submissions(df_analysis: pd.DataFrame,
                          url: str = DRIVE_INFO['original_form_spreadsheet_url'],
                          gc: gspread.client.Client = GC,
                          ):
    df: pd.DataFrame = read_spreadsheet_as_df(url=url, gc=gc)
    df = full_clean_data(df_raw=df)
    return find_new_entries(df_raw=df, df_analysis=df_analysis)


@st.cache_data
def load_data_analysis(url: str = DRIVE_INFO['final_form_spreadsheet_url'],
                       gc: gspread.client.Client = GC):
    return read_spreadsheet_as_df(url=url, gc=gc)

def main(lat_col: str = 'Latitude', 
         lon_col: str = 'Longitude',
         id_col: str = 'Nº de registo',
         colony_id_col: str = 'ID da colónia',
         coords_col: str = 'Coordenadas',
         distance_col: str = 'Distância (m)',
         n_timestamp_col: str = 'Timestamp',
         height_col: str = 'Altura (andares)',
         date_col: str = 'Data',
         species_col: str = 'Espécie',
         nest_structure_col: str = 'Estrutura de nidificação',
         email_col: str = 'Email',
         ):
    
    # Carregar dados
    df_validated = load_data_analysis()
    df_submissions: pd.DataFrame = load_data_submissions(df_analysis=df_validated)
    # Submissão a validar
    selected_row = get_submission_to_validate(df_submissions, n_timestamp_col)

    # Inicializar variáveis
    if 'continue_showing_results' not in st.session_state:
        st.session_state['continue_showing_results'] = False

    if 'map_selected_point' not in st.session_state:
        st.session_state['map_selected_point'] = None

    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = df_validated

    if 'submission_id' not in st.session_state:
        st.session_state['submission_id'] = max(df_validated[id_col]) + 1

    if 'new_lat' not in st.session_state:
        st.session_state['new_lat'] = selected_row[lat_col]

    if 'new_lon' not in st.session_state:
        st.session_state['new_lon'] = selected_row[lon_col]

    if 'new_colony_id' not in st.session_state:
        st.session_state['new_colony_id'] = st.session_state['submission_id']



    selected_row[lat_col] = st.session_state['new_lat']
    selected_row[lon_col] = st.session_state['new_lon']
    selected_row[coords_col] = f'{st.session_state['new_lat']}, {st.session_state['new_lon']}'
    selected_row[id_col] = st.session_state['submission_id']
    selected_row[colony_id_col] = st.session_state['new_colony_id']

    # Reorder selected_row
    selected_row = selected_row.reindex(list(CONVERSION.values()))

    st.title("Ferramenta de Validação de Submissões")

    # Sidebar
    st.sidebar.title("Submissões não validadas")
    st.sidebar.write("Usa os filtros para encontrar submissões similares à submissão selecionada.")
    st.sidebar.markdown("### Filtros")
    filters = create_filter_options()
    filter_params = create_filter_parameters(filters)
    apply_filters = st.sidebar.button("Aplicar filtros")

    if apply_filters:
        st.session_state['filtered_df'] = apply_all_filters(
        df_to_filter = df_validated,
        selected_row = selected_row,
        filters = filters,
        filter_params = filter_params,
        lat_col= lat_col,
        lon_col = lon_col,
        distance_col=distance_col,
        height_col = height_col,
        date_col = date_col,
        species_col= species_col,
        nest_structure_col=nest_structure_col,
        email_col=email_col
        )
        # Garantir que os resultados são sempre apresentados
        st.session_state['continue_showing_results'] = True
        apply_filters = False

    if st.session_state['continue_showing_results']:       
        col1, col2 = st.columns(2)
        with col1:
            st.session_state['edited_submission'] = show_selected_row_as_table(selected_row)
        with col2:
            st.session_state['map_selected_point'] = create_full_map(selected_row, st.session_state['filtered_df'])
            if st.session_state['map_selected_point']:
                update_coords_button = st.button(label='É a mesma colónia que a submissão?')
                st.write('As coordenadas serão atualizadas quando clicares "Submeter alterações" ou "Ver alterações"')
                if update_coords_button:
                    st.session_state['new_lat'] = st.session_state['map_selected_point'][lat_col]
                    st.session_state['new_lon'] = st.session_state['map_selected_point'][lon_col]
                    st.session_state['new_colony_id'] = st.session_state['map_selected_point'][colony_id_col]


        col3, col4 = st.columns(2, gap='small')
        with col3:
            preview_button = st.button(label='Ver alterações')
        with col4:
            submit_button = st.button(label='Submeter alterações')
        
        show_results(st.session_state['filtered_df'])


        if preview_button:
            pass

        if submit_button:
            data_to_save: list = st.session_state['edited_submission'].iloc[:, 0].tolist()
            append_row_to_spreadsheet(url=DRIVE_INFO['final_form_spreadsheet_url'], 
                                      gc= GC, 
                                      row_to_append=data_to_save)

            st.session_state['submission_id'] += 1

if __name__ == "__main__":
    main()
