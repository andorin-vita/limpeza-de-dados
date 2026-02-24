import gspread
import numpy as np
import pandas as pd
import streamlit as st

from limpeza_de_dados.clean_google_form_data import (
    CONVERSION,
    DISPLAY_COLUMNS,
    SAVE_COLUMNS,
    convert_column_names,
    full_clean_data,
)
from limpeza_de_dados.create_data_flow_to_google import (
    append_row_to_spreadsheet,
    get_drive_info,
    read_spreadsheet_as_df,
)
from limpeza_de_dados.create_map import create_full_map
from limpeza_de_dados.create_sidebar import (
    apply_all_filters,
    create_filter_options,
    create_filter_parameters,
    get_submission_to_validate,
    show_results,
    show_selected_row_as_table,
)
from limpeza_de_dados.utils import find_new_entries


def _infer_colony_id(
    selected_row: pd.Series,
    df_validated: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    species_col: str,
    colony_id_col: str,
    default_id: int,
) -> int:
    """Return existing colony ID if coordinates and species match, otherwise default_id."""
    lat = selected_row.get(lat_col)
    lon = selected_row.get(lon_col)
    species = selected_row.get(species_col)

    if df_validated.empty or pd.isna(lat) or pd.isna(lon):
        return default_id

    if colony_id_col not in df_validated.columns:
        return default_id

    coord_match = np.isclose(
        df_validated[lat_col].astype(float), float(lat), atol=1e-5
    ) & np.isclose(df_validated[lon_col].astype(float), float(lon), atol=1e-5)

    matching = df_validated[coord_match]

    if matching.empty:
        return default_id

    if pd.notna(species) and species_col in df_validated.columns:
        same_species = matching[matching[species_col] == species]
        if not same_species.empty:
            return int(same_species[colony_id_col].iloc[0])

    return default_id


def _format_for_pt_locale(value) -> str:
    """Format a value for Portuguese locale: use comma as decimal separator
    for numeric values, leave text values unchanged."""
    if pd.isna(value):
        return ""
    s = str(value)
    try:
        float(s)
        return s.replace(".", ",")
    except ValueError:
        return s


DRIVE_INFO: dict[str, str] = get_drive_info()
GC: gspread.client.Client = gspread.authorize(DRIVE_INFO["creds"])


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
def load_data_submissions(
    df_analysis: pd.DataFrame,
    url: str = DRIVE_INFO["original_form_spreadsheet_url"],
    gc: gspread.client.Client = GC,
):
    df: pd.DataFrame = read_spreadsheet_as_df(url=url, gc=gc)
    df = convert_column_names(df)
    df_new_sub: pd.DataFrame = find_new_entries(df_raw=df, df_analysis=df_analysis)
    return full_clean_data(df_raw=df_new_sub)


@st.cache_data
def load_data_analysis(
    url: str = DRIVE_INFO["final_form_spreadsheet_url"], gc: gspread.client.Client = GC
):
    return read_spreadsheet_as_df(url=url, gc=gc)


def main(
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    id_col: str = "Nº de registo",
    colony_id_col: str = "ID da colónia",
    coords_col: str = "Coordenadas",
    distance_col: str = "Distância (m)",
    n_timestamp_col: str = "Timestamp",
    height_col: str = "Altura (andares)",
    altitude_col: str = "Altitude (m)",
    date_col: str = "Data",
    species_col: str = "Espécie",
    nest_structure_col: str = "Estrutura de nidificação",
    bacia_col: str = "Bacia Hidrográfica",
    email_col: str = "Email",
):

    # Carregar dados
    df_validated = load_data_analysis()

    no_validated_data = (
        df_validated.empty
        or id_col not in df_validated.columns
        or df_validated[id_col].dropna().empty
    )

    if no_validated_data:
        final_url = DRIVE_INFO.get("final_form_spreadsheet_url", "URL não disponível")
        st.warning(f"Não existem dados validados. " f"Spreadsheet final: {final_url}")

    df_submissions: pd.DataFrame = load_data_submissions(df_analysis=df_validated)
    # Submissão a validar
    selected_row = get_submission_to_validate(df_submissions, n_timestamp_col)

    # Inicializar variáveis
    if "filtered_df" not in st.session_state:
        st.session_state["filtered_df"] = df_validated

    if "submission_id" not in st.session_state:
        if no_validated_data:
            st.session_state["submission_id"] = 1
        else:
            st.session_state["submission_id"] = int(max(df_validated[id_col]) + 1)

    selected_row[id_col] = st.session_state["submission_id"]
    selected_row[colony_id_col] = _infer_colony_id(
        selected_row,
        df_validated,
        lat_col,
        lon_col,
        species_col,
        colony_id_col,
        default_id=st.session_state["submission_id"],
    )

    # Derive campaign year from form submission timestamp
    ts_dt = pd.to_datetime(selected_row.get("Timestamp"), errors="coerce")
    selected_row["Ano de Campanha"] = ts_dt.year if pd.notna(ts_dt) else None

    selected_row["Dados em Falta"] = "Não"

    # Reorder for display (Altitude/Bacia after Freguesia, no Media)
    selected_row = selected_row.reindex(DISPLAY_COLUMNS)

    st.title("Ferramenta de Validação de Submissões")

    # Sidebar
    st.sidebar.title("Submissões não validadas")
    st.sidebar.write(
        "Usa os filtros para encontrar submissões similares à submissão selecionada."
    )
    st.sidebar.markdown("### Filtros")
    filters = create_filter_options()
    filter_params = create_filter_parameters(filters)

    st.session_state["filtered_df"] = apply_all_filters(
        df_to_filter=df_validated,
        selected_row=selected_row,
        filters=filters,
        filter_params=filter_params,
        lat_col=lat_col,
        lon_col=lon_col,
        distance_col=distance_col,
        height_col=height_col,
        altitude_col=altitude_col,
        date_col=date_col,
        species_col=species_col,
        nest_structure_col=nest_structure_col,
        bacia_col=bacia_col,
        email_col=email_col,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["edited_submission"] = show_selected_row_as_table(selected_row)
    with col2:
        selected_point = create_full_map(selected_row, st.session_state["filtered_df"])
        if selected_point and selected_point.get("Grupo") == "Validada":
            st.markdown(
                f"**Colónia existente:** ID da colónia = `{selected_point.get(colony_id_col)}` "
                f"| Coordenadas = `{selected_point.get(lat_col)}, {selected_point.get(lon_col)}`"
            )
            st.write(
                "Modifique coordenadas e ID da colónia na tabela se a observação é a mesma colónia."
            )

    submit_button = st.button(label="Submeter alterações")

    if st.session_state.pop("submit_success", False):
        st.toast("Submissão validada com sucesso!", icon="✅")

    if submit_button:
        with st.spinner("A guardar submissão..."):
            edited_series = st.session_state["edited_submission"].iloc[:, 0]
            save_series = edited_series.reindex(SAVE_COLUMNS)
            data_to_save: list = [
                _format_for_pt_locale(item) for item in save_series.tolist()
            ]
            append_row_to_spreadsheet(
                url=DRIVE_INFO["final_form_spreadsheet_url"],
                gc=GC,
                row_to_append=data_to_save,
            )

            st.session_state["submission_id"] += 1
            st.session_state["submit_success"] = True
            load_data_analysis.clear()
            load_data_submissions.clear()
        st.rerun()

    show_results(st.session_state["filtered_df"])


if __name__ == "__main__":
    main()
