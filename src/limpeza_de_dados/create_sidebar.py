"""
Ferramenta de Validação de Submissões - Streamlit App
------------------------------------------------------

Este script cria uma interface interativa para ajudar a validar submissões do formulário, 
facilitanto a identificação de edifícios duplicados.

Objetivo:
---------
Permitir ao utilizador escolher uma submissão não validada e encontrar outras submissões semelhantes,
de forma a identificar uma possível duplicação, com base em critérios como:
- proximidade geográfica (coordenadas),
- espécie,
- estrutura de nidificação.

Principais funcionalidades:
---------------------------
1. Carregamento dos de uma google sheet.
   - A função `load_data()` carrega o ficheiro de forma eficiente usando `@st.cache_data`.
   
2. Cálculo da distância geográfica entre dois pontos com a fórmula de Haversine.
   - Implementado na função `haversine()`, usado para comparar coordenadas atendendo à curvatura
   da terra.

3. Interface de utilizador:
   - O utilizador seleciona uma submissão (apresentadas por ordem cronológica).
   - A barra lateral permite aplicar filtros opcionais:
     - Comparação por distância (com slider para raio máximo em metros)
     - Comparação do número de andares (com slider para diferença absoluta máxima)
     - Comparação de datas (com slider para diferença absoluta máxima em dias)
     - Filtros adicionais por
        - Espécie
        - Estrutura de nidificação
        - Utilizador
   - Ao aplicar os filtros, são mostradas as submissões semelhantes na interface principal.

Estrutura do código:
--------------------
- Importações de bibliotecas e definições de funções no topo.
- Carregamento e preparação dos dados.
- Interface principal (`st.selectbox` e apresentação da submissão escolhida).
- Formulário na barra lateral (`st.sidebar.form`) com filtros.
- Aplicação dos filtros e exibição dos resultados.

Notas:
------
- Este é o primeiro protótipo funcional. A estrutura foi feita para facilitar futuras melhorias.
- A utilização de `st.cache_data` garante que o carregamento dos dados é feito de forma eficiente.

"""

from math import asin, cos, radians, sin, sqrt

import pandas as pd
import streamlit as st

from limpeza_de_dados.clean_google_form_data import (
    FIELD_ORIGIN,
    add_altitude,
    add_bacia_hidrografica,
    add_detailed_location,
)


def get_sorted_submission_options(df):
    """Returns a sorted list of display strings and a mapping to submission IDs."""
    new_df = df.copy()
    # Build display strings
    new_df["display"] = new_df.apply(
        lambda row: f"{row['Timestamp']} | {row['Data'].date()} | {row['Espécie']} | {row['Coordenadas']}",
        axis=1,
    )

    display_to_id = dict(zip(new_df["display"], new_df["Timestamp"]))

    return new_df["display"], display_to_id


def haversine(lat1, lon1, lat2, lon2):
    # Simple geographic distance calc in m
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 1000 * 6371 * c


def create_filter_options():
    """Create and return filter options from the sidebar."""
    filters = {
        "use_coordinates": st.sidebar.checkbox("Comparação de coordenadas", value=True),
        "use_andares_dif": st.sidebar.checkbox(
            "Comparação do número de andares", value=False
        ),
        "use_altitude_dif": st.sidebar.checkbox("Comparação de altitude", value=False),
        "use_dias_dif": st.sidebar.checkbox("Comparação datas", value=False),
        "use_especie": st.sidebar.checkbox("Mesma espécie?", value=False),
        "use_estrutura": st.sidebar.checkbox(
            "Mesma estrutura de nidificação?", value=False
        ),
        "use_bacia": st.sidebar.checkbox("Mesma bacia hidrográfica?", value=False),
        "use_utilizador": st.sidebar.checkbox("Mesmo utilizador?", value=False),
    }

    return filters


def create_filter_parameters(filters):
    """Create and return filter parameters based on selected filters."""
    params = {}

    if filters["use_coordinates"]:
        params["dist_m"] = st.sidebar.slider(
            "Distância (m)", min_value=10, max_value=200, value=10, step=1
        )

    if filters["use_andares_dif"]:
        params["diff_a"] = st.sidebar.slider(
            "Diferença (andares)", min_value=0, max_value=20, value=2, step=1
        )

    if filters["use_altitude_dif"]:
        params["diff_alt"] = st.sidebar.slider(
            "Diferença de altitude (m)", min_value=0, max_value=500, value=50, step=10
        )

    if filters["use_dias_dif"]:
        params["diff_d"] = st.sidebar.slider(
            "Diferença (dias)", min_value=0, max_value=600, value=365, step=1
        )

    return params


def apply_coordinate_filter(
    filtered_df, selected_row, distance_col, lat_col, lon_col, dist_m
):
    """Apply coordinate-based distance filter."""
    filtered_df[distance_col] = filtered_df.apply(
        lambda row: haversine(
            selected_row[lat_col], selected_row[lon_col], row[lat_col], row[lon_col]
        ),
        axis=1,
    )
    return filtered_df[filtered_df[distance_col] <= dist_m]


def apply_height_filter(filtered_df, selected_row, height_col, diff_a):
    """Apply height difference filter."""
    filtered_df[height_col] = pd.to_numeric(filtered_df[height_col], errors="coerce")
    selected_altura = pd.to_numeric(selected_row[height_col], errors="coerce")
    filtered_df["diff_andares"] = (filtered_df[height_col] - selected_altura).abs()
    return filtered_df[filtered_df["diff_andares"] <= diff_a]


def apply_altitude_filter(filtered_df, selected_row, altitude_col, diff_alt):
    """Apply altitude difference filter."""
    if altitude_col not in filtered_df.columns:
        return filtered_df
    filtered_df[altitude_col] = pd.to_numeric(
        filtered_df[altitude_col], errors="coerce"
    )
    selected_alt = pd.to_numeric(selected_row.get(altitude_col), errors="coerce")
    if pd.isna(selected_alt):
        return filtered_df
    filtered_df["diff_altitude"] = (filtered_df[altitude_col] - selected_alt).abs()
    return filtered_df[filtered_df["diff_altitude"] <= diff_alt]


def apply_date_filter(filtered_df, selected_row, date_col, diff_d):
    """Apply date difference filter."""
    filtered_df["date_dt"] = pd.to_datetime(filtered_df[date_col], errors="coerce")
    selected_date = pd.to_datetime(selected_row[date_col], errors="coerce")
    filtered_df["diff_dias"] = (filtered_df["date_dt"] - selected_date).abs().dt.days
    return filtered_df[filtered_df["diff_dias"] <= diff_d]


def apply_field_filters(
    filtered_df,
    selected_row,
    filters,
    species_col,
    nest_structure_col,
    bacia_col,
    email_col,
):
    """Apply exact match filters for species, structure, bacia, and user."""
    field_filters = [
        (species_col, filters["use_especie"]),
        (nest_structure_col, filters["use_estrutura"]),
        (bacia_col, filters["use_bacia"]),
        (email_col, filters["use_utilizador"]),
    ]

    for col_name, use_filter in field_filters:
        if use_filter and col_name in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df[col_name] == selected_row.get(col_name)
            ]
    return filtered_df


def apply_all_filters(
    df_to_filter: pd.DataFrame,
    selected_row: pd.Series,
    filters: dict[str, bool],
    filter_params: dict[str, str | float | int],
    lat_col: str,
    lon_col: str,
    distance_col: str,
    height_col: str,
    altitude_col: str,
    date_col: str,
    species_col: str,
    nest_structure_col: str,
    bacia_col: str,
    email_col: str,
) -> pd.DataFrame:
    # Start with validated dataset
    filtered_df = df_to_filter.copy()

    # Apply filters sequentially
    if filters["use_coordinates"]:
        filtered_df = apply_coordinate_filter(
            filtered_df,
            selected_row,
            distance_col,
            lat_col,
            lon_col,
            filter_params["dist_m"],
        )

    if filters["use_andares_dif"]:
        filtered_df = apply_height_filter(
            filtered_df, selected_row, height_col, filter_params["diff_a"]
        )

    if filters["use_altitude_dif"]:
        filtered_df = apply_altitude_filter(
            filtered_df, selected_row, altitude_col, filter_params["diff_alt"]
        )

    if filters["use_dias_dif"]:
        filtered_df = apply_date_filter(
            filtered_df, selected_row, date_col, filter_params["diff_d"]
        )

    # Apply remaining field filters
    filtered_df = apply_field_filters(
        filtered_df,
        selected_row,
        filters,
        species_col,
        nest_structure_col,
        bacia_col,
        email_col,
    )

    return filtered_df


def show_results(filtered_df: pd.DataFrame):
    if not filtered_df.empty:
        st.subheader(f"Submissões similares encontradas ({len(filtered_df)}):")
        st.dataframe(filtered_df)
    else:
        st.warning("✅ Nenhuma submissão similar encontrada.")


def get_submission_to_validate(df_new_submissions: pd.DataFrame, n_timestamp_col: str):
    st.markdown("## Seleciona uma submissão não validada")
    st.write(
        "Será comparada com outras submissões semelhantes, com base nos filtros escolhidos na barra lateral."
    )
    display_options, display_to_id = get_sorted_submission_options(df_new_submissions)
    selected_display = st.selectbox(
        label="Submissões (da mais recente para a mais antiga):",
        options=display_options,
        placeholder="Selecciona a submissão",
    )
    selected_id = display_to_id[selected_display]
    selected_row = df_new_submissions[
        df_new_submissions[n_timestamp_col] == selected_id
    ].iloc[0]
    selected_row = add_detailed_location(selected_row)
    selected_row = add_altitude(selected_row)
    selected_row = add_bacia_hidrografica(selected_row)

    return selected_row


def show_selected_row_as_table(selected_row: pd.Series, height: int = 810):
    st.subheader("Submissão seleccionada completa:")
    df_display = selected_row.to_frame(name="Valor")
    df_display.index.name = "Campo"
    df_display["Origem"] = df_display.index.map(FIELD_ORIGIN).fillna("")
    edited_submission = st.data_editor(
        df_display,
        height=height,
        disabled=["Origem"],
    )
    return edited_submission[["Valor"]]
