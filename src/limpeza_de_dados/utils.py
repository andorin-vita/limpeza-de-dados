import re
from copy import deepcopy

import numpy as np
import pandas as pd


def separar_coordenadas(
    dataframe: pd.DataFrame,
    coluna_coordenadas: str = "Coordenadas",
    coluna_latitude: str = "Latitude",
    coluna_longitude: str = "Longitude",
) -> pd.DataFrame:
    """
    Separa uma coluna de coordenadas em colunas de latitude e longitude.

    Parâmetros:
        dataframe (pd.DataFrame): DataFrame contendo a coluna de coordenadas.
        coluna_coordenadas (str): Nome da coluna com as coordenadas no formato "latitude,longitude".
        coluna_latitude (str): Nome da nova coluna para armazenar as latitudes.
        coluna_longitude (str): Nome da nova coluna para armazenar as longitudes.

    Retorna:
        pd.DataFrame: DataFrame com as novas colunas de latitude e longitude.
    """
    if not dataframe.empty:
        latitudes = []
        longitudes = []
        for _, row in dataframe.iterrows():
            try:
                latitude, longitude = row[coluna_coordenadas].split(",")
            except Exception:
                latitude, longitude = (np.nan, np.nan)
            latitudes.append(latitude)
            longitudes.append(longitude)
        dataframe[coluna_latitude] = pd.to_numeric(latitudes, errors="coerce")
        dataframe[coluna_longitude] = pd.to_numeric(longitudes, errors="coerce")
    return dataframe


def find_new_entries(
    df_raw: pd.DataFrame, df_analysis: pd.DataFrame, unique_col: str = "Timestamp"
):
    raw_ts = pd.to_datetime(df_raw[unique_col], errors="coerce")
    analysis_ts = pd.to_datetime(df_analysis[unique_col], errors="coerce")
    return df_raw[~raw_ts.isin(analysis_ts)]


def salvar_dados_limpos(caminho_arquivo: str, dataframe_limpo: pd.DataFrame) -> None:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Salva um DataFrame limpo em um arquivo CSV.

    Parâmetros:
        caminho_arquivo (str): Caminho completo onde o arquivo CSV será salvo.
        dataframe_limpo (pd.DataFrame): DataFrame contendo os dados limpos a serem salvos.
    """
    dataframe_limpo.to_csv(caminho_arquivo, index=False)


def atribuir_id_colonia(
    df: pd.DataFrame,
    coluna_latitude: str = "Latitude",
    coluna_longitude: str = "Longitude",
    coluna_numero_registo: str = "Nº de registo",
    coluna_novo_id: str = "ID da colónia",
) -> pd.DataFrame:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Atribui um identificador de colónia com base na localização geográfica.

    Para cada conjunto de registos que partilham a mesma Latitude e Longitude,
    é atribuído como ID da colónia o menor valor encontrado na coluna do número de registo.

    Parâmetros:
        df (pd.DataFrame): DataFrame que contém os dados.
        coluna_latitude (str): Nome da coluna que contém os valores de latitude.
        coluna_longitude (str): Nome da coluna que contém os valores de longitude.
        coluna_numero_registo (str): Nome da coluna que contém o número de registo.
        coluna_novo_id (str): Nome da nova coluna onde será guardado o ID da colónia.

    Retorna:
        pd.DataFrame: DataFrame com a nova coluna de ID da colónia adicionada.
    """

    df[coluna_novo_id] = (
        df.groupby([coluna_latitude, coluna_longitude])[coluna_numero_registo]
        .transform("min")
        .astype("Int64")
    )

    return df


def contar_valores_por_categoria(
    dataframe: pd.DataFrame,
    coluna_da_categoria: str,
    filtros: dict[str, list[str]],
    coluna_id_colonia: str = "ID da colónia",
    remover_duplicados_colonia: bool = True,
    normalizar: bool = True,
    multiplicador: int = 100,
) -> pd.Series:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Conta a frequência de valores em uma coluna (categoria), aplicando filtros e opções de normalização.

    Parâmetros:
        dataframe (pd.DataFrame): DataFrame a ser analisado.
        coluna_da_categoria (str): Nome da coluna cujos valores serão contados.
        filtros (dict[str, list[str]]): Dicionário com filtros a aplicar nas colunas.
        coluna_id_colonia (str): Nome da coluna de identificação da colónia para remoção de duplicados.
        remover_duplicados_colonia (bool): Se True, remove duplicados com base na coluna de colónia.
        normalizar (bool): Se True, retorna proporção em vez de contagem absoluta.
        multiplicador (int): Valor pelo qual multiplicar o resultado (ex: 100 para percentagem).

    Retorna:
        pd.Series: Série com a contagem (ou percentagem) dos valores da coluna filtrada.
    """
    new_df = deepcopy(dataframe)
    if remover_duplicados_colonia:
        new_df = new_df.drop_duplicates(subset=[coluna_id_colonia], keep="first")

    mascara: pd.Series = pd.Series(True, index=new_df.index)
    for chave, valores in filtros.items():
        mascara &= new_df[chave].isin(valores)

    df_filtrado: pd.DataFrame = new_df[mascara]
    resultado = (
        df_filtrado[coluna_da_categoria].value_counts(normalize=normalizar)
        * multiplicador
    )
    return resultado


def filtrar_valores_pequenos(serie: pd.Series, valor_minimo: float) -> pd.Series:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Filtra valores de uma série, mantendo apenas aqueles maiores ou iguais ao valor mínimo especificado.

    Parâmetros:
        serie (pd.Series): Série de dados a ser filtrada.
        valor_minimo (float): Valor mínimo para manter na série.

    Retorna:
        pd.Series: Série contendo apenas os valores maiores ou iguais ao valor mínimo.
    """
    return serie[serie >= valor_minimo]


def substituir_valores_na_coluna(
    dataframe: pd.DataFrame,
    coluna: str = "Estrutura de nidificação",
    substituicao: tuple[str, str] = (
        "Edifício de apoio (arrecadação, garagem, etc.)",
        "Ed. de apoio",
    ),
) -> pd.DataFrame:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Substitui valores específicos em uma coluna de um DataFrame por outro valor.

    Parâmetros:
        dataframe (pd.DataFrame): DataFrame a ser processado.
        coluna (str): Nome da coluna onde será feita a substituição.
        substituicao (tuple[str, str]): Tuplo com o valor original e o novo valor para substituição.

    Retorna:
        pd.DataFrame: DataFrame com os valores substituídos na coluna especificada.
    """
    dataframe[coluna] = dataframe[coluna].replace({substituicao[0]: substituicao[1]})
    return dataframe


def remover_espacos_extras(
    dataframe: pd.DataFrame, colunas: list[str] | None = None
) -> pd.DataFrame:
    # NOTA: Esta função não está a ser utilizada atualmente no projeto.
    """
    Remove espaços em branco no início, fim e múltiplos espaços consecutivos entre palavras, deixando apenas um espaço, em colunas selecionadas de um DataFrame.

    Parâmetros:
        dataframe (pd.DataFrame): DataFrame a ser processado.
        colunas (list[str] | None): Lista de colunas para remover espaços. Se None, aplica a todas as colunas.

    Retorna:
        pd.DataFrame: DataFrame com os valores de texto das colunas selecionadas sem espaços em branco nas extremidades e com apenas um espaço entre palavras.
    """
    if colunas is None:
        colunas = dataframe.columns
    dataframe = dataframe.copy()
    for coluna in colunas:
        if dataframe[coluna].dtype == "object":
            dataframe[coluna] = dataframe[coluna].apply(
                lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x
            )
    return dataframe
