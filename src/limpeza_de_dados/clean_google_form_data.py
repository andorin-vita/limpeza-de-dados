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

import functools
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from shapely.geometry import Point

from limpeza_de_dados.utils import split_coordinates

logger = logging.getLogger(__name__)

GEOLOCATOR: Nominatim = Nominatim(user_agent="my-agent")

SHAPEFILE_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "bacia-hidro"
    / "wise_vw_surfacewaterbody_basin_ptcont.shp"
)

OPEN_METEO_ELEVATION_URL: str = "https://api.open-meteo.com/v1/elevation"


@functools.lru_cache(maxsize=1)
def _load_bacias_gdf() -> gpd.GeoDataFrame | None:
    """Load the surface water body basins shapefile (cached after first call)."""
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        logger.info("Loaded %d basin polygons from %s", len(gdf), SHAPEFILE_PATH)
        return gdf
    except Exception:
        logger.warning("Could not load shapefile at %s", SHAPEFILE_PATH, exc_info=True)
        return None


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
    "Altitude (m)": "Altitude (m)",
    "Região Hidrográfica": "Região Hidrográfica",
    "Bacia Hidrográfica": "Bacia Hidrográfica",
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


def add_altitude(row: pd.Series) -> pd.Series:
    """Add altitude (elevation in metres) from the Open-Meteo Elevation API."""
    row["Altitude (m)"] = None
    try:
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        response = requests.get(
            OPEN_METEO_ELEVATION_URL,
            params={"latitude": lat, "longitude": lon},
            timeout=10,
        )
        response.raise_for_status()
        row["Altitude (m)"] = response.json()["elevation"][0]
    except Exception:
        logger.warning("Could not fetch altitude for (%s, %s)", row.get("Latitude"), row.get("Longitude"), exc_info=True)
    return row


def add_bacia_hidrografica(row: pd.Series, bacias_gdf: gpd.GeoDataFrame | None = None) -> pd.Series:
    """Add Região Hidrográfica and Bacia Hidrográfica via point-in-polygon lookup."""
    row["Região Hidrográfica"] = None
    row["Bacia Hidrográfica"] = None

    if bacias_gdf is None:
        bacias_gdf = _load_bacias_gdf()

    if bacias_gdf is None:
        return row

    try:
        point = Point(float(row["Longitude"]), float(row["Latitude"]))
        for _, bacia in bacias_gdf.iterrows():
            if bacia.geometry.contains(point):
                row["Região Hidrográfica"] = bacia["regiao_hid"]
                row["Bacia Hidrográfica"] = bacia["nome"]
                break
    except Exception:
        logger.warning("Could not determine bacia for (%s, %s)", row.get("Latitude"), row.get("Longitude"), exc_info=True)
    return row


def get_altitudes_batch(lats: list[float], lons: list[float]) -> list[float | None]:
    """Fetch altitudes for multiple coordinates in a single API call (batch mode)."""
    try:
        lat_str = ",".join(str(x) for x in lats)
        lon_str = ",".join(str(x) for x in lons)
        response = requests.get(
            OPEN_METEO_ELEVATION_URL,
            params={"latitude": lat_str, "longitude": lon_str},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["elevation"]
    except Exception:
        logger.warning("Batch altitude request failed", exc_info=True)
        return [None] * len(lats)


def get_bacias_batch(df: pd.DataFrame, bacias_gdf: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Assign Região Hidrográfica and Bacia Hidrográfica to all rows via spatial join."""
    if bacias_gdf is None:
        bacias_gdf = _load_bacias_gdf()

    if bacias_gdf is None:
        df["Região Hidrográfica"] = None
        df["Bacia Hidrográfica"] = None
        return df

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df["Longitude"], df["Latitude"])
    ]
    colonies_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    result = gpd.sjoin(colonies_gdf, bacias_gdf, how="left", predicate="within")

    df["Região Hidrográfica"] = result["regiao_hid"].values
    df["Bacia Hidrográfica"] = result["nome"].values
    return df


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
