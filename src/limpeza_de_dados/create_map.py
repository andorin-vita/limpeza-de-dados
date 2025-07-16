# 1. Create a map in streamlit using whatever map library is faster to loading.
# 2. Given a dataframe with cols latitude and longitude (cols names might not be these exactly),
# and a point coordinates:
    # 1. It puts a marker in the coordinates in color x
    # 2. Finds all other points in a dataframe within y m of distance

# Create a __main__ section where:
# 1. It has the map, a st.text_input and a st.button.
# 2. The map can only reload when the button is pressed;
# it can not reload each time the st.text_input is filled.
# this can be done by using @st.cache_resource and @st.fragment - 
# check https://github.com/dssg-pt/project-andorin/blob/form_attempt_using_folium/src/folium_form/folium_location.py
# The first loading can be when the button is pressed or when the app is open (what is easier)

import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# ---- LOAD AND PREP DATA ----
def _dummy_callable():
    pass

def color_selector(grupo, name_group: str = 'Não validada'):
    return [0, 128, 255, 180] if grupo == name_group else [255, 0, 80, 180]

def create_full_map(selected_row: pd.Series,
               df_validated: pd.DataFrame,
               lat_col: str = 'Latitude',
               lon_col: str = 'Longitude',
               code_col: str = 'Código'):
    selected_row['Grupo'] = 'Não validada'
    df_validated['Grupo'] = 'Validada'

    #st.write(type(df_validated[lat_col][0]), type(df_validated[lon_col][0]))
    df = pd.concat([df_validated, selected_row.to_frame().T], ignore_index=True)
    df[code_col] = df[code_col].fillna('Sem Código')

    df['color'] = df['Grupo'].apply(color_selector)
    df = df.dropna(subset=[lat_col, lon_col])
    zoom=17

    selected_lat, selected_lon = selected_row[lat_col], selected_row[lon_col]
    if np.isnan(selected_lat) or np.isnan(selected_lon):
        selected_lat = 39.69484
        selected_lon = -8.13031
        zoom=5

    # ---- SET VIEW STATE ----
    view_state = pdk.ViewState(
        latitude=selected_lat,
        longitude=selected_lon,
        zoom=zoom,      # High zoom for detail!
        pitch=0,
    )

    # ---- DECK LAYER ----
    layer = pdk.Layer(
        "ScatterplotLayer",
        id = code_col,
        data=df,
        get_position='[{}, {}]'.format(lon_col, lat_col),
        get_fill_color='color',
        get_radius=80,
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=15,
    )

    tooltip = {
        "html": "<b>Elevation Value:</b> {lat_col} <br/> <b>Color Value:</b> {lon_col}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }


    st.write('Selecciona um ponto para ver a localização')
    e = st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
           # tooltip=tooltip,
            map_style='light',
        ), height = 500, on_select=_dummy_callable
    )
    if e.selection and e.selection.get('objects'):
        selected_point = e.selection['objects'][code_col][0]
        url= f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={selected_point[lat_col]},{selected_point[lon_col]}"

        #st.markdown(f"**Google Street View para {selected_point[code_col]}**: [Abrir Street View]({url})")
        st.markdown(f"**{selected_point[code_col]}**: [Abrir Street View]({url})")
        return selected_point

    else:
        return None
        
        # st.write(url)

    # ---- STREET VIEW LINK ----
   # gsv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={selected_lat},{selected_lon}"
   # st.markdown(f"**Google Street View para {selected_row[code_col]}**: [Abrir Street View]({gsv_url})")

