import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Inicio", page_icon=":house:")


# Correr en local host
# streamlit run Inicio.py

#! Requerimientos

# Gráficos de líneas para mostrar la serie temporal original y los pronósticos.
# – Gráficos de autocorrelación (ACF) y autocorrelación parcial (PACF).
# – Visualización de los componentes descompuestos (tendencia, estacionalidad, ruido).
# – Resultados de los diferentes modelos aplicados.
# – Herramientas de interacción para seleccionar diferentes periodos de tiempo y visualizar pronósticos.

col1, col2, col3, col4 = st.columns(4)

with col2:
    st.image("UDC.png", width=200)

with col3:
    st.image("ICI.png", width=100)
st.html("""
            <div style="text-align: center">
                <p style="text-align: center">Universidad de Colima<br>
                Facultad de Ingeniería Mecánica y Eléctrica<br>
                Ingeniería en Computación Inteligente</p>
                <p style="text-align: center"><b>"Predicción de variables climatológicas<br> para este 2024 en el estado de Colima" <br></b>
                Análisis de Series Temporales </p>
                <p style="text-align: center"> <b> Olmos Romero Nydia Naomi  </b>20184561<br>
                <b> Larios Bravo Cristian Armando </b>20188165<br>
                6°D</p>
            </div>
            <div style="text-align: right">
                Coquimatlán, Col. Mx.<br>
                Mayo 31, 2024
            </div>
        """)

st.title("Predicción de variables climatológicas para este 2024 en el estado de Colima")
st.write("Introducción")
st.write("""
         El presente trabajo tiene como objetivo realizar un análisis de series temporales 
         para predecir las variables climatológicas en el estado de Colima para el año 2024. 
         Se utilizarán diferentes técnicas de análisis de series temporales para predecir 
         las temperaturas en el estado de Colima.
         """)


@st.cache_data
def Carga_datos():

    #! CARGA DE DATASETS
    # Cargar los datos de las temperaturas
    tMax = pd.read_csv('temperaturasMaxColima2013-2024.csv')
    tMed = pd.read_csv('temperaturasMedColima2013-2024.csv')
    tMin = pd.read_csv('temperaturasMinColima2013-2024.csv')

    #! UNIFICAMOS LOS DATOS DE LAS TEMPERATURAS
    # Crear un DataFrame con los datos de las temperaturas
    temperaturas = pd.DataFrame()
    # Unimos los dataframes por la fecha y la estación
    temperaturas = pd.merge(tMax, tMed, on=['Lon','Lat','Est','Fecha'], how='inner')
    # Reordenamos las columnas
    temperaturas = temperaturas[['Lon','Lat','Est','Fecha','Tmax','Tmed']]
    # temperaturas
    temperaturas = pd.merge(temperaturas, tMin, on=['Lon','Lat','Est','Fecha'], how='inner')
    # Renombramos las columnas
    temperaturas.columns = ['Lon','Lat','Est','Fecha', 'Tmax', 'Tmed', 'Tmin']

    #! RESETEAMOS EL INDEX
    temperaturas['Fecha'] = pd.to_datetime(temperaturas['Fecha'])
    temperaturas = temperaturas.set_index('Fecha')
    # Ponemos el index al mismo nivel que el resto de columnas
    temperaturas.reset_index(inplace=True)
    # Definimos la columna de la fecha como índice
    temperaturas.set_index('Fecha', inplace=True)
    # Ordenamos los datos por fecha
    temperaturas = temperaturas.sort_values('Fecha')


    #! UNIFICAMOS LAS ESTACIONES POR MUNICIPIOS
    # Unificamos las estaciones por municipios
    #* Manzanillo
    estaciones_manzanillo = ['MANZANILLO SEMAR', 'Observatorio Manzanillo Col. SMN*', 'Observatorio de Manzanillo Col. SMN*', 'Manzanillo Col. SEMAR*', 'Aeropuerto Internacional de Manzanillo-Costalegre Col.*', 'Aeropuerto Internacional de Manzanillo-Costalegre, Col.*', 'MANZANILLO', 'Observatorio de Manzanillo Col.', 'Manzanillo Col.', 'Observatorio de Manzanillo, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_manzanillo), 'Est'] = 'Manzanillo'

    #* Armería
    estaciones_armeria = ['Armeria Col. INIFAP*', 'Radar Col.', 'Radar, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_armeria), 'Est'] = 'Armería'

    #* Tecomán
    estaciones_tecoman = ['LAGUNA DE AMELA', 'CERRO DE ORTEGA', 'CALLEJONES', 'TECOMAN','DERIVADORA JALA', 'Tecom\x87n Col.', 'Presa derivadora Jala Col.', 'Derivadora Jala Col.', 'Derivadora Jala (Madrid) Col.', 'Presa derivadora Jala, Col.', 'Tecomán Col.', 'Tecomán, Col.', 'Laguna de Amela Col.', 'Laguna de Amela, Col.', 'Cerro de Ortega Col.', 'Cerro de Ortega, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_tecoman), 'Est'] = 'Tecomán'

    #* Ixtlahuacán
    estaciones_ixtlahuacan = ['Callejones Col.', 'Callejones, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_ixtlahuacan), 'Est'] = 'Ixtlahuacán'

    #* Coquimatlán
    estaciones_coquimatlan = ['COQUIMATLAN','COQUIMATLÁN','Coquimatl\x87n Col.','La Esperanza Col.', 'La Esperanza, Col.', 'Coquimatlán Col.', 'Coquimatlán, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_coquimatlan), 'Est'] = 'Coquimatlán'

    #* Villa de Álvarez
    estaciones_villa_de_alvarez = ['PE¤ITAS', 'Peñitas Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_villa_de_alvarez), 'Est'] = 'Villa de Álvarez'

    #* Comala
    estaciones_comala = ['Pe\x96itas Col.','Peñitas, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_comala), 'Est'] = 'Comala'

    #* Cuauhtémoc
    estaciones_cuauhtemoc = ['BUENAVISTA', 'CUAHTEMOC' ,'Cuauht\x8emoc Col.','Cuauhtémoc Col.', 'Cuauhtémoc, Col.', 'Aeropuerto Nacional de Colima Col.*', 'Aeropuerto Nacional de Colima, Col.*', 'Buenavista Col.', 'Buenavista, Col.', 'Buena Vista Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_cuauhtemoc), 'Est'] = 'Cuauhtémoc'

    #* Colima
    estaciones_colima = ['LA POSTA' ,'RADAR', 'OBSERVATORIO COLIMA', 'COLIMA (AUTOMATICA)','Observatorio de Colima Col. SMN*', 'Colima (Automática)', 'COLIMA', 'La Posta Col.', 'La Posta, Col.']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_colima), 'Est'] = 'Colima'

    #* Islas
    estaciones_islas = ['Isla Socorro Col. SEMAR*', 'ISLA SOCORRO SEMAR', 'Isla Clarión Col. SEMAR', 'Isla Clarión Col. SEMAR*']
    # Cambiamos el nombre de las estaciones por el nombre del municipio
    temperaturas.loc[temperaturas['Est'].isin(estaciones_islas), 'Est'] = 'Islas'


    #! LIMPIEZA DE DATOS
    # Eliminamos 'TROJES' ya que no es un municipio de Colima
    temperaturas = temperaturas[temperaturas['Est'] != 'TROJES']
    
    return temperaturas

@st.cache_data
def Mapa_estaciones(temperaturas):
    st.subheader("Mapa de las estaciones por municipio en Colima")
    #* Graficamos en un mapbox las estaciones de monitoreo
    fig = px.scatter_mapbox(temperaturas, lat='Lat', lon='Lon', color='Est', zoom=8, center={'lat': 19.1, 'lon': -104}, mapbox_style='carto-positron', labels={'Est':'Municipio'})
    # fig.update_layout(title='Mapa de las estaciones por municipio en Colima')
    st.plotly_chart(fig)

@st.cache_data
def Mapa_calor(temperaturas):
    st.subheader("Mapa de calor de las temperaturas máximas en Colima")
    # Mapa de calor de las temperaturas máximas
    # Coloreamos el mapa de acuerdo a las temperaturas máximas
    fig = px.density_mapbox(temperaturas, lat='Lat', lon='Lon', z='Tmax', radius=10, zoom=5, center={'lat': 18.5, 'lon': -109}, mapbox_style='carto-positron', color_continuous_scale='Solar')
    # fig.update_layout(title='Mapa de calor de las temperaturas máximas en Colima')
    st.plotly_chart(fig)

temperaturas = Carga_datos()
Mapa_estaciones(temperaturas)
Mapa_calor(temperaturas)

st.write("## Datos de las temperaturas obtenidos")
st.write("""
            La recolección de datos de las temperaturas se realizó a través de la página de la Comisión Nacional del Agua (CONAGUA)
            en conjunto con el Servicio Meteorológico Nacional (SMN). Los datos obtenidos son de las temperaturas máximas, medias y mínimas
            recolectados entre el 2013 y 2024 en el estado de Colima, de manera mensual. Extrayendolos en formato CSV, por medio de 
            web scraping.
        """)
st.write("Los datos obtenidos son los siguientes:")
st.dataframe(temperaturas)