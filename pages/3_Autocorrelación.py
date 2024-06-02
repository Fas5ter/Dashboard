import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
st.set_page_config(page_title="Autocorrelación", page_icon=":chart_with_upwards_trend:")

st.title("Análisis de la autocorrelación")
# – Gráficos de autocorrelación (ACF) y autocorrelación parcial (PACF).

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
def Promedio_temperaturas(temperaturas):
    # Promediamos cada mes por estación
    temperaturas_promediadas = temperaturas.groupby(['Est', temperaturas.index]).mean()
    temperaturas_promediadas = temperaturas_promediadas.reset_index()
    temperaturas_promediadas = temperaturas_promediadas.drop(columns=['Lat', 'Lon'])
    # Cambiamos el orden a Fecha, Est, Tmax, Tmed, Tmin
    temperaturas_promediadas = temperaturas_promediadas[['Fecha', 'Est', 'Tmax', 'Tmed', 'Tmin']]
    temperaturas_promediadas.sort_values('Fecha', inplace=True)
    temperaturas_promediadas.set_index('Fecha', inplace=True)
    temperaturas_promediadas.reset_index(inplace=True)

    return temperaturas_promediadas

@st.cache_data
def Promedio_fecha(temperaturas):
    temperaturas_promediadas_fu = temperaturas.drop(columns=['Lat', 'Lon', 'Est'])
    temperaturas_promediadas_fu = temperaturas_promediadas_fu.groupby(temperaturas.index).mean()
    temperaturas_promediadas_fu.reset_index(inplace=True)
    temperaturas_promediadas_fu = temperaturas_promediadas_fu[['Fecha', 'Tmax', 'Tmed', 'Tmin']]
    temperaturas_promediadas_fu.sort_values('Fecha', inplace=True)
    temperaturas_promediadas_fu['Fecha'] = pd.to_datetime(temperaturas_promediadas_fu['Fecha'])
    return temperaturas_promediadas_fu

@st.cache_data
def Autocorrelacion(temperaturas_promediadas_fu, var):
    fig = plt.figure(figsize=(10, 10))
    fig = plot_acf(temperaturas_promediadas_fu[var], lags=50, title='ACF')
    plt.grid()
    fig

@st.cache_data 
def AutocorrelacionParcial (temperaturas_promediadas_fu, var):
    fig = plt.figure(figsize=(10, 10))
    fig = plot_pacf(temperaturas_promediadas_fu[var], lags=50, title='PACF')
    plt.grid()
    fig


dif = st.sidebar.selectbox("Selecciona la serie con la que deseas trabajar", ['Serie original', 'Serie diferenciada no estacional', 'Serie diferenciada estacional'])
opc = st.sidebar.selectbox("Selecciona la variable a analizar", ['Temperatura máxima', 'Temperatura media', 'Temperatura mínima'])

if dif == 'Serie diferenciada no estacional':
    prom_fecha_index = Promedio_fecha(Carga_datos()).diff()
    prom_fecha_index = prom_fecha_index.dropna()
if dif == 'Serie diferenciada estacional':
    prom_fecha_index = Promedio_fecha(Carga_datos()).diff(12)
    prom_fecha_index = prom_fecha_index.dropna()
else:
    prom_fecha_index = Promedio_fecha(Carga_datos())

if opc == 'Temperatura máxima':
    st.write(f"## Autocorrelación de la temperatura máxima")
    Autocorrelacion(prom_fecha_index, 'Tmax')
    st.write(f"## Autocorrelación parcial de la temperatura máxima")
    AutocorrelacionParcial(prom_fecha_index, 'Tmax')
elif opc == 'Temperatura media':
    st.write(f"## Autocorrelación de la temperatura media")
    Autocorrelacion(prom_fecha_index, 'Tmed')
    st.write(f"## Autocorrelación parcial de la temperatura media")
    AutocorrelacionParcial(prom_fecha_index, 'Tmed')
else:
    st.write(f"## Autocorrelación de la temperatura mínima")
    Autocorrelacion(prom_fecha_index, 'Tmin')
    st.write(f"## Autocorrelación parcial de la temperatura mínima")
    AutocorrelacionParcial(prom_fecha_index, 'Tmin')