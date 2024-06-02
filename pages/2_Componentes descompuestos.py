import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import pandas as pd
st.set_page_config(page_title="Componentes descompuestos", page_icon=":calendar:")


st.title("Componentes de una serie temporal descompuestos")
# – Visualización de los componentes descompuestos (tendencia, estacionalidad, ruido).

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
def Descomposicion(temperaturas_promediadas):
    # Aplica la función de seasonal_decompose y graficamos los resultados para las ventas mensuales de los productos
    descomposition = [] # Lista para almacenar los resultados de la descomposición
    for i in ["Tmax", "Tmed", "Tmin"]:
        # Aplica la función en base a la temperatura en turno
        result = seasonal_decompose(temperaturas_promediadas[i], model='additive', period=12) # period=12 porque es el número de meses en un año
        descomposition.append(result) # Almacena los resultados en la lista
    # st.write(descomposition)
    return descomposition

@st.cache_data
def Plot_individual(temperaturas_promediadas, var):
    descomposition = Descomposicion(temperaturas_promediadas)
    fig = descomposition[var].plot()
    fig.set_size_inches(10, 8) # Ajusta el tamaño del gráfico
    # Ajustamos el label del eje x para que no se sobrepongan
    axes = fig.get_axes()
    plt.tight_layout()
    fig

@st.cache_data
def Plot_tendencia(temperaturas_promediadas):
    # Creamos la descomposición de las temperaturas
    descomposition = Descomposicion(temperaturas_promediadas)
    # Graficamos las tendencias de las temperaturas con plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[0].trend, mode='lines', name='Tmax'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[1].trend, mode='lines', name='Tmed'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[2].trend, mode='lines', name='Tmin'))
    fig.update_layout(title='Tendencias de las temperaturas en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)
    
@st.cache_data
def Plot_Estacionariedad(temperaturas_promediadas):
    descomposition = Descomposicion(temperaturas_promediadas)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[0].seasonal, mode='lines', name='Tmax'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[1].seasonal, mode='lines', name='Tmed'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[2].seasonal, mode='lines', name='Tmin'))
    fig.update_layout(title='Estacionalidades de las temperaturas en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)

@st.cache_data
def Plot_residuo(temperaturas_promediadas):
    descomposition = Descomposicion(temperaturas_promediadas)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[0].resid, mode='markers', name='Tmax'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[1].resid, mode='markers', name='Tmed'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas.index, y=descomposition[2].resid, mode='markers', name='Tmin'))
    fig.update_layout(title='Residuales de las temperaturas en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)

temperaturas = Carga_datos()
temperaturas_promediadas = Promedio_temperaturas(temperaturas)
temperaturas_promediadas.set_index('Fecha', inplace=True)
temperaturas_promediadas.index = pd.to_datetime(temperaturas_promediadas.index)
temperaturas_fecha = Promedio_fecha(temperaturas)

st.sidebar.write("## Datos")
data_type = st.sidebar.selectbox('Elige el tipo de datos que deseas visualizar', ['Promediadas por municipio', 'Promediadas por fecha'], key='datos')

st.write("## Descomposición individual de las temperaturas en Colima")
desc = Descomposicion(temperaturas_promediadas)
opc = ""
opc = st.selectbox('Elige la variable que deseas descomponer', ['Temperatura máxima', 'Temperatura media', 'Temperatura mínima'], key='variable')


if data_type == 'Promediadas por municipio':
    if opc == 'Temperatura máxima':
        Plot_individual(temperaturas_promediadas, 0)
    elif opc == 'Temperatura media':
        Plot_individual(temperaturas_promediadas, 1)
    else:
        Plot_individual(temperaturas_promediadas, 2)
else:
    if opc == 'Temperatura máxima':
        Plot_individual(temperaturas_fecha, 0)
    elif opc == 'Temperatura media':
        Plot_individual(temperaturas_fecha, 1)
    else:
        Plot_individual(temperaturas_fecha, 2)


st.write("## Componentes de las temperaturas en Colima")
dec = st.selectbox('Elige el componente que deseas visualizar', ['Tendencias', 'Estacionariedad', 'Residuales'], key='componente')
st.write(f"### {dec} de las temperaturas en Colima")

if data_type == 'Promediadas por municipio':
    if dec == 'Tendencias':
        Plot_tendencia(temperaturas_promediadas)
    elif dec == 'Estacionariedad':
        Plot_Estacionariedad(temperaturas_promediadas)
    else:
        Plot_residuo(temperaturas_promediadas)
else:
    if dec == 'Tendencias':
        Plot_tendencia(temperaturas_fecha)
    elif dec == 'Estacionariedad':
        Plot_Estacionariedad(temperaturas_fecha)
    else:
        Plot_residuo(temperaturas_fecha)

