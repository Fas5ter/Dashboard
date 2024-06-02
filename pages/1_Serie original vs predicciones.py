import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Original y predicciones", page_icon=":mostly_sunny:")

st.title("Serie temporal original")
# Gráficos de líneas para mostrar la serie temporal original y los pronósticos.

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
def Promedio_temperaturas_mun(temperaturas):
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
def Observables(temperaturas_promediadas, titulo:str):
    # Gráfica de líneas para observar la evolución de las temperaturas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperaturas_promediadas['Fecha'], y=temperaturas_promediadas['Tmax'], mode='lines', name='Tmax'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas['Fecha'], y=temperaturas_promediadas['Tmed'], mode='lines', name='Tmed'))
    fig.add_trace(go.Scatter(x=temperaturas_promediadas['Fecha'], y=temperaturas_promediadas['Tmin'], mode='lines', name='Tmin'))
    fig.update_layout(title=titulo, xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)
  
@st.cache_data  
def Por_municipio(temperaturas_promediadas, variable:str):
    # Gráfica de líneas para observar la evolución de las temperatura elegida por municipio
    fig = go.Figure()
    for municipio in temperaturas_promediadas['Est'].unique():
        datos = temperaturas_promediadas[temperaturas_promediadas['Est'] == municipio]
        datos.reset_index(inplace=True)
        fig.add_trace(go.Scatter(x=datos['Fecha'], y=datos[variable], mode='lines', name=municipio))
    if variable == 'Tmax':
        fig.update_layout(title='Temperaturas Máximas en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    elif variable == 'Tmed':
        fig.update_layout(title='Temperaturas Medias en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    else:
        fig.update_layout(title='Temperaturas Mínimas en Colima', xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)

# Funciones para predicción

def Tabla_df (predicciones, variable:str):
    fechas = []
    Cuauhtemoc = []
    Tecoman = []
    Manzanillo = []
    Colima = []
    Coquimatlan = []
    
    for fecha in predicciones['Fecha'].unique():
        fechas.append(fecha)
        Cuauhtemoc.append(predicciones[(predicciones['Fecha'] == fecha) & (predicciones['Est'] == 'Cuauhtémoc')][variable].values[0])
        Tecoman.append(predicciones[(predicciones['Fecha'] == fecha) & (predicciones['Est'] == 'Tecomán')][variable].values[0])
        Manzanillo.append(predicciones[(predicciones['Fecha'] == fecha) & (predicciones['Est'] == 'Manzanillo')][variable].values[0])
        Colima.append(predicciones[(predicciones['Fecha'] == fecha) & (predicciones['Est'] == 'Colima')][variable].values[0])
        Coquimatlan.append(predicciones[(predicciones['Fecha'] == fecha) & (predicciones['Est'] == 'Coquimatlán')][variable].values[0])
    
    tabla = pd.DataFrame({'Fecha': fechas, 'Cuauhtémoc': Cuauhtemoc, 'Tecomán': Tecoman, 'Manzanillo': Manzanillo, 'Colima': Colima, 'Coquimatlán': Coquimatlan})
    tabla.set_index('Fecha', inplace=True)
    return tabla

@st.cache_data
def Predictores_df (temperaturas):
    # Creamos un dataframe con las temperaturas añadiendo mes y año
    # Preparamos los datos para el modelo
    predictores = temperaturas.copy()
    # Creamos una columna para el mes
    predictores['Mes'] = temperaturas.index.month
    # Creamos una columna para el año
    predictores['Año'] = temperaturas.index.year
    
    return predictores

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
def Estacionaria(temperaturas_fu)->bool:
    # Prueba de Dickey-Fuller
    p_value = adfuller(temperaturas_fu)[1]
    # Si el p-value es menor a 0.05, la serie es estacionaria
    return p_value < 0.05

@st.cache_data
def Sarima_f(temperaturas_fu, var:str, d:int, p:int, q:int, D:int, P:int, Q:int) -> pd.DataFrame:
    modelo_sarima = SARIMAX(temperaturas_fu[var], order=(p, d, q), seasonal_order=(P, D, Q, 12))
    resultado_sarima= modelo_sarima.fit()
    
    # Predecimos los próximos 8 meses
    prediccion_sarima_serie = resultado_sarima.forecast(steps=8)
    # Creamos un DataFrame con las fechas de los próximos 8 meses
    fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
    # Creamos un DataFrame con las fechas y la serie predicha
    prediccion_sarima = pd.DataFrame({'Fecha': fechas, var: prediccion_sarima_serie.values})
    
    # Sacamos la ultima fecha de la serie original
    ultima_fecha = temperaturas_fu.index[-1]
    # Creamos un DataFrame con la fecha y la serie original
    serie_original = pd.DataFrame({'Fecha': [ultima_fecha], var: [temperaturas_fu[var].iloc[-1]]})
    # Concatenamos los datos
    prediccion_sarima = pd.concat([serie_original, prediccion_sarima], ignore_index=True)
    
    prediccion_sarima.set_index('Fecha', inplace=True)
    prediccion_sarima.index = pd.to_datetime(prediccion_sarima.index)
    
    return prediccion_sarima

@st.cache_data
def Graph_arima(temperaturas_fu, var:str, prediccion_arima):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temperaturas_fu.index, y=temperaturas_fu[var], mode='lines', name='Real'))
    fig.add_trace(go.Scatter(x=prediccion_arima.index, y=prediccion_arima[var], mode='lines', name='Predicción', line=dict(dash='dot')))
    fig.update_layout(title='Predicción de la temperatura máxima con ARIMA', xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)  

@st.cache_data
def Arbol(estaciones, predictores, predicciones_arbol):
    # Predecimos para los próximos 8 meses en las estaciones especificadas
    for est in estaciones:
        # Filtramos los datos de la estación
        datos = predictores[predictores['Est'] == est]
        X = datos[['Mes', 'Año']]
        
        # Creamos un modelo para la temperatura máxima
        modelo_tmax = DecisionTreeRegressor()
        # Entrenamos el modelo
        modelo_tmax.fit(X, datos['Tmax'])
        
        # Creamos un modelo para la temperatura máxima
        modelo_tmed = DecisionTreeRegressor()
        # Entrenamos el modelo
        modelo_tmed.fit(X, datos['Tmed'])
        
        # Creamos un modelo para la temperatura máxima
        modelo_tmin = DecisionTreeRegressor()
        # Entrenamos el modelo
        modelo_tmin.fit(X, datos['Tmin'])
        
        
        # Creamos un dataframe con las fechas de los próximos 8 meses
        fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
        # Creamos un dataframe con las fechas y la estación
        pred = pd.DataFrame({'Fecha': fechas, 'Est': est})
        # Creamos una columna para el mes
        pred['Mes'] = pred['Fecha'].dt.month
        # Creamos una columna para el año
        pred['Año'] = pred['Fecha'].dt.year
        
        # Predecimos la temperaturas
        pred['Tmax'] = modelo_tmax.predict(pred[['Mes', 'Año']])
        pred['Tmed'] = modelo_tmed.predict(pred[['Mes', 'Año']])
        pred['Tmin'] = modelo_tmin.predict(pred[['Mes', 'Año']])
        # Concatenamos los datos
        predicciones_arbol = pd.concat([predicciones_arbol, pred], ignore_index=True)

    return predicciones_arbol

@st.cache_data
def Graficas_pred(estaciones:list, temperaturas_promediadas, predicciones, var:str, titulo:str):
    # Gráfica de las temperaturas máximas
    fig = go.Figure()
    for municipio in estaciones:
        datos = temperaturas_promediadas[temperaturas_promediadas['Est'] == municipio]
        fig.add_trace(go.Scatter(x=datos['Fecha'], y=datos[var], mode='lines', name=municipio))
        datos = predicciones[predicciones['Est'] == municipio]
        fig.add_trace(go.Scatter(x=datos['Fecha'], y=datos[var], mode='lines', name=municipio + ' predicción', line=dict(dash='dot')))
    fig.update_layout(title=titulo, xaxis_title='Fecha', yaxis_title='Temperatura')
    st.plotly_chart(fig)


temperatura = Carga_datos()
temperaturas_promediadas_mun = Promedio_temperaturas_mun(temperatura)
temperaturas_promediadas_fecha = Promedio_fecha(temperatura)

# Elige el tipo de gráfico que deseas visualizar
st.sidebar.write('## Configuración de datos originales')
opc = st.sidebar.selectbox('Elige que datos "originales" deseas ver:', ['Promediado por municipio', 'Promediado por fecha','Original'], key='originales')

st.write('## Temperaturas en Colima')
if opc == 'Original':
    temperatura = temperatura.reset_index()
    Observables(temperatura, 'Temperaturas en Colima')
elif opc == 'Promediado por municipio':
    Observables(temperaturas_promediadas_mun,'Temperaturas en Colima Promediadas por municipio')
else:
    Observables(temperaturas_promediadas_fecha,'Temperaturas en Colima Promediadas por fecha')

if opc == 'Promediado por municipio' or opc == 'Original':
    st.write('## Temperaturas por municipio')
        
    elec = st.selectbox('Elige el tipo de gráfico que deseas visualizar', ['Temperatura máxima', 'Temperatura media', 'Temperatura mínima'])

    if opc == 'Original':
        if elec == 'Temperatura máxima':
            Por_municipio(temperatura, 'Tmax')
        elif elec == 'Temperatura media':
            Por_municipio(temperatura, 'Tmed')
        else:
            Por_municipio(temperatura, 'Tmin')
    else:
        if elec == 'Temperatura máxima':
            Por_municipio(temperaturas_promediadas_mun, 'Tmax')
        elif elec == 'Temperatura media':
            Por_municipio(temperaturas_promediadas_mun, 'Tmed')
        else:
            Por_municipio(temperaturas_promediadas_mun, 'Tmin')

st.sidebar.write('## Configuración de las predicciones')
pred_opc = st.sidebar.selectbox('Elige el tipo de predicción que deseas ver:', ['Por municipio', 'Por fecha'], key='pred')
pred_elec = st.sidebar.selectbox('Elige el tipo de gráfico que deseas visualizar', ['Temperatura máxima', 'Temperatura media', 'Temperatura mínima'], key='variable')

st.title('Predicciones')

#? VARIABLES GLOBALES
# Municipios que se usarán para predecir
estaciones = ["Cuauhtémoc", "Tecomán", "Manzanillo", "Colima", "Coquimatlán"]
predictores = Predictores_df(Carga_datos())

if pred_opc == 'Por municipio':
    st.write("### Resultados de las predicciones con Árbol de decisión")
    
    # Dataframe para almacenar las predicciones
    predicciones_arbol = pd.DataFrame()
    
    predicciones_arbol = Arbol(estaciones, predictores, predicciones_arbol)
    
    if pred_elec == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        st.markdown("##### Tabla de predicciones")
        arbol_tmax = Tabla_df(predicciones_arbol, "Tmax")
        st.write(arbol_tmax)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), predicciones_arbol, "Tmax", "Predicciones de la temperatura máxima")
    elif pred_elec == "Temperatura media":
        st.markdown("#### Predicciones de la temperatura media")
        st.markdown("##### Tabla de predicciones")
        arbol_tmed = Tabla_df(predicciones_arbol, "Tmed")
        st.write(arbol_tmed)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), predicciones_arbol, "Tmed", "Predicciones de la temperatura media")
    else:
        st.markdown("#### Predicciones de la temperatura mínima")
        st.markdown("##### Tabla de predicciones")
        arbol_tmin = Tabla_df(predicciones_arbol, "Tmin")
        st.write(arbol_tmin)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), predicciones_arbol, "Tmin", "Predicciones de la temperatura mínima")
else:
    st.write("## Resultados de las predicciones con SARIMA")
    
    #? Datos para ARIMA y SARIMA
    prom_fecha_index = Promedio_fecha(Carga_datos())

    prom_fecha_index.set_index(prom_fecha_index['Fecha'], inplace=True)
    prom_fecha_index.drop(columns=['Fecha'], inplace=True)
    # Nos aseguramos que el indice sea de tipo datetime
    prom_fecha_index.index = pd.to_datetime(prom_fecha_index.index)
    # Fijamos una frecuencia mensual
    prom_fecha_index = prom_fecha_index.asfreq('MS')

    
    if pred_elec == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        #! Parametros para SARIMA
        d,p,q,D,P,Q = 1,2,9,1,0,0
        
        pred_sarima = Sarima_f(prom_fecha_index, "Tmax", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmax", pred_sarima)
        
    elif pred_elec == "Temperatura media":
        #! Parametros para SARIMA
        d,p,q,D,P,Q = 1,4,6,0,0,0
        
        st.markdown("#### Predicciones de la temperatura media")
        pred_sarima = Sarima_f(prom_fecha_index, "Tmed", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmed", pred_sarima)
    else:
        #! Parametros para SARIMA
        d,p,q,D,P,Q = 1,4,6,0,1,1
        
        st.markdown("#### Predicciones de la temperatura mínima")
        pred_sarima = Sarima_f(prom_fecha_index, "Tmin", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmin", pred_sarima)