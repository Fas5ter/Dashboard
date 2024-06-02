import pandas as pd
import streamlit as st
from sklearn.svm import SVR
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Predicciones", page_icon=":thermometer:")


st.title("Resultados obtenidos en los diferentes modelos de predicción de temperaturas en Colima")

#? FUNCIONES
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
def Estacionaria(temperaturas_fu)->bool:
    # Prueba de Dickey-Fuller
    p_value = adfuller(temperaturas_fu)[1]
    # Si el p-value es menor a 0.05, la serie es estacionaria
    return p_value < 0.05

@st.cache_data
def Arima_f(temperaturas_fu, var:str, d:int, p:int, q:int) -> pd.DataFrame:
    # Aplicamos el modelo ARIMA
    modelo = ARIMA(temperaturas_fu[var], order=(p, d, q))
    resultado_arima = modelo.fit()
    
    # Predecimos los próximos 8 meses
    prediccion_arima_serie = resultado_arima.forecast(steps=8)
    
    # Creamos un DataFrame con las fechas de los próximos 8 meses
    fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
    # Creamos un DataFrame con las fechas y la serie predicha
    prediccion_arima = pd.DataFrame({'Fecha': fechas, var: prediccion_arima_serie.values})
    
    # Sacamos la ultima fecha de la serie original
    ultima_fecha = temperaturas_fu.index[-1]
    # Creamos un DataFrame con la fecha y la serie original
    serie_original = pd.DataFrame({'Fecha': [ultima_fecha], var: [temperaturas_fu[var].iloc[-1]]})
    # Concatenamos los datos
    prediccion_arima = pd.concat([serie_original, prediccion_arima], ignore_index=True)
    
    prediccion_arima.set_index('Fecha', inplace=True)
    prediccion_arima.index = pd.to_datetime(prediccion_arima.index)
    
    return prediccion_arima

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
def Regresion_lineal(estaciones, predictores, pred_linear):
    # Predecimos las temperaturas de los próximos 8 meses con el modelo de regresión lineal
    for municipio in estaciones:
        # Filtramos los datos del municipio
        datos = predictores[predictores['Est'] == municipio]
        # Separamos los datos en entrenamiento y prueba
        X_train = datos[['Mes', 'Año']]
        
        # Creamos el modelo de regresión lineal
        modelo_tmax = LinearRegression()
        # Entrenamos el modelo
        modelo_tmax.fit(X_train, datos['Tmax'])
        # Creamos el modelo de regresión lineal
        modelo_tmed = LinearRegression()
        # Entrenamos el modelo
        modelo_tmed.fit(X_train, datos['Tmed'])
        # Creamos el modelo de regresión lineal
        modelo_tmin = LinearRegression()
        # Entrenamos el modelo
        modelo_tmin.fit(X_train, datos['Tmin'])
        
        # Creamos un DataFrame con las fechas de los próximos 8 meses
        fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
        # Creamos un DataFrame con las fechas y el municipio
        pred = pd.DataFrame({'Fecha': fechas, 'Est': municipio})
        # Creamos una columna para el mes
        pred['Mes'] = pred['Fecha'].dt.month
        # Creamos una columna para el año
        pred['Año'] = pred['Fecha'].dt.year
        # Predecimos las temperaturas máximas
        
        # Predecimos la temperaturas
        pred['Tmax'] = modelo_tmax.predict(pred[['Mes', 'Año']])
        pred['Tmed'] = modelo_tmed.predict(pred[['Mes', 'Año']])
        pred['Tmin'] = modelo_tmin.predict(pred[['Mes', 'Año']])
        # Concatenamos los datos
        pred_linear = pd.concat([pred_linear, pred], ignore_index=True)
    
    return pred_linear

@st.cache_data
def Regresion_logistica(estaciones, predictores, pred_logistica):
    for municipio in estaciones:
        # Filtramos los datos del municipio
        datos = predictores[predictores['Est'] == municipio]
        
        #* tMax > 33
        # Creamos el modelo de regresión logística
        modelo_tmax = LinearRegression()
        # Entrenamos el modelo
        # (x,y)
        modelo_tmax.fit((datos[['Mes', 'Año']]), (datos['Tmax'] > 33))
        
        #* tMed > 27
        modelo_tmed = LinearRegression()
        modelo_tmed.fit((datos[['Mes', 'Año']]), (datos['Tmed'] > 27))
        
        #* tMin > 20
        modelo_tmin = LinearRegression()
        modelo_tmin.fit((datos[['Mes', 'Año']]), (datos['Tmin'] > 20))
        
        # Creamos un DataFrame con las fechas de los próximos 8 meses
        fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
        # Creamos un DataFrame con las fechas y el municipio
        pred = pd.DataFrame({'Fecha': fechas, 'Est': municipio})
        # Creamos una columna para el mes
        pred['Mes'] = pred['Fecha'].dt.month
        # Creamos una columna para el año
        pred['Año'] = pred['Fecha'].dt.year
        
        
        # Predecimos si la temperatura máxima será mayor a 33 grados
        pred['Tmax'] = modelo_tmax.predict(pred[['Mes', 'Año']]) 
        # Predecimos si la temperatura media será mayor a 27 grados
        pred['Tmed'] = modelo_tmed.predict(pred[['Mes', 'Año']])
        # Predecimos si la temperatura mínima será mayor a 20 grados
        pred['Tmin'] = modelo_tmin.predict(pred[['Mes', 'Año']])
        
        # Concatenamos los datos
        pred_logistica = pd.concat([pred_logistica, pred], ignore_index=True)
    
    return pred_logistica

@st.cache_data
def MSV(estaciones, predictores, pred_svm):
    # Predecimos las temperaturas de los próximos 8 meses con el modelo de SVM
    for est in estaciones:
        # Filtramos los datos del municipio
        datos = predictores[predictores['Est'] == est]
        # Separamos los datos en entrenamiento y prueba
        X_train = datos[['Mes', 'Año']]
        
        # Creamos el modelo de regresión lineal
        modelo_tmax = SVR(kernel='linear')
        # Entrenamos el modelo
        modelo_tmax.fit(X_train, datos['Tmax'])
        # Creamos el modelo de regresión lineal
        modelo_tmed = SVR(kernel='linear')
        # Entrenamos el modelo
        modelo_tmed.fit(X_train, datos['Tmed'])
        # Creamos el modelo de regresión lineal
        modelo_tmin = SVR(kernel='linear')
        # Entrenamos el modelo
        modelo_tmin.fit(X_train, datos['Tmin'])
        
        # Creamos un DataFrame con las fechas de los próximos 8 meses
        fechas = pd.date_range(start='2024-05-01', periods=8, freq='MS')
        # Creamos un DataFrame con las fechas y el municipio
        pred = pd.DataFrame({'Fecha': fechas, 'Est': est})
        # Creamos una columna para el mes
        pred['Mes'] = pred['Fecha'].dt.month
        # Creamos una columna para el año
        pred['Año'] = pred['Fecha'].dt.year
        # Predecimos las temperaturas máximas
        
        # Predecimos la temperaturas
        pred['Tmax'] = modelo_tmax.predict(pred[['Mes', 'Año']])
        pred['Tmed'] = modelo_tmed.predict(pred[['Mes', 'Año']])
        pred['Tmin'] = modelo_tmin.predict(pred[['Mes', 'Año']])
        # Concatenamos los datos
        pred_svm = pd.concat([pred_svm, pred], ignore_index=True)
    
    return pred_svm

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



#? VARIABLES GLOBALES
# Municipios que se usarán para predecir
estaciones = ["Cuauhtémoc", "Tecomán", "Manzanillo", "Colima", "Coquimatlán"]
predictores = Predictores_df(Carga_datos())

#? Modelos
st.sidebar.title("Selecciona los modelo que deseas ver")
arima_opc = st.sidebar.checkbox('ARIMA', key='arima', value=True)
sarima_opc = st.sidebar.checkbox('SARIMA', key='sarima', value=True)
regresion_lineal_opc = st.sidebar.checkbox('Regresión lineal', key='Rlin', value=True)
regresion_logistica_opc = st.sidebar.checkbox('Regresión logística', key='Rlog', value=True)
svm_opc = st.sidebar.checkbox('Máquinas de vectores de soporte', key='svm', value=True)
arbol_opc = st.sidebar.checkbox('Árbol de decisión', key='arbol', value=True)

#? Variable a predecir
st.sidebar.title("Selecciona la variable a predecir")
opc = st.sidebar.selectbox("Selecciona la variable a predecir", ["Temperatura máxima", "Temperatura media", "Temperatura mínima"], key='predicciones')
# opc = st.selectbox("Selecciona la variable a predecir", ["Temperatura máxima", "Temperatura media", "Temperatura mínima"], key='lineal')


#? Datos para ARIMA y SARIMA
prom_fecha_index = Promedio_fecha(Carga_datos())

prom_fecha_index.set_index(prom_fecha_index['Fecha'], inplace=True)
prom_fecha_index.drop(columns=['Fecha'], inplace=True)
# Nos aseguramos que el indice sea de tipo datetime
prom_fecha_index.index = pd.to_datetime(prom_fecha_index.index)
# Fijamos una frecuencia mensual
prom_fecha_index = prom_fecha_index.asfreq('MS')

# Argumentos para ARIMA y SARIMA
d= 0

if arima_opc:
    st.write("## Resultados de las predicciones con ARIMA")
    if opc == "Temperatura máxima":
        
        # Verificamos la estacionariedad de la serie
        if not Estacionaria(prom_fecha_index['Tmax']):
            d = 1
        p = 2
        q = 9
        
        st.markdown("#### Predicciones de la temperatura máxima")
        pred_arima = Arima_f(prom_fecha_index, "Tmax", d, p, q)
        st.write('##### Tabla de predicciones')
        st.write(pred_arima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmax", pred_arima)
    elif opc == "Temperatura media":
        
        # Verificamos la estacionariedad de la serie
        if not Estacionaria(prom_fecha_index['Tmed']):
            d = 1
        p = 15
        q = 6
        
        st.markdown("#### Predicciones de la temperatura media")
        pred_arima = Arima_f(prom_fecha_index, "Tmed", d, p, q)
        st.write('##### Tabla de predicciones')
        st.write(pred_arima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmed", pred_arima)
    else:
        
        # Verificamos la estacionariedad de la serie
        if not Estacionaria(prom_fecha_index['Tmin']):
            d = 1
        p = 16
        q = 15
        
        st.markdown("#### Predicciones de la temperatura mínima")
        pred_arima = Arima_f(prom_fecha_index, "Tmin", d, p, q)
        st.write('##### Tabla de predicciones')
        st.write(pred_arima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmin", pred_arima)

if sarima_opc:
    
    st.write("## Resultados de las predicciones con SARIMA")
    if opc == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        
        #! Parametros para SARIMA
        if not Estacionaria(prom_fecha_index['Tmax']):
            d = 1
        
        # Diferenciar estacionalmente
        prom_fecha_index['Tmax_seasonal_diff'] = prom_fecha_index['Tmax'] - prom_fecha_index['Tmax'].shift(12)
        # Prueba ADF para la serie diferenciada estacionalmente
        result_seasonal_diff = adfuller(prom_fecha_index['Tmax_seasonal_diff'].dropna())
        # Si después de la diferenciación estacional la serie aún no es estacionaria, D = 1
        if result_seasonal_diff[1] > 0.05:
            D = 1
        else:
            D = 0
        P = 0 # PACF
        Q = 0 # 
        
        p = 2
        q = 9
        
        pred_sarima = Sarima_f(prom_fecha_index, "Tmax", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmax", pred_sarima)
    elif opc == "Temperatura media":
        
        #! Parametros para SARIMA
        if not Estacionaria(prom_fecha_index['Tmed']):
            d = 1
        
        # Diferenciar estacionalmente
        prom_fecha_index['Tmed_seasonal_diff'] = prom_fecha_index['Tmed'] - prom_fecha_index['Tmed'].shift(12)
        # Prueba ADF para la serie diferenciada estacionalmente
        result_seasonal_diff = adfuller(prom_fecha_index['Tmed_seasonal_diff'].dropna())
        # Si después de la diferenciación estacional la serie aún no es estacionaria, D = 1
        if result_seasonal_diff[1] > 0.05:
            D = 1
        else:
            D = 0
        P = 0 # PASCF
        Q = 0 # ACF
        
        p = 4
        q = 6        
        
        st.markdown("#### Predicciones de la temperatura media")
        pred_sarima = Sarima_f(prom_fecha_index, "Tmed", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmed", pred_sarima)
    else:

        #! Parametros para SARIMA
        if not Estacionaria(prom_fecha_index['Tmin']):
            d = 1
        
        # Diferenciar estacionalmente
        prom_fecha_index['Tmin_seasonal_diff'] = prom_fecha_index['Tmin'] - prom_fecha_index['Tmin'].shift(12)
        # Prueba ADF para la serie diferenciada estacionalmente
        result_seasonal_diff = adfuller(prom_fecha_index['Tmin_seasonal_diff'].dropna())
        # Si después de la diferenciación estacional la serie aún no es estacionaria, D = 1
        if result_seasonal_diff[1] > 0.05:
            D = 1
        else:
            D = 0
        P = 1 # PASCF
        Q = 1 # ACF
        
        p = 4
        q = 6
        
        
        st.markdown("#### Predicciones de la temperatura mínima")
        pred_sarima = Sarima_f(prom_fecha_index, "Tmin", d, p, q, D, P, Q)
        st.write('##### Tabla de predicciones')
        st.write(pred_sarima)
        st.markdown("##### Gráficas de las predicciones")
        Graph_arima(prom_fecha_index, "Tmin", pred_sarima)

if regresion_lineal_opc:
    st.write("## Resultados de las predicciones con Regresión Lineal")
    # Creamos un DataFrame con las temperaturas predichas por la regresión lineal
    pred_linear = pd.DataFrame()

    pred_linear = Regresion_lineal(estaciones, predictores, pred_linear)
    
    if opc == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        st.markdown("##### Tabla de predicciones")
        rli_tmax = Tabla_df(pred_linear, "Tmax")
        st.write(rli_tmax)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_linear, "Tmax", "Predicciones de la temperatura máxima")
    elif opc == "Temperatura media":
        st.markdown("#### Predicciones de la temperatura media")
        st.markdown("##### Tabla de predicciones")
        rli_tmed = Tabla_df(pred_linear, "Tmed")
        st.write(rli_tmed)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_linear, "Tmed", "Predicciones de la temperatura media")
    else:
        st.markdown("#### Predicciones de la temperatura mínima")
        st.markdown("##### Tabla de predicciones")
        rli_tmin = Tabla_df(pred_linear, "Tmin")
        st.write(rli_tmin)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_linear, "Tmin", "Predicciones de la temperatura mínima")

if regresion_logistica_opc:
    st.write("## Resultados de las predcciones de Regresión logístisca")

    # Creamos un DataFrame con las temperaturas predichas por la regresión logística
    pred_logistica = pd.DataFrame()

    pred_logistica = Regresion_logistica(estaciones, predictores, pred_logistica)

    if opc == "Temperatura máxima":
        st.markdown("#### Probabilidad de que la temperatura máxima sea mayor a 33 grados")
        st.markdown("##### Tabla de predicciones")
        rlog_tmax = Tabla_df(pred_logistica, "Tmax")
        st.write(rlog_tmax)
    elif opc == "Temperatura media":
        st.markdown("#### Probabilidad de que la temperatura media sea mayor a 27 grados")
        st.markdown("##### Tabla de predicciones")
        rlog_tmed = Tabla_df(pred_logistica, "Tmed")
        st.write(rlog_tmed)
    else:
        st.markdown("#### Probabilidad de que la temperatura mínima sea mayor a 20 grados")
        st.markdown("##### Tabla de predicciones")
        rlog_tmin = Tabla_df(pred_logistica, "Tmin")
        st.write(rlog_tmin)

if svm_opc:
    st.write("## Resultados de las predicciones con Máquinas de vectores de soporte")
    
    # Creamos un DataFrame con las temperaturas predichas por el modelo de SVM
    pred_svm = pd.DataFrame()
    
    pred_svm = MSV(estaciones, predictores, pred_svm)
    
    if opc == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        st.markdown("##### Tabla de predicciones")
        svm_tmax = Tabla_df(pred_svm, "Tmax")
        st.write(svm_tmax)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_svm, "Tmax", "Predicciones de la temperatura máxima")
    elif opc == "Temperatura media":
        st.markdown("#### Predicciones de la temperatura media")
        st.markdown("##### Tabla de predicciones")
        svm_tmed = Tabla_df(pred_svm, "Tmed")
        st.write(svm_tmed)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_svm, "Tmed", "Predicciones de la temperatura media")
    else:
        st.markdown("#### Predicciones de la temperatura mínima")
        st.markdown("##### Tabla de predicciones")
        svm_tmin = Tabla_df(pred_svm, "Tmin")
        st.write(svm_tmin)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), pred_svm, "Tmin", "Predicciones de la temperatura mínima")
        
if arbol_opc:
    st.write("## Resultados de las predicciones con Árbol de decisión")
    
    # Dataframe para almacenar las predicciones
    predicciones_arbol = pd.DataFrame()
    
    predicciones_arbol = Arbol(estaciones, predictores, predicciones_arbol)
    
    if opc == "Temperatura máxima":
        st.markdown("#### Predicciones de la temperatura máxima")
        st.markdown("##### Tabla de predicciones")
        arbol_tmax = Tabla_df(predicciones_arbol, "Tmax")
        st.write(arbol_tmax)
        st.markdown("##### Gráficas de las predicciones")
        Graficas_pred(estaciones, Promedio_temperaturas(Carga_datos()), predicciones_arbol, "Tmax", "Predicciones de la temperatura máxima")
    elif opc == "Temperatura media":
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
        