import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis de Delitos - ML",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para estilo
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .risk-high { background-color: #ff6b6b; color: white; padding: 5px; border-radius: 5px; }
    .risk-medium { background-color: #ffd166; color: black; padding: 5px; border-radius: 5px; }
    .risk-low { background-color: #06d6a0; color: white; padding: 5px; border-radius: 5px; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🔍 Sistema Inteligente de Análisis de Delitos</h1>', unsafe_allow_html=True)
st.markdown("### Predicción de Zonas de Riesgo usando Machine Learning")

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "🤖 Predicciones ML", "📈 Análisis Temporal", "🗺️ Mapas Interactivos", "🔍 Análisis Detallado"]
)

# Cargar datos
@st.cache_data(ttl=3600)
def load_data():
    # URL de tu CSV en Dropbox
    dropbox_url = "https://www.dropbox.com/scl/fi/auw6jeeasstr5rqp3z7r6/SAT-Propiedad-BU_2017-2023.csv?rlkey=ta6swmxwc14uq5c06yhu56z1y&st=cgazccvq&dl=1"
    
    try:
        response = requests.get(dropbox_url)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), sep=';')
        st.success("✅ Datos cargados correctamente")
        return df
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        return pd.DataFrame()

@st.cache_data
def prepare_ml_data(df):
    """Preparar datos para ML"""
    datos_agrupados = df.groupby(['provincia_nombre', 'departamento_nombre']).agg({
        'cantidad_hechos': 'sum',
        'cantidad_hechos_lugar_via_publ': 'sum',
        'cantidad_hechos_lugar_establec': 'sum',
        'cantidad_hechos_lugar_dom_part': 'sum',
        'cantidad_hechos_arma_de_fuego': 'sum',
        'cantidad_hechos_arma_otra': 'sum',
        'cant_hechos_agrav_por_lesiones': 'sum',
        'cantidad_inculpados': 'sum',
        'cantidad_inculpados_edad_0_15': 'sum',
        'cantidad_inculpados_edad_16_17': 'sum',
        'cantidad_inculpados_edad_mas_18': 'sum'
    }).reset_index()

    # Crear features de riesgo
    datos_agrupados['tasa_violencia'] = datos_agrupados['cantidad_hechos_arma_de_fuego'] / (datos_agrupados['cantidad_hechos'] + 1)
    datos_agrupados['tasa_lesiones'] = datos_agrupados['cant_hechos_agrav_por_lesiones'] / (datos_agrupados['cantidad_hechos'] + 1)
    datos_agrupados['tasa_via_publica'] = datos_agrupados['cantidad_hechos_lugar_via_publ'] / (datos_agrupados['cantidad_hechos'] + 1)
    datos_agrupados['tasa_jovenes'] = (datos_agrupados['cantidad_inculpados_edad_0_15'] + datos_agrupados['cantidad_inculpados_edad_16_17']) / (datos_agrupados['cantidad_inculpados'] + 1)
    
    datos_agrupados = datos_agrupados.replace([np.inf, -np.inf], 0).fillna(0)
    
    return datos_agrupados

@st.cache_resource
def train_model(_datos_agrupados):
    """Entrenar modelo de ML"""
    # Definir clases de riesgo
    percentiles = _datos_agrupados['cantidad_hechos'].quantile([0.25, 0.75, 0.90])
    
    def definir_riesgo(cantidad):
        if cantidad <= percentiles[0.25]:
            return 'BAJO'
        elif cantidad <= percentiles[0.75]:
            return 'MEDIO'
        elif cantidad <= percentiles[0.90]:
            return 'ALTO'
        else:
            return 'MUY_ALTO'
    
    _datos_agrupados['riesgo_real'] = _datos_agrupados['cantidad_hechos'].apply(definir_riesgo)
    
    # Features y target
    features = ['cantidad_hechos', 'tasa_violencia', 'tasa_lesiones', 'tasa_via_publica', 'tasa_jovenes']
    X = _datos_agrupados[features]
    y = _datos_agrupados['riesgo_real']
    
    # Codificar y escalar
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predicciones
    _datos_agrupados['riesgo_predicho'] = label_encoder.inverse_transform(model.predict(X_scaled))
    _datos_agrupados['prob_alto_riesgo'] = model.predict_proba(X_scaled)[:, label_encoder.transform(['ALTO'])[0]]
    
    return model, label_encoder, scaler, features, X_test, y_test, _datos_agrupados

# Cargar datos
df = load_data()
datos_agrupados = prepare_ml_data(df)

# SECCIÓN INICIO
if seccion == "🏠 Inicio":
    st.markdown('<h2 class="section-header">📊 Resumen Ejecutivo</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Hechos", f"{df['cantidad_hechos'].sum():,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Período Analizado", f"{df['anio'].min()}-{df['anio'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Provincias", f"{df['provincia_nombre'].nunique()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tipos de Delito", f"{df['nombre_delito_sat_prop'].nunique()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico rápido de evolución
    st.subheader("Evolución Anual de Hechos")
    hechos_anio = df.groupby('anio')['cantidad_hechos'].sum().reset_index()
    fig = px.line(hechos_anio, x='anio', y='cantidad_hechos', 
                 title='Tendencia de Delitos 2017-2023',
                 markers=True)
    st.plotly_chart(fig, use_container_width=True)

# SECCIÓN PREDICCIONES ML
elif seccion == "🤖 Predicciones ML":
    st.markdown('<h2 class="section-header">🤖 Modelo de Machine Learning - Zonas de Riesgo</h2>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Información del Modelo", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Período de Análisis:**")
            st.write("- Datos históricos: 2017-2023 (7 años)")
            st.write("- Entrenamiento con datos agregados")
            st.write("- Predicciones para el período actual")
            
        with col2:
            st.markdown("**🔍 Criterios Considerados:**")
            st.write("- Cantidad total de hechos")
            st.write("- Tasa de violencia (armas de fuego)")
            st.write("- Tasa de lesiones en hechos")
            st.write("- Frecuencia en vía pública")
            st.write("- Participación de jóvenes")
    
    # Entrenar modelo
    with st.spinner("Entrenando modelo de Machine Learning..."):
        model, label_encoder, scaler, features, X_test, y_test, datos_con_pred = train_model(datos_agrupados)
    
    # Métricas del modelo
    st.subheader("📊 Performance del Modelo")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precisión General", f"{accuracy:.1%}")
    
    with col2:
        st.metric("Zonas Analizadas", len(datos_con_pred))
    
    with col3:
        alto_riesgo = len(datos_con_pred[datos_con_pred['riesgo_predicho'].isin(['ALTO', 'MUY_ALTO'])])
        st.metric("Zonas Alto Riesgo", alto_riesgo)
    
    # Importancia de features
    st.subheader("🎯 Factores Más Importantes")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(feature_importance, x='importance', y='feature', 
                 title='Importancia de Variables en la Predicción',
                 labels={'importance': 'Importancia', 'feature': 'Variable'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Resultados por zona
    st.subheader("📍 Clasificación por Zona Geográfica")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        provincia_filtro = st.selectbox("Filtrar por Provincia:", 
                                       ['Todas'] + list(datos_con_pred['provincia_nombre'].unique()))
    with col2:
        riesgo_filtro = st.selectbox("Filtrar por Nivel de Riesgo:",
                                   ['Todos'] + list(datos_con_pred['riesgo_predicho'].unique()))
    
    # Aplicar filtros
    datos_filtrados = datos_con_pred.copy()
    if provincia_filtro != 'Todas':
        datos_filtrados = datos_filtrados[datos_filtrados['provincia_nombre'] == provincia_filtro]
    if riesgo_filtro != 'Todos':
        datos_filtrados = datos_filtrados[datos_filtrados['riesgo_predicho'] == riesgo_filtro]
    
    # Mostrar resultados
    st.dataframe(
        datos_filtrados[['provincia_nombre', 'departamento_nombre', 'cantidad_hechos', 
                        'riesgo_predicho', 'prob_alto_riesgo']].sort_values('prob_alto_riesgo', ascending=False),
        use_container_width=True,
        height=400
    )

# SECCIÓN ANÁLISIS TEMPORAL (Recordando gráficos anteriores)
elif seccion == "📈 Análisis Temporal":
    st.markdown('<h2 class="section-header">📈 Análisis Temporal y Evolución</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolución anual
        hechos_anio = df.groupby('anio')['cantidad_hechos'].sum().reset_index()
        fig = px.line(hechos_anio, x='anio', y='cantidad_hechos',
                     title='Evolución Anual de Hechos',
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tipos de delito por año
        delitos_anio = df.groupby(['anio', 'nombre_delito_sat_prop'])['cantidad_hechos'].sum().reset_index()
        fig = px.area(delitos_anio, x='anio', y='cantidad_hechos', color='nombre_delito_sat_prop',
                     title='Distribución de Tipos de Delito por Año')
        st.plotly_chart(fig, use_container_width=True)
    
    # Mapa de calor temporal
    st.subheader("🗓️ Mapa de Calor Temporal")
    heatmap_data = df.groupby(['anio', 'mes'])['cantidad_hechos'].sum().unstack(fill_value=0)
    fig = px.imshow(heatmap_data, aspect='auto', title='Hechos por Año y Mes')
    st.plotly_chart(fig, use_container_width=True)

# SECCIÓN MAPAS INTERACTIVOS

# SECCIÓN MAPAS INTERACTIVOS
elif seccion == "🗺️ Mapas Interactivos":
    st.markdown('<h2 class="section-header">🗺️ Análisis Geográfico Interactivo</h2>', unsafe_allow_html=True)
    
    # Verificar qué columnas geográficas están disponibles
    st.subheader("📍 Mapa de Burbujas Interactivo")
    
    
    # Coordenadas básicas de capitales provinciales (solo las esenciales)
    coordenadas = {
        'Buenos Aires': (-58.3816, -34.6037),
        'Ciudad Autónoma de Buenos Aires': (-58.3816, -34.6037),
        'Catamarca': (-65.7785, -28.4696),
        'Chaco': (-58.9833, -27.4667),
        'Chubut': (-65.1026, -43.2999),
        'Córdoba': (-64.1919, -31.4201),
        'Corrientes': (-58.8304, -27.4678),
        'Entre Ríos': (-60.5233, -31.7441),
        'Formosa': (-58.2285, -26.1849),
        'Jujuy': (-65.2971, -24.1858),
        'La Pampa': (-64.2833, -36.6167),
        'La Rioja': (-66.8500, -29.4131),
        'Mendoza': (-68.8420, -32.8908),
        'Misiones': (-55.8962, -27.3621),
        'Neuquén': (-68.0621, -38.9517),
        'Río Negro': (-67.4987, -39.0284),
        'Salta': (-65.4117, -24.7883),
        'San Juan': (-68.5364, -31.5375),
        'San Luis': (-66.3354, -33.2995),
        'Santa Cruz': (-69.2241, -51.6226),
        'Santa Fe': (-60.7089, -31.6107),
        'Santiago del Estero': (-64.2641, -27.7951),
        'Tierra del Fuego': (-68.3020, -54.8072),
        'Tucumán': (-65.2098, -26.8083)
    }
    
    # Opción 1: Si hay columnas de lat/lon en el CSV, usarlas directamente
    if all(col in df.columns for col in ['latitud', 'longitud']):
        st.info("📍 Usando coordenadas del dataset")
        mapa_data = df.groupby('provincia_nombre').agg({
            'cantidad_hechos': 'sum',
            'latitud': 'first',
            'longitud': 'first'
        }).reset_index()
        
    # Opción 2: Si no hay coordenadas, usar el diccionario básico
    else:
        st.info("📍 Usando coordenadas predefinidas")
        hechos_provincia = df.groupby('provincia_nombre')['cantidad_hechos'].sum().reset_index()
        
        mapa_data = []
        for _, row in hechos_provincia.iterrows():
            provincia = row['provincia_nombre']
            if provincia in coordenadas:
                lon, lat = coordenadas[provincia]
                mapa_data.append({
                    'provincia': provincia,
                    'cantidad_hechos': row['cantidad_hechos'],
                    'latitud': lat,
                    'longitud': lon
                })
            else:
                # Para provincias no encontradas, usar coordenadas aproximadas
                mapa_data.append({
                    'provincia': provincia,
                    'cantidad_hechos': row['cantidad_hechos'],
                    'latitud': -34.0,
                    'longitud': -64.0
                })
        
        mapa_data = pd.DataFrame(mapa_data)
    
    # Crear el mapa principal
    if not mapa_data.empty:
        fig = px.scatter_mapbox(mapa_data, 
                               lat="latitud", 
                               lon="longitud", 
                               size="cantidad_hechos",
                               color="cantidad_hechos",
                               size_max=40,
                               zoom=3, 
                               height=600,
                               hover_name="provincia", 
                               hover_data={"cantidad_hechos": True},
                               title="Distribución Geográfica de Hechos por Provincia",
                               color_continuous_scale="reds",
                               labels={"cantidad_hechos": "Cantidad de Hechos"})
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 10, "t": 50, "l": 10, "b": 10},
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar datos en tabla
        with st.expander("📋 Ver datos del mapa"):
            tabla_datos = mapa_data[['provincia', 'cantidad_hechos']].sort_values('cantidad_hechos', ascending=False)
            tabla_datos['cantidad_hechos'] = tabla_datos['cantidad_hechos'].apply(lambda x: f"{x:,}")
            st.dataframe(tabla_datos, use_container_width=True)

    # Análisis por región si existe la columna en el CSV
    st.subheader("📈 Análisis por Regiones")
    
    # Verificar si existe columna de región
    columnas_posibles_regiones = ['region', 'region_nombre', 'zona', 'area']
    columna_region = None
    
    for col in columnas_posibles_regiones:
        if col in df.columns:
            columna_region = col
            break
    
    if columna_region:
        st.info(f"📊 Usando columna '{columna_region}' para análisis regional")
        
        # Análisis por región
        datos_region = df.groupby(columna_region).agg({
            'cantidad_hechos': 'sum',
            'provincia_nombre': 'nunique'
        }).reset_index()
        
        datos_region = datos_region.rename(columns={
            'provincia_nombre': 'cantidad_provincias',
            'cantidad_hechos': 'Total Hechos'
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_region = px.bar(datos_region, x=columna_region, y='Total Hechos',
                               title='Distribución de Hechos por Región',
                               color='Total Hechos',
                               color_continuous_scale='viridis')
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(datos_region, values='Total Hechos', names=columna_region,
                            title='Porcentaje de Hechos por Región')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Mostrar tabla de regiones
        with st.expander("📋 Detalles por región"):
            st.dataframe(datos_region.sort_values('Total Hechos', ascending=False), 
                        use_container_width=True)
    
    else:
        
        # Crear regiones automáticamente basado en provincias
        st.subheader("🗺️ Regiones Automáticas (Agrupadas)")
        
        # Agrupar provincias en regiones automáticamente
        regiones_auto = {
            'Centro': ['Buenos Aires', 'Ciudad Autónoma de Buenos Aires', 'Córdoba', 'Santa Fe', 'Entre Ríos', 'La Pampa'],
            'Noroeste': ['Jujuy', 'Salta', 'Tucumán', 'Catamarca', 'Santiago del Estero'],
            'Noreste': ['Formosa', 'Chaco', 'Corrientes', 'Misiones'],
            'Cuyo': ['Mendoza', 'San Juan', 'San Luis'],
            'Patagonia': ['Río Negro', 'Neuquén', 'Chubut', 'Santa Cruz', 'Tierra del Fuego']
        }
        
        def asignar_region_auto(provincia):
            for region, provincias in regiones_auto.items():
                if provincia in provincias:
                    return region
            return 'Otra'
        
        # Aplicar agrupación regional
        df_temp = df.copy()
        df_temp['region_auto'] = df_temp['provincia_nombre'].apply(asignar_region_auto)
        
        datos_region_auto = df_temp.groupby('region_auto').agg({
            'cantidad_hechos': 'sum',
            'provincia_nombre': 'nunique'
        }).reset_index()
        
        # Mostrar solo el gráfico de torta (sin gráfico de barras ni tablas)
        fig_pie_auto = px.pie(datos_region_auto, values='cantidad_hechos', names='region_auto',
                              title='Distribución por Región Automática')
        st.plotly_chart(fig_pie_auto, use_container_width=True)

    # Mapa alternativo
    st.subheader("🎨 Vista Alternativa")
    
    if not mapa_data.empty:
        fig2 = px.scatter_mapbox(mapa_data, 
                                lat="latitud", 
                                lon="longitud", 
                                size="cantidad_hechos",
                                color="cantidad_hechos",
                                size_max=35,
                                zoom=3, 
                                height=400,
                                hover_name="provincia",
                                title="Vista Compacta del Mapa",
                                color_continuous_scale="bluered")
        
        fig2.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 10, "t": 40, "l": 10, "b": 10}
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# SECCIÓN ANÁLISIS DETALLADO
elif seccion == "🔍 Análisis Detallado":
    st.markdown('<h2 class="section-header">🔍 Análisis Detallado por Categorías</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lugares de los hechos
        lugares = {
            'Vía Pública': df['cantidad_hechos_lugar_via_publ'].sum(),
            'Establecimiento': df['cantidad_hechos_lugar_establec'].sum(),
            'Domicilio': df['cantidad_hechos_lugar_dom_part'].sum()
        }
        fig = px.pie(values=list(lugares.values()), names=list(lugares.keys()),
                    title='Distribución por Lugar del Hecho')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Armas utilizadas
        armas = {
            'Arma de Fuego': df['cantidad_hechos_arma_de_fuego'].sum(),
            'Otra Arma': df['cantidad_hechos_arma_otra'].sum(),
            'Sin Arma': df['cantidad_hechos_arma_sin_arma'].sum()
        }
        fig = px.pie(values=list(armas.values()), names=list(armas.keys()),
                    title='Tipo de Arma Utilizada')
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Edad de inculpados
        edades = {
            '0-17 años': (df['cantidad_inculpados_edad_0_15'] + df['cantidad_inculpados_edad_16_17']).sum(),
            '18+ años': df['cantidad_inculpados_edad_mas_18'].sum()
        }
        fig = px.bar(x=list(edades.keys()), y=list(edades.values()),
                    title='Edad de los Inculpados',
                    labels={'x': 'Grupo Etario', 'y': 'Cantidad'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Agravantes
        agravantes = {
            'Con Lesiones': df['cant_hechos_agrav_por_lesiones'].sum(),
            'Sin Lesiones': df['cant_hechos_agrav_sin_lesiones'].sum()
        }
        fig = px.bar(x=list(agravantes.keys()), y=list(agravantes.values()),
                    title='Hechos con y sin Lesiones',
                    color=list(agravantes.keys()),
                    color_discrete_map={'Con Lesiones': 'red', 'Sin Lesiones': 'blue'})
        st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    "**Sistema desarrollado para análisis predictivo de delitos** | "
    "Datos: SAT 2017-2023 | "
    "Modelo: Random Forest"
)