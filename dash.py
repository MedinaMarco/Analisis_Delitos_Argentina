import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis de Delitos",
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
st.markdown('<h1 class="main-header">🔍 Sistema de Análisis de Delitos SAT</h1>', unsafe_allow_html=True)
st.markdown("### Análisis de datos del Sistema de Análisis Territorial 2017-2023")

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📈 Análisis Temporal", "🗺️ Análisis Geográfico", "🔍 Análisis Detallado"]
)

# Cargar datos desde Dropbox
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

# Cargar datos
df = load_data()
if df.empty:
    st.stop()

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
    
    # Gráficos nativos de Streamlit
    st.subheader("📈 Evolución Anual de Hechos")
    hechos_anio = df.groupby('anio')['cantidad_hechos'].sum()
    st.line_chart(hechos_anio)
    
    st.subheader("🏙️ Top 10 Provincias")
    top_provincias = df.groupby('provincia_nombre')['cantidad_hechos'].sum().nlargest(10)
    st.bar_chart(top_provincias)

# SECCIÓN ANÁLISIS TEMPORAL
# SECCIÓN ANÁLISIS TEMPORAL
elif seccion == "📈 Análisis Temporal":
    st.markdown('<h2 class="section-header">📈 Análisis Temporal</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Evolución Mensual")
        df['periodo'] = df['anio'].astype(str) + '-' + df['mes']
        hechos_mensuales = df.groupby('periodo')['cantidad_hechos'].sum().tail(24)
        st.line_chart(hechos_mensuales)
        st.caption("Últimos 24 meses de actividad")
    
    with col2:
        st.subheader("📊 Distribución por Mes")
        hechos_por_mes = df.groupby('mes')['cantidad_hechos'].sum()
        st.bar_chart(hechos_por_mes)
        st.caption("Acumulado histórico por mes")
    
    # Heatmap mejorado - VERSIÓN PROFESIONAL
    st.subheader("🗓️ Matriz Temporal: Hechos por Año y Mes")
    
    heatmap_data = df.groupby(['anio', 'mes'])['cantidad_hechos'].sum().unstack(fill_value=0)
    
    # 1. Mostrar tabla con formato mejorado
    st.dataframe(
        heatmap_data.style.format('{:,.0f}'),  # Formato con separadores de miles
        use_container_width=True,
        height=400
    )
    
    # 2. Agregar métricas clave
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mes_max = heatmap_data.sum(axis=1).idxmax()
        st.metric("Año con más hechos", f"{mes_max}", f"{heatmap_data.sum(axis=1).max():,}")
    
    with col2:
        mes_mas_activo = heatmap_data.sum().idxmax()
        st.metric("Mes más activo", mes_mas_activo, f"{heatmap_data.sum().max():,}")
    
    with col3:
        crecimiento = ((heatmap_data.sum(axis=1).iloc[-1] - heatmap_data.sum(axis=1).iloc[0]) / heatmap_data.sum(axis=1).iloc[0] * 100)
        st.metric("Crecimiento total", f"{crecimiento:+.1f}%")
    
    # 3. Gráfico de tendencia anual
    st.subheader("📈 Tendencia Anual Comparada")
    
    # Ordenar meses lógicamente
    meses_orden = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    # Reordenar columns
    heatmap_data = heatmap_data.reindex(columns=meses_orden, fill_value=0)
    
    # Mostrar como área chart por año
    for año in sorted(heatmap_data.index, reverse=True)[:3]:  # Últimos 3 años
        datos_año = heatmap_data.loc[año]
        st.write(f"**{año}** - Total: {datos_año.sum():,} hechos")
        st.area_chart(datos_año)

# SECCIÓN ANÁLISIS GEOGRÁFICO
elif seccion == "🗺️ Análisis Geográfico":
    st.markdown('<h2 class="section-header">🗺️ Análisis Geográfico</h2>', unsafe_allow_html=True)
    
    # Métricas rápidas geográficas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_provincias = df['provincia_nombre'].nunique()
        st.metric("Provincias con datos", total_provincias)
    
    with col2:
        total_departamentos = df['departamento_nombre'].nunique()
        st.metric("Departamentos", total_departamentos)
    
    with col3:
        provincia_max = df.groupby('provincia_nombre')['cantidad_hechos'].sum().idxmax()
        st.metric("Provincia líder", provincia_max)
    
    st.markdown("---")
    
    # Top provincias - MEJORADO
    st.subheader("🏆 Ranking Provincial")
    top_provincias = df.groupby('provincia_nombre')['cantidad_hechos'].sum().nlargest(15)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(top_provincias)
    
    with col2:
        # Mostrar ranking numérico
        st.write("**Top 5 Provincias:**")
        for i, (provincia, total) in enumerate(top_provincias.head().items(), 1):
            st.write(f"{i}. **{provincia}**: {total:,}")
    
    st.markdown("---")
    
    # Análisis por departamento - MEJORADO
    st.subheader("📍 Análisis Departamental")

    # Crear datos agrupados
    departamentos_data = df.groupby(['provincia_nombre', 'departamento_nombre'])['cantidad_hechos'].sum().reset_index()
    departamentos_data = departamentos_data.rename(columns={
        'provincia_nombre': 'Provincia', 
        'departamento_nombre': 'Departamento',
        'cantidad_hechos': 'Total Hechos'
    })

    # Filtros de búsqueda
    col1, col2 = st.columns(2)

    with col1:
        # Selector de provincia
        todas_provincias = ['Todas'] + sorted(departamentos_data['Provincia'].unique())
        provincia_seleccionada = st.selectbox("🔍 Filtrar por Provincia:", todas_provincias)

    with col2:
        # Selector de departamento (dependiente de la provincia)
        if provincia_seleccionada != 'Todas':
            departamentos_filtrados = sorted(departamentos_data[departamentos_data['Provincia'] == provincia_seleccionada]['Departamento'].unique())
        else:
            departamentos_filtrados = sorted(departamentos_data['Departamento'].unique())
    
        todos_departamentos = ['Todos'] + departamentos_filtrados
        departamento_seleccionado = st.selectbox("🏙️ Filtrar por Departamento:", todos_departamentos)

    # Aplicar filtros
    datos_filtrados = departamentos_data.copy()

    if provincia_seleccionada != 'Todas':
        datos_filtrados = datos_filtrados[datos_filtrados['Provincia'] == provincia_seleccionada]

    if departamento_seleccionado != 'Todos':
        datos_filtrados = datos_filtrados[datos_filtrados['Departamento'] == departamento_seleccionado]

    # Ordenar por Total Hechos (descendente) y agregar ranking
    datos_filtrados = datos_filtrados.sort_values('Total Hechos', ascending=False)
    datos_filtrados['Ranking'] = range(1, len(datos_filtrados) + 1)
    datos_filtrados = datos_filtrados[['Ranking', 'Provincia', 'Departamento', 'Total Hechos']]

    # Mostrar resultados
    st.write(f"**Resultados encontrados:** {len(datos_filtrados)} departamentos")

    st.dataframe(
        datos_filtrados.style.format({'Total Hechos': '{:,}'}),
        use_container_width=True,
        height=500
    )

# SECCIÓN ANÁLISIS DETALLADO
elif seccion == "🔍 Análisis Detallado":
    st.markdown('<h2 class="section-header">🔍 Análisis Detallado por Categorías</h2>', unsafe_allow_html=True)
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_hechos = df['cantidad_hechos'].sum()
        st.metric("Hechos Totales", f"{total_hechos:,}")
    
    with col2:
        total_inculpados = df['cantidad_inculpados'].sum()
        st.metric("Inculpados", f"{total_inculpados:,}")
    
    with col3:
        tasa_lesiones = (df['cant_hechos_agrav_por_lesiones'].sum() / total_hechos * 100)
        st.metric("Tasa Lesiones", f"{tasa_lesiones:.1f}%")
    
    with col4:
        tasa_armas = (df['cantidad_hechos_arma_de_fuego'].sum() / total_hechos * 100)
        st.metric("Tasa Armas Fuego", f"{tasa_armas:.1f}%")
    
    st.markdown("---")
    
    # PRIMERA FILA: Lugares y Armas
    st.subheader("📍 Contexto de los Hechos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏢 Lugares de Ocurrencia**")
        lugares_data = {
            'Vía Pública': df['cantidad_hechos_lugar_via_publ'].sum(),
            'Establecimiento': df['cantidad_hechos_lugar_establec'].sum(),
            'Domicilio': df['cantidad_hechos_lugar_dom_part'].sum(),
            'Sin Dato': df['cantidad_hechos_lugar_sd'].sum()
        }
        
        lugares_df = pd.DataFrame(list(lugares_data.items()), columns=['Lugar', 'Total'])
        lugares_df['Porcentaje'] = (lugares_df['Total'] / total_hechos * 100).round(1)
        
        # Mostrar con formato mejorado
        for _, row in lugares_df.iterrows():
            st.write(f"• **{row['Lugar']}**: {row['Total']:,} ({row['Porcentaje']}%)")
        
        # Gráfico de torta alternativo
        st.bar_chart(lugares_df.set_index('Lugar')['Total'])
    
    with col2:
        st.markdown("**🔫 Medios Utilizados**")
        armas_data = {
            'Arma de Fuego': df['cantidad_hechos_arma_de_fuego'].sum(),
            'Otra Arma': df['cantidad_hechos_arma_otra'].sum(),
            'Sin Arma': df['cantidad_hechos_arma_sin_arma'].sum(),
            'Sin Dato': df['cantidad_hechos_arma_sd'].sum()
        }
        
        armas_df = pd.DataFrame(list(armas_data.items()), columns=['Tipo Arma', 'Total'])
        armas_df['Porcentaje'] = (armas_df['Total'] / total_hechos * 100).round(1)
        
        # Mostrar con formato mejorado
        for _, row in armas_df.iterrows():
            st.write(f"• **{row['Tipo Arma']}**: {row['Total']:,} ({row['Porcentaje']}%)")
        
        st.bar_chart(armas_df.set_index('Tipo Arma')['Total'])
    
    st.markdown("---")
    
    # SEGUNDA FILA: Perfil de Inculpados y Agravantes
    st.subheader("👤 Perfil de los Inculpados")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**📊 Distribución por Edad**")
        edades_data = {
            '0-15 años': df['cantidad_inculpados_edad_0_15'].sum(),
            '16-17 años': df['cantidad_inculpados_edad_16_17'].sum(),
            '18+ años': df['cantidad_inculpados_edad_mas_18'].sum(),
            'Sin Dato': df['cantidad_inculpados_edad_sd'].sum()
        }
        
        edades_df = pd.DataFrame(list(edades_data.items()), columns=['Grupo Edad', 'Total'])
        if total_inculpados > 0:
            edades_df['Porcentaje'] = (edades_df['Total'] / total_inculpados * 100).round(1)
        
        for _, row in edades_df.iterrows():
            porcentaje = row['Porcentaje'] if total_inculpados > 0 else 0
            st.write(f"• **{row['Grupo Edad']}**: {row['Total']:,} ({porcentaje}%)")
        
        st.bar_chart(edades_df.set_index('Grupo Edad')['Total'])
    
    with col4:
        st.markdown("**🚨 Gravedad de los Hechos**")
        agravantes_data = {
            'Con Lesiones': df['cant_hechos_agrav_por_lesiones'].sum(),
            'Sin Lesiones': df['cant_hechos_agrav_sin_lesiones'].sum()
        }
        
        agravantes_df = pd.DataFrame(list(agravantes_data.items()), columns=['Agravante', 'Total'])
        agravantes_df['Porcentaje'] = (agravantes_df['Total'] / total_hechos * 100).round(1)
        
        for _, row in agravantes_df.iterrows():
            st.write(f"• **{row['Agravante']}**: {row['Total']:,} ({row['Porcentaje']}%)")
        
        st.bar_chart(agravantes_df.set_index('Agravante')['Total'])
    
    st.markdown("---")
    
    # TIPOS DE DELITO - MEJORADO
    st.subheader("📋 Tipología Delictiva")
    
    delitos_comunes = df.groupby('nombre_delito_sat_prop')['cantidad_hechos'].sum().nlargest(12)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(delitos_comunes)
    
    with col2:
        st.write("**Top 6 Delitos:**")
        for i, (delito, total) in enumerate(delitos_comunes.head(6).items(), 1):
            porcentaje = (total / total_hechos * 100)
            st.write(f"{i}. **{delito}**")
            st.write(f"   {total:,} ({porcentaje:.1f}%)")

# Footer
st.markdown("---")
st.markdown("**Dashboard Delitos SAT** • Datos 2017-2023 • Desarrollado con Streamlit")