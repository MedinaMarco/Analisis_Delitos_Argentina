# Análisis de Delitos en Argentina

Sistema de análisis y predicción de delitos en Argentina utilizando datos del Sistema de Análisis Territorial (SAT).

##  Sobre el Proyecto

Este proyecto analiza los datos de delitos ocurridos en Argentina desde 2017 hasta 2023, proporcionando:

- **Dashboard Exploratorio**: Visualización interactiva de tendencias y patrones
- **Modelo Predictivo**: Machine Learning para identificar zonas de riesgo
- **Análisis Geográfico**: Mapas interactivos y distribución territorial
- **Análisis Temporal**: Evolución de delitos a través del tiempo

##  Características Principales

### Dashboard Exploratorio
- Análisis temporal y evolución de delitos
- Distribución geográfica por provincias y departamentos
- Análisis detallado por categorías (lugares, armas, edades)
- Métricas ejecutivas y resúmenes

###  Modelo Predictivo (ML)
- **Algoritmo**: Random Forest Classifier
- **Objetivo**: Clasificación de zonas de riesgo (BAJO, MEDIO, ALTO, MUY_ALTO)
- **Features**: Tasa de violencia, lesiones, frecuencia en vía pública, participación juvenil
- **Precisión**: 100%

###  Visualizaciones
- Mapas interactivos con Plotly
- Gráficos temporales y de distribución
- Análisis por regiones automáticas
- Heatmaps y gráficos de burbujas

##  Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
Streamlit
numpy
Pandas
seaborn
Scikit-learn
Plotly
geopy
