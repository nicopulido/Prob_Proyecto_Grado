# Construcción de modelo LSTM para el pronóstico de evapotranspiración de referencia y heladas meteorológicas (V1)

## Flujo de trabajo

El proyecto se desarrolla mediante las siguientes etapas:

### 1. Recolección y organización de datos

* Obtención de datos meteorológicos de las estaciones seleccionadas.
* Identificación de variables, periodo y resolución temporal.
* Integración de los diferentes conjuntos de datos.
* Organización y estandarización de fechas y unidades.

### 2. Control de calidad y preprocesamiento

* Identificación de valores faltantes.
* Detección de registros duplicados o inconsistentes.
* Verificación de rangos físicamente posibles.
* Análisis de valores extremos.
* Tratamiento de datos faltantes y registros inconsistentes.

### 3. Análisis exploratorio

* Estadística descriptiva de las variables.
* Distribución de las variables meteorológicas.
* Análisis de correlaciones.
* Identificación de relaciones entre las variables de entrada y las variables objetivo.

### 4. Análisis temporal

* Visualización de las series temporales.
* Identificación de tendencias y patrones estacionales.
* Análisis de variabilidad temporal.
* Identificación de periodos con cambios o comportamientos atípicos.
* Análisis de la frecuencia y distribución temporal de eventos de helada.

### 5. Construcción de variables objetivo

* Cálculo de la evapotranspiración de referencia (ET₀) mediante FAO-56 Penman-Monteith, cuando los datos disponibles lo permitan.
* Definición de la ocurrencia de heladas mediante un umbral de temperatura mínima.
* Evaluación de la distribución y frecuencia de las clases.

### 6. Preparación de datos para aprendizaje automático

* Selección de variables de entrada.
* Construcción de secuencias temporales.
* Definición de ventanas temporales para el modelo.
* División cronológica en conjuntos de entrenamiento, validación y prueba.
* Normalización de las variables utilizando únicamente los datos de entrenamiento.

### 7. Desarrollo de modelos predictivos

Se implementarán modelos para dos tareas:

**Pronóstico de ET₀**

* Modelo base.
* Modelos de aprendizaje automático.
* Red neuronal LSTM.

**Predicción de heladas**

* Modelo base.
* Modelos de aprendizaje automático.
* Red neuronal LSTM.

### 8. Evaluación de los modelos

Para ET₀:

* MAE.
* RMSE.
* R².

Para heladas:

* Precision.
* Recall.
* F1-score.
* Matriz de confusión.
* PR-AUC, cuando sea necesario debido al desbalance de clases.

### 9. Análisis de resultados

* Comparación entre modelos.
* Análisis de errores de predicción.
* Identificación de condiciones en las que los modelos presentan mayor dificultad.
* Evaluación del comportamiento ante eventos extremos.
* Análisis del efecto de diferentes ventanas temporales.

### 10. Visualización y plataforma

* Visualización de predicciones y resultados.
* Consulta de datos meteorológicos.
* Presentación de métricas de evaluación.
* Integración de los resultados en una plataforma web como herramienta de consulta y apoyo.

## Flujo general

```text
Datos meteorológicos
        ↓
Control de calidad
        ↓
Preprocesamiento
        ↓
Análisis exploratorio
        ↓
Análisis temporal
        ↓
Construcción de ET₀ y heladas
        ↓
Preparación de secuencias
        ↓
Entrenamiento de modelos
        ↓
Evaluación
        ↓
Análisis de resultados
        ↓
Visualización / Plataforma web
```
