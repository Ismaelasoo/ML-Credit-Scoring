# Credit Scoring

## Tabla de Contenidos
- [Credit Scoring](#credit-scoring)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Descripción](#descripción)
  - [Resumen](#resumen)
  - [Database](#database)
  - [Análisis](#análisis)
  - [Modelado](#modelado)

## Descripción
Este proyecto se centra en el desarrollo de un modelo de **Credit Scoring** utilizando técnicas de **Machine Learning**. El objetivo principal es predecir el riesgo crediticio de los clientes de una institución financiera en base a un conjunto de variables financieras. A través del uso de modelos de regresión y técnicas avanzadas de interpretación como **SHAP**, se busca mejorar la precisión de las predicciones mientras se garantiza la transparencia y el cumplimiento de regulaciones.

El modelo se entrena con un conjunto de datos realista que contiene características relacionadas con el perfil financiero de los clientes. La implementación de **XAI (Inteligencia Artificial Explicable)** permite interpretar y visualizar los factores que influyen en las predicciones, lo que facilita la toma de decisiones informadas.

## Resumen
Este proyecto explora cómo el uso de modelos de **Machine Learning** puede mejorar la predicción del riesgo crediticio en comparación con los enfoques tradicionales. Se comparan varios algoritmos de regresión, como **Regresión Lineal con Regularización (Ridge, Lasso)**, **Árboles de Decisión**, **Random Forest**, **XGBoost** y **Redes Neuronales**. Además, se incorpora la librería **SHAP** para mejorar la interpretabilidad de los modelos, permitiendo desglosar el impacto de cada variable en la predicción. El proyecto demuestra que, a pesar de la alta precisión de los modelos avanzados, la interpretabilidad es clave para su implementación efectiva en entornos regulados.

## Database
El conjunto de datos utilizado proviene de una fuente realista de clientes bancarios, que contiene diversas variables financieras, como ingresos, historial de pagos, nivel de deuda y otras características relacionadas con el comportamiento financiero. El dataset se procesa mediante una serie de pasos de **limpieza** y **preprocesamiento** para eliminar valores nulos, manejar valores atípicos y normalizar las variables, asegurando la calidad y representatividad de los datos. 

Además, se realiza un análisis exploratorio para identificar patrones y correlaciones significativas entre las variables que ayudan a predecir el riesgo crediticio.

## Análisis
En esta etapa, se realiza un análisis estadístico y gráfico de las variables, explorando la relación entre las características financieras de los clientes y el riesgo crediticio. Se utilizan técnicas como:

- **Análisis Descriptivo**: Para identificar la distribución y características clave de las variables.
- **Correlación**: Para identificar qué variables tienen una relación significativa con la variable objetivo (riesgo crediticio).
- **Visualización**: Mediante gráficos de barras, histogramas, y diagramas de dispersión para explorar visualmente las relaciones entre las variables.

A través de estas técnicas, se obtienen conocimientos valiosos que guían la selección de modelos y el preprocesamiento de datos.

## Modelado
El modelo se desarrolla utilizando diversas técnicas de **Machine Learning** para predecir el riesgo crediticio. Los pasos incluyen:

1. **Selección de modelos**: Se prueban varios algoritmos, como **Regresión Lineal**, **Árboles de Decisión**, **Random Forest**, **XGBoost** y **Redes Neuronales**.
2. **Ajuste de hiperparámetros**: Se utiliza **Grid Search** para optimizar los parámetros de los modelos y mejorar su rendimiento.
3. **Evaluación de modelos**: Se evalúan los modelos usando métricas como el **Error Cuadrático Medio (MSE)**, el **Error Absoluto Medio (MAE)** y el **Coeficiente de Determinación (R²)**.
4. **Interpretabilidad**: Para garantizar que los modelos sean comprensibles, se utiliza la librería **SHAP** para desglosar la contribución de cada variable en la predicción final, mejorando la transparencia del modelo.

El resultado es un modelo capaz de predecir con alta precisión el riesgo crediticio, con la capacidad de explicar las decisiones tomadas por el modelo, lo que es esencial en un entorno financiero regulado.
