import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def agregacion_pca(df, n, *col):
    """
    Realiza una agregación de variables utilizando PCA, reduciendo las dimensiones
    de las columnas seleccionadas y manteniendo la variabilidad temporal.
    
    Parámetros:
    df: DataFrame de entrada con las columnas a analizar.
    n: Número de componentes principales a retener.
    *col: Columnas que se usarán para el PCA.
    
    Retorna:
    Una Serie o DataFrame con las componentes principales obtenidas.
    """
    # Comprobar que las columnas existen en el DataFrame
    for c in col:
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no está en el DataFrame.")
    
    # Selecciona y estandariza las columnas indicadas
    df_subset = df[list(col)]
    df_scaled = StandardScaler().fit_transform(df_subset)
    
    # Aplica PCA
    pca = PCA(n_components=n)
    df_pca = pca.fit_transform(df_scaled)
    
    # Si n=1, devolver una Serie con la componente principal
    if n == 1:
        return pd.Series(df_pca.flatten(), index=df.index)
    
    # Si n > 1, devolver un DataFrame con las componentes principales
    return pd.DataFrame(df_pca, index=df.index, columns=[f'PC{i+1}' for i in range(n)])


def evaluacion_coeficientes(modelo1, modelo2, X_train, y_train):
    """
    Evalúa la evolución de los coeficientes de un modelo de regresión (por ejemplo, Ridge, Lasso, ElasticNet) en función
    de diferentes valores de alpha, graficando la evolución de los coeficientes según el cambio de regularización.
    
    Parámetros:
        modelo1: El modelo entrenado (por ejemplo, RidgeCV, LassoCV, ElasticNetCV) que contiene el conjunto de alphas.
        modelo2: El modelo base para entrenamiento (por ejemplo, Ridge, Lasso o ElasticNet) que será entrenado para cada alpha.
        X_train: Datos de entrenamiento (características) del modelo.
        y_train: Datos de entrenamiento (variable objetivo) del modelo.
        
    Resultado:
        None: La función genera un gráfico de la evolución de los coeficientes y no devuelve ningún valor.
    """
    
    # Obtener los alphas que se evaluarán para el modelo
    alphas = modelo1.alphas
    
    # Inicializar una lista para almacenar los coeficientes para cada valor de alpha
    coefs = []
    
    # Entrenamiento del modelo base para cada valor de alpha
    for alpha in alphas:
        # Inicializa el modelo base con el valor de alpha actual y sin el intercepto
        modelo_temp = modelo2(alpha=alpha, fit_intercept=False)
        
        # Entrenar el modelo con los datos de entrenamiento
        modelo_temp.fit(X_train, y_train)
        
        # Guardar los coeficientes aplanados del modelo entrenado
        coefs.append(modelo_temp.coef_.flatten())
    
    # Graficar la evolución de los coeficientes en función de los valores de alpha
    graficar_evolucion_coeficientes(alphas, coefs, modelo1)
    
    
def graficar_error_cv(modelo):
    """
    Esta función calcula el error medio cuadrático (RMSE) y su desviación estándar
    para un modelo de regresión (RidgeCV, LassoCV, ElasticNetCV), y grafica el resultado
    con los valores óptimos de alpha.

    Parámetros:
        modelo: El modelo entrenado (por ejemplo, RidgeCV, LassoCV, ElasticNetCV).

    Resultado:
        float: El valor óptimo de alpha encontrado.
    """
    # Verificar si el modelo tiene mse_path_ (para LassoCV, ElasticNetCV, etc.)
    if hasattr(modelo, 'mse_path_'):
        mse_cv = modelo.mse_path_.mean(axis=1)
        mse_sd = modelo.mse_path_.std(axis=1)
        # Cálculo del error cuadrático medio (RMSE) y su desviación estándar
        rmse_cv, rmse_sd = np.sqrt(mse_cv), np.sqrt(mse_sd)

        # Selección del mejor valor de alpha y el valor de alpha + 1 desviación estándar
        optimo, optimo_1sd = seleccionar_alpha_optimo(rmse_cv, rmse_sd, modelo.alphas_)

        # Gráfico del error con intervalo de +- 1 desviación estándar
        fig, ax = plt.subplots(figsize=(7, 3.84))
        ax.plot(modelo.alphas_, rmse_cv)
        ax.fill_between(
            modelo.alphas_,
            rmse_cv + rmse_sd,
            rmse_cv - rmse_sd,
            color="red",
            alpha=0.2
        )

    elif hasattr(modelo, 'cv_values_'):
        mse_cv = modelo.cv_values_.reshape((-1, 200)).mean(axis=0)
        mse_sd = modelo.cv_values_.reshape((-1, 200)).std(axis=0)
        rmse_cv, rmse_sd = np.sqrt(mse_cv), np.sqrt(mse_sd)

        # Selección del mejor valor de alpha y el valor de alpha + 1 desviación estándar
        optimo, optimo_1sd = seleccionar_alpha_optimo(rmse_cv, rmse_sd, modelo.alphas)

        # Gráfico del error con intervalo de +- 1 desviación estándar
        fig, ax = plt.subplots(figsize=(7, 3.84))
        ax.plot(modelo.alphas, rmse_cv)
        ax.fill_between(
            modelo.alphas,
            rmse_cv + rmse_sd,
            rmse_cv - rmse_sd,
            color="red",
            alpha=0.2
        )
    else:
        raise ValueError("El modelo no contiene 'mse_path_' ni 'cv_values_'. Verifique el ajuste del modelo.")

    # Líneas verticales indicando los valores óptimos de alpha
    ax.axvline(
        x=optimo,
        c="gray",
        linestyle="--",
        label="Óptimo"
    )

    ax.axvline(
        x=optimo_1sd,
        c="blue",
        linestyle="--",
        label="Óptimo + 1 std"
    )

    # Ajustes de la visualización
    ax.set_xscale("log")
    ax.set_ylim([0, None])
    ax.set_title("Evolución del error CV en función de la regularización")
    ax.set_xlabel("alpha")
    ax.set_ylabel("RMSE")
    plt.legend()

    # Impresión de los mejores valores de alpha encontrados
    print(f"Mejor valor de alpha encontrado: {optimo}")
    print(f"Mejor valor de alpha encontrado + 1 desviación estándar: {optimo_1sd}")
    
    return optimo


def evaluar_modelos(modelo, X_test, y_test):
    """
    Evalúa un modelo y almacena sus métricas en una tabla persistente dentro de la función.
    
    Parámetros:
        modelo: El objeto del modelo entrenado.
        X_test: Datos de prueba para las características.
        y_test: Datos de prueba para la variable objetivo.

    Resultado:
        pandas.DataFrame: La tabla con las métricas actualizadas.
    """
    
    # Si la tabla de métricas no existe, la crea
    if not hasattr(evaluar_modelos, "tabla_metricas"):
        evaluar_modelos.tabla_metricas = pd.DataFrame()
    
    # Obtiene el nombre del modelo
    nombre_modelo = modelo.__class__.__name__  

    # Verifica si el modelo ya está en la tabla
    if nombre_modelo in evaluar_modelos.tabla_metricas.index:  
        print(f"El modelo '{nombre_modelo}' ya ha sido evaluado. No se añadirá nuevamente.")
        return evaluar_modelos.tabla_metricas  
    
    # Genera predicciones
    y_pred = modelo.predict(X_test)  

    # Calcula métricas de evaluación
    r2 = r2_score(y_test, y_pred)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  

    # Crea un DataFrame con las métricas del modelo
    resultados_modelo = pd.DataFrame({nombre_modelo: {'R^2': r2, 'RMSE': rmse}}).T  

    # Concatena el nuevo resultado con la tabla existente
    evaluar_modelos.tabla_metricas = pd.concat([evaluar_modelos.tabla_metricas, resultados_modelo])  

    # Retorna la tabla con las métricas actualizadas
    return evaluar_modelos.tabla_metricas  


def seleccionar_alpha_optimo(rmse_cv, rmse_sd, alphas):
    """ 
    Selecciona el valor óptimo de alpha para regularización usando la regla del mínimo 
    más una desviación estándar.

    Parámetros:
    rmse_cv (numpy array): Array con los valores de RMSE obtenidos en validación cruzada.
    rmse_sd (numpy array): Array con las desviaciones estándar del RMSE para cada alpha.
    alphas (numpy array): Array con los valores de alpha evaluados.

    Resultado:
    tuple: (alpha óptimo, alpha óptimo con la regla de 1 desviación estándar)
    """
    
    # Encuentra el valor mínimo de RMSE en validación cruzada
    min_rmse = np.min(rmse_cv)

    # Obtiene la desviación estándar asociada al mínimo RMSE
    sd_min_rmse = rmse_sd[np.argmin(rmse_cv)]

    # Calcula el umbral del mínimo RMSE + 1 desviación estándar
    min_rmse_1sd = np.max(rmse_cv[rmse_cv <= min_rmse + sd_min_rmse])

    # Alpha correspondiente aametrosl RMSE mínimo
    optimo = alphas[np.argmin(rmse_cv)]

    # Alpha más pequeño cuyo RMSE esté dentro del umbral mínimo + 1 desviación estándar
    optimo_1sd = alphas[rmse_cv == min_rmse_1sd][0]

    return optimo, optimo_1sd


def graficar_evolucion_coeficientes(alphas, coefs, modelo):
    """
    Grafica la evolución de los coeficientes en función de alpha.

    Parámetros:
    -----------
    alphas : array-like
        Valores de regularización alpha.

    coefs : array-like
        Coeficientes del modelo para cada alpha.

    modelo : object
        Modelo ajustado, cuyo nombre se extraerá automáticamente.

    """
    fig, ax = plt.subplots(figsize=(7, 3.84))
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('coeficientes')
    ax.set_title(f'Evolución de coeficientes - {modelo.__class__.__name__}')
    plt.axis('tight')
    
def graficar_coeficientes(modelo, X_train):
    """
    Grafica los coeficientes del modelo ajustado, excluyendo los coeficientes igual a 0 en modelos como Lasso y ElasticNet.
    
    Parámetros:
    modelo : object
        Modelo ajustado con el atributo `coef_`.
    X_train : pandas.DataFrame
        Datos de entrada que fueron usados para entrenar el modelo.
    titulo : str
        Título para la gráfica.
    """
    # Filtrar coeficientes no nulos en Lasso y ElasticNet
    df_coeficientes = pd.DataFrame({
        'predictor': X_train.columns,
        'coef': modelo.coef_.flatten()
    })
    
    # Filtrar coeficientes que son distintos de 0
    df_coeficientes = df_coeficientes[df_coeficientes.coef != 0]
    
    fig, ax = plt.subplots(figsize=(11, 3.84))
    ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
    plt.xticks(rotation=90, ha='right', size=5)
    ax.set_xlabel('variable')
    ax.set_ylabel('coeficientes')
    ax.set_title(f"Coeficientes del modelo: {modelo.__class__.__name__}")
    
def pred_vs_real(model, X_test, y_test):
    """
    Compara la predicción del modelo con el valor real para una muestra.

    Parámetros:
    model: Modelo entrenado.
    X_test: Datos de prueba (características).
    y_test: Datos de prueba (valores reales).

    Resultado:
    Muestra la predicción del modelo y el valor real correspondiente.
    """
    # Obtener la predicción del modelo para la misma muestra
    prediccion = model.predict(X_test.iloc[0:1, :])[0]
    
    # Obtener el valor real de la muestra
    valor_real = y_test.iloc[0]

    # Mostrar la comparación
    print(f"Predicción del modelo: {prediccion}")
    print(f"Valor real: {valor_real}")
    
    
def df_best_params(search):
    """
    Esta función toma el resultado de una búsqueda y 
    muestra la mejor combinación de hiperparámetros encontrada durante el proceso de 
    optimización.

    Parámetros:
        search: Objeto de RandomizedSearchCV o GrindSearchCV que contiene los resultados de 
                       la búsqueda aleatoria, incluyendo los mejores hiperparámetros 
                       en el atributo `best_params_`.

    Resultado:
        Imprime en consola los mejores hiperparámetros en un formato de tabla.
    """
    # Mejor combinación de hiperparámetros
    top_params = search.best_params_
    # Convertirlo a un DataFrame de pandas para mostrarlo de manera más limpia
    df_top_params = pd.DataFrame(list(top_params.items()), columns=["Hiperparámetro", "Valor"])

    # Imprimir el DataFrame
    print("\nMejores hiperparámetros encontrados:")
    print(df_top_params)