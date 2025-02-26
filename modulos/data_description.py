import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, probplot, norm, shapiro, chi2_contingency
import numpy as np
from itertools import combinations


def head_tail(df, n):
    """
    Muestra las primeras y últimas 'n' filas de un DataFrame, separadas por una fila de separación con '...'.
    
    Parámetros:
    df (DataFrame): El DataFrame del que se extraen las filas.
    n (int): El número de filas que se muestran al principio y al final del DataFrame.
    
    Resultado:
    DataFrame con las primeras 'n' filas, una fila separadora con '...' y las últimas 'n' filas.
    """
    separator = pd.DataFrame([["..."] * df.shape[1]], columns=df.columns)
    return pd.concat([df.head(n), separator, df.tail(n)], ignore_index = True)


def describe_columna(df, col):
    """
    Muestra un resumen descriptivo de una columna de un DataFrame, incluyendo su tipo de datos, 
    el número de valores nulos, el número de valores distintos y los valores más frecuentes.
    
    Parámetros:
    df (DataFrame): El DataFrame que contiene la columna.
    col (str): El nombre de la columna para la que se genera el resumen descriptivo.
    
    Resultado:
    Muestra en consola información sobre la columna, como el tipo de datos, valores nulos, valores distintos 
    y los 10 valores más frecuentes.
    """
    print(f'Columna: {col}  -  Tipo de datos: {df[col].dtype}')
    print(f'Número de valores nulos: {df[col].isnull().sum()}  -  Número de valores distintos: {df[col].nunique()}')
    print('Valores más frecuentes:')
    for i, v in df[col].value_counts().iloc[:10].items() :
        print(i, '\t', v)
        
        
def correlacion_pearson(col1, col2, df):
    """
    Calcula la correlación de Pearson entre dos variables de un DataFrame e imprime el resultado.
    
    Parámetros:
    col1 (str): Nombre de la primera columna.
    col2 (str): Nombre de la segunda columna.
    df (DataFrame): DataFrame que contiene las variables.

    Resultado:
    Muestra la correlación de Pearson con su p-valor de manera formateada.
    """
    v = df.dropna(subset=[col1, col2])
    
    corr, p_value = pearsonr(v[col1], v[col2])
    
    print(f"Correlación de Pearson: {corr:.2f}, P-valor: {p_value:.4f}")
    
    if p_value < 0.05:
        print(" La correlación es estadísticamente significativa.")
    else:
        print(" No hay suficiente evidencia para concluir que la correlación es significativa.")
        

def comprueba_normalidad(df, return_type='axes', title='Comprobación de normalidad'):
    """
    Genera Q-Q plots para comprobar la normalidad de cada columna de un DataFrame y realiza una prueba de Shapiro-Wilk para cada una.
    
    Parámetros:
    df : pandas.DataFrame
        DataFrame cuyas columnas se evaluarán para comprobar la normalidad.
    return_type : str, opcional
        Tipo de retorno esperado ('axes' para devolver los gráficos, 'stats' para devolver resultados estadísticos).
    title : str, opcional
        Título del gráfico principal.

    Retorna:
    
    matplotlib.axes._subplots.AxesSubplot o dict
        Si return_type es 'axes', devuelve los gráficos generados. Si es 'stats', devuelve un diccionario con el resultado de la prueba Shapiro-Wilk para cada columna.
    """
    
    fig_tot = len(df.columns)
    fig_por_fila = 3  
    tamanio_fig = 4.0

    num_filas = -(-fig_tot // fig_por_fila)
    plt.figure(figsize=(tamanio_fig * fig_por_fila, tamanio_fig * num_filas))
    plt.suptitle(title, fontsize=16)
    
    shapiro_results = {}
    for i, col in enumerate(df.columns):
        ax = plt.subplot(num_filas, fig_por_fila, i + 1)
        probplot(df[col], dist=norm, plot=ax)
        plt.title(col)
        
        # Agrega el resultado de la prueba de Shapiro-Wilk
        stat, p_value = shapiro(df[col].dropna())  # Se usa dropna para evitar errores con valores faltantes
        shapiro_results[col] = {'statistic': stat, 'p_value': p_value}
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    if return_type == 'stats':
        return shapiro_results
    
    
def cramers_v(contingency_table):
    '''
    Calcula Cramér's V para una tabla de contingencia.
    
    Args:
        contingency_table (pd.DataFrame): Tabla de contingencia entre dos variables.
    
    Returns:
        float: Valor de Cramér's V entre 0 y 1.
    '''
    # Valor de chi-cuadrado
    chi2 = chi2_contingency(contingency_table)[0]
    
    # Total de observaciones
    n = contingency_table.sum().sum()
    
    # Número de categorías en cada variable
    k, r = contingency_table.shape
    
    # Fórmula de Cramér's V
    return np.sqrt(chi2 / (n * (min(k-1, r-1))))


def cramers_v_relationships(df, cat_features):
    '''
    Evalúa la fuerza de la relación entre todas las variables categóricas de un dataset usando Cramér's V.
    
    Args:
        df (pd.DataFrame): Dataset con las variables categóricas.
        cat_features (list): Lista de nombres de columnas categóricas en el dataset.
    
    Returns:
        pd.DataFrame: DataFrame con las combinaciones de variables y sus valores de Cramér's V.
    '''
    results = []
    
    # Recorre todas las combinaciones de variables categóricas
    for var1, var2 in combinations(cat_features, 2):
        
        # Genera la tabla de contingencia
        contingency_table = pd.crosstab(df[var1], df[var2])
        
        # Calcula Cramér's V
        cramers_value = cramers_v(contingency_table)
        
        # Almacena los resultados
        results.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Cramérs V': cramers_value
        })
    
    # Devuelve los resultados como un DataFrame
    return pd.DataFrame(results)