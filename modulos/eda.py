
import pandas as pd
from scipy.stats import pearsonr


def head_tail ( df, n ):
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