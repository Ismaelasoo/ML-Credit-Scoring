
import pandas as pd

def head_tail ( df, n ):
    separator = pd.DataFrame([["..."] * df.shape[1]], columns=df.columns)
    return pd.concat([df.head(n), separator, df.tail(n)], ignore_index = True)

def describe_columna(df, col):
    print(f'Columna: {col}  -  Tipo de datos: {df[col].dtype}')
    print(f'Número de valores nulos: {df[col].isnull().sum()}  -  Número de valores distintos: {df[col].nunique()}')
    print('Valores más frecuentes:')
    for i, v in df[col].value_counts().iloc[:10].items() :
        print(i, '\t', v)