import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import shap
from IPython.display import Markdown, display


def plot_boxplot(data, title):
    """
    Representa un boxplot de los datos proporcionados, destacando la mediana, la media,
    y los valores atípicos.
    
    Parámetros:
    data: Serie o DataFrame con los datos para el boxplot.
    title: Título para el gráfico.
    """
    # Eliminar valores nulos antes de crear el boxplot
    data_clean = data.dropna()

    # Crear el boxplot
    plt.boxplot(data_clean, 
                notch=True, 
                patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'), 
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'),
                showmeans=True, 
                meanprops=dict(marker='o', markerfacecolor='green', markersize=8))
    
    # Configurar título
    plt.title(title)
    
    # Mostrar el gráfico
    plt.show()

def count_outliers_iqr(df, numeric_features):
    """
    Calcula el número de valores atípicos (outliers) en cada variable numérica 
    utilizando el método del rango intercuartílico (IQR).

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene las variables numéricas a analizar.
    numeric_features (list): Lista con los nombres de las variables numéricas.

    Retorna:
    dict: Diccionario con el recuento de outliers por variable.
    """
    outlier_count = {}

    for column in numeric_features:
        # Calcular el rango intercuartílico y los límites
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        # Contar los outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count[column] = outliers.shape[0]

    # Mostrar el recuento de outliers
    print('Recuento de outliers por feature:')
    for column, count in outlier_count.items():
        if count > 0:
            print(f'{column}: {count}')
    
def plot_outliers_boxplots(df, numeric_features, rows, columns, palette='Set2', figsize=(12, 20)):
    """
    Dibuja boxplots para detectar outliers en las características numéricas.

    Parámetros:
    df: DataFrame que contiene los datos a visualizar.
    numeric_features: Lista de nombres de las columnas numéricas a analizar.
    palette: Paleta de colores para los gráficos (por defecto 'Set2').
    figsize: Tamaño de la figura para los subgráficos (por defecto (12, 60)).
    """
    # Configuración de estilo
    sns.set_palette(palette)

    color_viridis = sns.color_palette('viridis')[2]
    plt.figure(figsize=figsize)

    # Dibujar un boxplot por cada columna numérica
    for i, column in enumerate(numeric_features):
        plt.subplot(rows, columns, i + 1)
        sns.boxplot(data=df[[column]], color=color_viridis) 

    # Mostrar título y ajustar el layout
    plt.suptitle('Outliers en las features cuantitativas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def scatter_variable_relation(df, x, y, title=None):
    """
    Representa la relación entre dos variables numéricas a través de un gráfico de dispersión.
    
    Parámetros:
    df: DataFrame que contiene los datos.
    x: Nombre de la columna en el eje X.
    y: Nombre de la columna en el eje Y.
    title: Título del gráfico (opcional). Si no se proporciona, se asigna un título predeterminado.
    """
    # Si no se pasa un título, se asigna uno predeterminado
    if title is None:
        title = f"Relación entre {x} y {y}"

    # Crear gráfico de dispersión
    plt.figure(figsize=(10, 6))
    plt.scatter(data=df, x=x, y=y, alpha=0.7, c='blue', edgecolors='k')

    # Configurar título y etiquetas
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()
    
def Count_Cat(df, cat_feat, subrep_name):
    """
    Genera un análisis descriptivo para una variable categórica en un DataFrame.

    Parámetros:
    df (DataFrame): El DataFrame que contiene la variable categórica.
    cat_feat (str): El nombre de la columna categórica a analizar.
    subrep_name (str): El nombre del subdirectorio donde se guardarán las imágenes.

    Resultado:
    - Muestra en consola el número de categorías únicas y una tabla con las 30 categorías más frecuentes.
    - Genera un gráfico de barras mostrando el porcentaje de cada categoría.
    - Presenta una tabla con los recuentos y porcentajes de las categorías más frecuentes.
    - Guarda la gráfica en la carpeta 'images/sub_repositorio'.
    """

    # Verificación inicial
    if cat_feat not in df.columns:
        raise ValueError(f"La columna '{cat_feat}' no existe en el DataFrame.")

    # Crear carpetas si no existen
    save_dir = os.path.join("images", subrep_name)
    os.makedirs(save_dir, exist_ok=True)

    # Valores únicos
    unique_values = df[cat_feat].nunique()
    
    # Encabezado
    print('\n\n')
    formatted_text = f'**{cat_feat.upper()}**'
    try:
        display(Markdown(formatted_text)) 
    except:
        print(formatted_text)
    
    print(f"El número de categorías distintas en la variable '{cat_feat}' es {unique_values}.")

    # Datos
    feat_count = df[cat_feat].value_counts()
    feat_perc = (feat_count / len(df)) * 100
    feat_res = pd.DataFrame({'Recuento': feat_count, '%': round(feat_perc, 2)})\
               .sort_values(by='Recuento', ascending=False).head(30)

    # Configuración de estilo
    sns.set(style='whitegrid')
    fig = plt.figure(figsize=(18, 10))
    spec = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    # Gráfico de barras
    ax0 = plt.subplot(spec[0])
    sns.barplot(y='%', x=feat_res.index, data=feat_res, palette='viridis', ax=ax0, hue=feat_res.index)

    ax0.set_title(f"Porcentaje de instancias por {cat_feat}", fontsize=14)
    ax0.set_ylabel('% de instancias', fontsize=14)
    ax0.set_xlabel(cat_feat, fontsize=14)

    # Configurar correctamente los ticks antes de asignar etiquetas
    ax0.set_xticks(range(len(feat_res.index)))
    ax0.set_xticklabels(feat_res.index, rotation=45, ha='right')

    # Tabla
    ax1 = plt.subplot(spec[1])
    ax1.axis('off')
    cell_text = feat_res.reset_index().values.tolist()
    col_labels = ['Categoría', 'Recuento', '%']
    table = ax1.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#f0f0f0'] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Guardar la gráfica solo si no existe
    file_path = os.path.join(save_dir, f"{cat_feat}.png")
    if not os.path.exists(file_path):
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Gráfica guardada en: {file_path}")
    else:
        print(f"La gráfica ya existe en: {file_path}")

    # Mostrar gráfico
    plt.show()
    
def Count_Quant(df, quant_feat, subrep_name):
    """
    Genera un análisis descriptivo para una variable cuantitativa en un DataFrame.

    Parámetros:
    df (DataFrame): El DataFrame que contiene la variable numérica.
    quant_feat (str): El nombre de la columna numérica a analizar.
    subrep_name (str): El nombre del subdirectorio donde se guardarán las imágenes.

    Resultado:
    - Muestra en consola los principales estadísticos descriptivos (mínimo, máximo, promedio y desviación estándar).
    - Genera un histograma con una estimación de densidad (KDE) y un boxplot de la variable.
    """ 

    # Encabezado
    print()
    print()
    formatted_text = f'**{quant_feat.upper()}**'
    try:
        display(Markdown(formatted_text))
    except:
        print(formatted_text)

    # Estadísticos descriptivos
    print('Estadísticos')
    print()
    print(f'Mínimo: {round(df[quant_feat].min(),2)}')
    print(f'Máximo: {round(df[quant_feat].max(),2)}')
    print(f'Promedio: {round(df[quant_feat].mean(), 2)}')
    print(f'Std.dev: {round(df[quant_feat].std(),2)}')
    print()
    print(f'Histograma y Boxplot de {quant_feat}.')

    # Configuración de estilo
    color = '#EE9414'
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 7))

    # Histograma con KDE corregido
    sns.histplot(data=df, x=quant_feat, ax=axes[0], kde=True, color=color, kde_kws={'bw_adjust': 0.3, 'clip': (0, 1)})
    axes[0].set_xlabel('')
    axes[0].set_xlim(0, 1)  # Limitar eje X

    # Boxplot
    sns.boxplot(data=df, x=quant_feat, ax=axes[1], color=color)
    axes[1].set_xlabel('')
    axes[1].set_xlim(0, 1)  # Limitar eje X

    # Crear carpeta para guardar imágenes si no existe
    save_dir = os.path.join("images", subrep_name)
    os.makedirs(save_dir, exist_ok=True)

    # Guardar la gráfica solo si no existe
    file_path = os.path.join(save_dir, f"{quant_feat}.png")
    if not os.path.exists(file_path):
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Gráfica guardada en: {file_path}")
    else:
        print(f"La gráfica ya existe en: {file_path}")

    # Mostrar gráficos
    plt.show()

def Analyze_Categorical_Features_Density(df, cat_feat, target_column, subrep_name):
    '''
    Genera diagramas de densidad superpuestos para cada variable categórica,
    mostrando las distribuciones para cada valor distinto en la misma visualización.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - cat_feat: Lista de variables categóricas en el DataFrame.
    - target_column: Nombre de la variable objetivo numérica.
    - subrep_name: Nombre de la subcarpeta donde se guardarán las gráficas.
    '''

    print(f"\nAnálisis de variables categóricas respecto a la columna objetivo '{target_column}':\n")

    # Crear carpetas si no existen
    save_dir = os.path.join("images", subrep_name)
    os.makedirs(save_dir, exist_ok=True)

    for v in cat_feat:
        if v == target_column:
            continue

        print(f"\nVariable categórica: {v}")
        print(f"Número de categorías únicas: {df[v].nunique()}")
        print(f"Frecuencia de las categorías principales:\n{df[v].value_counts().head()}\n")

        plt.figure(figsize=(12, 8))  # Ajustar tamaño para mejor visualización

        # Dibujar la densidad para cada valor único de la variable categórica
        for valor in df[v].unique():
            subset = df[df[v] == valor]
            sns.kdeplot(subset[target_column], label=str(valor), fill=True)  # Añadir etiqueta y rellenar

        plt.title(f"Distribución de {target_column} para diferentes valores de {v}")
        plt.xlabel(target_column)
        plt.ylabel("Densidad")
        plt.legend()  # Mostrar leyenda para identificar los grupos

        # Guardar la gráfica solo si no existe
        file_path = os.path.join(save_dir, f'density_{v}.png')
        if not os.path.exists(file_path):
            plt.savefig(file_path, bbox_inches='tight')
            print(f"Gráfica guardada en: {file_path}")
        else:
            print(f"La gráfica ya existe en: {file_path}")

        plt.show()
        
def Analyze_Numeric_Features_Scatter(df, num_features, target_column, hue_column=None, subrep_name=None):
    '''
    Analiza variables numéricas respecto a una variable objetivo numérica
    mediante scatter plots con una línea de regresión lineal.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - num_features: Lista de variables numéricas en el DataFrame.
    - target_column: Nombre de la variable objetivo numérica.
    - hue_column: (Opcional) Nombre de la variable categórica para definir el color de los puntos.
    - subrep_name: Nombre de la subcarpeta donde se guardarán las gráficas.
    '''
    
    print(f"Análisis de variables numéricas respecto a la columna objetivo '{target_column}':\n")
    
    # Crear carpetas si no existen
    if subrep_name:
        save_dir = os.path.join("images", subrep_name)
        os.makedirs(save_dir, exist_ok=True)

    for v_num in num_features:
        if v_num == target_column:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Graficar scatter plot, con color definido por la variable categórica si se proporciona
        sns.scatterplot(x=df[v_num], y=df[target_column], alpha=0.6, hue=df[hue_column] if hue_column else None)
        
        # Regresión lineal
        model = LinearRegression()
        model.fit(df[[v_num]], df[target_column])
        y_pred = model.predict(df[[v_num]])
        
        # Graficar la línea de regresión lineal
        plt.plot(df[v_num], y_pred, color='red', linestyle='--', label='Regresión lineal')
        
        # Configurar el gráfico
        plt.title(f"Relación entre {v_num} y {target_column}")
        plt.xlabel(v_num)
        plt.ylabel(target_column)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Guardar la gráfica si se especifica la subcarpeta
        if subrep_name:
            file_path = os.path.join(save_dir, f'scatter_{v_num}.png')
            if not os.path.exists(file_path):  # Solo guardar si no existe
                plt.savefig(file_path, bbox_inches='tight')
                print(f"Gráfica guardada en: {file_path}")
            else:
                print(f"La gráfica ya existe en: {file_path}")

        plt.show()

def shap_visualization(shap_values, X_test):
    """
    Visualiza los valores SHAP de un modelo.

    Argumentos:
        shap_values (numpy.ndarray): Valores SHAP generados para las observaciones.
        X_test (pandas.DataFrame): DataFrame que contiene las características de las observaciones.

    Gráficos:
        - Bar plot para la importancia global de las variables.
        - Waterfall plot para la primera observación.
        - Force plot para las primeras 50 observaciones.
        - Beeswarm plot para la distribución de los efectos SHAP por variable.
        - Gráficos de dispersión de los valores SHAP para las variables más influyentes.
    """

    # Iteramos del 1 al 5 para ejecutar cada tipo de visualización
    for i in range(1, 6):  
        try:
            if i == 1:
                print(f"Ejecutando visualización {i}: Bar Plot")
                # Bar Plot: Importancia global de las variables
                shap.plots.bar(shap_values)

            elif i == 2:
                print(f"Ejecutando visualización {i}: Waterfall Plot")
                # Waterfall Plot: Muestra el impacto de cada variable en la predicción de una observación específica
                shap.plots.waterfall(shap_values[0])

            elif i == 3:
                print(f"Ejecutando visualización {i}: Force Plot")
                # Force Plot: Explica la predicción de varias observaciones 
                shap.force_plot(shap_values[:50], matplotlib=True)

            elif i == 4:
                print(f"Ejecutando visualización {i}: Beeswarm Plot")
                # Beeswarm Plot: Muestra la distribución de los valores SHAP por variable
                shap.plots.beeswarm(shap_values)

            elif i == 5:
                print(f"Ejecutando visualización {i}: Scatter Plots")
                # Scatter Plots: Gráficos de dispersión de SHAP para las variables más influyentes
                
                # Calculamos la importancia media absoluta de cada variable
                shap_importance = np.abs(shap_values.values).mean(axis=0)
                
                # Seleccionamos las 3 variables más influyentes en base a su importancia SHAP
                top_features = X_test.columns[np.argsort(shap_importance)[::-1]][:3]
                
                # Generamos un scatter plot para cada una de las variables seleccionadas
                for feature in top_features:
                    shap.plots.scatter(shap_values[:, feature])

        except Exception as e:
            # Si ocurre un error en alguna visualización, lo imprimimos y continuamos con las siguientes
            print(f"Error en la visualización {i}: {e}")