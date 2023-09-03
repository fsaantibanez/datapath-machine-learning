import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def nan_checking(dataframe):
    # Calcula el número de valores nulos en cada columna
    nulos_por_columna = dataframe.isnull().sum()
    # Crea un DataFrame que muestre la cantidad de nulos por columna
    tabla = pd.DataFrame({'Columna': nulos_por_columna.index, 'Nulos': nulos_por_columna.values})
    # Ordena la tabla de manera descendente por la cantidad de nulos
    tabla = tabla.sort_values(by='Nulos', ascending=False)

    return tabla

def plot_kde(dataframe, num_rows, num_cols):
    """
    Crea gráficos KDE para las columnas especificadas en un DataFrame y los distribuye en filas y columnas.

    Args:
    dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    columns (list): Lista de nombres de columnas para las cuales se generarán los gráficos KDE.
    num_rows (int): Número de filas para la disposición de subgráficos.
    num_cols (int): Número de columnas para la disposición de subgráficos.

    Returns:
    None
    """
    # Calcula el número total de subgráficos
    num_plots = len(dataframe.columns)

    # Calcula el número de filas y columnas según los parámetros o de forma automática si no se proporcionan
    if num_rows is None or num_cols is None:
        num_cols = int(math.ceil(num_plots ** 0.5))
        num_rows = int(math.ceil(num_plots / num_cols))

    # Verifica que el número total de subgráficos no sea mayor que el número de subplots disponibles
    if num_plots > num_rows * num_cols:
        print(f"El número de subgráficos ({num_plots}) es mayor que el número de subplots disponibles ({num_rows}x{num_cols}).")
        return

    # Crea subplots para organizar los gráficos en filas y columnas
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6))
    axs = axs.flatten()  # Convierte la matriz de subplots en una lista unidimensional

    # Itera a través de las columnas y crea los gráficos KDE correspondientes
    for i, column in enumerate(dataframe.columns):
        ax = axs[i]
        sns.kdeplot(data=dataframe[column], ax=ax, color='blue', alpha=0.4, fill=True, common_norm=False)
        ax.set_title(f"{column}")
        #ax.legend()

    # Ajusta la disposición de los subplots
    plt.tight_layout()
    plt.show()

def plot_categorical(column, title_name=None):
    """
    Genera gráficos de barras y de torta para una columna categórica.

    Args:
    column (pd.Series): La columna categórica que se desea visualizar.
    title_name (str): El nombre del conjunto de datos (por defecto es 'Dataset').

    Returns:
    None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.4)

    # Gráfico de Barras
    sns.countplot(data=df, x=column, ax=axs[0])
    axs[0].set_title(f'Bar Plot para "{column.name}"')
    axs[0].set_xlabel(column.name)
    axs[0].set_ylabel('Count')
    axs[0].set_ylim(0, len(df))  # Ajusta los límites del eje y para mostrar ambas categorías

    # Gráfico de Torta
    counts = column.value_counts()
    labels = counts.index
    axs[1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[1].set_title(f'Pie Chart para "{column.name}"')

    axs[1].legend(loc='upper right', labels=labels)

    plt.suptitle(f'{title_name} - Gráficos Categóricos', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation_heatmap(dataframe, title='Matriz de Correlación'):
    """
    Genera y muestra una matriz de correlación como un mapa de calor.

    Args:
    dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    title (str): El título del gráfico (por defecto es 'Matriz de Correlación').

    Returns:
    None
    """
    # Calcula la matriz de correlación
    corr_matrix = dataframe.corr()

    # Crear una máscara para la mitad superior de la matriz
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Crea una figura y un eje para el mapa de calor
    plt.figure(figsize=(10, 8))

    # Genera el mapa de calor
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask = mask)

    # Establece el título
    plt.title(title, fontsize=16)

    # Muestra el gráfico
    plt.show()

    return corr_matrix