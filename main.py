import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from models import load_model_from_pickle, generate_classification_report


parser = argparse.ArgumentParser(description='Realizar predicciones en un conjunto de datos de prueba.')
parser.add_argument('data_file', type=str, help='Ruta al archivo CSV de datos de prueba')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

df = pd.read_csv(args.data_file)

X_test = df.drop(columns=['target'])  
y_test = df['target']

model = load_model_from_pickle('logreg_balanced.pkl')
y_pred = model.predict(X_test)

print(generate_classification_report(y_test, y_pred))
print(y_pred)


