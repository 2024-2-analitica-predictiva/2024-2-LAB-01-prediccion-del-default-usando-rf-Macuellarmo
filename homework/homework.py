# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import gzip
import json
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

#Paso 1
print('Paso 1')
#Cargamos los datos
df_train = pd.read_csv('files/input/train_data.csv.zip', index_col=False, compression="zip")
df_test = pd.read_csv('files/input/test_data.csv.zip', index_col=False, compression="zip")

#Renombrar la columna "default payment next month" a "default"
df_train = df_train.rename(columns={"default payment next month": "default"})
df_test = df_test.rename(columns={"default payment next month": "default"})

#Remover la columna "ID"
df_train = df_train.drop(columns=["ID"])
df_test = df_test.drop(columns=["ID"])

#Eliminación de los registro de información no disponible
df_train = df_train.dropna()
df_train = df_train[df_train['EDUCATION'] != 0]
df_train = df_train[df_train['MARRIAGE'] != 0]
df_test = df_test.dropna()
df_test = df_test[df_test['EDUCATION'] != 0]
df_test = df_test[df_test['MARRIAGE'] != 0]

#Agrupación de los valore de EDUCATION > 4 en la categoría "others"
df_train.loc[df_train["EDUCATION"] > 4, "EDUCATION"] = 4
df_test.loc[df_test["EDUCATION"] > 4, "EDUCATION"] = 4

#Paso 2
print('Paso 2')
#Dividir los datasets en x_train, y_train, x_test, y_test
x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]
x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]

#Paso 3
print('Paso 3')
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).

#Crear el transformer
transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(dtype="int"), ["SEX", "EDUCATION", "MARRIAGE"]),
    ],
    remainder="passthrough",
)

#Creamos el pipeline
pipeline = Pipeline(
    steps=[
        ("transformer", transformer),
        ("classifier", RandomForestClassifier(max_features=10, min_samples_split=10, min_samples_leaf=3, n_jobs=-1)),
    ],
    verbose=False,
)

#Paso 4
print('Paso 4')
params = {
    "classifier__n_estimators": [80, 90, 100, 110, 120],
}
grid = GridSearchCV(pipeline, params, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True)
grid.fit(x_train, y_train)

print('Mejores hiperparametros:', grid.best_params_)
print('score_test:', grid.score(x_test, y_test))

#Paso 5
print('Paso 5')
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
if not os.path.exists("files/models"):
        os.makedirs("files/models")

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

#Paso 6
print('Paso 6')
metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, grid.predict(x_train)),
    "balanced_accuracy": balanced_accuracy_score(y_train, grid.predict(x_train)),
    "recall": recall_score(y_train, grid.predict(x_train)),
    "f1_score": f1_score(y_train, grid.predict(x_train))
}

if not os.path.exists("files/output"):
    os.makedirs("files/output")

with open("files/output/metrics.json", "w") as f:
    json.dump(metrics, f)

metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_test, grid.predict(x_test)),
    "balanced_accuracy": balanced_accuracy_score(y_test, grid.predict(x_test)),
    "recall": recall_score(y_test, grid.predict(x_test)),
    "f1_score": f1_score(y_test, grid.predict(x_test))
}

with open("files/output/metrics.json", "a") as f:
    f.write('\n')

with open("files/output/metrics.json", "a") as f:
    f.write(json.dumps(metrics) + '\n')

#Paso 7
print('Paso 7')
cm_train = confusion_matrix(y_train, grid.predict(x_train))
cm_test = confusion_matrix(y_test, grid.predict(x_test))
metrics = {
    "type": "cm_matrix", 
    "dataset": "train",
    "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
    "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
}
with open("files/output/metrics.json", "a") as f:
    f.write(json.dumps(metrics) + '\n')

metrics = {
    "type": "cm_matrix", 
    "dataset": "test",
    "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
    "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
}
with open("files/output/metrics.json", "a") as f:
    f.write(json.dumps(metrics))



