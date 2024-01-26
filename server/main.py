from fastapi import FastAPI, HTTPException
from typing import List
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

app = FastAPI()
df = pd.read_csv("heart.csv")
# Cargar modelo y escalador
modelo = joblib.load("modelo_logreg.pkl")
# Para cargar el escalador más tarde
scaler = joblib.load("escalador_robusto.pkl")

# Función para realizar predicciones
def predictor(nuevos_datos):
    """
    Realizar predicciones utilizando el modelo cargado.

    Parameters:
    - nuevos_datos (list of dict): Lista de diccionarios con datos para realizar predicciones.

    Returns:
    - predictions: Predicciones del modelo.
    """
    print(nuevos_datos)
    try:
        # Convertir la lista de diccionarios a un DataFrame de pandas
        print("Converting to DataFrame...")
        df_nuevos_datos = pd.DataFrame([nuevos_datos])
        cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
        con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
        
        df_nuevos_datos = pd.DataFrame([nuevos_datos]
        )
        cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
        con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

        # Codificar y escalar los nuevos datos de la misma manera que los datos de entrenamiento
        df_uniques = {
            'sex': [0, 1],
            'exng': [0, 1],
            'caa': [0, 1, 2, 3, 4],
            'cp': [0, 1, 2, 3],
            'fbs': [0, 1],
            'restecg': [0, 1, 2],
            'slp': [0, 1, 2],
            'thall': [0, 1, 2, 3],
        }

        for col in cat_cols:
            nuevo_dummie = pd.get_dummies(df_nuevos_datos[col].astype('category').cat.set_categories(df_uniques[col]), prefix=col,drop_first=True)
            df_nuevos_datos = df_nuevos_datos.join(nuevo_dummie)
            df_nuevos_datos.drop(col, axis=1, inplace=True)

        df_nuevos_datos[con_cols] = scaler.transform(df_nuevos_datos[con_cols])
        print(df_nuevos_datos)
        print("Predicting...")
        y_pred_proba = modelo.predict_proba(df_nuevos_datos)
        y_pred = np.argmax(y_pred_proba,axis=1)
        return {"predictions": y_pred_proba.tolist(), "y_pred": y_pred.tolist()}
    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

# Ruta raíz
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Ruta para realizar predicciones
@app.post("/predict")
async def predict(datos: dict):
    """
    Recibe una lista de diccionarios con los datos de entrada para el modelo.

    Parameters:
    - datos (list of dict): Lista de diccionarios con datos para realizar predicciones.
    
    Returns:
    - predictions: Predicciones del modelo.
    """
    return predictor(datos)
