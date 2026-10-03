from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Ventas SRI Ecuador",
    description="API REST para estimar ventas futuras por provincia, cantón y sector económico mediante modelos de Minería de Datos (UEA 2026).",
    version="1.0.0"
)

# Cargar el pipeline del modelo empaquetado
MODEL_PATH = "modelo_ventas_sri.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# Definición del esquema de entrada de datos con Pydantic
class VentasInput(BaseModel):
    PROVINCIA: str = Field(..., example="PICHINCHA")
    CANTON: str = Field(..., example="QUITO")
    CODIGO_SECTOR_N1: str = Field(..., example="G")
    AÑO: int = Field(..., example=2026)
    MES: int = Field(..., example=5)
    VENTAS_NETAS_TARIFA_GRAVADA: float = Field(..., example=150000.0)
    VENTAS_NETAS_TARIFA_0: float = Field(..., example=25000.0)
    VENTAS_NETAS_TARIFA_VARIABLE: float = Field(..., example=0.0)
    VENTAS_NETAS_TARIFA_5: float = Field(..., example=0.0)
    EXPORTACIONES: float = Field(..., example=10000.0)
    COMPRAS_NETAS_TARIFA_GRAVADA: float = Field(..., example=90000.0)
    COMPRAS_NETAS_TARIFA_0: float = Field(..., example=15000.0)
    IMPORTACIONES: float = Field(..., example=5000.0)
    COMPRAS_RISE: float = Field(..., example=500.0)
    TOTAL_COMPRAS: float = Field(..., example=110500.0)
    TOTAL_VENTAS: float = Field(..., example=185000.0)
    VENTAS_MES_ANTERIOR: float = Field(..., example=170000.0)


@app.get("/")
def home():
    """Endpoint raíz de verificación de estado."""
    return {
        "status": "online",
        "mensaje": "API de Predicción de Ventas SRI Ecuador activa.",
        "universidad": "Universidad Estatal Amazónica (UEA)",
        "documentacion": "/docs"
    }


@app.post("/predict")
def predict_ventas(data: VentasInput):
    """Endpoint de predicción que recibe las variables tributarias y retorna la estimación de ventas futuras."""
    if model is None:
        raise HTTPException(status_code=500, detail="El archivo modelo_ventas_sri.joblib no se encuentra en el servidor.")

    try:
        # Convertir los datos de entrada a DataFrame de pandas
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])

        # Calcular variables financieras derivadas necesarias
        df_input['MARGEN_COMERCIAL'] = df_input['TOTAL_VENTAS'] - df_input['TOTAL_COMPRAS']
        df_input['VARIACION_VENTAS'] = df_input['TOTAL_VENTAS'] - df_input['VENTAS_MES_ANTERIOR']
        df_input['PROPORCION_GRAVADA'] = np.where(
            df_input['TOTAL_VENTAS'] > 0,
            df_input['VENTAS_NETAS_TARIFA_GRAVADA'] / df_input['TOTAL_VENTAS'],
            0.0
        )

        # Realizar la predicción
        prediccion = model.predict(df_input)[0]
        prediccion_estimada = max(0.0, float(prediccion))

        return {
            "provincia": data.PROVINCIA,
            "canton": data.CANTON,
            "sector_economico": data.CODIGO_SECTOR_N1,
            "total_ventas_actuales_usd": data.TOTAL_VENTAS,
            "prediccion_ventas_futuras_usd": round(prediccion_estimada, 2),
            "margen_comercial_usd": round(float(df_input['MARGEN_COMERCIAL'].iloc[0]), 2),
            "variacion_ventas_usd": round(float(df_input['VARIACION_VENTAS'].iloc[0]), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")