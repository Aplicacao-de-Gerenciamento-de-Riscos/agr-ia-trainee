# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import joblib
from database import get_db
from data_processing import load_data

app = FastAPI()

# Carregar o modelo Random Forest treinado
model = joblib.load('random_forest_model.pkl')


@app.get("/predict/sprint/{cod_sprint}")
def predict_sprint_delay(cod_sprint: int, db: Session = Depends(get_db)):
    # Carregar os dados do banco de dados
    data = load_data(db)

    # Filtrar os dados da sprint fornecida
    sprint_data = data[data['cod_sprint'] == cod_sprint]
    if sprint_data.empty:
        raise HTTPException(status_code=404, detail="Sprint not found")

    # Fazer a previsão
    X = sprint_data.drop(columns=['cod_sprint', 'timespent', 'time_estimate'])
    prediction = model.predict(X)

    # Retornar o resultado da previsão (1 = atrasado, 0 = no prazo)
    return {"cod_sprint": cod_sprint, "atrasado": bool(prediction[0])}
