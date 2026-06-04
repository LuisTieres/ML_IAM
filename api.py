from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="IAM Risk Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo   = None
encoders = None

MODELO_PATH   = "modelo_iam_risk.pkl"
ENCODERS_PATH = "encoders_iam_risk.pkl"

COLUNAS_CATEGORICAS = ["cargo", "departamento", "sistema", "tipo_acesso", "criticidade"]
COLUNAS_NUMERICAS   = ["tempo_empresa_meses", "acessos_ativos", "aprovacoes_anteriores",
                       "revogacoes_anteriores", "violacoes_historicas"]
COLUNAS_BOOLEANAS   = ["conflito_sod", "conformidade_ok"]
FEATURES            = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS


@app.on_event("startup")
def startup():
    global modelo, encoders
    if os.path.exists(MODELO_PATH) and os.path.exists(ENCODERS_PATH):
        modelo   = joblib.load(MODELO_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        print("[OK] Modelo carregado de disco")
    else:
        print("[INFO] Modelo nao encontrado, treinando agora...")
        from iam_risk_predictor import (
            gerar_dados_sinteticos, preprocessar,
            treinar_modelo, salvar_modelo
        )
        df = gerar_dados_sinteticos(n=500)
        X, y, enc = preprocessar(df)
        mod, _, _ = treinar_modelo(X, y)
        salvar_modelo(mod, enc)
        modelo   = mod
        encoders = enc
        print("[OK] Modelo treinado e salvo")


class Solicitacao(BaseModel):
    cargo: str
    departamento: str
    sistema: str
    tipo_acesso: str
    criticidade: str
    tempo_empresa_meses: int
    acessos_ativos: int
    aprovacoes_anteriores: int
    revogacoes_anteriores: int
    violacoes_historicas: int
    conflito_sod: int
    conformidade_ok: int


@app.get("/")
def health():
    return {"status": "ok", "modelo": "IAM Risk Predictor", "version": "1.0.0"}


@app.post("/prever-risco")
def prever_risco(solicitacao: Solicitacao):
    try:
        df = pd.DataFrame([solicitacao.dict()])

        for col in COLUNAS_CATEGORICAS:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

        proba    = modelo.predict_proba(df[FEATURES])[0]
        pred_idx = int(np.argmax(proba))
        classes  = list(encoders["risco"].classes_)
        risco    = classes[pred_idx]

        # Score baseado no risco real (0=baixo, 50=médio, 100=alto)
        prob_baixo = float(proba[classes.index("Baixo")]) if "Baixo" in classes else 0
        prob_medio = float(proba[classes.index("Medio")]) if "Medio" in classes else 0
        prob_alto  = float(proba[classes.index("Alto")])  if "Alto"  in classes else 0
        score = int(round((prob_medio * 50) + (prob_alto * 100)))

        recomendacao = {"Baixo": "APROVAR", "Medio": "REVISAR", "Alto": "REJEITAR"}

        return {
            "risco":          risco,
            "score":          score,
            "probabilidades": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "recomendacao":   recomendacao.get(risco, "REVISAR"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))