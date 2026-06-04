# IAM Risk Predictor — ML Backend

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model Manually

```bash
python iam_risk_predictor.py
```

This generates the following files:

* `modelo_iam_risk.pkl`
* `encoders_iam_risk.pkl`

## Start the API

```bash
uvicorn api:app --reload --port 8000
```

The API will be available at:

http://localhost:8000

If the model files do not exist, the model will be trained automatically during startup.

## Endpoints

| Method | Route         | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| GET    | /             | Health check                                 |
| POST   | /prever-risco | Receives an access request and predicts risk |

## Example Request

```bash
curl -X POST http://localhost:8000/prever-risco \
  -H "Content-Type: application/json" \
  -d '{
    "cargo": "Intern",
    "departamento": "Finance",
    "sistema": "Core Banking",
    "tipo_acesso": "Administrator",
    "criticidade": "Critical",
    "tempo_empresa_meses": 3,
    "acessos_ativos": 12,
    "aprovacoes_anteriores": 1,
    "revogacoes_anteriores": 0,
    "violacoes_historicas": 0,
    "conflito_sod": 1,
    "conformidade_ok": 0
  }'
```

## Expected Response

```json
{
  "risk": "High",
  "score": 91,
  "probabilities": {
    "High": 0.91,
    "Low": 0.03,
    "Medium": 0.06
  },
  "recommendation": "REJECT"
}
```

## Frontend Environment Variable

Create a `.env` file in the frontend root directory:

```env
VITE_ML_API_URL=https://luistieres-iam-risk-predictor.hf.space
```
