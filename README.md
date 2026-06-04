# IAM Risk Predictor — Backend ML

## Instalação

```bash
pip install -r requirements.txt
```

## Treinar o modelo manualmente

```bash
python iam_risk_predictor.py
```

Gera os arquivos:
- `modelo_iam_risk.pkl`
- `encoders_iam_risk.pkl`

## Subir a API

```bash
uvicorn api:app --reload --port 8000
```

A API sobe em: http://localhost:8000

Se o modelo ainda não existir, ele é treinado automaticamente no startup.

## Endpoints

| Método | Rota            | Descrição                        |
|--------|-----------------|----------------------------------|
| GET    | /               | Health check                     |
| POST   | /prever-risco   | Recebe solicitação, retorna risco |

## Exemplo de chamada

```bash
curl -X POST http://localhost:8000/prever-risco \
  -H "Content-Type: application/json" \
  -d '{
    "cargo": "Estagiario",
    "departamento": "Financeiro",
    "sistema": "Core Bancario",
    "tipo_acesso": "Administrador",
    "criticidade": "Critica",
    "tempo_empresa_meses": 3,
    "acessos_ativos": 12,
    "aprovacoes_anteriores": 1,
    "revogacoes_anteriores": 0,
    "violacoes_historicas": 0,
    "conflito_sod": 1,
    "conformidade_ok": 0
  }'
```

## Resposta esperada

```json
{
  "risco": "Alto",
  "score": 91,
  "probabilidades": { "Alto": 0.91, "Baixo": 0.03, "Medio": 0.06 },
  "recomendacao": "REJEITAR"
}
```

## Variável de ambiente no front

Crie um `.env` na raiz do front:

```
VITE_ML_API_URL=http://localhost:8000
```
