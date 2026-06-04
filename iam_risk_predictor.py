import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
import joblib
import warnings
warnings.filterwarnings("ignore")


COLUNAS_CATEGORICAS = [
    "cargo",
    "departamento",
    "sistema",
    "tipo_acesso",
    "criticidade"
]

COLUNAS_NUMERICAS = [
    "tempo_empresa_meses",
    "acessos_ativos",
    "aprovacoes_anteriores",
    "revogacoes_anteriores",
    "violacoes_historicas"
]

COLUNAS_BOOLEANAS = [
    "conflito_sod",        # 1 = tem conflito de Segregacao de Funcoes
    "conformidade_ok"      # 1 = esta em conformidade
]

# Coluna alvo
TARGET = "risco"  # valores: "Baixo", "Medio", "Alto"

# Caminho para salvar o modelo treinado
MODELO_PATH = "modelo_iam_risk.pkl"
ENCODERS_PATH = "encoders_iam_risk.pkl"



def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega o dataset de solicitacoes IAM a partir de um arquivo CSV.

    Colunas esperadas no CSV:
    - cargo, departamento, sistema, tipo_acesso, criticidade
    - tempo_empresa_meses, acessos_ativos, aprovacoes_anteriores
    - revogacoes_anteriores, violacoes_historicas
    - conflito_sod, conformidade_ok
    - risco  (coluna alvo: Baixo / Medio / Alto)
    """
    df = pd.read_csv(caminho_csv)
    print(f"[OK] Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def gerar_dados_sinteticos(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Gera um dataset sintetico realista para treino e testes iniciais.
    Substitua por dados reais quando disponiveis.
    """
    np.random.seed(seed)

    cargos       = ["Analista Jr", "Analista Pl", "Analista Sr",
                    "Coordenador", "Gerente", "Diretor",
                    "Estagiario", "Consultor"]
    departamentos = ["TI", "Financeiro", "RH", "Juridico",
                     "Operacoes", "Comercial", "Auditoria", "Compliance"]
    sistemas     = ["ERP SAP", "Core Bancario", "CRM Salesforce",
                    "BI Tableau", "Active Directory", "AWS Console",
                    "Folha de Pagamento", "Sistema Fiscal"]
    tipos_acesso = ["Leitura", "Escrita", "Administrador",
                    "Aprovador", "Auditor", "Super Usuario"]
    criticidades = ["Baixa", "Media", "Alta", "Critica"]

    df = pd.DataFrame({
        "cargo":               np.random.choice(cargos, n),
        "departamento":        np.random.choice(departamentos, n),
        "sistema":             np.random.choice(sistemas, n),
        "tipo_acesso":         np.random.choice(tipos_acesso, n),
        "criticidade":         np.random.choice(criticidades, n),
        "tempo_empresa_meses": np.random.randint(1, 120, n),
        "acessos_ativos":      np.random.randint(0, 30, n),
        "aprovacoes_anteriores": np.random.randint(0, 20, n),
        "revogacoes_anteriores": np.random.choice([0,0,0,1,2], n),
        "violacoes_historicas":  np.random.choice([0,0,0,0,1,2,3], n),
        "conflito_sod":         np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "conformidade_ok":      np.random.choice([0, 1], n, p=[0.1, 0.9]),
    })

    df["risco"] = df.apply(_calcular_risco_sintetico, axis=1)

    print(f"[OK] Dataset sintetico gerado: {n} amostras")
    print(df["risco"].value_counts().to_string())
    return df


def _calcular_risco_sintetico(row) -> str:
    """Logica interna para rotular o risco no dataset sintetico."""
    score = 0

    priv_map = {"Leitura": 5, "Auditor": 15, "Escrita": 25,
                "Aprovador": 35, "Administrador": 55, "Super Usuario": 75}
    crit_map = {"Baixa": 0, "Media": 10, "Alta": 20, "Critica": 35}

    score += priv_map.get(row["tipo_acesso"], 20)
    score += crit_map.get(row["criticidade"], 10)
    score += 25 if row["conflito_sod"] else 0
    score += row["violacoes_historicas"] * 8
    score += row["revogacoes_anteriores"] * 6
    score += 15 if row["tempo_empresa_meses"] < 6 else (8 if row["tempo_empresa_meses"] < 12 else 0)
    score += 10 if row["acessos_ativos"] > 15 else 0
    score += 12 if not row["conformidade_ok"] else 0
    score += np.random.randint(-5, 6)  # ruido

    if score <= 30:
        return "Baixo"
    elif score <= 70:
        return "Medio"
    else:
        return "Alto"


def preprocessar(df: pd.DataFrame):
    """
    Converte colunas categoricas para numerico via LabelEncoder.
    Retorna X (features), y (target) e o dicionario de encoders.
    """
    df = df.copy()
    encoders = {}

    for col in COLUNAS_CATEGORICAS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df[TARGET] = le_target.fit_transform(df[TARGET])
    encoders[TARGET] = le_target

    features = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS
    X = df[features]
    y = df[TARGET]

    print(f"[OK] Pre-processamento concluido. Features: {X.shape[1]} colunas")
    return X, y, encoders


def treinar_modelo(X: pd.DataFrame, y: pd.Series):
    """
    Treina o Random Forest com validacao cruzada estratificada.
    Retorna o modelo treinado e as metricas de validacao.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",   
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"[CV] F1-Macro medio: {scores_cv.mean():.4f} (+/- {scores_cv.std():.4f})")

    modelo.fit(X_train, y_train)

    return modelo, X_test, y_test


def avaliar_modelo(modelo, X_test, y_test, encoders):
    """
    Exibe metricas completas de avaliacao no conjunto de teste.
    """
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    le_target = encoders[TARGET]
    classes   = le_target.classes_

    print("\n" + "="*60)
    print("RELATORIO DE CLASSIFICACAO")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=classes))

    print("MATRIZ DE CONFUSAO")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"\nAUC-ROC (macro OvR): {auc:.4f}")

    return y_pred


# =============================================================================
# BLOCO 7 — IMPORTANCIA DAS FEATURES
# =============================================================================

def importancia_features(modelo, X: pd.DataFrame):
    """
    Exibe as features mais importantes para o modelo.
    """
    features = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS
    importancias = pd.Series(modelo.feature_importances_, index=features)
    importancias = importancias.sort_values(ascending=False)

    print("\n" + "="*60)
    print("IMPORTANCIA DAS FEATURES")
    print("="*60)
    for feat, val in importancias.items():
        barra = "█" * int(val * 50)
        print(f"  {feat:<30} {barra} {val:.4f}")

    return importancias


# =============================================================================
# BLOCO 8 — PREDICAO PARA NOVA SOLICITACAO
# =============================================================================

def prever_risco(modelo, encoders, solicitacao: dict) -> dict:
    """
    Recebe uma nova solicitacao de acesso e retorna o score de risco.

    Exemplo de solicitacao:
    {
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
    }
    """
    df = pd.DataFrame([solicitacao])

    # Encode categoricas com os encoders salvos
    for col in COLUNAS_CATEGORICAS:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    features = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS
    X = df[features]

    proba    = modelo.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    classes  = encoders[TARGET].classes_
    score    = int(round(max(proba) * 100))

    # Mapear para recomendacao
    risco_label = classes[pred_idx]
    recomendacoes = {
        "Baixo":  "APROVAR",
        "Medio":  "REVISAR",
        "Alto":   "REJEITAR"
    }

    resultado = {
        "risco":         risco_label,
        "score":         score,
        "probabilidades": {c: round(p, 4) for c, p in zip(classes, proba)},
        "recomendacao":  recomendacoes[risco_label]
    }

    print(f"\n  Risco:        {resultado['risco']}")
    print(f"  Score:        {resultado['score']}/100")
    print(f"  Probabilidades: {resultado['probabilidades']}")
    print(f"  Recomendacao: {resultado['recomendacao']}")

    return resultado


# =============================================================================
# BLOCO 9 — SALVAR E CARREGAR MODELO
# =============================================================================

def salvar_modelo(modelo, encoders):
    joblib.dump(modelo,   MODELO_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\n[OK] Modelo salvo em '{MODELO_PATH}'")
    print(f"[OK] Encoders salvos em '{ENCODERS_PATH}'")


def carregar_modelo():
    modelo   = joblib.load(MODELO_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("[OK] Modelo e encoders carregados")
    return modelo, encoders


# =============================================================================
# BLOCO 10 — EXECUCAO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  IAM RISK PREDICTOR — Iniciando pipeline")
    print("=" * 60)

    # 1. Gerar dados (troque por carregar_dados("seu_arquivo.csv") quando tiver dados reais)
    df = gerar_dados_sinteticos(n=500)

    # 2. Pre-processar
    X, y, encoders = preprocessar(df)

    # 3. Treinar
    modelo, X_test, y_test = treinar_modelo(X, y)

    # 4. Avaliar
    avaliar_modelo(modelo, X_test, y_test, encoders)

    # 5. Importancia das features
    importancia_features(modelo, X)

    # 6. Salvar
    salvar_modelo(modelo, encoders)

    # 7. Teste com uma nova solicitacao
    print("\n" + "=" * 60)
    print("  TESTE — Nova solicitacao de acesso")
    print("=" * 60)

    nova_solicitacao = {
        "cargo":                 "Estagiario",
        "departamento":          "Financeiro",
        "sistema":               "Core Bancario",
        "tipo_acesso":           "Administrador",
        "criticidade":           "Critica",
        "tempo_empresa_meses":   3,
        "acessos_ativos":        12,
        "aprovacoes_anteriores": 1,
        "revogacoes_anteriores": 0,
        "violacoes_historicas":  0,
        "conflito_sod":          1,
        "conformidade_ok":       0
    }

    prever_risco(modelo, encoders, nova_solicitacao)
