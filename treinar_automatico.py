import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    f1_score, fbeta_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")

from iam_risk_predictor import (
    gerar_dados_sinteticos, preprocessar,
    MODELO_PATH, ENCODERS_PATH, TARGET
)


SEEDS    = [42, 123, 456, 789, 2024, 2025, 314, 99, 7, 777]
TAMANHOS = [400, 600, 800, 1000]

MODELOS_CONFIG = [
    {
        "nome": "RandomForest_default",
        "classe": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 3,
                   "class_weight": "balanced", "n_jobs": -1}
    },
    {
        "nome": "RandomForest_deep",
        "classe": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2,
                   "max_features": "sqrt", "class_weight": "balanced", "n_jobs": -1}
    },
    {
        "nome": "RandomForest_wide",
        "classe": RandomForestClassifier,
        "params": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5,
                   "class_weight": "balanced", "n_jobs": -1}
    },
    {
        "nome": "GradientBoosting",
        "classe": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5,
                   "learning_rate": 0.1, "subsample": 0.8}
    },
]

RESULTADOS_DIR = "resultados_treinamento"
HISTORICO_PATH = os.path.join(RESULTADOS_DIR, "historico.json")


def avaliar_completo(modelo, X_test, y_test, encoders):
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    classes = list(encoders[TARGET].classes_)

    acuracia     = accuracy_score(y_test, y_pred)
    bal_acuracia = balanced_accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    f1_macro        = f1_score(y_test, y_pred,        average="macro", zero_division=0)
    f2_macro        = fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0)
    f1_weighted     = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    f2_weighted     = fbeta_score(y_test, y_pred, beta=2, average="weighted", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    por_classe = {}
    for i, cls in enumerate(classes):
        y_bin_true = (y_test == i).astype(int)
        y_bin_pred = (y_pred == i).astype(int)
        por_classe[cls] = {
            "precision": round(precision_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_bin_true, y_bin_pred, zero_division=0), 4),
            "f2":        round(fbeta_score(y_bin_true, y_bin_pred, beta=2, zero_division=0), 4),
        }

    score_composto = (
        f2_macro     * 0.35 +
        auc          * 0.30 +
        f1_macro     * 0.20 +
        bal_acuracia * 0.15
    )
    if "Alto" in por_classe:
        score_composto += por_classe["Alto"]["f2"] * 0.10
        score_composto /= 1.10

    return {
        "acuracia": round(acuracia, 4), "bal_acuracia": round(bal_acuracia, 4),
        "precision_macro": round(precision_macro, 4), "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4), "f2_macro": round(f2_macro, 4),
        "precision_weighted": round(precision_weighted, 4), "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4), "f2_weighted": round(f2_weighted, 4),
        "auc_roc": round(auc, 4),
        "por_classe": por_classe,
        "score_composto": round(score_composto, 4),
        "y_pred": y_pred,
        "y_test": y_test,
    }

def alerta_f2_alto(f2_alto: float) -> str:
    """Classifica o F2 da classe Alto com threshold correto."""
    if f2_alto >= 0.90:
        return "✅ Excelente cobertura da classe Alto"
    elif f2_alto >= 0.80:
        return "⚠  Atencao ao recall da classe Alto"
    else:
        return "🚨 Modelo deixando passar riscos altos"

def imprimir_resultado(m, config_nome, n_amostras, seed, tempo, is_melhor=False):
    prefixo = "  🏆" if is_melhor else "  "
    print(f"\n{prefixo} {'─'*53}")
    print(f"  Modelo:   {config_nome} | n={n_amostras} | seed={seed} | {tempo:.1f}s")
    print(f"  {'─'*53}")
    print(f"  {'Metrica':<22} {'Macro':>8} {'Weighted':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*10}")
    print(f"  {'Accuracy':<22} {m['acuracia']:>8.4f}  (bal) {m['bal_acuracia']:.4f}")
    print(f"  {'Precision':<22} {m['precision_macro']:>8.4f}        {m['precision_weighted']:.4f}")
    print(f"  {'Recall':<22} {m['recall_macro']:>8.4f}        {m['recall_weighted']:.4f}")
    print(f"  {'F1-Score':<22} {m['f1_macro']:>8.4f}        {m['f1_weighted']:.4f}")
    print(f"  {'F2-Score':<22} {m['f2_macro']:>8.4f}        {m['f2_weighted']:.4f}")
    print(f"  {'AUC-ROC':<22} {m['auc_roc']:>8.4f}")
    print(f"  {'Score Composto':<22} {m['score_composto']:>8.4f}  <- criterio de selecao")
    print(f"\n  Por classe:")
    print(f"  {'Classe':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F2':>8}  Status")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*35}")
    for cls, vals in m["por_classe"].items():
        status = alerta_f2_alto(vals["f2"]) if cls == "Alto" else ""
        print(f"  {cls:<10} {vals['precision']:>10.4f} {vals['recall']:>8.4f} "
              f"{vals['f1']:>8.4f} {vals['f2']:>8.4f}  {status}")

def imprimir_matriz_confusao(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  MATRIZ DE CONFUSAO")
    print(f"  {'':>10}", end="")
    for c in classes:
        print(f"  {'Prev_'+c:>10}", end="")
    print()
    for i, cls in enumerate(classes):
        print(f"  {'Real_'+cls:>10}", end="")
        for j in range(len(classes)):
            marker = " ←OK" if i == j else ""
            print(f"  {cm[i][j]:>10}{marker[:3]}", end="")
        print()
    total_erros = cm.sum() - np.diag(cm).sum()
    print(f"\n  Total de erros: {total_erros} / {cm.sum()}")
    # Erros criticos: Alto predito como Baixo
    idx_alto  = list(classes).index("Alto")  if "Alto"  in classes else -1
    idx_baixo = list(classes).index("Baixo") if "Baixo" in classes else -1
    if idx_alto >= 0 and idx_baixo >= 0:
        falsos_neg = cm[idx_alto][idx_baixo]
        print(f"  Erros criticos (Alto->Baixo): {falsos_neg}  <- risco nao detectado")

def imprimir_importancia_features(modelo, feature_names, classes):
    if not hasattr(modelo, "feature_importances_"):
        return
    imp = pd.Series(modelo.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    print(f"\n  IMPORTANCIA DAS FEATURES")
    print(f"  {'Feature':<28} {'Importancia':>12}  Barra")
    print(f"  {'─'*28} {'─'*12}  {'─'*25}")
    for feat, val in imp.items():
        barra = "█" * int(val * 50)
        print(f"  {feat:<28} {val:>12.4f}  {barra}")

def barra_progresso(atual, total, largura=40):
    pct   = atual / total
    cheio = int(pct * largura)
    barra = "█" * cheio + "░" * (largura - cheio)
    return f"[{barra}] {atual}/{total} ({pct*100:.0f}%)"


def rodar_experimento(config, seed, n_amostras):
    df = gerar_dados_sinteticos(n=n_amostras, seed=seed)
    X, y, encoders = preprocessar(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    params = {**config["params"], "random_state": seed}
    if config["classe"] == GradientBoostingClassifier:
        params.pop("n_jobs", None)
    modelo = config["classe"](**params)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo  = time.time() - inicio
    metricas = avaliar_completo(modelo, X_test, y_test, encoders)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1_macro")
    metricas["cv_f1_mean"] = round(cv_scores.mean(), 4)
    metricas["cv_f1_std"]  = round(cv_scores.std(), 4)
    feature_names = list(X.columns)
    return modelo, encoders, metricas, tempo, feature_names, X_train


def main():
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTADOS_DIR, "modelos"), exist_ok=True)

    historico = {}
    if os.path.exists(HISTORICO_PATH):
        with open(HISTORICO_PATH) as f:
            historico = json.load(f)

    print("=" * 60)
    print("  IAM RISK PREDICTOR — Automacao de Treinamentos v2")
    print("=" * 60)
    total = len(SEEDS) * len(TAMANHOS) * len(MODELOS_CONFIG)
    print(f"  Total experimentos: {total}")
    print("=" * 60)

    melhor_score    = historico.get("melhor", {}).get("score_composto", 0) if historico else 0
    melhor_modelo   = None
    melhor_encoders = None
    melhor_config   = None
    melhor_features = None
    contador = 0
    novos    = []

    for config in MODELOS_CONFIG:
        for n_amostras in TAMANHOS:
            for seed in SEEDS:
                contador += 1
                print(f"\n{barra_progresso(contador, total)}")
                print(f"  -> {config['nome']} | n={n_amostras} | seed={seed}")
                try:
                    modelo, encoders, metricas, tempo, feat_names, _ = rodar_experimento(
                        config, seed, n_amostras
                    )
                    metricas["modelo_nome"] = config["nome"]
                    metricas["n_amostras"]  = n_amostras
                    metricas["timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    is_melhor = metricas["score_composto"] > melhor_score
                    imprimir_resultado(metricas, config["nome"], n_amostras, seed, tempo, is_melhor)
                    novos.append(metricas)

                    if is_melhor:
                        melhor_score    = metricas["score_composto"]
                        melhor_modelo   = modelo
                        melhor_encoders = encoders
                        melhor_config   = metricas.copy()
                        melhor_features = feat_names
                        ckpt = os.path.join(RESULTADOS_DIR, "modelos",
                                            f"ckpt_{config['nome']}_n{n_amostras}_s{seed}.pkl")
                        joblib.dump(modelo, ckpt)
                except Exception as e:
                    print(f"  Erro: {e}")
                    continue

    print("\n" + "=" * 60)
    print("  RANKING FINAL — TOP 10")
    print("=" * 60)

    if novos:
        rows = []
        for r in novos:
            row = {k: v for k, v in r.items() if k not in ("por_classe","y_pred","y_test")}
            for cls, vals in r.get("por_classe", {}).items():
                for met, val in vals.items():
                    row[f"{cls}_{met}"] = val
            rows.append(row)

        df_res = pd.DataFrame(rows).sort_values("score_composto", ascending=False)
        print(f"\n  {'Modelo':<26} {'n':>6} {'F2':>7} {'F1':>7} {'AUC':>7} {'Score':>7}")
        print(f"  {'─'*26} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for _, row in df_res.head(10).iterrows():
            print(f"  {row['modelo_nome']:<26} {int(row['n_amostras']):>6} "
                  f"{row['f2_macro']:>7.4f} {row['f1_macro']:>7.4f} "
                  f"{row['auc_roc']:>7.4f} {row['score_composto']:>7.4f}")

        csv_path = os.path.join(RESULTADOS_DIR,
                                f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\n  CSV salvo: {csv_path}")

    # ── Melhor Modelo — Analise Completa ───────────────────────────
    if melhor_modelo is not None:
        joblib.dump(melhor_modelo,   MODELO_PATH)
        joblib.dump(melhor_encoders, ENCODERS_PATH)

        classes = list(melhor_encoders[TARGET].classes_)

        print("\n" + "=" * 60)
        print("  ANALISE COMPLETA DO MELHOR MODELO")
        print("=" * 60)
        print(f"  Modelo:    {melhor_config['modelo_nome']}")
        print(f"  Amostras:  {melhor_config['n_amostras']} | Seed: {melhor_config['seed']}")
        print(f"  F2-Macro:  {melhor_config['f2_macro']}")
        print(f"  F1-Macro:  {melhor_config['f1_macro']}")
        print(f"  AUC-ROC:   {melhor_config['auc_roc']}")
        print(f"  Score:     {melhor_config['score_composto']}")

        if "Alto" in melhor_config.get("por_classe", {}):
            f2_alto = melhor_config["por_classe"]["Alto"]["f2"]
            print(f"\n  F2 Alto Risco: {f2_alto}  {alerta_f2_alto(f2_alto)}")

        # Matriz de confusao
        imprimir_matriz_confusao(
            melhor_config["y_test"],
            melhor_config["y_pred"],
            classes
        )

        # Importancia das features
        if melhor_features:
            imprimir_importancia_features(melhor_modelo, melhor_features, classes)

        historico["melhor"] = {k: v for k, v in melhor_config.items()
                                if k not in ("y_pred", "y_test", "por_classe")}
        historico["melhor"]["por_classe"] = melhor_config.get("por_classe", {})
        historico.setdefault("experimentos", [])
        historico["experimentos"].extend(
            [{k: v for k, v in r.items() if k not in ("y_pred","y_test")} for r in novos]
        )
        with open(HISTORICO_PATH, "w") as f:
            json.dump(historico, f, indent=2)

        print(f"\n  Modelo salvo: {MODELO_PATH}")

    print("\n" + "=" * 60)
    print("  Automacao concluida!")
    print("=" * 60)


if __name__ == "__main__":
    main()