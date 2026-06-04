from iam_risk_predictor import (
    gerar_dados_sinteticos,
    preprocessar,
    treinar_modelo,
    avaliar_modelo
)

df = gerar_dados_sinteticos(n=500)
X, y, encoders = preprocessar(df)

modelo, X_test, y_test = treinar_modelo(X, y)

avaliar_modelo(modelo, X_test, y_test, encoders)