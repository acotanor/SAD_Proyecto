import joblib
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from scipy.sparse import hstack
from textblob import TextBlob

# Cargar modelos y vectorizadores
knn_model = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\modelo_knn.pkl")
knn_vectorizer = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\vectorizador_tfidf.pkl")
stacking_model = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\modelo_clasificacion.pkl")
stacking_vectorizer = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\vectorizador_clasificacion.pkl")
stacking_scaler = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\scaler_clasificacion.pkl")

def predecir_sentimiento(review):
    X_tfidf = knn_vectorizer.transform([review])
    pred = knn_model.predict(X_tfidf)[0]
    if pred == 1:
        return "Positive"
    else:
        return "Negative"

def extraer_features_review(review):
    review_length = len(review.split())
    num_exclam = review.count('!')
    num_question = review.count('?')
    num_upper = sum(1 for c in review if c.isupper())
    polarity = TextBlob(review).sentiment.polarity
    return pd.DataFrame([[review_length, num_exclam, num_question, num_upper, polarity]],
                        columns=['review_length', 'num_exclam', 'num_question', 'num_upper', 'polarity'])

def predecir_score_stacking(review):
    X_vec = stacking_vectorizer.transform([review])
    features = extraer_features_review(review)
    features_scaled = stacking_scaler.transform(features)
    X_final = hstack([X_vec, features_scaled])
    pred = stacking_model.predict(X_final)[0]
    return int(pred) + 1  # Si entrenaste con etiquetas 0-4

def score_final(sentimiento, score_stacking):
    if sentimiento.lower() == "negative":
        return score_stacking  # 1-5
    else:
        return score_stacking + 4  # 5-9

def predecir_review(review):
    sentimiento = predecir_sentimiento(review)
    score_stacking = predecir_score_stacking(review)
    puntuacion = score_final(sentimiento, score_stacking)
    print(f"Sentimiento: {sentimiento}")
    print(f"Score Stacking (1-5): {score_stacking}")
    print(f"Puntuación final (1-9): {puntuacion}")
    return sentimiento, score_stacking, puntuacion

def predecir_reviews_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if 'Review' not in df.columns:
        raise ValueError("El archivo debe tener una columna 'Review'")
    resultados = []
    for review in tqdm(df['Review'], desc="Procesando reviews"):
        sentimiento = predecir_sentimiento(review)
        score_stacking = predecir_score_stacking(review)
        puntuacion = score_final(sentimiento, score_stacking)
        resultados.append({
            'Review': review,
            'Sentimiento': sentimiento,
            'Score_Stacking_1_5': score_stacking,
            'Puntuacion_Final_1_9': puntuacion
        })
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        predecir_reviews_csv(input_csv, output_csv)
    else:
        review = input("Introduce una review: ")
        predecir_review(review)