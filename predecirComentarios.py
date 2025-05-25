import joblib
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from scipy.sparse import hstack
from textblob import TextBlob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Ruta del archivo csv sobre el que entrenar el modelo.",type=str)
parser.add_argument("-o","--output",help="Ruta del archivo csv sobre el que volcar los resultados.",type=str)
parser.add_argument("--knn_model",help="Ruta del modelo knn.",type=str, default="Modelos/modelo_knn.pkl")
parser.add_argument("--knn_vectorizer",help="Ruta del knn vectorizer.",type=str, default="Modelos/vectorizador_tfidf.pkl")
parser.add_argument("--stacking_model",help="Ruta del modelo stacking.",type=str, default="Modelos/modelo_clasificacion.pkl")
parser.add_argument("--stacking_scaler",help="Ruta del modelo stacking.",type=str, default="Modelos/scaler_clasificacion.pkl")
parser.add_argument("--stacking_vectorizer",help="Ruta del stacking vectorizer.",type=str, default="Modelos/vectorizador_clasificacion.pkl")
args = parser.parse_args()


# Cargar modelos y vectorizadores
knn_model = joblib.load(args.knn_model)
knn_vectorizer = joblib.load(args.knn_vectorizer)
stacking_model = joblib.load(args.stacking_model)
stacking_vectorizer = joblib.load(args.stacking_vectorizer)
stacking_scaler = joblib.load(args.stacking_scaler)

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
    for review in tqdm(df['Review'].values.astype('U'), desc="Procesando reviews"):
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
    if args.input != None and args.output != None :
        predecir_reviews_csv(args.input, args.output)
    else:
        review = input("Introduce una review: ")
        predecir_review(review)