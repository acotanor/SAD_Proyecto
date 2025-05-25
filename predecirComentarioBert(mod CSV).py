import joblib
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Cargar modelos y vectorizadores
knn_model = joblib.load("modelo_knn.pkl")
knn_vectorizer = joblib.load("vectorizador_tfidf.pkl")
bert_model = BertForSequenceClassification.from_pretrained("modelo_bert_multiclase")
bert_tokenizer = BertTokenizer.from_pretrained("modelo_bert_multiclase")
bert_model.eval()

def predecir_sentimiento(review):
    X_tfidf = knn_vectorizer.transform([review])
    pred = knn_model.predict(X_tfidf)[0]
    if pred == 1:
        return "Positive"
    else:
        return "Negative"

def predecir_score_bert(review):
    inputs = bert_tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return pred + 1  # 1-5

def score_final(sentimiento, score_bert):
    if sentimiento.lower() == "negative":
        return score_bert  # 1-5
    else:
        return score_bert + 4  # 5-9

def predecir_review(review):
    sentimiento = predecir_sentimiento(review)
    score_bert = predecir_score_bert(review)
    puntuacion = score_final(sentimiento, score_bert)
    print(f"Sentimiento: {sentimiento}")
    print(f"Score BERT (1-5): {score_bert}")
    print(f"Puntuación final (1-9): {puntuacion}")
    return sentimiento, score_bert, puntuacion

def predecir_reviews_csv(input_csv, output_csv, columna_review='review'):
    df = pd.read_csv(input_csv)

    if columna_review not in df.columns:
        raise ValueError(f"El archivo debe tener una columna '{columna_review}'")

    # Añadir columnas de resultados directamente al DataFrame original
    sentimientos = []
    scores_bert = []
    puntuaciones_finales = []

    for r in tqdm(df[columna_review], desc="Procesando reviews"):
        # Saltar si la review está vacía o es NaN
        if pd.isna(r) or not str(r).strip():
            sentimientos.append(None)
            scores_bert.append(None)
            puntuaciones_finales.append(None)
            continue

        try:
            sentimiento = predecir_sentimiento(r)
            score_bert = predecir_score_bert(r)
            puntuacion = score_final(sentimiento, score_bert)
        except Exception as e:
            print(f"Error procesando review: {r}\n{e}")
            sentimiento = None
            score_bert = None
            puntuacion = None

        sentimientos.append(sentimiento)
        scores_bert.append(score_bert)
        puntuaciones_finales.append(puntuacion)

    df['Sentimiento'] = sentimientos
    df['Score_BERT_1_5'] = scores_bert
    df['Puntuacion_Final_1_9'] = puntuaciones_finales

    df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {output_csv}")



if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        predecir_reviews_csv(input_csv, output_csv)
    else:
        review = input("Introduce una review: ")
        predecir_review(review)