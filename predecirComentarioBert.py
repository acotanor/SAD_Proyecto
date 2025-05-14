import joblib
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Cargar modelos y vectorizadores
knn_model = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\modelo_knn.pkl")
knn_vectorizer = joblib.load("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\vectorizador_tfidf.pkl")
bert_model = BertForSequenceClassification.from_pretrained("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\ScriptsApoyo\\modelo_bert_multiclase")
bert_tokenizer = BertTokenizer.from_pretrained("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\ScriptsApoyo\\modelo_bert_multiclase")
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

def predecir_reviews_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if 'Review' not in df.columns:
        raise ValueError("El archivo debe tener una columna 'Review'")
    resultados = []
    for review in tqdm(df['Review'], desc="Procesando reviews"):
        sentimiento = predecir_sentimiento(review)
        score_bert = predecir_score_bert(review)
        puntuacion = score_final(sentimiento, score_bert)
        resultados.append({
            'Review': review,
            'Sentimiento': sentimiento,
            'Score_BERT_1_5': score_bert,
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