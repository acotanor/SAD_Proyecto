import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Ruta del archivo csv sobre el que entrenar el modelo.",type=str)
parser.add_argument("-o","--output",help="Ruta del archivo csv sobre el que volcar los resultados.",type=str)
parser.add_argument("--knn_model",help="Ruta del modelo knn.",type=str, default="Modelos/modelo_knn.pkl")
parser.add_argument("--knn_vectorizer",help="Ruta del vectorizer.",type=str, default="Modelos/vectorizador_tfidf.pkl")
parser.add_argument("--bert_model",help="Ruta del modelo bert.",type=str, default="Modelos/modelo_bert_multiclase")
parser.add_argument("--bert_tokenizer",help="Ruta del tokenizer.",type=str, default="Modelos/modelo_bert_multiclase")
args = parser.parse_args()

# Cargar modelos y vectorizadores
knn_model = joblib.load(args.knn_model)
knn_vectorizer = joblib.load(args.knn_vectorizer)
bert_model = BertForSequenceClassification.from_pretrained(args.bert_model)
bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
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
    for review in tqdm(df['Review'].values.astype('U'), desc="Procesando reviews"):
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
    if args.input != None and args.output != None :
        predecir_reviews_csv(args.input, args.output)
    else:
        review = input("Introduce una review: ")
        predecir_review(review)