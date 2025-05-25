import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Ruta del archivo csv sobre el que entrenar el modelo.",type=str,required=True)
parser.add_argument("-m","--model_path",help="Ruta donde almacenar el modelo.",type=str, default="Modelos/")
parser.add_argument("-v","--vectorizer_path",help="Ruta donde almacenar el vectorizer.",type=str, default="Modelos/")
parser.add_argument("-s","--scaler_path",help="Ruta donde almacenar el scaler.",type=str, default="Modelos/")
args = parser.parse_args()

def cargar_datos_tripadvisor(csv_path):
    df = pd.read_csv(csv_path)
    if 'Review' not in df.columns or 'Rating' not in df.columns:
        raise ValueError("El archivo CSV debe contener las columnas 'Review' y 'Rating'.")
    df = df.dropna(subset=['Review', 'Rating'])
    return df

def preprocesar_texto(texto):
    texto = re.sub(r'[^a-zA-Z\s]', '', texto).lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    palabras = texto.split()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras)

def extraer_features_review(review):
    review_length = len(review.split())
    num_exclam = review.count('!')
    num_question = review.count('?')
    num_upper = sum(1 for c in review if c.isupper())
    polarity = TextBlob(review).sentiment.polarity
    char_length = len(review)
    prop_upper = num_upper / max(1, char_length)
    prop_exclam = num_exclam / max(1, review_length)
    prop_question = num_question / max(1, review_length)
    return pd.DataFrame([[review_length, num_exclam, num_question, num_upper, polarity, char_length, prop_upper, prop_exclam, prop_question]],
        columns=['review_length', 'num_exclam', 'num_question', 'num_upper', 'polarity', 'char_length', 'prop_upper', 'prop_exclam', 'prop_question'])

def extraer_features(df):
    tqdm.pandas(desc="Extrayendo features numéricas")
    df['review_length'] = df['Review'].progress_apply(lambda x: len(x.split()))
    df['num_exclam'] = df['Review'].progress_apply(lambda x: x.count('!'))
    df['num_question'] = df['Review'].progress_apply(lambda x: x.count('?'))
    df['num_upper'] = df['Review'].progress_apply(lambda x: sum(1 for c in x if c.isupper()))
    df['polarity'] = df['Review'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
    df['char_length'] = df['Review'].progress_apply(len)
    df['prop_upper'] = df.apply(lambda row: row['num_upper'] / max(1, row['char_length']), axis=1)
    df['prop_exclam'] = df.apply(lambda row: row['num_exclam'] / max(1, row['review_length']), axis=1)
    df['prop_question'] = df.apply(lambda row: row['num_question'] / max(1, row['review_length']), axis=1)
    return df

def entrenar_stacking(csv_path):
    df = cargar_datos_tripadvisor(csv_path)
    df['Review'] = df['Review'].fillna("").apply(preprocesar_texto)
    df = extraer_features(df)
    X = df['Review']
    y = df['Rating'].astype(int) - 1  # Etiquetas de 0 a 4

    # TF-IDF mejorado
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3), stop_words='english')
    X_tfidf = vectorizer.fit_transform(tqdm(X, desc="Vectorizando texto"))

    # Features numéricas mejoradas
    features_cols = ['review_length', 'num_exclam', 'num_question', 'num_upper', 'polarity', 'char_length', 'prop_upper', 'prop_exclam', 'prop_question']
    scaler = StandardScaler()
    features_numericas = scaler.fit_transform(df[features_cols])
    X_final = hstack([X_tfidf, features_numericas])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

    # GridSearch para RandomForest
    print("\nBuscando mejores hiperparámetros para RandomForest...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid_rf, cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    print("Mejores parámetros para RandomForest:", grid_rf.best_params_)

    # GridSearch para LogisticRegression
    print("\nBuscando mejores hiperparámetros para LogisticRegression...")
    param_grid_lr = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs']
    }
    grid_lr = GridSearchCV(LogisticRegression(max_iter=2000), param_grid_lr, cv=3, n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    print("Mejores parámetros para LogisticRegression:", grid_lr.best_params_)

    # Stacking ensemble mejorado
    print("\nEntrenando SOLO modelo StackingClassifier...")
    stacking = StackingClassifier(
        estimators=[
            ('rf', grid_rf.best_estimator_),
            ('lr', grid_lr.best_estimator_)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1,
        cv=5
    )
    stacking.fit(X_train, y_train)

    # Evaluación
    y_pred = stacking.predict(X_test)
    y_pred_original = y_pred + 1
    y_test_original = y_test + 1
    print(f"\nReporte de clasificación para Stacking:")
    print(classification_report(y_test_original, y_pred_original))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test_original, y_pred_original))
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy del modelo Stacking: {accuracy:.4f}")

    return stacking, vectorizer, scaler

def guardar_modelo(model, vectorizer, scaler, model_path, vectorizer_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo guardado en: {model_path}")
    print(f"Vectorizador guardado en: {vectorizer_path}")
    print(f"Scaler guardado en: {scaler_path}")

if __name__ == "__main__":
    csv_path = args.input
    modelo, vectorizador, scaler = entrenar_stacking(csv_path)
    guardar_modelo(modelo, vectorizador, scaler,
                   args.model_path + "modelo_clasificacion.pkl",
                   args.vectorizer_path + "vectorizador_clasificacion.pkl",
                   args.scaler_path + "scaler_clasificacion.pkl")