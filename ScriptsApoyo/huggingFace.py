import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from transformers import pipeline
import torch
from tqdm import tqdm

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

def preprocesar_texto(texto):
    """
    Preprocesa el texto eliminando caracteres especiales, stopwords y aplicando lematización.
    """
    # Eliminar caracteres especiales y convertir a minúsculas
    texto = re.sub(r'[^a-zA-Z\s]', '', texto).lower()

    # Dividir en palabras
    palabras = texto.split()

    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    palabras = [palabra for palabra in palabras if palabra not in stop_words]

    # Aplicar lematización
    lematizador = WordNetLemmatizer()
    palabras = [lematizador.lemmatize(palabra) for palabra in palabras]

    # Unir las palabras procesadas
    return ' '.join(palabras)

def clasificar_comentarios_kmeans(input_csv, output_csv, n_clusters=9):
    """
    Clasifica los comentarios en clusters usando K-Means clustering.

    :param input_csv: Ruta del archivo CSV de entrada con la columna 'Review'.
    :param output_csv: Ruta del archivo CSV de salida con las clasificaciones.
    :param n_clusters: Número de clusters (por defecto 9).
    """
    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Verificar que la columna 'Review' exista
    if 'Review' not in df.columns:
        raise ValueError("El archivo de entrada debe contener la columna 'Review'.")

    # Preprocesar los comentarios
    df['Review_Procesado'] = df['Review'].fillna("").apply(preprocesar_texto)

    # Convertir los comentarios en vectores TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Review_Procesado'])

    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Analizar los clusters
    for cluster in range(n_clusters):
        print(f"Cluster {cluster}:")
        print(df[df['Cluster'] == cluster]['Review'].head(5))  # Muestra 5 comentarios por cluster
        print("\n")

    # Guardar el archivo con las clasificaciones
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

def clasificar_comentarios_huggingface(input_csv, output_csv, batch_size=16):
    """
    Clasifica los comentarios usando un modelo de Hugging Face y los mapea a una escala del 1 al 9.

    :param input_csv: Ruta del archivo CSV de entrada con la columna 'Review'.
    :param output_csv: Ruta del archivo CSV de salida con las clasificaciones.
    :param batch_size: Tamaño del lote para procesar los comentarios.
    """
    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Verificar que la columna 'Review' exista
    if 'Review' not in df.columns:
        raise ValueError("El archivo de entrada debe contener la columna 'Review'.")

    # Inicializar el pipeline de análisis de sentimientos con un modelo preentrenado
    device = 0 if torch.cuda.is_available() else -1  # Usa la GPU si está disponible, de lo contrario usa la CPU
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        device=device
    )

    # Dividir los comentarios en lotes
    comentarios = df['Review'].fillna("").tolist()
    resultados = []

    for i in tqdm(range(0, len(comentarios), batch_size)):
        batch = comentarios[i:i + batch_size]
        batch_resultados = sentiment_pipeline(batch)
        resultados.extend(batch_resultados)

    # Mapear los resultados a una escala del 1 al 9
    def mapear_puntuacion(resultado):
        if resultado['label'] == 'POSITIVE':
            return round(5 + (resultado['score'] * 4))
        else:
            return round(5 - (resultado['score'] * 4))

    df['Predicted_Rating'] = [mapear_puntuacion(r) for r in resultados]

    # Guardar el archivo con las clasificaciones
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo CSV de entrada
    input_csv = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\tripadvisor_hotel_reviews.csv"
    
    # Ruta del archivo CSV de salida para K-Means
    output_csv_kmeans = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\clasificacion_comentarios_kmeans.csv"
    
    # Ruta del archivo CSV de salida para Hugging Face
    output_csv_huggingface = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\clasificacion_comentarios_huggingface.csv"
    
    # Llamar a las funciones
    clasificar_comentarios_kmeans(input_csv, output_csv_kmeans)
    clasificar_comentarios_huggingface(input_csv, output_csv_huggingface, batch_size=16)