import pandas as pd
from collections import Counter
import ast
import string
from nltk.sentiment import SentimentIntensityAnalyzer

def analizar_sentimientos_y_frecuencia(input_csv, output_csv):
    """
    Genera un CSV con las columnas: pais, palabra, frecuencia_total, sentimiento.
    Solo incluye palabras con sentimiento positivo o negativo (según VADER).
    """
    # Inicializar VADER
    sia = SentimentIntensityAnalyzer()

    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Verificar que las columnas necesarias existan
    if 'pais' not in df.columns or 'reviews' not in df.columns:
        raise ValueError("El archivo de entrada debe contener las columnas 'pais' y 'reviews'.")

    # Diccionario para acumular palabras por país
    pais_palabras = {}

    # Procesar cada fila
    for _, row in df.iterrows():
        pais = row['pais']
        reviews = row['reviews']

        if pd.isna(reviews):
            continue

        # Procesar el formato de las reviews
        comentarios = []
        try:
            if reviews.startswith("[") and reviews.endswith("]"):
                reviews_list = ast.literal_eval(reviews)
                comentarios = [item['comments'] for item in reviews_list if 'comments' in item]
            else:
                comentarios = reviews.split('|')
        except Exception as e:
            print(f"Error procesando reviews: {e}")
            continue

        # Dividir los comentarios en palabras
        palabras = []
        for comentario in comentarios:
            for palabra in comentario.split():
                palabra_limpia = palabra.strip(string.punctuation).lower()
                if palabra_limpia:  # Evita palabras vacías
                    palabras.append(palabra_limpia)

        # Acumular palabras por país
        if pais not in pais_palabras:
            pais_palabras[pais] = []
        pais_palabras[pais].extend(palabras)

    # Ahora contamos globalmente por país
    resultados = []
    for pais, palabras in pais_palabras.items():
        contador = Counter(palabras)
        for palabra, frecuencia in contador.items():
            vader_score = sia.polarity_scores(palabra)['compound']
            if vader_score > 0:
                sentimiento = "positivo"
            elif vader_score < 0:
                sentimiento = "negativo"
            else:
                continue  # Saltar palabras neutras
            resultados.append({
                'pais': pais,
                'palabra': palabra,
                'frecuencia': frecuencia,
                'sentimiento': sentimiento
            })

    # Crear un DataFrame con los resultados
    resultados_df = pd.DataFrame(resultados)

    # Guardar el DataFrame en un archivo CSV
    resultados_df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

# Ejemplo de uso
if __name__ == "__main__":
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    input_csv = "portugal_id_reviews.csv"
    output_csv = "portugal_palabras_analisis_sentimientos.csv"
    analizar_sentimientos_y_frecuencia(input_csv, output_csv)