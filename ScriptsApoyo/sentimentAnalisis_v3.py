import pandas as pd
from collections import Counter
from textblob import TextBlob
import ast

def analizar_sentimientos_y_frecuencia(input_csv, output_csv):
    """
    Procesa las reviews para generar un archivo CSV con las columnas:
    pais, palabra, frecuencia, sentimiento.

    :param input_csv: Ruta del archivo CSV de entrada con columnas 'reviews'.
    :param output_csv: Ruta del archivo CSV de salida.
    """
    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Añadir la columna 'pais' con el valor 'Portugal'
    df['pais'] = 'Portugal'

    # Verificar que las columnas necesarias existan
    if 'pais' not in df.columns or 'reviews' not in df.columns:
        raise ValueError("El archivo de entrada debe contener las columnas 'pais' y 'reviews'.")

    # Lista para almacenar los resultados
    resultados = []

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
                # Si es una lista de diccionarios
                reviews_list = ast.literal_eval(reviews)
                comentarios = [item['comments'] for item in reviews_list if 'comments' in item]
            else:
                # Si están separadas por '|'
                comentarios = reviews.split('|')
        except Exception as e:
            print(f"Error procesando reviews: {e}")
            continue

        # Dividir los comentarios en palabras y contar la frecuencia
        palabras = []
        for comentario in comentarios:
            palabras.extend(comentario.split())

        contador = Counter(palabras)

        # Analizar el sentimiento de cada palabra
        for palabra, frecuencia in contador.items():
            sentimiento = TextBlob(palabra).sentiment.polarity
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
    # Ruta del archivo CSV de entrada
    input_csv = "airbnb_portugal_id_reviews.csv"
    
    # Ruta del archivo CSV de salida
    output_csv = "analisis_sentimientos.csv"
    
    # Llamar a la función
    analizar_sentimientos_y_frecuencia(input_csv, output_csv)