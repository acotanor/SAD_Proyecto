import pandas as pd
from textblob import TextBlob

def analyze_sentiment(review):
    """
    Analiza el sentimiento de una reseña y clasifica como positiva, neutral o negativa.
    También devuelve la polaridad de la reseña.
    """
    if not review or not isinstance(review, str) or review.strip() == "":
        return None, None  # Ignorar reseñas vacías devolviendo None para sentimiento y polaridad
    
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "positive", polarity
    elif polarity < 0:
        return "negative", polarity
    else:
        return "neutral", polarity

def determine_overall_sentiment(sentiments):
    """
    Determina el sentimiento general basado en la mayoría de los sentimientos.
    """
    if not sentiments:
        return "neutral"
    
    sentiment_counts = pd.Series(sentiments).value_counts()
    return sentiment_counts.idxmax()  # Devuelve el sentimiento con mayor frecuencia

def process_reviews(input_csv, output_csv, review_column):
    """
    Procesa las reseñas de un archivo CSV, analiza su sentimiento y calcula el sentimiento general.
    También incluye la polaridad de cada reseña en el archivo de salida.
    """
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)
    
    if review_column not in df.columns:
        raise ValueError(f"La columna '{review_column}' no existe en el archivo CSV.")
    
    # Filtrar filas con reseñas vacías
    df = df[df[review_column].notna() & (df[review_column].str.strip() != "")]
    
    # Crear una lista para almacenar los resultados
    results = []
    
    # Procesar cada fila
    for _, row in df.iterrows():
        review = row[review_column]
        
        # Dividir las reseñas concatenadas en listas de reseñas individuales
        separated_reviews = review.split("\n")  # Dividir por saltos de línea
        separated_reviews = [r.strip() for r in separated_reviews if r.strip()]  # Eliminar reseñas vacías
        
        # Analizar el sentimiento y la polaridad de cada reseña
        sentiments = []
        polarities = []
        for review in separated_reviews:
            sentiment, polarity = analyze_sentiment(review)
            sentiments.append(sentiment)
            polarities.append(polarity)
        
        # Determinar el sentimiento general
        overall_sentiment = determine_overall_sentiment(sentiments)
        
        # Agregar los resultados
        for review, sentiment, polarity in zip(separated_reviews, sentiments, polarities):
            results.append({
                review_column: review,
                "sentiment": sentiment,
                "polarity": polarity,
                "overall_sentiment": overall_sentiment
            })
    
    # Crear un nuevo DataFrame con los resultados
    results_df = pd.DataFrame(results)
    
    # Guardar los resultados en un nuevo archivo CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Análisis completado. Resultados guardados en '{output_csv}'.")

# Ejemplo de uso
if __name__ == "__main__":
    input_csv = "AirBNBReviews.csv"  # Cambia esto por el nombre de tu archivo CSV de entrada
    output_csv = "reviews_with_sentiment.csv"
    review_column = "Review"  # Cambia esto por el nombre de la columna que contiene las reseñas
    process_reviews(input_csv, output_csv, review_column)