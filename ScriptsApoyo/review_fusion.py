import pandas as pd
from sklearn.utils import shuffle


def cargar_y_limpiar_dataset_binario(ruta_csv):
    """Carga el dataset binario (0/1) manejando nombres de columnas variables."""
    df = pd.read_csv(ruta_csv, encoding='utf-8', sep=None, engine='python')

    # Buscar automáticamente la columna de review (acepta varios nombres)
    review_col = next((col for col in df.columns
                       if col.lower() in ['review', 'texto', 'comentario', 'text', 'comment']), None)

    # Buscar automáticamente la columna binaria (acepta varios nombres)
    sentiment_col = next((col for col in df.columns
                          if col.lower() in ['positive or negative', 'sentimiento', 'label', 'binary', 'class']), None)

    if not review_col or not sentiment_col:
        raise ValueError("No se encontraron las columnas requeridas (review y sentimiento)")

    # Limpieza
    df = df.dropna(subset=[review_col, sentiment_col])
    df[sentiment_col] = df[sentiment_col].apply(lambda x: 0 if str(x).strip() in ['0', 'negative', 'neg'] else 1)

    return df.rename(columns={review_col: 'Review', sentiment_col: 'Positive or Negative'})[
        ['Review', 'Positive or Negative']]


def cargar_y_limpiar_dataset_nota(ruta_csv):
    """Carga el dataset de notas (1-5) buscando automáticamente la columna 'Rating' o similares."""
    df = pd.read_csv(ruta_csv, encoding='utf-8', sep=None, engine='python')

    # Buscar columna de review
    review_col = next((col for col in df.columns
                       if col.lower() in ['review', 'texto', 'comentario', 'text', 'comment']), None)

    # Buscar específicamente "Rating" u otros nombres comunes para puntuaciones
    rating_col = next((col for col in df.columns
                       if col.lower() in ['rating', 'nota', 'score', 'puntuacion', 'stars']), None)

    if not review_col or not rating_col:
        raise ValueError("No se encontraron las columnas requeridas (review y rating/nota)")

    # Limpieza
    df = df.dropna(subset=[review_col, rating_col])
    df = df[~df[rating_col].astype(str).str.strip().isin(['3'])]  # Filtrar neutrales
    df['Positive or Negative'] = df[rating_col].apply(lambda x: 0 if float(x) <= 2 else 1)

    return df.rename(columns={review_col: 'Review'})[['Review', 'Positive or Negative']]


def fusionar_datasets(ruta_binario, ruta_nota, ruta_salida):
    """Fusión robusta con manejo de múltiples formatos de CSV."""
    try:
        print("🔍 Procesando datasets...")
        df_bin = cargar_y_limpiar_dataset_binario(ruta_binario)
        df_nota = cargar_y_limpiar_dataset_nota(ruta_nota)

        print(f"\n📊 Datasets cargados:")
        print(f"- Dataset binario: {len(df_bin):,} reviews")
        print(f"- Dataset de notas: {len(df_nota):,} reviews")

        df_final = shuffle(pd.concat([df_bin, df_nota], ignore_index=True), random_state=42)
        df_final.to_csv(ruta_salida, sep=';', index=False, encoding='utf-8')

        print(f"\n✅ Fusionado exitoso. Datos guardados en: '{ruta_salida}'")
        print(f"📝 Total de reviews: {len(df_final):,}")
        print("⚙️ Configuración aplicada:")
        print("- Encoding: UTF-8")
        print("- Delimitador: ;")
        print("- Columnas estandarizadas: 'Review' y 'Positive or Negative'")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Posibles soluciones:")
        print("1. Verifica los nombres de las columnas en tus archivos CSV")
        print("2. Asegúrate de que los archivos existan en las rutas especificadas")
        print("3. Si hay caracteres especiales, guarda los CSVs originales en UTF-8")


if __name__ == "__main__":
    fusionar_datasets(
        ruta_binario='AirBNBReviews.csv',
        ruta_nota='tripadvisor_hotel_reviews.csv',
        ruta_salida='reviews_fusionadas.csv'
    )