import pandas as pd
import ast
import re

def eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path, output_id_reviews_path):
    """
    Elimina las columnas especificadas de un archivo CSV, procesa la columna 'reviews' para quedarse
    con los comentarios y fechas en forma de lista asociada, y extrae la primera palabra después de 'street' en la columna 'address',
    guardando el resultado en un nuevo archivo. Además, crea un archivo con solo las columnas '_id' y 'reviews'.

    :param csv_path: Ruta del archivo CSV original.
    :param columnas_a_eliminar: Lista de nombres de columnas a eliminar.
    :param output_path: Ruta donde se guardará el nuevo archivo CSV.
    :param output_id_reviews_path: Ruta donde se guardará el archivo con '_id' y 'reviews'.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # Guardar el nuevo archivo CSV con las columnas procesadas
    df.to_csv(output_path, index=False)
    print(f"Archivo guardado en: {output_path}")
    
    # Crear un nuevo DataFrame con solo '_id' y 'reviews'
    if '_id' in df.columns and 'reviews' in df.columns:
        id_reviews_df = df[['_id', 'reviews']].copy()
        id_reviews_df.to_csv(output_id_reviews_path, index=False)
        print(f"Archivo con '_id' y 'reviews' guardado en: {output_id_reviews_path}")
    else:
        print("Las columnas '_id' o 'reviews' no están presentes en el archivo procesado.")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo CSV original
    csv_path = "comentarios_traducidos_Portugal.csv"
    
    # Columnas a eliminar
    columnas_a_eliminar = ["listing_url", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "bed_type", "minimum_nights", "maximum_nights", "cancellation_policy", "accommodates", "bedrooms", "beds", "bathrooms", "amenities", "extra_people", "guests_included", "images", "host", "availability", "weekly_price", "monthly_price", "text_embeddings", "image_embeddings"]
    
    # Ruta del archivo CSV de salida
    output_path = "airbnb_portugal_limpio.csv"
    
    # Ruta del archivo CSV con '_id' y 'reviews'
    output_id_reviews_path = "airbnb_portugal_id_reviews.csv"
    
    # Llamar a la función
    eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path, output_id_reviews_path)