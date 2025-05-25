import pandas as pd
import ast

def eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path, output_id_reviews_path):
    """
    Elimina las columnas especificadas de un archivo CSV y procesa la columna 'reviews' para extraer
    comment_id y comment, guardando el resultado en un nuevo archivo CSV.

    :param csv_path: Ruta del archivo CSV original.
    :param columnas_a_eliminar: Lista de nombres de columnas a eliminar.
    :param output_path: Ruta donde se guardará el nuevo archivo CSV procesado.
    :param output_id_reviews_path: Ruta donde se guardará el archivo con '_id', 'comment_id' y 'comment'.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # Guardar el nuevo archivo CSV con las columnas procesadas
    df.to_csv(output_path, index=False)
    print(f"Archivo guardado en: {output_path}")
    
    # Procesar las reviews para extraer comment_id y comment
    if '_id' in df.columns and 'reviews' in df.columns:
        rows = []
        for _, row in df.iterrows():
            try:
                reviews_list = ast.literal_eval(row['reviews'])
            except (SyntaxError, ValueError):
                continue  # Ignorar filas con reviews mal formadas
            for review in reviews_list:
                comment_id = review.get('_id')
                comment = review.get('comments')
                if comment_id and comment:
                    rows.append({
                        '_id': row['_id'],
                        'comment_id': comment_id,
                        'comment': comment
                    })
        # Crear DataFrame y guardar
        reviews_df = pd.DataFrame(rows)
        reviews_df.to_csv(output_id_reviews_path, index=False)
        print(f"Archivo con '_id', 'comment_id' y 'comment' guardado en: {output_id_reviews_path}")
    else:
        print("Las columnas '_id' o 'reviews' no están presentes en el archivo procesado.")

# Ejemplo de uso
if __name__ == "__main__":
    csv_path = "Datos/spain_traducido.csv"
    columnas_a_eliminar = ["listing_url", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "bed_type", "minimum_nights", "maximum_nights", "cancellation_policy", "accommodates", "bedrooms", "beds", "bathrooms", "amenities", "extra_people", "guests_included", "images", "host", "availability", "weekly_price", "monthly_price", "text_embeddings", "image_embeddings"]
    output_path = "Datos/airbnb_spain_limpio_trans.csv"
    output_id_reviews_path = "Datos/airbnb_spain_id_comments.csv"
    
    eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path, output_id_reviews_path)