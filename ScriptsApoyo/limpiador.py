import pandas as pd
import ast
import re

def eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path):
    """
    Elimina las columnas especificadas de un archivo CSV, procesa la columna 'reviews' para quedarse
    con los comentarios y fechas en forma de lista asociada, y extrae la primera palabra después de 'street' en la columna 'address',
    guardando el resultado en un nuevo archivo.

    :param csv_path: Ruta del archivo CSV original.
    :param columnas_a_eliminar: Lista de nombres de columnas a eliminar.
    :param output_path: Ruta donde se guardará el nuevo archivo CSV.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # Procesar la columna 'reviews' para asociar comentarios con fechas
    if 'reviews' in df.columns:
        def procesar_reviews(review):
            try:
                # Intentar interpretar listas o diccionarios
                parsed_review = ast.literal_eval(review) if isinstance(review, str) and review.startswith(('{', '[')) else review
                
                # Asociar comentarios con fechas
                if isinstance(parsed_review, list):
                    comments = [
                        str(item.get('comments', '')) 
                        for item in parsed_review 
                        if isinstance(item, dict) and 'comments' in item
                    ]
                    dates = [
                        str(item.get('date', '')) 
                        for item in parsed_review 
                        if isinstance(item, dict) and 'date' in item
                    ]
                    # Asociar cada comentario con su fecha correspondiente
                    return [{"comment": c, "date": d} for c, d in zip(comments, dates)]
                else:
                    return parsed_review  # Devolver el valor original si no es una lista
            except Exception as e:
                print(f"Error procesando review: {e}")
                return review  # Devolver el valor original si hay un error
        
        df['reviews'] = df['reviews'].apply(procesar_reviews)
    
    # Procesar la columna 'address' para extraer la primera palabra después de 'street'
    if 'address' in df.columns:
        def extraer_palabra_despues_de_street(address):
            try:
                # Intentar interpretar el contenido como un diccionario
                parsed_address = ast.literal_eval(address) if isinstance(address, str) and address.startswith('{') else address
                if isinstance(parsed_address, dict) and 'street' in parsed_address:
                    # Extraer el valor de 'street'
                    street_value = parsed_address['street']
                    # Dividir el valor y obtener la primera palabra
                    return street_value.split(',')[0].strip()
                return None
            except Exception as e:
                print(f"Error procesando address: {e}")
                return None
        
        df['street'] = df['address'].apply(extraer_palabra_despues_de_street)
    
    # Guardar el nuevo archivo CSV
    df.to_csv(output_path, index=False)
    print(f"Archivo guardado en: {output_path}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo CSV original
    csv_path = "airbnb_portugal.csv"
    
    # Columnas a eliminar
    columnas_a_eliminar = ["listing_url", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "bed_type", "minimum_nights", "maximum_nights", "cancellation_policy", "accommodates", "bedrooms", "beds", "bathrooms", "amenities", "extra_people", "guests_included", "images", "host", "availability", "weekly_price", "monthly_price", "text_embeddings", "image_embeddings"]
    
    # Ruta del archivo CSV de salida
    output_path = "airbnb_portugal_limpio.csv"
    
    # Llamar a la función
    eliminar_columnas_y_procesar_reviews(csv_path, columnas_a_eliminar, output_path)